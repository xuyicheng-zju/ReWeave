# Modified for the ReWeave namespace.
"""Bidirectional requests on one private SSH stdio stream; no reconnect/replay.

Frames have an integer request id, method and params, or the matching id and
result/error. The two endpoints own separate id namespaces. EOF fails every
pending request and cancels its handlers, including remote physical work.
Caller cancellation retains its issued id until one late reply is drained;
waiting for the write lock does not issue a request. Cancellation inside an
incomplete frame closes the connection rather than corrupting another frame.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from typing import Any


class SshChannel:
    """One connection with complete JSON frames and concurrent reverse requests."""

    def __init__(
        self,
        reader,
        writer,
        handler: Callable[[str, dict], Awaitable[Any]],
        *,
        write_chunk_bytes: int = 65536,
        write_interval_s: float = 0.0,
    ):
        if type(write_chunk_bytes) is not int or write_chunk_bytes <= 0:
            raise ValueError("SSH write_chunk_bytes must be a positive integer")
        if type(write_interval_s) not in (int, float):
            raise ValueError("SSH write_interval_s must be finite and nonnegative")
        try:
            interval = float(write_interval_s)
        except OverflowError as error:
            raise ValueError(
                "SSH write_interval_s must be finite and nonnegative"
            ) from error
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("SSH write_interval_s must be finite and nonnegative")
        self.reader = reader
        self.writer = writer
        self.handler = handler
        self.write_chunk_bytes = write_chunk_bytes
        self.write_interval_s = interval
        self.pending: dict[int, asyncio.Future] = {}
        self.handlers: set[asyncio.Task] = set()
        self.sequence = 0
        self.closed = False
        self.write_lock = asyncio.Lock()
        self._pump: asyncio.Task | None = None
        self._failure: BaseException | None = None

    def _abort_write(self, error: BaseException) -> None:
        self.closed = True
        if self._failure is None:
            self._failure = error
        try:
            self.writer.close()
        except Exception as close_error:
            error.add_note(f"SSH writer cleanup failed: {close_error!r}")
        if self._pump is not None:
            self._pump.cancel()

    async def _write_frame(self, raw: bytes) -> None:
        """Write one encoded frame while the caller holds write_lock."""
        if self.closed:
            raise ConnectionError(
                "SSH author connection is closed; explicit recovery required"
            )
        written = 0
        try:
            for offset in range(0, len(raw), self.write_chunk_bytes):
                chunk = raw[offset : offset + self.write_chunk_bytes]
                self.writer.write(chunk)
                written += len(chunk)
                await self.writer.drain()
                if written < len(raw) and self.write_interval_s:
                    await asyncio.sleep(self.write_interval_s)
        except asyncio.CancelledError:
            if 0 < written < len(raw):
                self._abort_write(
                    ConnectionError(
                        "SSH frame transmission was cancelled before completion; "
                        "work was not replayed"
                    )
                )
            raise
        except Exception as error:
            self._abort_write(error)
            raise

    async def send(self, message: dict) -> None:
        raw = json.dumps(message, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        async with self.write_lock:
            await self._write_frame(raw)

    async def call(self, method: str, params: dict) -> Any:
        future = None
        try:
            async with self.write_lock:
                if self.closed:
                    raise ConnectionError(
                        "SSH author connection is closed; explicit recovery required"
                    )
                identity = self.sequence + 1
                raw = (
                    json.dumps(
                        {"id": identity, "method": method, "params": params},
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode()
                    + b"\n"
                )
                self.sequence = identity
                future = asyncio.get_running_loop().create_future()
                self.pending[identity] = future
                await self._write_frame(raw)
            return await future
        except asyncio.CancelledError:
            if future is not None:
                future.cancel()
            raise
        finally:
            # The peer can still settle a request whose caller reached its deadline.
            if future is not None and not future.cancelled():
                self.pending.pop(identity, None)

    async def _dispatch(self, frame: dict) -> None:
        try:
            result = await self.handler(frame["method"], frame["params"])
        except Exception as error:
            await self.send(
                {
                    "id": frame["id"],
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        else:
            await self.send({"id": frame["id"], "result": result})

    async def run(self) -> None:
        buffer = bytearray()
        last_request = 0
        pump = asyncio.current_task()
        self._pump = pump

        def finished(task):
            self.handlers.discard(task)
            if not task.cancelled() and (error := task.exception()) is not None:
                if self._failure is None:
                    self._failure = error
                pump.cancel()

        try:
            while chunk := await self.reader.read(65536):
                buffer.extend(chunk)
                while (end := buffer.find(b"\n")) >= 0:
                    raw = bytes(buffer[:end])
                    del buffer[: end + 1]
                    frame = json.loads(raw)
                    if not isinstance(frame, dict) or type(frame.get("id")) is not int:
                        raise ValueError("Malformed SSH author frame")
                    if "method" in frame:
                        if (
                            set(frame) != {"id", "method", "params"}
                            or not isinstance(frame["params"], dict)
                            or not isinstance(frame["method"], str)
                            or frame["id"] != last_request + 1
                        ):
                            raise ValueError("Malformed SSH author request")
                        last_request = frame["id"]
                        task = asyncio.create_task(self._dispatch(frame))
                        self.handlers.add(task)
                        task.add_done_callback(finished)
                    else:
                        future = self.pending.get(frame["id"])
                        if future is None or (future.done() and not future.cancelled()):
                            raise ValueError("Unmatched SSH author response")
                        if set(frame) not in ({"id", "result"}, {"id", "error"}):
                            raise ValueError("Malformed SSH author response")
                        if future.cancelled():
                            self.pending.pop(frame["id"])
                            continue
                        if set(frame) == {"id", "result"}:
                            future.set_result(frame["result"])
                        else:
                            future.set_exception(
                                RuntimeError(
                                    f"Remote {frame['error']['type']}: {frame['error']['message']}"
                                )
                            )
            if buffer:
                raise ConnectionError("SSH author stream ended inside a frame")
        except asyncio.CancelledError:
            if self._failure is not None:
                raise self._failure
            raise
        finally:
            self.closed = True
            self._pump = None
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(
                        ConnectionError("SSH disconnected; work was not replayed")
                    )
            self.pending.clear()
            tasks = tuple(self.handlers)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
