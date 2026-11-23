"""Episode-scoped native decisions, without ownership of physical execution."""

from __future__ import annotations

import copy
import fcntl
import json
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import jsonschema

from reweave.utils.persistence import atomic_text


class NativeSessionError(RuntimeError):
    """A native decision was not available; this is not an environment verdict."""


class NativeTransportError(NativeSessionError):
    """A connection failed, with the best known submission boundary."""

    def __init__(self, message: str, *, request_sent: bool = True):
        super().__init__(message)
        self.request_sent = request_sent


class NativeProtocolError(NativeSessionError):
    """The provider did not satisfy its declared decision protocol."""


class NativeCompletedInvalidDecision(NativeProtocolError):
    """A completed decision is invalid and has no unsettled native effects."""


class NativeAuthenticationError(NativeSessionError):
    """The required local account is unavailable; no alternate login is tried."""


class NativeSubmissionUnknown(NativeSessionError):
    """Submission may have happened; reading history could not identify it."""


class NativeTurnFailed(NativeSessionError):
    """An existing native turn ended unsuccessfully; it must not be replayed."""

    def __init__(self, status: str, error: dict | None):
        super().__init__(f"Native turn ended with status {status}")
        self.status = status
        self.error = error


class NativeTurnRecoveryRequired(NativeTurnFailed):
    """A verified failed turn permits one budgeted continuation in the same SID."""

    def __init__(self, error: dict | None, *, reason: str, continuation: int):
        super().__init__("failed", error)
        self.reason = reason
        self.continuation = continuation


class NativeRecoveryBlocked(NativeTurnFailed):
    """Recovery is parked; another decide call must not start a model turn."""

    def __init__(self, error: dict | None, *, reason: str, continuations: int):
        super().__init__("failed", error)
        self.reason = reason
        self.continuations = continuations


class NativeRpcError(NativeSessionError):
    """An explicit JSON-RPC error, retained separately from transport failure."""

    def __init__(self, method: str, error: dict):
        super().__init__(f"Native RPC {method} was rejected (code {error.get('code')})")
        self.method = method
        self.error = error


class NativeDecisionSession(ABC):
    """One logical episode, one provider session, serial decisions.

    A request id binds one exact input. Re-reading a completed decision returns
    the same cached object; action deduplication belongs to the Host ledger.
    """

    @property
    @abstractmethod
    def session_id(self) -> str | None:
        """The provider's persistent session identity, once known."""

    @property
    @abstractmethod
    def context_generation(self) -> int:
        """In-process count of native context compactions for this episode.

        This records provider-owned memory management; it does not instruct the
        Host to reconstruct history or repeat unchanged AGENTS.md contracts.
        """

    @abstractmethod
    async def start(self, instructions: str, cwd: Path) -> None:
        """Create once, or reconnect to the persisted session identity."""

    @abstractmethod
    async def decide(
        self,
        text: str,
        images: tuple[Path, ...],
        output_schema: dict,
        request_id: str,
    ) -> dict:
        """Read one structured native decision, without executing its proposals."""

    @abstractmethod
    async def close(self) -> None:
        """Close owned resources, retaining session and request evidence."""


def _canonical(value: dict) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _decision_text(text: str) -> str:
    """Unwrap one terminal JSON object without changing its payload."""
    raw = text.strip()
    lines = raw.splitlines()
    if len(lines) >= 3 and lines[0] in {"```", "```json"} and lines[-1] == "```":
        raw = "\n".join(lines[1:-1]).strip()
    start = raw.find("{")
    if start > 0:
        try:
            _, end = json.JSONDecoder().raw_decode(raw, start)
        except json.JSONDecodeError:
            return raw
        # Never select a later object, nested fragment, or trailing explanation.
        suffix = raw[end:].strip()
        opening = raw[:start].rstrip().splitlines()[-1]
        if not suffix or suffix == "```" and opening in {"```", "```json"}:
            return raw[start:end]
    return raw


def _same_bytes(left: Path, right: Path) -> bool:
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            before = first.read(1024 * 1024)
            after = second.read(1024 * 1024)
            if before != after:
                return False
            if not before:
                return True


class DurableDecisionSession(NativeDecisionSession):
    """Episode state on disk: identity, owner lock, event log and request inputs.

    A subclass sets ``directory``, ``_state_path``, ``_state``, ``_model``,
    ``_context_generation`` and ``_lease`` before claiming the episode.
    """

    PROVIDER: str
    DEFAULT_MODEL: str

    @classmethod
    def resolve_model(cls, model: str | None) -> str:
        """Resolve an explicit native model name; only omission uses the default."""
        if model is None:
            return cls.DEFAULT_MODEL
        if not isinstance(model, str) or not model or model != model.strip():
            raise ValueError("Native model must be a nonempty model name")
        return model

    @property
    def model(self) -> str:
        """Exact selected model bound to this logical episode's durable identity."""
        return self._model

    @property
    def session_id(self) -> str | None:
        return self._state["session_id"]

    @property
    def context_generation(self) -> int:
        return self._context_generation

    def _identity(self) -> dict:
        """Fields a persisted session must match before this process may use it."""
        return {"version": 1, "provider": self.PROVIDER, "model": self.model}

    def _load_state(self) -> dict:
        state = json.loads(self._state_path.read_text())
        if any(state.get(key) != value for key, value in self._identity().items()):
            raise NativeProtocolError(
                "Native session identity does not match the selected model or backend"
            )
        return state

    def _save(self) -> None:
        atomic_text(
            self._state_path,
            json.dumps(self._state, ensure_ascii=False, allow_nan=False) + "\n",
        )

    def _record(self, kind: str, **values) -> None:
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {"timestamp": time.time(), "kind": kind, **values},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()

    def _claim(self) -> None:
        if self._lease is not None:
            return
        lease = (self.directory / "owner.lock").open("a+b")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lease.close()
            raise NativeSessionError(
                "This native episode already has an active owner"
            ) from error
        # Another owner may have completed work after this object was constructed.
        try:
            if self._state_path.exists():
                self._state = self._load_state()
        except BaseException:
            lease.close()
            raise
        self._lease = lease

    def _release(self) -> None:
        if self._lease is not None:
            fcntl.flock(self._lease, fcntl.LOCK_UN)
            self._lease.close()
            self._lease = None

    def _request_directory(self, request_id: str) -> Path:
        name = self._state["requests"][request_id]["directory"]
        if uuid.UUID(name).hex != name:
            raise NativeProtocolError(
                "Native request directory is not an ordinary UUID"
            )
        return self.directory / "requests" / name

    def _input(self, text: str, images: tuple[Path, ...], output_schema: dict) -> dict:
        jsonschema.validators.validator_for(output_schema).check_schema(output_schema)
        return {
            "text": text,
            "images": [{"path": str(Path(image).resolve())} for image in images],
            "output_schema": copy.deepcopy(output_schema),
        }

    @staticmethod
    def _image_path(directory: Path, image: dict, index: int) -> Path:
        return directory / "images" / f"{index}{Path(image['path']).suffix}"

    def _same_input(self, directory: Path, inputs: dict) -> bool:
        """Whether a reused request id carries its archived input and image bytes."""
        original = json.loads((directory / "input.json").read_text())
        return _canonical(original) == _canonical(inputs) and all(
            _same_bytes(Path(image["path"]), self._image_path(directory, image, index))
            for index, image in enumerate(inputs["images"])
        )
