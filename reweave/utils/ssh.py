"""Reconnect transport only; never retry bytes or a remote operation."""

import threading

import paramiko


class ReconnectingSSH:
    def __init__(self, connector):
        self.connector = connector
        self.lock = threading.Lock()
        self.client = None
        self.generation = 0
        self._borrowers = {}
        self._retired = {}
        self._closed = False

    def acquire(self):
        # One reconnect owner: concurrent HTTP connections share a live SSH link.
        with self.lock:
            return self._acquire_locked()

    def _acquire_locked(self):
        if self._closed:
            raise OSError("SSH connection owner is closed")
        if self.client is not None:
            transport = self.client.get_transport()
            if transport is not None and transport.is_active():
                return self.client
            self.client.close()
            self.client = None
        self.client = self.connector()
        self.generation += 1
        return self.client

    def borrow(self):
        """Pin a generation until its forwarding channel has finished."""
        with self.lock:
            client = self._acquire_locked()
            key = id(client)
            self._borrowers[key] = self._borrowers.get(key, 0) + 1
            return client

    def retire(self, client):
        """Use a fresh route next time without severing unrelated in-flight RPCs.

        is_active() is only a local flag: an unresponsive SSH peer can still
        report active. A failed channel must not poison every receipt query.
        """
        with self.lock:
            if self.client is client:
                self.client = None
                if self._borrowers.get(id(client), 0):
                    self._retired[id(client)] = client
                else:
                    client.close()

    def release(self, client):
        with self.lock:
            key = id(client)
            count = self._borrowers[key] - 1
            if count:
                self._borrowers[key] = count
            else:
                del self._borrowers[key]
                retired = self._retired.pop(key, None)
                if retired is not None:
                    retired.close()

    def invalidate(self, client):
        with self.lock:
            if self.client is client:
                self.client.close()
                self.client = None

    def close(self):
        with self.lock:
            self._closed = True
            if self.client is not None:
                self.client.close()
                self.client = None
            for client in self._retired.values():
                client.close()
            self._retired.clear()

    def maintain(self, stopping, report):
        delay = 1.0
        connected = False
        while not stopping.is_set():
            try:
                self.acquire()
                if not connected:
                    report("connected", self.generation)
                connected = True
                delay = 1.0
            except (OSError, EOFError, paramiko.SSHException) as error:
                if connected:
                    report("disconnected", type(error).__name__)
                connected = False
                delay = min(delay * 2, 10.0)
            stopping.wait(delay)
