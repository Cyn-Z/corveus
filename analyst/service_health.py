"""Content-free worker readiness shared by local API and worker processes."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
from threading import Event, Thread
import time


class WorkerHealth:
    def __init__(self, database_path, *, clock=time.time):
        database = Path(database_path)
        self.path = database.with_name(database.name + ".worker-status.json")
        self.clock = clock

    def router_ready(self):
        try:
            with self.path.open(encoding="utf-8") as source:
                payload = json.loads(source.read(4097))
            stamp = payload["updated_at"]
            age = self.clock() - stamp
            return (
                payload["router_ready"] is True
                and isinstance(payload["worker_id"], str)
                and bool(payload["worker_id"])
                and type(stamp) in (int, float)
                and math.isfinite(stamp)
                and 0 <= age <= 20
            )
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def _publish(self, worker_id):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as target:
                temporary = Path(target.name)
                json.dump({
                    "worker_id": worker_id,
                    "updated_at": self.clock(),
                    "router_ready": True,
                }, target, allow_nan=False)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @contextmanager
    def heartbeat(self, worker_id):
        """Publish only after router initialization; renew during blocking work.

        Five-second updates expire after twenty seconds. A heartbeat proves
        router-process readiness, never inference termination or GPU capacity.
        """
        stop = Event()
        self._publish(worker_id)

        def renew():
            while not stop.wait(5):
                try:
                    self._publish(worker_id)
                except Exception as exc:
                    print(f"[Worker Health] {type(exc).__name__}", flush=True)
                    return

        thread = Thread(target=renew, name="worker-health", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("worker_id") == worker_id:
                    self.path.unlink(missing_ok=True)
            except (OSError, ValueError, AttributeError):
                pass
