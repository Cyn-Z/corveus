"""Worker readiness must expire independently of inference ownership."""

import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))
from service_health import WorkerHealth


class WorkerHealthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = 100.0
        self.health = WorkerHealth(
            Path(self.directory.name) / "jobs.sqlite3", clock=lambda: self.now,
        )

    def test_missing_file_is_offline(self):
        self.assertFalse(self.health.router_ready())

    def test_live_heartbeat_is_removed_on_graceful_exit(self):
        with self.health.heartbeat("worker-one"):
            self.assertTrue(self.health.router_ready())
            payload = json.loads(self.health.path.read_text())
            self.assertEqual(set(payload), {"worker_id", "updated_at", "router_ready"})
        self.assertFalse(self.health.router_ready())

    def test_abrupt_stop_expires_without_releasing_any_job(self):
        self.health._publish("worker-one")
        self.now += 20
        self.assertTrue(self.health.router_ready())
        self.now += 0.1
        self.assertFalse(self.health.router_ready())

    def test_invalid_or_future_status_is_offline(self):
        for content in (
            "invalid", "[]", "null", "{}",
            '{"worker_id":"one","router_ready":true,"updated_at":101}',
            '{"worker_id":"one","router_ready":true,"updated_at":true}',
            '{"worker_id":"one","router_ready":true,"updated_at":NaN}',
            '{"worker_id":"one","router_ready":false,"updated_at":100}',
        ):
            with self.subTest(content=content):
                self.health.path.write_text(content)
                self.assertFalse(self.health.router_ready())

    def test_shutdown_preserves_a_replacement_workers_status(self):
        with self.health.heartbeat("old-worker"):
            self.health._publish("new-worker")
        self.assertTrue(self.health.router_ready())

    def test_failed_atomic_write_preserves_old_status_and_cleans_temp_file(self):
        self.health._publish("old-worker")
        with patch("service_health.os.replace", side_effect=OSError):
            with self.assertRaises(OSError):
                self.health._publish("new-worker")
        self.assertEqual(json.loads(self.health.path.read_text())["worker_id"], "old-worker")
        self.assertEqual(list(self.health.path.parent.glob("*.tmp")), [])

    def test_heartbeat_cleans_up_after_worker_exception(self):
        with self.assertRaises(RuntimeError):
            with self.health.heartbeat("worker-one"):
                raise RuntimeError("worker failed")
        self.assertFalse(self.health.router_ready())

    def test_api_reads_the_same_status_file_as_worker(self):
        from service_api import LocalJobAPI
        from service_inference import InferenceGate
        from service_job_store import JobStore
        from service_results import JobResults
        from service_scheduler import JobScheduler
        from service_settings import ServiceSettings

        store = JobStore(Path(self.directory.name) / "jobs.sqlite3", ServiceSettings())
        api = LocalJobAPI(JobResults(InferenceGate(JobScheduler(store))), profile_id="default")
        self.assertFalse(api.router_available())
        with WorkerHealth(store.path).heartbeat("worker-one"):
            self.assertTrue(api.router_available())
        self.assertFalse(api.router_available())

    def test_entrypoint_publishes_only_for_worker_lifetime(self):
        import service_entrypoint as entry

        database = Path(self.directory.name) / "jobs.sqlite3"
        runtime = SimpleNamespace(worker=SimpleNamespace(
            store=SimpleNamespace(path=database), worker_id="worker-one",
        ))
        health = WorkerHealth(database)

        def work(*args):
            self.assertTrue(health.router_ready())

        with patch.object(entry.signal, "signal"), patch.object(
            entry, "build_runtime", return_value=(runtime, None),
        ), patch.object(entry, "wait_for_model", return_value=True), patch.object(
            entry, "serve_worker", side_effect=work,
        ) as worker:
            entry.main(["worker"])
        worker.assert_called_once()
        self.assertFalse(health.router_ready())
