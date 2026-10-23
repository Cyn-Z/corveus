from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "analyst")
)

from memory_config import MemorySettings
from memory_runtime import build_memory_manager


class MemoryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = (
            Path(self.temporary_directory.name) / "memory.sqlite3"
        )

    def test_disabled_startup_does_not_open_storage(self):
        settings = MemorySettings(
            enabled=False,
            database_path=self.database_path,
        )

        with patch("memory_manager.MemoryStore") as store_class:
            manager = build_memory_manager(settings)

            self.assertFalse(manager.enabled)
            store_class.assert_not_called()

        self.assertFalse(self.database_path.exists())

    def test_configuration_is_forwarded_without_implicit_operations(self):
        settings = MemorySettings(
            enabled=True,
            database_path=self.database_path,
            scope="workspace-a",
            retention_days=30,
            busy_timeout_ms=12000,
        )

        with patch(
            "memory_runtime.MemoryManager", autospec=True
        ) as manager_class:
            manager = build_memory_manager(settings)

            self.assertIs(manager, manager_class.return_value)
            manager_class.assert_called_once_with(
                enabled=True,
                database_path=self.database_path,
                scope="workspace-a",
                retention_days=30,
                busy_timeout_ms=12000,
            )
            self.assertEqual(
                manager_class.return_value.mock_calls, []
            )

    def test_enabled_startup_initializes_and_reopens_storage(self):
        settings = MemorySettings(
            enabled=True,
            database_path=self.database_path,
            scope="workspace-a",
        )

        manager = build_memory_manager(settings)

        self.assertTrue(manager.enabled)
        self.assertTrue(self.database_path.is_file())

        conversation = manager.create_conversation()
        exchange = manager.record_completed_exchange(
            conversation.id,
            user_content="Question",
            assistant_content="Answer",
        )

        restarted = build_memory_manager(settings)

        self.assertEqual(
            restarted.get_conversation(conversation.id).id,
            conversation.id,
        )
        self.assertEqual(
            restarted.get_messages(conversation.id),
            list(exchange),
        )

    def test_missing_parent_causes_startup_failure(self):
        missing_parent = (
            Path(self.temporary_directory.name) / "missing"
        )
        settings = MemorySettings(
            enabled=True,
            database_path=missing_parent / "memory.sqlite3",
        )

        with self.assertRaises(ValueError):
            build_memory_manager(settings)

        self.assertFalse(missing_parent.exists())

    def test_initialization_failure_is_not_silently_disabled(self):
        settings = MemorySettings(
            enabled=True,
            database_path=self.database_path,
        )

        with patch(
            "memory_runtime.MemoryManager",
            side_effect=RuntimeError("Simulated initialization failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Simulated initialization failure"
            ):
                build_memory_manager(settings)


if __name__ == "__main__":
    unittest.main()