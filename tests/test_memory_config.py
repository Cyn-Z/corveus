from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "analyst")
)

from memory_config import MemorySettings, load_memory_settings


class MemoryConfigTests(unittest.TestCase):
    def test_defaults_leave_memory_disabled(self):
        settings = load_memory_settings({})

        self.assertFalse(settings.enabled)
        self.assertEqual(
            settings.database_path,
            Path("/app/data/memory.sqlite3"),
        )
        self.assertEqual(settings.scope, "local-operator")
        self.assertIsNone(settings.retention_days)
        self.assertEqual(settings.busy_timeout_ms, 5000)

    def test_explicit_configuration(self):
        settings = load_memory_settings(
            {
                "MEMORY_ENABLED": " TRUE ",
                "MEMORY_DATABASE_PATH": "/tmp/custom-memory.sqlite3",
                "MEMORY_SCOPE": " workspace-a ",
                "MEMORY_RETENTION_DAYS": "30",
                "MEMORY_BUSY_TIMEOUT_MS": "12000",
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(
            settings.database_path,
            Path("/tmp/custom-memory.sqlite3"),
        )
        self.assertEqual(settings.scope, "workspace-a")
        self.assertEqual(settings.retention_days, 30)
        self.assertEqual(settings.busy_timeout_ms, 12000)

    def test_reads_process_environment(self):
        with patch.dict(
            os.environ,
            {
                "MEMORY_ENABLED": "true",
                "MEMORY_SCOPE": "configured-scope",
            },
            clear=True,
        ):
            settings = load_memory_settings()

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.scope, "configured-scope")

    def test_empty_retention_disables_pruning(self):
        for value in ("", "   "):
            with self.subTest(value=value):
                settings = load_memory_settings(
                    {"MEMORY_RETENTION_DAYS": value}
                )
                self.assertIsNone(settings.retention_days)

    def test_invalid_environment_values_are_rejected(self):
        invalid_values = [
            ("MEMORY_ENABLED", ""),
            ("MEMORY_ENABLED", "1"),
            ("MEMORY_ENABLED", "yes"),
            ("MEMORY_DATABASE_PATH", ""),
            ("MEMORY_DATABASE_PATH", ":memory:"),
            ("MEMORY_DATABASE_PATH", "relative/memory.sqlite3"),
            ("MEMORY_SCOPE", " "),
            ("MEMORY_RETENTION_DAYS", "0"),
            ("MEMORY_RETENTION_DAYS", "-1"),
            ("MEMORY_RETENTION_DAYS", "1.5"),
            ("MEMORY_BUSY_TIMEOUT_MS", ""),
            ("MEMORY_BUSY_TIMEOUT_MS", "0"),
            ("MEMORY_BUSY_TIMEOUT_MS", "-1"),
            ("MEMORY_BUSY_TIMEOUT_MS", "fast"),
        ]

        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    load_memory_settings({name: value})

    def test_configuration_does_not_create_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = (
                Path(directory) / "missing-parent" / "memory.sqlite3"
            )

            settings = load_memory_settings(
                {
                    "MEMORY_ENABLED": "true",
                    "MEMORY_DATABASE_PATH": str(database_path),
                }
            )

            self.assertEqual(settings.database_path, database_path)
            self.assertFalse(database_path.parent.exists())

    def test_direct_construction_validates_types_and_ranges(self):
        invalid_arguments = [
            {"enabled": "true"},
            {"database_path": "/tmp/memory.sqlite3"},
            {"database_path": Path("relative.sqlite3")},
            {"scope": ""},
            {"retention_days": True},
            {"retention_days": 0},
            {"busy_timeout_ms": True},
            {"busy_timeout_ms": 0},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    MemorySettings(**arguments)


if __name__ == "__main__":
    unittest.main()