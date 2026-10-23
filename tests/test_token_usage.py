from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from memory_manager import MemoryManager
from memory_store import ConversationNotFoundError, MemoryStore, SCHEMA_VERSION
from token_usage import TokenUsage, summarize_usage, token_count


class TokenUsageTests(unittest.TestCase):
    def test_zero_is_measured_and_invalid_counts_are_unknown(self):
        self.assertEqual(token_count(0), 0)
        for value in (True, False, -1, 1.2, "12", float("nan"), 2**63):
            with self.subTest(value=value):
                self.assertIsNone(token_count(value))

    def test_snapshots_derive_total_from_measured_parts(self):
        usage = TokenUsage("model", 1000, 100, 0)
        self.assertEqual(usage.total_tokens, 100)
        for arguments in (
            {"model_name": ""}, {"context_size": 0},
            {"input_tokens": True}, {"output_tokens": -1},
            {"input_tokens": 100, "output_tokens": 25, "total_tokens": 999},
            {"input_tokens": 100, "total_tokens": 50},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    TokenUsage(**{"model_name": "model", "context_size": 1000, **arguments})

    def test_empty_and_unmeasured_history_are_distinct(self):
        empty = summarize_usage([])
        unknown = summarize_usage([None])
        self.assertEqual(empty["total_tokens"], 0)
        self.assertFalse(empty["is_partial"])
        self.assertIsNone(unknown["total_tokens"])
        self.assertTrue(unknown["is_partial"])
        self.assertEqual(unknown["total_tokens_measured_exchanges"], 0)

    def test_component_coverage_and_latest_missing_measurement(self):
        summary = summarize_usage([
            TokenUsage("old-model", 1000, 100, 25),
            TokenUsage("new-model", 2000, 50),
            None,
        ])
        self.assertEqual(summary["input_tokens"], 150)
        self.assertEqual(summary["output_tokens"], 25)
        self.assertEqual(summary["total_tokens"], 125)
        self.assertEqual(summary["input_tokens_measured_exchanges"], 2)
        self.assertEqual(summary["total_tokens_measured_exchanges"], 1)
        self.assertIsNone(summary["latest_usage"])
        self.assertTrue(summary["is_partial"])


class PersistentUsageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.sqlite3"
        self.now = datetime(2026, 10, 20, tzinfo=timezone.utc)
        self.store = self.make_store()
        self.conversation = self.store.create_conversation(title="Usage test")
        self.usage = TokenUsage("model-a", 1000, 100, 25)

    def make_store(self, scope="workspace-a", **options):
        return MemoryStore(self.path, scope=scope, clock=lambda: self.now, **options)

    def save(self, usage):
        return self.store.append_exchange(
            self.conversation.id, user_content="Question", assistant_content="Answer", usage=usage
        )

    def test_metrics_survive_restart_and_preserve_historical_model_capacity(self):
        self.save(self.usage)
        self.save(TokenUsage("model-b", 4000, 200, 50))
        summary = self.make_store().get_conversation_usage(self.conversation.id)
        self.assertEqual(summary["exchange_count"], 2)
        self.assertEqual(summary["input_tokens"], 300)
        self.assertEqual(summary["output_tokens"], 75)
        self.assertEqual(summary["total_tokens"], 375)
        self.assertEqual(summary["latest_usage"]["model_name"], "model-b")
        self.assertEqual(summary["latest_usage"]["context_size"], 4000)
        self.assertEqual(summary["latest_usage"]["total_tokens"], 250)
        self.assertFalse(summary["is_partial"])

    def test_usage_is_scoped(self):
        self.save(self.usage)
        with self.assertRaises(ConversationNotFoundError):
            self.make_store(scope="workspace-b").get_conversation_usage(self.conversation.id)

    def test_unknown_legacy_exchange_does_not_become_zero_usage(self):
        self.save(None)
        summary = self.store.get_conversation_usage(self.conversation.id)
        self.assertEqual(summary["exchange_count"], 1)
        self.assertEqual(summary["recorded_exchanges"], 0)
        self.assertIsNone(summary["input_tokens"])
        self.assertIsNone(summary["latest_usage"])
        self.save(self.usage)
        summary = self.store.get_conversation_usage(self.conversation.id)
        self.assertEqual(summary["total_tokens"], 125)
        self.assertTrue(summary["is_partial"])

    def test_usage_failure_rolls_back_messages_and_activity(self):
        before = self.store.get_conversation(self.conversation.id)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TRIGGER reject_usage BEFORE INSERT ON exchange_usage
                BEGIN SELECT RAISE(ABORT, 'Simulated metrics write failure'); END"""
            )
        self.now += timedelta(hours=1)
        with self.assertRaises(sqlite3.IntegrityError):
            self.save(self.usage)
        self.assertEqual(self.store.get_messages(self.conversation.id), [])
        self.assertEqual(self.store.get_conversation(self.conversation.id), before)
        self.assertEqual(self.store.get_conversation_usage(self.conversation.id)["exchange_count"], 0)

    def test_concurrent_writes_count_each_exchange_once(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda number: self.save(self.usage), range(12)))
        summary = self.store.get_conversation_usage(self.conversation.id)
        self.assertEqual(summary["exchange_count"], 12)
        self.assertEqual(summary["recorded_exchanges"], 12)
        self.assertEqual(summary["total_tokens"], 1500)

    def test_conversation_deletion_cascades_usage(self):
        self.save(self.usage)
        self.store.delete_conversation(self.conversation.id)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM exchange_usage").fetchone()[0], 0)
        with self.assertRaises(ConversationNotFoundError):
            self.store.get_conversation_usage(self.conversation.id)

    def test_retention_cascades_usage(self):
        self.save(self.usage)
        self.now += timedelta(days=2)
        self.assertEqual(self.make_store(retention_days=1).prune_expired_conversations(), 1)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM exchange_usage").fetchone()[0], 0)

    def version_two_fixture(self):
        path = Path(self.directory.name) / "version-two.sqlite3"
        with sqlite3.connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            MemoryStore._migrate_0_to_1(connection)
            MemoryStore._migrate_1_to_2(connection)
            connection.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                ("legacy", "workspace-a", "2026-10-20", "2026-10-20", "Legacy title"),
            )
            for sequence, role in ((1, "user"), (2, "assistant")):
                connection.execute(
                    "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                    (str(sequence), "legacy", sequence, role, "Legacy content", "2026-10-20"),
                )
            connection.execute("PRAGMA user_version = 2")
        return path

    def test_version_two_migration_preserves_messages_and_marks_usage_unknown(self):
        path = self.version_two_fixture()
        store = MemoryStore(path, scope="workspace-a")
        self.assertEqual(store.get_conversation("legacy").title, "Legacy title")
        self.assertEqual(len(store.get_messages("legacy")), 2)
        self.assertIsNone(store.get_conversation_usage("legacy")["total_tokens"])
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_failed_version_three_migration_rolls_back(self):
        path = self.version_two_fixture()
        original = MemoryStore._migrate_2_to_3

        def fail(connection):
            original(connection)
            raise RuntimeError("Simulated migration failure")

        with patch.object(MemoryStore, "_migrate_2_to_3", side_effect=fail):
            with self.assertRaises(RuntimeError):
                MemoryStore(path, scope="workspace-a")
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='exchange_usage'").fetchone())
        self.assertEqual(len(MemoryStore(path, scope="workspace-a").get_messages("legacy")), 2)

    def test_disabled_manager_has_no_usage_and_enabled_manager_forwards_snapshot(self):
        self.assertIsNone(MemoryManager().get_conversation_usage("unused"))
        manager = MemoryManager(enabled=True, database_path=self.path, scope="workspace-a")
        manager.record_completed_exchange(
            self.conversation.id, user_content="Question", assistant_content="Answer", usage=self.usage
        )
        self.assertEqual(manager.get_conversation_usage(self.conversation.id)["total_tokens"], 125)


if __name__ == "__main__":
    unittest.main()
