"""Batch 3B.1: retry-safe conversation writes using durable job receipts."""

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
from memory_store import ConversationNotFoundError, ExchangeConflictError, MemoryStore
from token_usage import TokenUsage


class JobReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.sqlite3"
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.store = self.open_store()
        self.conversation = self.store.create_conversation()
        self.job_id = "a" * 32
        self.usage = TokenUsage("model", 100, 10, 5)

    def open_store(self, scope="workspace-a"):
        return MemoryStore(self.path, scope=scope, clock=lambda: self.now)

    def save(self, store=None, **changes):
        arguments = {
            "user_content": "Question", "assistant_content": "Answer",
            "usage": self.usage, "job_id": self.job_id,
        }
        arguments.update(changes)
        return (store or self.store).append_exchange(self.conversation.id, **arguments)

    def test_restart_returns_original_pair_without_duplicate_messages(self):
        first = self.save()
        retry = self.save(self.open_store())
        self.assertEqual(first, retry)
        self.assertEqual(self.store.get_messages(self.conversation.id), list(first))

    def test_retry_does_not_refresh_conversation_retention_timestamp(self):
        self.save()
        original = self.store.get_conversation(self.conversation.id)
        self.now += timedelta(hours=1)
        self.save()
        self.assertEqual(self.store.get_conversation(self.conversation.id).updated_at,
                         original.updated_at)

    def test_changed_content_or_usage_conflicts_without_mutating_exchange(self):
        original = self.save()
        for change in (
            {"user_content": "Changed"}, {"assistant_content": "Changed"},
            {"usage": None}, {"usage": TokenUsage("other-model", 100, 10, 5)},
            {"usage": TokenUsage("model", 100, 11, 5)},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ExchangeConflictError):
                    self.save(**change)
        self.assertEqual(self.store.get_messages(self.conversation.id), list(original))

    def test_concurrent_retries_commit_one_pair(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: self.save(self.open_store()), range(16)))
        self.assertEqual(len({pair[1].id for pair in results}), 1)
        self.assertEqual(len(self.store.get_messages(self.conversation.id)), 2)

    def test_receipt_insert_failure_rolls_back_messages_usage_and_timestamp(self):
        original = self.store.get_conversation(self.conversation.id)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TRIGGER reject_receipt BEFORE INSERT ON job_exchange_receipts
                BEGIN SELECT RAISE(ABORT, 'simulated receipt failure'); END
            """)
        self.now += timedelta(seconds=10)
        with self.assertRaises(sqlite3.IntegrityError):
            self.save()
        self.assertEqual(self.store.get_messages(self.conversation.id), [])
        self.assertEqual(self.store.get_conversation_usage(self.conversation.id)["exchange_count"], 0)
        self.assertEqual(self.store.get_conversation(self.conversation.id).updated_at, original.updated_at)

    def test_usage_is_counted_once_for_retried_job(self):
        self.save()
        self.save()
        usage = self.store.get_conversation_usage(self.conversation.id)
        self.assertEqual(usage["exchange_count"], 1)
        self.assertEqual(usage["total_tokens"], 15)

    def test_calls_without_job_id_keep_independent_append_behavior(self):
        self.save(job_id=None)
        self.save(job_id=None)
        self.assertEqual(len(self.store.get_messages(self.conversation.id)), 4)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM job_exchange_receipts").fetchone()[0], 0)

    def test_receipts_are_isolated_by_configured_scope(self):
        self.save()
        other = self.open_store("workspace-b")
        with self.assertRaises(ConversationNotFoundError):
            self.save(other)
        conversation = other.create_conversation()
        pair = other.append_exchange(conversation.id, user_content="Other question",
                                     assistant_content="Other answer", job_id=self.job_id)
        self.assertEqual(len(pair), 2)

    def test_same_scoped_job_cannot_be_redirected_to_another_conversation(self):
        self.save()
        other = self.store.create_conversation()
        with self.assertRaises(ExchangeConflictError):
            self.store.append_exchange(other.id, user_content="Question",
                                       assistant_content="Answer", usage=self.usage,
                                       job_id=self.job_id)
        self.assertEqual(self.store.get_messages(other.id), [])

    def test_deletion_removes_receipt_and_retry_does_not_restore_conversation(self):
        self.save()
        self.store.delete_conversation(self.conversation.id)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM job_exchange_receipts").fetchone()[0], 0)
        with self.assertRaises(ConversationNotFoundError):
            self.save()

    def test_schema_three_migration_preserves_historical_messages(self):
        original = self.save(job_id=None)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE job_exchange_receipts")
            connection.execute("PRAGMA user_version = 3")
        migrated = self.open_store()
        self.assertEqual(migrated.get_messages(self.conversation.id), list(original))
        self.save(migrated)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)

    def test_failed_migration_is_transactionally_rolled_back(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE job_exchange_receipts")
            connection.execute("PRAGMA user_version = 3")
        def fail(connection):
            connection.execute("CREATE TABLE partial_receipt_migration(value TEXT)")
            raise RuntimeError("simulated migration failure")
        with patch.object(MemoryStore, "_migrate_3_to_4", side_effect=fail):
            with self.assertRaises(RuntimeError):
                self.open_store()
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='partial_receipt_migration'").fetchone())

    def test_invalid_job_identifiers_are_rejected(self):
        for value in (False, "", "bad", "A" * 32, "a" * 33):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.save(job_id=value)
        self.assertEqual(self.store.get_messages(self.conversation.id), [])

    def test_manager_forwards_job_identifier_and_disabled_mode_remains_optional(self):
        manager = MemoryManager(enabled=True, database_path=self.path, scope="workspace-a")
        arguments = {"user_content": "Question", "assistant_content": "Answer",
                     "job_id": self.job_id, "usage": self.usage}
        first = manager.record_completed_exchange(self.conversation.id, **arguments)
        self.assertEqual(manager.record_completed_exchange(self.conversation.id, **arguments), first)
        self.assertIsNone(MemoryManager(enabled=False).record_completed_exchange(
            self.conversation.id, **arguments
        ))


if __name__ == "__main__":
    unittest.main()
