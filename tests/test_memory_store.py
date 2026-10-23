from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


# The existing application uses direct imports from the analyst directory.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "analyst")
)

from memory_store import (  # noqa: E402
    ConversationNotFoundError,
    MemoryStore,
    SCHEMA_VERSION,
    UnsupportedSchemaError,
)


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = (
            Path(self.temporary_directory.name) / "memory.sqlite3"
        )
        self.now = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
        self.store = self.make_store()

    def make_store(self, scope="workspace-a", **options):
        return MemoryStore(
            self.database_path,
            scope=scope,
            clock=lambda: self.now,
            **options,
        )

    def test_initialization_is_versioned_and_idempotent(self):
        self.make_store()

        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0],
                SCHEMA_VERSION,
            )
            tables = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                    """
                )
            }

        self.assertEqual(tables, {
            "conversations", "messages", "exchange_usage", "job_exchange_receipts"
        })

    def test_newer_schema_is_rejected(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                f"PRAGMA user_version = {SCHEMA_VERSION + 1}"
            )

        with self.assertRaises(UnsupportedSchemaError):
            self.make_store()

    def test_nonempty_unversioned_database_is_not_claimed(self):
        other_path = (
            Path(self.temporary_directory.name) / "unrelated.sqlite3"
        )

        with sqlite3.connect(other_path) as connection:
            connection.execute("CREATE TABLE unrelated(value TEXT)")

        with self.assertRaises(UnsupportedSchemaError):
            MemoryStore(other_path, scope="workspace-a")

        with sqlite3.connect(other_path) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0],
                0,
            )
            self.assertIsNone(
                connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE name = 'conversations'
                    """
                ).fetchone()
            )

    def test_failed_migration_rolls_back_schema(self):
        other_path = (
            Path(self.temporary_directory.name) / "failed.sqlite3"
        )

        def fail_migration(connection):
            connection.execute("CREATE TABLE partial(value TEXT)")
            raise RuntimeError("Simulated migration failure")

        with patch.object(
            MemoryStore, "_migrate_0_to_1", side_effect=fail_migration
        ):
            with self.assertRaises(RuntimeError):
                MemoryStore(other_path, scope="workspace-a")

        with sqlite3.connect(other_path) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                    """
                ).fetchall(),
                [],
            )

        MemoryStore(other_path, scope="workspace-a")

    def test_creation_and_restart_persistence(self):
        conversation = self.store.create_conversation()
        messages = self.store.append_exchange(
            conversation.id,
            user_content="Explain the local configuration.",
            assistant_content="The configuration is local.",
        )

        restarted = self.make_store()

        self.assertEqual(
            restarted.get_conversation(conversation.id), conversation
        )
        self.assertEqual(
            restarted.get_messages(conversation.id), list(messages)
        )
        self.assertEqual(
            restarted.list_conversations(), [conversation]
        )

    def test_ordering_recent_history_and_utc_timestamps(self):
        self.now = datetime(
            2026, 10, 20, 8, 0,
            tzinfo=timezone(timedelta(hours=-4)),
        )
        conversation = self.store.create_conversation()

        for number in range(3):
            self.store.append_exchange(
                conversation.id,
                user_content=f"Question {number}",
                assistant_content=f"Answer {number}",
            )

        messages = self.store.get_messages(conversation.id)

        self.assertEqual(
            [message.sequence for message in messages],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            [message.role for message in messages],
            ["user", "assistant"] * 3,
        )
        self.assertEqual(
            self.store.get_messages(
                conversation.id, recent_exchanges=1
            ),
            messages[-2:],
        )
        self.assertEqual(
            conversation.created_at,
            "2026-10-20T12:00:00.000000+00:00",
        )
        self.assertTrue(
            all(
                message.created_at == conversation.created_at
                for message in messages
            )
        )

    def test_conversations_are_isolated(self):
        first = self.store.create_conversation()
        second = self.store.create_conversation()

        self.assertNotEqual(first.id, second.id)

        self.store.append_exchange(
            first.id, user_content="First", assistant_content="Reply"
        )

        self.assertEqual(self.store.get_messages(second.id), [])
        self.assertEqual(len(self.store.get_messages(first.id)), 2)

    def test_scope_isolation_covers_reads_writes_and_deletion(self):
        conversation = self.store.create_conversation()
        other = self.make_store(scope="workspace-b")

        self.assertEqual(other.list_conversations(), [])

        operations = [
            lambda: other.get_conversation(conversation.id),
            lambda: other.get_messages(conversation.id),
            lambda: other.append_exchange(
                conversation.id,
                user_content="Unauthorized",
                assistant_content="Unauthorized",
            ),
            lambda: other.delete_conversation(conversation.id),
        ]

        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ConversationNotFoundError):
                    operation()

        self.assertEqual(
            self.store.get_conversation(conversation.id), conversation
        )
        self.assertEqual(self.store.get_messages(conversation.id), [])

    def test_validation_does_not_save_half_an_exchange(self):
        conversation = self.store.create_conversation()

        with self.assertRaises(ValueError):
            self.store.append_exchange(
                conversation.id,
                user_content="Question",
                assistant_content="   ",
            )

        self.assertEqual(self.store.get_messages(conversation.id), [])
        self.assertEqual(
            self.store.get_conversation(conversation.id), conversation
        )

    def test_database_failure_rolls_back_both_messages_and_timestamp(self):
        conversation = self.store.create_conversation()

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_assistant
                BEFORE INSERT ON messages
                WHEN NEW.role = 'assistant'
                BEGIN
                    SELECT RAISE(ABORT, 'Simulated insertion failure');
                END
                """
            )

        self.now += timedelta(hours=1)

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.append_exchange(
                conversation.id,
                user_content="Question",
                assistant_content="Answer",
            )

        self.assertEqual(self.store.get_messages(conversation.id), [])
        self.assertEqual(
            self.store.get_conversation(conversation.id), conversation
        )

    def test_concurrent_exchanges_are_complete_and_contiguously_ordered(self):
        conversation = self.store.create_conversation()

        def save_exchange(number):
            self.store.append_exchange(
                conversation.id,
                user_content=f"Question {number}",
                assistant_content=f"Answer {number}",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(save_exchange, range(20)))

        messages = self.store.get_messages(conversation.id)

        self.assertEqual(len(messages), 40)
        self.assertEqual(
            [message.sequence for message in messages],
            list(range(1, 41)),
        )
        self.assertEqual(len({message.id for message in messages}), 40)

        for index in range(0, len(messages), 2):
            user, assistant = messages[index:index + 2]
            self.assertEqual(user.role, "user")
            self.assertEqual(assistant.role, "assistant")
            self.assertEqual(
                user.content.removeprefix("Question "),
                assistant.content.removeprefix("Answer "),
            )

    def test_concurrent_initialization(self):
        other_path = (
            Path(self.temporary_directory.name) / "concurrent.sqlite3"
        )

        def create_store(number):
            store = MemoryStore(other_path, scope="workspace-a")
            return store.create_conversation().id

        with ThreadPoolExecutor(max_workers=4) as executor:
            ids = list(executor.map(create_store, range(8)))

        store = MemoryStore(other_path, scope="workspace-a")
        self.assertEqual(
            {item.id for item in store.list_conversations()}, set(ids)
        )

    def test_deletion_cascades_and_survives_restart(self):
        conversation = self.store.create_conversation()
        self.store.append_exchange(
            conversation.id,
            user_content="Question",
            assistant_content="Answer",
        )

        self.store.delete_conversation(conversation.id)
        restarted = self.make_store()

        with self.assertRaises(ConversationNotFoundError):
            restarted.get_messages(conversation.id)

        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM messages"
                ).fetchone()[0],
                0,
            )

    def test_retention_is_scoped_and_uses_latest_activity(self):
        retained = self.store.create_conversation()
        expired = self.store.create_conversation()
        other = self.make_store(scope="workspace-b")
        other_conversation = other.create_conversation()

        self.now += timedelta(days=2)
        self.store.append_exchange(
            retained.id,
            user_content="Recent activity",
            assistant_content="Recent response",
        )

        pruning_store = self.make_store(retention_days=1)

        self.assertEqual(pruning_store.prune_expired_conversations(), 1)
        self.assertEqual(
            [item.id for item in pruning_store.list_conversations()],
            [retained.id],
        )
        with self.assertRaises(ConversationNotFoundError):
            pruning_store.get_conversation(expired.id)

        self.assertEqual(
            other.get_conversation(other_conversation.id),
            other_conversation,
        )

    def test_retention_boundary_and_disabled_retention(self):
        conversation = self.store.create_conversation()

        self.now += timedelta(days=1)
        pruning_store = self.make_store(retention_days=1)

        self.assertEqual(pruning_store.prune_expired_conversations(), 0)

        self.now += timedelta(microseconds=1)
        self.assertEqual(self.store.prune_expired_conversations(), 0)
        self.assertEqual(pruning_store.prune_expired_conversations(), 1)

        with self.assertRaises(ConversationNotFoundError):
            self.store.get_conversation(conversation.id)

    def test_invalid_configuration_and_arguments(self):
        for options in (
            {"scope": ""},
            {"retention_days": 0},
            {"retention_days": True},
            {"busy_timeout_ms": -1},
        ):
            with self.subTest(options=options):
                arguments = {"scope": "workspace-a", **options}
                with self.assertRaises(ValueError):
                    MemoryStore(self.database_path, **arguments)

        conversation = self.store.create_conversation()

        with self.assertRaises(ValueError):
            self.store.get_messages(
                conversation.id, recent_exchanges=0
            )

        with self.assertRaises(ValueError):
            self.store.list_conversations(offset=-1)

        with self.assertRaises(ValueError):
            MemoryStore(":memory:", scope="workspace-a")

        with self.assertRaises(ValueError):
            MemoryStore(
                self.database_path,
                scope="workspace-a",
                clock=None,
            )

    def test_naive_clock_is_rejected(self):
        store = MemoryStore(
            self.database_path,
            scope="workspace-a",
            clock=lambda: datetime(2026, 10, 20),
        )

        with self.assertRaises(ValueError):
            store.create_conversation()

        self.assertEqual(store.list_conversations(), [])

    def test_backward_clock_does_not_reverse_activity_timestamp(self):
        conversation = self.store.create_conversation()
        self.now -= timedelta(hours=1)

        self.store.append_exchange(
            conversation.id,
            user_content="Question",
            assistant_content="Answer",
        )

        self.assertEqual(
            self.store.get_conversation(conversation.id).updated_at,
            conversation.updated_at,
        )


if __name__ == "__main__":
    unittest.main()
