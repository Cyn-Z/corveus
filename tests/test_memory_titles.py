from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "analyst")
)

from memory_manager import MemoryManager
from memory_store import (
    DEFAULT_CONVERSATION_TITLE,
    ConversationNotFoundError,
    MemoryStore,
    SCHEMA_VERSION,
)


class MemoryTitleTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = (
            Path(self.temporary_directory.name) / "memory.sqlite3"
        )
        self.now = datetime(
            2026, 10, 20, 12, 0, tzinfo=timezone.utc
        )

    def make_store(self, scope="workspace-a", **options):
        return MemoryStore(
            self.database_path,
            scope=scope,
            clock=lambda: self.now,
            **options,
        )

    def create_version_one_database(self):
        timestamp = "2026-10-20T12:00:00.000000+00:00"

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            MemoryStore._migrate_0_to_1(connection)

            connection.executemany(
                """
                INSERT INTO conversations(
                    id, scope, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    ("a" * 32, "workspace-a", timestamp, timestamp),
                    ("b" * 32, "workspace-b", timestamp, timestamp),
                ],
            )
            connection.executemany(
                """
                INSERT INTO messages(
                    id, conversation_id, sequence,
                    role, content, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "1" * 32,
                        "a" * 32,
                        1,
                        "user",
                        "Original question",
                        timestamp,
                    ),
                    (
                        "2" * 32,
                        "a" * 32,
                        2,
                        "assistant",
                        "Original answer",
                        timestamp,
                    ),
                ],
            )
            connection.execute("PRAGMA user_version = 1")

    def test_named_creation_survives_restart(self):
        store = self.make_store()
        conversation = store.create_conversation(
            title="  Blue Lantern  "
        )

        self.assertEqual(conversation.title, "Blue Lantern")

        restarted = self.make_store()
        self.assertEqual(
            restarted.get_conversation(conversation.id),
            conversation,
        )
        self.assertEqual(
            restarted.list_conversations(), [conversation]
        )

    def test_default_title_preserves_existing_creation_calls(self):
        store = self.make_store()
        conversation = store.create_conversation()

        self.assertEqual(
            conversation.title, DEFAULT_CONVERSATION_TITLE
        )

    def test_invalid_titles_are_rejected_without_writes(self):
        store = self.make_store()
        conversation = store.create_conversation(title="Original")

        for title in (
            None,
            "",
            "   ",
            "x" * 121,
            "Line one\nLine two",
            "Tab\tinside",
            "Null\x00inside",
            "Hidden\u202econtrol",
        ):
            with self.subTest(title=repr(title)):
                with self.assertRaises(ValueError):
                    store.create_conversation(title=title)

                with self.assertRaises(ValueError):
                    store.rename_conversation(
                        conversation.id, title=title
                    )

        self.assertEqual(
            store.list_conversations(), [conversation]
        )

    def test_rename_preserves_identity_messages_and_activity(self):
        store = self.make_store()
        conversation = store.create_conversation(title="Original")
        exchange = store.append_exchange(
            conversation.id,
            user_content="Question",
            assistant_content="Answer",
        )
        before = store.get_conversation(conversation.id)

        self.now += timedelta(days=1)

        renamed = store.rename_conversation(
            conversation.id, title="Updated name"
        )

        self.assertEqual(renamed.id, before.id)
        self.assertEqual(renamed.scope, before.scope)
        self.assertEqual(renamed.created_at, before.created_at)
        self.assertEqual(renamed.updated_at, before.updated_at)
        self.assertEqual(renamed.title, "Updated name")
        self.assertEqual(
            store.get_messages(conversation.id), list(exchange)
        )
        self.assertEqual(
            self.make_store().get_conversation(conversation.id),
            renamed,
        )

    def test_rename_is_scoped(self):
        store = self.make_store()
        conversation = store.create_conversation(title="Private")
        other = self.make_store(scope="workspace-b")

        with self.assertRaises(ConversationNotFoundError):
            other.rename_conversation(
                conversation.id, title="Unauthorized"
            )

        self.assertEqual(
            store.get_conversation(conversation.id).title,
            "Private",
        )

    def test_missing_conversation_cannot_be_renamed(self):
        store = self.make_store()

        with self.assertRaises(ConversationNotFoundError):
            store.rename_conversation(
                "0" * 32, title="Missing"
            )

        self.assertEqual(store.list_conversations(), [])

    def test_manager_forwards_named_creation_and_rename(self):
        with patch(
            "memory_manager.MemoryStore", autospec=True
        ) as store_class:
            manager = MemoryManager(
                enabled=True,
                database_path=self.database_path,
                scope="workspace-a",
            )
            store = store_class.return_value

            created = manager.create_conversation(
                title="Blue Lantern"
            )
            renamed = manager.rename_conversation(
                "a" * 32, title="Blue Lantern investigation"
            )

            self.assertIs(
                created, store.create_conversation.return_value
            )
            self.assertIs(
                renamed, store.rename_conversation.return_value
            )
            store.create_conversation.assert_called_once_with(
                title="Blue Lantern"
            )
            store.rename_conversation.assert_called_once_with(
                "a" * 32,
                title="Blue Lantern investigation",
            )

    def test_disabled_manager_performs_no_title_operations(self):
        with patch("memory_manager.MemoryStore") as store_class:
            manager = MemoryManager()

            self.assertIsNone(
                manager.create_conversation(title="Unused")
            )
            self.assertIsNone(
                manager.rename_conversation(
                    "a" * 32, title="Unused"
                )
            )
            store_class.assert_not_called()

        self.assertFalse(self.database_path.exists())

    def test_version_one_migration_preserves_records(self):
        self.create_version_one_database()
        store = self.make_store()

        conversation = store.get_conversation("a" * 32)

        self.assertEqual(
            conversation.title, DEFAULT_CONVERSATION_TITLE
        )
        self.assertEqual(
            [message.content for message in store.get_messages("a" * 32)],
            ["Original question", "Original answer"],
        )
        self.assertEqual(
            len(store.list_conversations()), 1
        )

        other = self.make_store(scope="workspace-b")
        self.assertEqual(
            other.get_conversation("b" * 32).title,
            DEFAULT_CONVERSATION_TITLE,
        )

        store.append_exchange(
            "a" * 32,
            user_content="Next question",
            assistant_content="Next answer",
        )
        self.assertEqual(
            [
                message.sequence
                for message in store.get_messages("a" * 32)
            ],
            [1, 2, 3, 4],
        )

        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0],
                SCHEMA_VERSION,
            )
            self.assertEqual(
                connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall(),
                [],
            )

    def test_failed_title_migration_rolls_back(self):
        self.create_version_one_database()
        original_migration = MemoryStore._migrate_1_to_2

        def fail_after_alter(connection):
            original_migration(connection)
            raise RuntimeError("Simulated title migration failure")

        with patch.object(
            MemoryStore,
            "_migrate_1_to_2",
            side_effect=fail_after_alter,
        ):
            with self.assertRaises(RuntimeError):
                self.make_store()

        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0],
                1,
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(conversations)"
                )
            }
            self.assertNotIn("title", columns)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM messages"
                ).fetchone()[0],
                2,
            )

        self.assertEqual(
            self.make_store().get_conversation("a" * 32).title,
            DEFAULT_CONVERSATION_TITLE,
        )

    def test_concurrent_initialization_migrates_once(self):
        self.create_version_one_database()

        def initialize(number):
            store = self.make_store()
            return store.get_conversation("a" * 32).title

        with ThreadPoolExecutor(max_workers=4) as executor:
            titles = list(executor.map(initialize, range(8)))

        self.assertEqual(
            titles, [DEFAULT_CONVERSATION_TITLE] * 8
        )

    def test_rename_does_not_extend_retention(self):
        store = self.make_store(retention_days=1)
        conversation = store.create_conversation(title="Original")

        self.now += timedelta(days=2)
        store.rename_conversation(
            conversation.id, title="Renamed"
        )

        self.assertEqual(
            store.prune_expired_conversations(), 1
        )


if __name__ == "__main__":
    unittest.main()