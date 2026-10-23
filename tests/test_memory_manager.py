from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "analyst")
)

from memory_manager import MemoryManager
from memory_store import (
    Conversation,
    ConversationNotFoundError,
    MemoryStore,
    Message,
)


class MemoryManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = (
            Path(self.temporary_directory.name) / "memory.sqlite3"
        )

    def make_manager(self, **options):
        arguments = {
            "enabled": True,
            "database_path": self.database_path,
            "scope": "workspace-a",
        }
        arguments.update(options)
        return MemoryManager(**arguments)

    def test_disabled_manager_never_constructs_a_store(self):
        with patch("memory_manager.MemoryStore") as store_class:
            manager = MemoryManager(
                enabled=False,
                database_path=self.database_path,
                scope="workspace-a",
            )

            self.assertFalse(manager.enabled)
            self.assertIsNone(manager.create_conversation())
            self.assertIsNone(manager.get_conversation("unused"))
            self.assertEqual(manager.list_conversations(), [])
            self.assertEqual(manager.get_messages("unused"), [])
            self.assertIsNone(
                manager.record_completed_exchange(
                    "unused",
                    user_content="Question",
                    assistant_content="Answer",
                )
            )
            self.assertFalse(manager.delete_conversation("unused"))
            self.assertEqual(
                manager.prune_expired_conversations(), 0
            )

            store_class.assert_not_called()

        self.assertFalse(self.database_path.exists())

    def test_default_manager_requires_no_storage_configuration(self):
        manager = MemoryManager()

        self.assertFalse(manager.enabled)
        self.assertEqual(manager.list_conversations(), [])
        self.assertFalse(self.database_path.exists())

    def test_enabled_configuration_is_required(self):
        invalid_options = [
            {"enabled": "true"},
            {"enabled": True},
            {
                "enabled": True,
                "database_path": self.database_path,
            },
            {
                "enabled": True,
                "database_path": self.database_path,
                "scope": " ",
            },
        ]

        with patch("memory_manager.MemoryStore") as store_class:
            for options in invalid_options:
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):
                        MemoryManager(**options)

            store_class.assert_not_called()

    def test_constructor_passes_storage_configuration(self):
        with patch(
            "memory_manager.MemoryStore", autospec=True
        ) as store_class:
            manager = self.make_manager(
                retention_days=30,
                busy_timeout_ms=12000,
            )

            self.assertTrue(manager.enabled)
            store_class.assert_called_once_with(
                self.database_path,
                scope="workspace-a",
                retention_days=30,
                busy_timeout_ms=12000,
            )

    def test_conversation_operations_survive_restart(self):
        manager = self.make_manager()
        conversation = manager.create_conversation()

        self.assertIsNotNone(conversation)

        exchange = manager.record_completed_exchange(
            conversation.id,
            user_content="Explain the configuration.",
            assistant_content="The configuration is local.",
        )

        restarted = self.make_manager()

        self.assertEqual(
            restarted.get_conversation(conversation.id).id,
            conversation.id,
        )
        self.assertEqual(
            restarted.get_messages(conversation.id),
            list(exchange),
        )
        self.assertEqual(
            [item.id for item in restarted.list_conversations()],
            [conversation.id],
        )

    def test_recent_exchange_selection(self):
        manager = self.make_manager()
        conversation = manager.create_conversation()

        for number in range(3):
            manager.record_completed_exchange(
                conversation.id,
                user_content=f"Question {number}",
                assistant_content=f"Answer {number}",
            )

        messages = manager.get_messages(
            conversation.id,
            recent_exchanges=1,
        )

        self.assertEqual(
            [message.content for message in messages],
            ["Question 2", "Answer 2"],
        )

    def test_scope_isolation(self):
        owner = self.make_manager()
        conversation = owner.create_conversation()
        other = self.make_manager(scope="workspace-b")

        self.assertEqual(other.list_conversations(), [])

        operations = [
            lambda: other.get_conversation(conversation.id),
            lambda: other.get_messages(conversation.id),
            lambda: other.record_completed_exchange(
                conversation.id,
                user_content="Question",
                assistant_content="Answer",
            ),
            lambda: other.delete_conversation(conversation.id),
        ]

        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ConversationNotFoundError):
                    operation()

        self.assertEqual(owner.get_messages(conversation.id), [])

    def test_deletion_removes_conversation(self):
        manager = self.make_manager()
        conversation = manager.create_conversation()

        manager.record_completed_exchange(
            conversation.id,
            user_content="Question",
            assistant_content="Answer",
        )

        self.assertTrue(
            manager.delete_conversation(conversation.id)
        )
        self.assertEqual(manager.list_conversations(), [])

        with self.assertRaises(ConversationNotFoundError):
            manager.get_messages(conversation.id)

    def test_validation_errors_propagate(self):
        manager = self.make_manager()
        conversation = manager.create_conversation()

        with self.assertRaises(ValueError):
            manager.record_completed_exchange(
                conversation.id,
                user_content="Question",
                assistant_content=" ",
            )

        self.assertEqual(
            manager.get_messages(conversation.id), []
        )

    def test_storage_errors_propagate(self):
        with patch(
            "memory_manager.MemoryStore", autospec=True
        ) as store_class:
            store = store_class.return_value
            store.append_exchange.side_effect = (
                sqlite3.OperationalError("Simulated database failure")
            )

            manager = self.make_manager()

            with self.assertRaises(sqlite3.OperationalError):
                manager.record_completed_exchange(
                    "conversation-id",
                    user_content="Question",
                    assistant_content="Answer",
                )

    def test_methods_forward_arguments_and_results(self):
        timestamp = "2026-10-20T12:00:00.000000+00:00"
        conversation = Conversation(
            id="conversation-id",
            scope="workspace-a",
            created_at=timestamp,
            updated_at=timestamp,
        )
        exchange = (
            Message(
                id="user-id",
                conversation_id=conversation.id,
                sequence=1,
                role="user",
                content="Question",
                created_at=timestamp,
            ),
            Message(
                id="assistant-id",
                conversation_id=conversation.id,
                sequence=2,
                role="assistant",
                content="Answer",
                created_at=timestamp,
            ),
        )

        with patch(
            "memory_manager.MemoryStore", autospec=True
        ) as store_class:
            store = store_class.return_value
            store.create_conversation.return_value = conversation
            store.get_conversation.return_value = conversation
            store.list_conversations.return_value = [conversation]
            store.get_messages.return_value = list(exchange)
            store.append_exchange.return_value = exchange
            store.prune_expired_conversations.return_value = 3

            manager = self.make_manager()

            self.assertEqual(
                manager.create_conversation(), conversation
            )
            self.assertEqual(
                manager.get_conversation(conversation.id),
                conversation,
            )
            self.assertEqual(
                manager.list_conversations(limit=5, offset=2),
                [conversation],
            )
            self.assertEqual(
                manager.get_messages(
                    conversation.id, recent_exchanges=2
                ),
                list(exchange),
            )
            self.assertEqual(
                manager.record_completed_exchange(
                    conversation.id,
                    user_content="Question",
                    assistant_content="Answer",
                ),
                exchange,
            )
            self.assertTrue(
                manager.delete_conversation(conversation.id)
            )
            self.assertEqual(
                manager.prune_expired_conversations(), 3
            )

            store.create_conversation.assert_called_once_with(
                title="Untitled conversation"
            )
            store.list_conversations.assert_called_once_with(
                limit=5, offset=2
            )
            store.get_messages.assert_called_once_with(
                conversation.id, recent_exchanges=2
            )
            store.append_exchange.assert_called_once_with(
                conversation.id,
                user_content="Question",
                assistant_content="Answer",
            )
            store.delete_conversation.assert_called_once_with(
                conversation.id
            )
            store.prune_expired_conversations.assert_called_once_with()

    def test_retention_deletes_expired_conversations(self):
        manager = self.make_manager(retention_days=7)
        conversation = manager.create_conversation()

        # A deterministic fixture timestamp avoids sleeps and clock changes.
        # The store's clock injection is tested in test_memory_store.py.
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE conversations
                SET created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "2000-01-01T00:00:00.000000+00:00",
                    "2000-01-01T00:00:00.000000+00:00",
                    conversation.id,
                ),
            )

        self.assertEqual(
            manager.prune_expired_conversations(), 1
        )
        self.assertEqual(manager.list_conversations(), [])

    def test_retention_is_disabled_by_default(self):
        manager = self.make_manager()
        conversation = manager.create_conversation()

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    "2000-01-01T00:00:00.000000+00:00",
                    conversation.id,
                ),
            )

        self.assertEqual(
            manager.prune_expired_conversations(), 0
        )
        self.assertEqual(
            manager.get_conversation(conversation.id).id,
            conversation.id,
        )


if __name__ == "__main__":
    unittest.main()