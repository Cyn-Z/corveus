"""Local conversation persistence for a single-operator application.

Scope is a server-configured partition, not an authenticated identity.
Conversation IDs are identifiers, not authorization credentials.

Stored content is untrusted data. This module does not construct prompts,
interpret instructions, execute content, or depend on inference models.

Database files and their parent directory require appropriate OS permissions.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator
from token_usage import TokenUsage, summarize_usage

SCHEMA_VERSION = 4
DEFAULT_CONVERSATION_TITLE = "Untitled conversation"
MAX_CONVERSATION_TITLE_LENGTH = 120


@dataclass(frozen=True)
class Conversation:
    id: str
    scope: str
    created_at: str
    updated_at: str
    title: str = DEFAULT_CONVERSATION_TITLE


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    sequence: int
    role: str
    content: str
    created_at: str


class ConversationNotFoundError(LookupError):
    """Conversation is missing or belongs to another configured scope."""


class UnsupportedSchemaError(RuntimeError):
    """Database schema cannot be used by this version of the application."""

class ExchangeConflictError(ValueError):
    """A scoped job identifier was reused with a different exchange."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Clock must return a timezone-aware datetime.")

    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _nonempty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _conversation_title(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Conversation title must be a string.")

    title = value.strip()

    if not title or len(title) > MAX_CONVERSATION_TITLE_LENGTH:
        raise ValueError(
            "Conversation title must contain between "
            f"1 and {MAX_CONVERSATION_TITLE_LENGTH} characters."
        )

    if not all(character.isprintable() for character in title):
        raise ValueError("Conversation title must contain printable characters only.")

    return title


class MemoryStore:
    """SQLite-backed conversation store bound to one configured scope.

    Every operation opens its own connection. Connections are never shared
    between threads. SQLite serializes writes using BEGIN IMMEDIATE;
    busy_timeout bounds how long an operation waits for another writer.

    retention_days=None disables age-based pruning. Otherwise, explicitly
    call prune_expired_conversations() at a suitable application boundary.
    Retention measures inactivity since creation or the latest saved exchange.

    The parent directory must already exist. No database is opened until
    this class is instantiated. Use a persistent filesystem path, not
    ':memory:', because connections are deliberately short-lived.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        scope: str,
        retention_days: int | None = None,
        busy_timeout_ms: int = 5000,
        clock: Callable[[], datetime] = _utc_now,
    ):
        if str(database_path) == ":memory:":
            raise ValueError("MemoryStore requires a persistent database path.")

        self.database_path = Path(database_path).expanduser().resolve()
        self.scope = _nonempty_string(scope, "scope")
        self.retention_days = (
            None
            if retention_days is None
            else _positive_integer(retention_days, "retention_days")
        )
        self.busy_timeout_ms = _positive_integer(busy_timeout_ms, "busy_timeout_ms")

        if not callable(clock):
            raise ValueError("clock must be callable.")

        self._clock = clock

        if not self.database_path.parent.is_dir():
            raise ValueError("Database parent directory must already exist.")

        self._initialize()

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )

        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")

            yield connection

            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        # Schema inspection and migration hold the same writer transaction.
        # Concurrent constructors cannot apply a migration twice.
        with self._transaction(write=True) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]

            if version > SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"Database schema {version} is newer than supported "
                    f"schema {SCHEMA_VERSION}."
                )

            if version == 0:
                # Do not claim an unrelated, unversioned SQLite database.
                tables = connection.execute("""
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name NOT LIKE 'sqlite_%'
                    """).fetchall()

                if tables:
                    raise UnsupportedSchemaError(
                        "Refusing to initialize a nonempty unversioned database."
                    )

                self._migrate_0_to_1(connection)
                connection.execute("PRAGMA user_version = 1")
                version = 1

            if version == 1:
                self._migrate_1_to_2(connection)
                connection.execute("PRAGMA user_version = 2")
                version = 2

            if version == 2:
                self._migrate_2_to_3(connection)
                connection.execute("PRAGMA user_version = 3")
                version = 3

            if version == 3:
                self._migrate_3_to_4(connection)
                connection.execute("PRAGMA user_version = 4")
                version = 4

            if version != SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"No migration path from schema {version}."
                )

    @staticmethod
    def _migrate_0_to_1(connection: sqlite3.Connection) -> None:
        # Individual execute calls preserve the surrounding transaction.
        # executescript() is intentionally avoided because it can commit
        # a pending transaction before executing its statements.
        connection.execute("""
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY NOT NULL,
                scope TEXT NOT NULL CHECK(length(trim(scope)) > 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
        connection.execute("""
            CREATE INDEX conversations_scope_updated
            ON conversations(scope, updated_at, id)
            """)
        connection.execute("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY NOT NULL,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL CHECK(sequence >= 1),
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL CHECK(length(trim(content)) > 0),
                created_at TEXT NOT NULL,
                UNIQUE(conversation_id, sequence)
            )
            """)

    @staticmethod
    def _migrate_1_to_2(connection: sqlite3.Connection) -> None:
        connection.execute("""
            ALTER TABLE conversations
            ADD COLUMN title TEXT NOT NULL
                DEFAULT 'Untitled conversation'
                CHECK(length(trim(title)) BETWEEN 1 AND 120)
            """)

    def create_conversation(
        self,
        *,
        title: str = DEFAULT_CONVERSATION_TITLE,
    ) -> Conversation:
        title = _conversation_title(title)
        conversation_id = uuid.uuid4().hex

        with self._transaction(write=True) as connection:
            now = _timestamp(self._clock())

            connection.execute(
                """
                INSERT INTO conversations(
                    id, scope, created_at, updated_at, title
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, self.scope, now, now, title),
            )

        return Conversation(
            id=conversation_id,
            scope=self.scope,
            created_at=now,
            updated_at=now,
            title=title,
        )

    def _require_conversation(
        self,
        connection: sqlite3.Connection,
        conversation_id: str,
    ) -> sqlite3.Row:
        _nonempty_string(conversation_id, "conversation_id")

        row = connection.execute(
            """
            SELECT id, scope, created_at, updated_at, title
            FROM conversations
            WHERE id = ? AND scope = ?
            """,
            (conversation_id, self.scope),
        ).fetchone()

        if row is None:
            # The same result covers nonexistent and inaccessible IDs.
            raise ConversationNotFoundError(
                "Conversation was not found in the configured scope."
            )

        return row

    def get_conversation(self, conversation_id: str) -> Conversation:
        with self._transaction() as connection:
            row = self._require_conversation(connection, conversation_id)
            return Conversation(**dict(row))

    def rename_conversation(
        self,
        conversation_id: str,
        *,
        title: str,
    ) -> Conversation:
        title = _conversation_title(title)

        with self._transaction(write=True) as connection:
            self._require_conversation(connection, conversation_id)

            connection.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE id = ? AND scope = ?
                """,
                (title, conversation_id, self.scope),
            )

            row = self._require_conversation(connection, conversation_id)
            return Conversation(**dict(row))

    def list_conversations(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[Conversation]:
        _positive_integer(limit, "limit")

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")

        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT id, scope, created_at, updated_at, title
                FROM conversations
                WHERE scope = ?
                ORDER BY updated_at DESC, id ASC
                LIMIT ? OFFSET ?
                """,
                (self.scope, limit, offset),
            ).fetchall()

        return [Conversation(**dict(row)) for row in rows]

    def append_exchange(
        self,
        conversation_id: str,
        *,
        user_content: str,
        assistant_content: str,
        usage: TokenUsage | None = None,
        job_id: str | None = None,
    ) -> tuple[Message, Message]:
        """Atomically save a completed exchange and optional job receipt.

        Without job_id, each call appends a new exchange.

        With job_id, an identical retry returns the original message pair.
        Reusing that scoped job ID with different content, conversation,
        or usage raises ExchangeConflictError.

        Job IDs must come from trusted service code. They are identifiers,
        not authorization credentials.
        """
        _nonempty_string(user_content, "user_content")
        _nonempty_string(assistant_content, "assistant_content")
        if usage is not None and not isinstance(usage, TokenUsage):
            raise ValueError("usage must be TokenUsage or None")

        fingerprint = None
        if job_id is not None:
            if (
                not isinstance(job_id, str)
                or re.fullmatch(r"[0-9a-f]{32}", job_id) is None
            ):
                raise ValueError(
                    "job_id must contain 32 lowercase hexadecimal characters."
                )

            canonical = json.dumps(
                {
                    "conversation_id": conversation_id,
                    "user_content": user_content,
                    "assistant_content": assistant_content,
                    "usage": None if usage is None else asdict(usage),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            fingerprint = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()

        with self._transaction(write=True) as connection:
            conversation = self._require_conversation(
                connection, conversation_id
            )

            if job_id is not None:
                receipt = connection.execute(
                    """
                    SELECT conversation_id, fingerprint,
                           user_message_id, assistant_message_id
                    FROM job_exchange_receipts
                    WHERE scope = ? AND job_id = ?
                    """,
                    (self.scope, job_id),
                ).fetchone()

                if receipt is not None:
                    if (
                        receipt["conversation_id"] != conversation_id
                        or receipt["fingerprint"] != fingerprint
                    ):
                        raise ExchangeConflictError(
                            "The job identifier already belongs to "
                            "a different exchange."
                        )

                    rows = connection.execute(
                        """
                        SELECT id, conversation_id, sequence,
                               role, content, created_at
                        FROM messages
                        WHERE conversation_id = ?
                          AND id IN (?, ?)
                        ORDER BY sequence ASC
                        """,
                        (
                            conversation_id,
                            receipt["user_message_id"],
                            receipt["assistant_message_id"],
                        ),
                    ).fetchall()

                    if (
                        len(rows) != 2
                        or rows[0]["id"] != receipt["user_message_id"]
                        or rows[1]["id"] != receipt["assistant_message_id"]
                        or rows[0]["role"] != "user"
                        or rows[1]["role"] != "assistant"
                    ):
                        raise RuntimeError(
                            "The stored job exchange receipt is inconsistent."
                        )

                    return Message(**dict(rows[0])), Message(**dict(rows[1]))

            last_sequence = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0)
                FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()[0]

            now = max(
                _timestamp(self._clock()),
                conversation["updated_at"],
            )

            user = Message(
                id=uuid.uuid4().hex,
                conversation_id=conversation_id,
                sequence=last_sequence + 1,
                role="user",
                content=user_content,
                created_at=now,
            )
            assistant = Message(
                id=uuid.uuid4().hex,
                conversation_id=conversation_id,
                sequence=last_sequence + 2,
                role="assistant",
                content=assistant_content,
                created_at=now,
            )

            connection.executemany(
                """
                INSERT INTO messages(
                    id, conversation_id, sequence, role, content, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        message.id,
                        message.conversation_id,
                        message.sequence,
                        message.role,
                        message.content,
                        message.created_at,
                    )
                    for message in (user, assistant)
                ],
            )

            if usage is not None:
                connection.execute(
                    """
                    INSERT INTO exchange_usage(
                        assistant_message_id, model_name, context_size,
                        input_tokens, output_tokens, total_tokens
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assistant.id,
                        usage.model_name,
                        usage.context_size,
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.total_tokens,
                    ),
                )

            if job_id is not None:
                connection.execute(
                    """
                    INSERT INTO job_exchange_receipts(
                        scope, job_id, conversation_id, fingerprint,
                        user_message_id, assistant_message_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.scope,
                        job_id,
                        conversation_id,
                        fingerprint,
                        user.id,
                        assistant.id,
                    ),
                )

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ? AND scope = ?
                """,
                (now, conversation_id, self.scope),
            )

        return user, assistant

    @staticmethod
    def _migrate_3_to_4(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE job_exchange_receipts (
                scope TEXT NOT NULL CHECK(length(trim(scope)) > 0),
                job_id TEXT NOT NULL CHECK(length(job_id) = 32),
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                fingerprint TEXT NOT NULL CHECK(length(fingerprint) = 64),
                user_message_id TEXT NOT NULL UNIQUE
                    REFERENCES messages(id) ON DELETE CASCADE,
                assistant_message_id TEXT NOT NULL UNIQUE
                    REFERENCES messages(id) ON DELETE CASCADE,
                PRIMARY KEY(scope, job_id),
                CHECK(user_message_id != assistant_message_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX job_exchange_receipts_conversation
            ON job_exchange_receipts(conversation_id)
            """
        )

    @staticmethod
    def _migrate_2_to_3(connection: sqlite3.Connection) -> None:
        connection.execute("""
            CREATE TABLE exchange_usage (
                assistant_message_id TEXT PRIMARY KEY NOT NULL
                    REFERENCES messages(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL CHECK(length(trim(model_name)) > 0),
                context_size INTEGER NOT NULL CHECK(context_size > 0),
                input_tokens INTEGER CHECK(input_tokens >= 0),
                output_tokens INTEGER CHECK(output_tokens >= 0),
                total_tokens INTEGER CHECK(total_tokens >= 0),
                CHECK(input_tokens IS NULL OR output_tokens IS NULL
                    OR (total_tokens IS NOT NULL
                        AND total_tokens = input_tokens + output_tokens)),
                CHECK(total_tokens IS NULL OR input_tokens IS NULL
                    OR total_tokens >= input_tokens),
                CHECK(total_tokens IS NULL OR output_tokens IS NULL
                    OR total_tokens >= output_tokens)
            )
            """)

    def get_conversation_usage(self, conversation_id: str) -> dict:
        """Read all-exchange totals and the latest request snapshot in scope."""
        with self._transaction() as connection:
            self._require_conversation(connection, conversation_id)
            rows = connection.execute(
                """
                SELECT u.assistant_message_id, u.model_name, u.context_size,
                       u.input_tokens, u.output_tokens, u.total_tokens
                FROM messages AS m
                LEFT JOIN exchange_usage AS u ON u.assistant_message_id = m.id
                WHERE m.conversation_id = ? AND m.role = 'assistant'
                ORDER BY m.sequence ASC
                """,
                (conversation_id,),
            ).fetchall()
            usages = [
                (
                    None
                    if row["assistant_message_id"] is None
                    else TokenUsage(
                        model_name=row["model_name"],
                        context_size=row["context_size"],
                        input_tokens=row["input_tokens"],
                        output_tokens=row["output_tokens"],
                        total_tokens=row["total_tokens"],
                    )
                )
                for row in rows
            ]
            return {"conversation_id": conversation_id, **summarize_usage(usages)}

    def get_messages(
        self,
        conversation_id: str,
        *,
        recent_exchanges: int | None = None,
    ) -> list[Message]:
        """Inspect all messages or the newest N complete exchanges.

        Results always appear in ascending sequence order.
        """
        if recent_exchanges is not None:
            _positive_integer(recent_exchanges, "recent_exchanges")

        with self._transaction() as connection:
            self._require_conversation(connection, conversation_id)

            if recent_exchanges is None:
                rows = connection.execute(
                    """
                    SELECT id, conversation_id, sequence,
                           role, content, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY sequence ASC
                    """,
                    (conversation_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, conversation_id, sequence,
                           role, content, created_at
                    FROM (
                        SELECT id, conversation_id, sequence,
                               role, content, created_at
                        FROM messages
                        WHERE conversation_id = ?
                        ORDER BY sequence DESC
                        LIMIT ?
                    )
                    ORDER BY sequence ASC
                    """,
                    (conversation_id, recent_exchanges * 2),
                ).fetchall()

        return [Message(**dict(row)) for row in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a scoped conversation and all its messages atomically."""
        with self._transaction(write=True) as connection:
            self._require_conversation(connection, conversation_id)
            connection.execute(
                "DELETE FROM conversations WHERE id = ? AND scope = ?",
                (conversation_id, self.scope),
            )

    def prune_expired_conversations(self) -> int:
        """Delete inactive conversations in this scope; return their count.

        A conversation exactly at the cutoff is retained. Calling this method
        with retention disabled is a no-op. No background scheduler is started.
        """
        if self.retention_days is None:
            return 0

        with self._transaction(write=True) as connection:
            now = self._clock()
            # Validate timezone awareness before computing the cutoff.
            _timestamp(now)
            cutoff = _timestamp(
                now.astimezone(timezone.utc) - timedelta(days=self.retention_days)
            )

            cursor = connection.execute(
                """
                DELETE FROM conversations
                WHERE scope = ? AND updated_at < ?
                """,
                (self.scope, cutoff),
            )
            return cursor.rowcount
