"""SQLite test helpers with explicit connection lifetime."""
from __future__ import annotations

from contextlib import contextmanager
import sqlite3


@contextmanager
def closed_sqlite_connect(path, *args, **kwargs):
    """Commit/rollback and always close SQLite connections on Windows.

    ``sqlite3.Connection`` context management handles transactions but does not
    close the connection on Python 3.15. Explicit closure is required before
    Windows temporary-directory cleanup can remove the database file.
    """
    connection = sqlite3.connect(path, *args, **kwargs)
    try:
        with connection:
            yield connection
    finally:
        connection.close()
