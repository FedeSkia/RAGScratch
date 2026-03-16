import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from typing import List, Dict
from contextlib import contextmanager
from rag_app.config import settings


class DatabaseManager:
    """Manages a thread-safe pool of database connections.

    Why a pool instead of a single connection?
    - FastAPI handles multiple requests at the same time (threads)
    - A single connection shared across threads would cause errors
    - A pool gives each thread its own connection, safely
    """

    def __init__(self):
        # The pool is created once, and manages multiple connections
        self._pool = pool.ThreadedConnectionPool(
            minconn=2,   # always keep 2 connections ready
            maxconn=10,  # allow up to 10 concurrent connections
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password
        )

    @contextmanager
    def _get_connection(self):
        """Borrow a connection from the pool, return it when done.

        Usage:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT ...")

        The 'with' block guarantees the connection goes back to the pool,
        even if an error occurs.
        """
        conn = self._pool.getconn()
        try:
            register_vector(conn)
            yield conn
        finally:
            self._pool.putconn(conn)

    def create_conversation(self, user_id: str, title: str = None) -> int:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id",
                (user_id, title)
            )
            conversation_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            return conversation_id

    def save_message(self, conversation_id: int, role: str, content: str, embedding: List[float] = None) -> int:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content, embedding) VALUES (%s, %s, %s, %s) RETURNING id",
                (conversation_id, role, content, embedding)
            )
            message_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            return message_id

    def get_conversation_history(self, conversation_id: int, limit: int = 50) -> List[Dict]:
        with self._get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT id, role, content, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at LIMIT %s",
                (conversation_id, limit)
            )
            return cur.fetchall()

    def get_all_conversations(self, user_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,)
            )
            return cur.fetchall()

    def close(self):
        """Close all connections in the pool."""
        self._pool.closeall()
