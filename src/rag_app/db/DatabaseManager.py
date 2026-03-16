import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from typing import List, Dict
from rag_app.config import settings


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password
            )
            register_vector(self.conn)
            print(f"✓ Connected to {settings.db_name}")
        except psycopg2.Error as e:
            print(f"✗ Connection error DB: {e}")
            raise

    def create_conversation(self, user_id: str, title: str = None) -> int:
        """Create a new conversation"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO conversations (user_id, title)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, title))
            conversation_id = cur.fetchone()[0]
            self.conn.commit()
            return conversation_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"✗ Error creating conversation: {e}")
            raise
        finally:
            cur.close()

    def save_message(self, conversation_id: int, role: str, content: str, embedding: List[float] = None) -> int:
        """Save a message with optional embedding"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO messages (conversation_id, role, content, embedding)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (conversation_id, role, content, embedding))
            message_id = cur.fetchone()[0]
            self.conn.commit()
            return message_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"✗ Error saving message: {e}")
            raise
        finally:
            cur.close()

    def get_conversation_history(self, conversation_id: int, limit: int = 50) -> List[Dict]:
        """Retrieve conversation history"""
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                SELECT id, role, content, created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (conversation_id, limit))
            messages = cur.fetchall()
            return list(reversed(messages))  # Sort from oldest to newest
        except psycopg2.Error as e:
            print(f"✗ Error retrieving history: {e}")
            return []
        finally:
            cur.close()

    def search_similar_messages(self, conversation_id: int, embedding: List[float], limit: int = 5) -> List[Dict]:
        """Semantic search for similar messages using pgvector"""
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                SELECT id, role, content, 1 - (embedding <=> %s::vector) as similarity
                FROM messages
                WHERE conversation_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, conversation_id, embedding, limit))
            results = cur.fetchall()
            return results
        except psycopg2.Error as e:
            print(f"✗ Error in semantic search: {e}")
            return []
        finally:
            cur.close()

    def get_all_conversations(self, user_id: str) -> List[Dict]:
        """Retrieve all conversations for a user"""
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = %s
                ORDER BY updated_at DESC
            """, (user_id,))
            return cur.fetchall()
        except psycopg2.Error as e:
            print(f"✗ Error retrieving conversations: {e}")
            return []
        finally:
            cur.close()

    def close(self):
        """Close the connection"""
        if self.conn:
            self.conn.close()
            print("✓ Connection closed")

database_manager = DatabaseManager()