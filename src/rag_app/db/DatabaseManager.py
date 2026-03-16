import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from ..config import settings
from typing import List, Dict, Optional


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        """Connettiti al database PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password
            )
            # Registra il tipo vector per pgvector
            register_vector(self.conn)
            print(f"✓ Connesso a {settings.db_name}")
        except psycopg2.Error as e:
            print(f"✗ Errore connessione DB: {e}")
            raise

    def init_database(self):
        """Crea le tabelle necessarie"""
        try:
            cur = self.conn.cursor()

            # Abilita pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Tabella conversazioni
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    title VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Tabella messaggi con embedding
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indice per ricerca semantica
            cur.execute("""
                CREATE INDEX IF NOT EXISTS messages_embedding_idx 
                ON messages USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
            """)

            # Indice per ricerche temporali
            cur.execute("""
                CREATE INDEX IF NOT EXISTS messages_conversation_time_idx 
                ON messages(conversation_id, created_at)
            """)

            self.conn.commit()
            cur.close()
            print("✓ Database inizializzato")
        except psycopg2.Error as e:
            print(f"✗ Errore inizializzazione: {e}")
            self.conn.rollback()

    def create_conversation(self, user_id: str, title: str = None) -> int:
        """Crea una nuova conversazione"""
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
            print(f"✗ Errore creazione conversazione: {e}")
            raise
        finally:
            cur.close()

    def save_message(self, conversation_id: int, role: str, content: str, embedding: List[float] = None) -> int:
        """Salva un messaggio con embedding opzionale"""
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
            print(f"✗ Errore salvataggio messaggio: {e}")
            raise
        finally:
            cur.close()

    def get_conversation_history(self, conversation_id: int, limit: int = 50) -> List[Dict]:
        """Recupera la storia di una conversazione"""
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
            return list(reversed(messages))  # Ordina dal più vecchio al più recente
        except psycopg2.Error as e:
            print(f"✗ Errore recupero storia: {e}")
            return []
        finally:
            cur.close()

    def search_similar_messages(self, conversation_id: int, embedding: List[float], limit: int = 5) -> List[Dict]:
        """Ricerca semantica di messaggi simili usando pgvector"""
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
            print(f"✗ Errore ricerca semantica: {e}")
            return []
        finally:
            cur.close()

    def get_all_conversations(self, user_id: str) -> List[Dict]:
        """Recupera tutte le conversazioni di un utente"""
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
            print(f"✗ Errore recupero conversazioni: {e}")
            return []
        finally:
            cur.close()

    def close(self):
        """Chiudi la connessione"""
        if self.conn:
            self.conn.close()
            print("✓ Connessione chiusa")