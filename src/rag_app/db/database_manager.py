import uuid
from datetime import datetime
from typing import List

from sqlalchemy import update, func
from sqlalchemy.orm import Session

from rag_app.db.orm_models import Conversation, Message, Document
from rag_app.ingestion.model.models import EmbeddedDocument


class DatabaseManager:

    def __init__(self, db: Session):
        self.db = db

    def create_conversation(self, user_id: str, thread_id: str = None) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            thread_id=uuid.UUID(thread_id) if thread_id else uuid.uuid4()
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def update_conversation_with_summary(self, thread_id: uuid.UUID, summary: str):
        stmt = (update(Conversation)
                .where(Conversation.thread_id == thread_id)
                .values(summary=summary, summary_generated_at=func.now()))
        self.db.execute(stmt)
        self.db.commit()

    def save_message(self, thread_id: uuid.UUID, role: str, content: str) -> Message:
        message = Message(thread_id=thread_id, role=role, content=content)
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_conversation_history(self, thread_id: uuid.UUID, limit: int = 50) -> List[Message]:
        """Retrieve the most recent messages for a conversation, ordered chronologically.

        :param thread_id: the conversation thread identifier
        :param limit: maximum number of messages to return (default 50)
        :returns: list of messages ordered from oldest to newest
        """
        return (self.db.query(Message)
                .filter(Message.thread_id == thread_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .all()[::-1])

    def count_messages_for_thread(self, thread_id: uuid.UUID) -> int:
        """Count messages for a certain thread."""
        return self.db.query(func.count(Message.id)).scalar()

    def count_messages_since(self, thread_id: uuid.UUID, since: datetime) -> int:
        """Count messages created after a given timestamp.

        :param thread_id: the conversation thread identifier
        :param since: count messages created after this timestamp
        :returns: number of messages since the given timestamp
        """
        return (self.db.query(func.count(Message.id))
                .filter(Message.thread_id == thread_id, Message.created_at > since)
                .scalar())

    def get_all_conversations(self, user_id: str) -> List[Conversation]:
        return (self.db.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
                .all())

    def get_conversation(self, thread_id: uuid.UUID) -> Conversation:
        return (self.db.query(Conversation)
                .filter(Conversation.thread_id == thread_id)
                .first())

    def save_embedded_document(self, docs: list[EmbeddedDocument]) -> None:
        sources = {doc.metadata["source"] for doc in docs}
        for source in sources:
            self.db.query(Document).filter(Document.doc_metadata["source"].astext == source).delete(synchronize_session=False)
        orm_docs = [
            Document(
                content=doc.content,
                doc_metadata=doc.metadata,
                embedding=doc.embedding,
            )
            for doc in docs
        ]
        self.db.add_all(orm_docs)
        self.db.commit()
