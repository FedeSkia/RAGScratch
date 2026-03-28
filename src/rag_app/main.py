import uuid

from anthropic import Anthropic
from chainlit.utils import mount_chainlit
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from rag_app.chat_service.chat_service import ChatService
from rag_app.config import settings
from rag_app.db.database import get_db
from rag_app.db.database_manager import DatabaseManager
from rag_app.ingestion.embedder.base import Embedder
from rag_app.ingestion.embedder.embedders import get_embedder
from rag_app.ingestion.ingestor import ingest_directory
from rag_app.models import InputData
from rag_app.retrieval.base import Retriever
from rag_app.retrieval.retrieval import PgVectorRetriever

app = FastAPI()


def get_retriever(db: Session = Depends(get_db), embedder: Embedder = Depends(get_embedder)) -> Retriever:
    return PgVectorRetriever(embedder, db)


def get_anthropic_client() -> Anthropic:
    return Anthropic()


@app.post("/query")
def read_root(data: InputData, db: Session = Depends(get_db), retriever: Retriever = Depends(get_retriever), client: Anthropic = Depends(get_anthropic_client)):
    chat_service = ChatService(db, retriever, client)
    if data.thread_id is None:
        text, thread_id = chat_service.add_new_conversation(data)
        return text
    return chat_service.send_message_with_history(user_input=data)


@app.get("/conversations/{user_id}")
def get_conversations(user_id: str, db: Session = Depends(get_db)):
    return DatabaseManager(db).get_all_conversations(user_id)


@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: uuid.UUID, limit: int = 50, db: Session = Depends(get_db)):
    return DatabaseManager(db).get_conversation_history(conversation_id, limit)


@app.get("/start-ingesting")
def start_ingesting(db: Session = Depends(get_db), embedder: Embedder = Depends(get_embedder)):
    docs = ingest_directory(settings.path_to_files_to_be_ingested)
    embedded_docs = embedder.embed(docs)
    DatabaseManager(db).save_embedded_document(embedded_docs)


def main():
    import uvicorn
    mount_chainlit(app=app, target="cl_app.py", path="/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
