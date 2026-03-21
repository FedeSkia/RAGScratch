from openai import OpenAI

from rag_app.config import settings
from rag_app.ingestion.embedder.base import Embedder
from rag_app.ingestion.model.models import Document, EmbeddedDocument


class OpenAIEmbedder(Embedder):
    def __init__(self):
        self.open_ai_client = OpenAI(api_key=settings.open_ai_api_key)

    def embed(self, documents: list[Document]) -> list[EmbeddedDocument]:
        documents_text = [doc.content for doc in documents]
        response = self.open_ai_client.embeddings.create(input=documents_text, model=settings.embedding_model,
                                                         dimensions=settings.embedding_dims)
        return [
            EmbeddedDocument(content=doc.content, metadata=doc.metadata, embedding=item.embedding)
            for doc, item in zip(documents, response.data)
        ]


def get_embedder() -> Embedder:
    return OpenAIEmbedder()
