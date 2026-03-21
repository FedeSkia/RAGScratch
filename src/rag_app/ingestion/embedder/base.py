from abc import ABC

from rag_app.ingestion.model.models import Document, EmbeddedDocument


class Embedder(ABC):
    def embed(self, texts: list[Document]) -> list[EmbeddedDocument]:
        pass