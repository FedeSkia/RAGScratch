from abc import ABC, abstractmethod

from rag_app.ingestion.model.models import RetrievedDocument


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievedDocument]:
        pass
