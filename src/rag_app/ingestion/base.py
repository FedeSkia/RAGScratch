from abc import ABC, abstractmethod

from rag_app.ingestion.model.models import Document


class DocumentLoader(ABC):

    @abstractmethod
    def load(self, **kwargs) -> list[Document]:
        pass
