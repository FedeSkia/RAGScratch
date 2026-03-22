from sqlalchemy.orm import Session

from rag_app.db.orm_models import Document as DocumentORM
from rag_app.ingestion.embedder.base import Embedder
from rag_app.ingestion.model.models import Document, RetrievedDocument
from rag_app.retrieval.base import Retriever


class PgVectorRetriever(Retriever):
    def __init__(self, embedder: Embedder, db: Session):
        self.embedder = embedder
        self.db = db

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        query_doc = Document(content=query, metadata={})
        query_embedding = self.embedder.embed([query_doc])[0].embedding
        results = (
            self.db.query(
                DocumentORM,
                (1 - DocumentORM.embedding.cosine_distance(query_embedding)).label("score"),
            )
            .order_by(DocumentORM.embedding.cosine_distance(query_embedding))
            .limit(k)
            .all()
        )
        return [
            RetrievedDocument(content=row.Document.content, metadata=row.Document.doc_metadata, score=row.score)
            for row in results
        ]
