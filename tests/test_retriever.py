import unittest
from unittest.mock import MagicMock, patch

from rag_app.ingestion.model.models import Document, EmbeddedDocument, RetrievedDocument
from rag_app.retrieval.retrieval import PgVectorRetriever


class TestPgVectorRetriever(unittest.TestCase):

    def setUp(self):
        self.mock_embedder = MagicMock()
        self.mock_db = MagicMock()
        self.retriever = PgVectorRetriever(self.mock_embedder, self.mock_db)

    def _setup_embedder(self, embedding=None):
        if embedding is None:
            embedding = [0.1, 0.2, 0.3]
        embedded = EmbeddedDocument(content="query", metadata={}, embedding=embedding)
        self.mock_embedder.embed.return_value = [embedded]

    @patch("rag_app.retrieval.retrieval.DocumentORM")
    def test_retrieve_embeds_query(self, mock_orm):
        self._setup_embedder()
        self.mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []

        self.retriever.retrieve("test query", k=3)

        self.mock_embedder.embed.assert_called_once()
        doc_arg = self.mock_embedder.embed.call_args[0][0][0]
        self.assertEqual(doc_arg.content, "test query")

    @patch("rag_app.retrieval.retrieval.DocumentORM")
    def test_retrieve_returns_retrieved_documents(self, mock_orm):
        self._setup_embedder()
        row = MagicMock()
        row.Document.content = "result text"
        row.Document.doc_metadata = {"source": "file.md"}
        row.score = 0.95
        self.mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = [row]

        results = self.retriever.retrieve("query", k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "result text")
        self.assertEqual(results[0].metadata["source"], "file.md")
        self.assertAlmostEqual(results[0].score, 0.95)

    @patch("rag_app.retrieval.retrieval.DocumentORM")
    def test_retrieve_respects_k(self, mock_orm):
        self._setup_embedder()
        self.mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []

        self.retriever.retrieve("query", k=10)

        self.mock_db.query.return_value.order_by.return_value.limit.assert_called_with(10)

    @patch("rag_app.retrieval.retrieval.DocumentORM")
    def test_retrieve_empty_results(self, mock_orm):
        self._setup_embedder()
        self.mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []

        results = self.retriever.retrieve("query")

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
