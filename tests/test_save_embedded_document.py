import unittest
from unittest.mock import MagicMock, patch

from rag_app.db.database_manager import DatabaseManager
from rag_app.ingestion.model.models import EmbeddedDocument


class TestSaveEmbeddedDocument(unittest.TestCase):

    def setUp(self):
        self.mock_db = MagicMock()
        self.dm = DatabaseManager(self.mock_db)

    def _make_doc(self, source: str, content: str = "text") -> EmbeddedDocument:
        return EmbeddedDocument(
            content=content,
            metadata={"source": source},
            embedding=[0.1, 0.2, 0.3],
        )

    @patch("rag_app.db.database_manager.Document")
    def test_save_adds_and_commits(self, mock_orm):
        docs = [self._make_doc("a.md"), self._make_doc("a.md")]

        self.dm.save_embedded_document(docs)

        self.mock_db.add_all.assert_called_once()
        self.assertEqual(len(self.mock_db.add_all.call_args[0][0]), 2)
        self.mock_db.commit.assert_called_once()

    @patch("rag_app.db.database_manager.Document")
    def test_save_deletes_existing_before_insert(self, mock_orm):
        docs = [self._make_doc("a.md")]

        self.dm.save_embedded_document(docs)

        self.mock_db.query.assert_called()
        self.mock_db.add_all.assert_called_once()

    @patch("rag_app.db.database_manager.Document")
    def test_save_deletes_each_unique_source(self, mock_orm):
        docs = [
            self._make_doc("a.md"),
            self._make_doc("b.md"),
            self._make_doc("a.md"),
        ]

        self.dm.save_embedded_document(docs)

        delete_mock = self.mock_db.query.return_value.filter.return_value.delete
        self.assertEqual(delete_mock.call_count, 2)

    @patch("rag_app.db.database_manager.Document")
    def test_save_empty_list(self, mock_orm):
        self.dm.save_embedded_document([])

        self.mock_db.add_all.assert_called_once_with([])
        self.mock_db.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
