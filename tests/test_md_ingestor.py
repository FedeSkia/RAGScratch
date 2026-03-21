import unittest
from unittest.mock import patch, mock_open

from rag_app.ingestion.ingestors.md_ingestor import MDIngestor


SAMPLE_MD = """# Title

First paragraph with some content.

## Section 1

Content of section one.

## Section 2

Content of section two.
"""


class TestMDIngestor(unittest.TestCase):

    @patch("rag_app.ingestion.md_ingestor.settings")
    def setUp(self, mock_settings):
        mock_settings.embedding_chunk_size = 100
        mock_settings.embedding_chunk_overlap = 20
        self.ingestor = MDIngestor()

    @patch("builtins.open", mock_open(read_data=SAMPLE_MD))
    def test_load_returns_documents(self):
        docs = self.ingestor.load(path_to_file="test.md")
        self.assertGreater(len(docs), 0)
        for doc in docs:
            self.assertIsInstance(doc.content, str)
            self.assertIn("source", doc.metadata)
            self.assertIn("filename", doc.metadata)
            self.assertIn("file_type", doc.metadata)
            self.assertIn("chunk_index", doc.metadata)
            self.assertIn("chunk_total", doc.metadata)

    @patch("builtins.open", mock_open(read_data=SAMPLE_MD))
    def test_metadata_values(self):
        docs = self.ingestor.load(path_to_file="/some/path/readme.md")
        for doc in docs:
            self.assertEqual(doc.metadata["source"], "/some/path/readme.md")
            self.assertEqual(doc.metadata["filename"], "readme.md")
            self.assertEqual(doc.metadata["file_type"], ".md")
            self.assertEqual(doc.metadata["chunk_total"], len(docs))

    @patch("builtins.open", mock_open(read_data=SAMPLE_MD))
    def test_chunk_indices_are_sequential(self):
        docs = self.ingestor.load(path_to_file="test.md")
        indices = [doc.metadata["chunk_index"] for doc in docs]
        self.assertEqual(indices, list(range(len(docs))))

    @patch("builtins.open", mock_open(read_data="short"))
    def test_single_chunk_for_small_content(self):
        docs = self.ingestor.load(path_to_file="small.md")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["chunk_index"], 0)
        self.assertEqual(docs[0].metadata["chunk_total"], 1)

    def test_load_missing_kwarg_raises(self):
        with self.assertRaises(KeyError):
            self.ingestor.load()


if __name__ == "__main__":
    unittest.main()
