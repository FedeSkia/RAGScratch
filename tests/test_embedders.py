import unittest
from unittest.mock import patch, MagicMock

from rag_app.ingestion.embedder.embedders import OpenAIEmbedder
from rag_app.ingestion.model.models import Document


class TestOpenAIEmbedder(unittest.TestCase):

    @patch("rag_app.ingestion.embedder.embedders.settings")
    @patch("rag_app.ingestion.embedder.embedders.OpenAI")
    def setUp(self, mock_openai_cls, mock_settings):
        mock_settings.open_ai_api_key = "fake-key"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.embedding_dims = 1536
        self.mock_client = mock_openai_cls.return_value
        self.embedder = OpenAIEmbedder()

    def _make_embedding_response(self, n: int):
        response = MagicMock()
        items = []
        for i in range(n):
            item = MagicMock()
            item.embedding = [0.1 * i] * 3
            items.append(item)
        response.data = items
        return response

    def test_embed_returns_embedded_documents(self):
        docs = [
            Document(content="hello", metadata={"source": "a.md"}),
            Document(content="world", metadata={"source": "a.md"}),
        ]
        self.mock_client.embeddings.create.return_value = self._make_embedding_response(2)

        result = self.embedder.embed(docs)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "hello")
        self.assertEqual(result[1].content, "world")
        self.assertIsInstance(result[0].embedding, list)

    def test_embed_preserves_metadata(self):
        docs = [Document(content="text", metadata={"source": "b.md", "chunk_index": 0})]
        self.mock_client.embeddings.create.return_value = self._make_embedding_response(1)

        result = self.embedder.embed(docs)

        self.assertEqual(result[0].metadata["source"], "b.md")
        self.assertEqual(result[0].metadata["chunk_index"], 0)

    def test_embed_calls_openai_with_correct_params(self):
        docs = [Document(content="test", metadata={})]
        self.mock_client.embeddings.create.return_value = self._make_embedding_response(1)

        self.embedder.embed(docs)

        self.mock_client.embeddings.create.assert_called_once_with(
            input=["test"],
            model="text-embedding-3-small",
            dimensions=1536,
        )

    def test_embed_empty_list(self):
        self.mock_client.embeddings.create.return_value = self._make_embedding_response(0)

        result = self.embedder.embed([])

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
