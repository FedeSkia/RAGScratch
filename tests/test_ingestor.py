import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from rag_app.ingestion.ingestor import ingest_directory


class TestIngestDirectory(unittest.TestCase):

    @patch("rag_app.ingestion.ingestor.LOADER_REGISTRY")
    def test_ingest_calls_loader_for_matching_extension(self, mock_registry):
        mock_loader = MagicMock()
        mock_loader.load.return_value = [MagicMock()]
        mock_registry.get.return_value = mock_loader

        with patch.object(Path, "rglob") as mock_rglob:
            mock_file = MagicMock(spec=Path)
            mock_file.suffix = ".md"
            mock_rglob.return_value = [mock_file]

            result = ingest_directory("/fake/dir")

        self.assertEqual(len(result), 1)
        mock_loader.load.assert_called_once()

    @patch("rag_app.ingestion.ingestor.LOADER_REGISTRY")
    def test_ingest_skips_unsupported_extension(self, mock_registry):
        mock_registry.get.return_value = None

        with patch.object(Path, "rglob") as mock_rglob:
            mock_file = MagicMock(spec=Path)
            mock_file.suffix = ".pdf"
            mock_rglob.return_value = [mock_file]

            result = ingest_directory("/fake/dir")

        self.assertEqual(result, [])

    @patch("rag_app.ingestion.ingestor.LOADER_REGISTRY")
    def test_ingest_empty_directory(self, mock_registry):
        with patch.object(Path, "rglob", return_value=[]):
            result = ingest_directory("/empty/dir")

        self.assertEqual(result, [])

    @patch("rag_app.ingestion.ingestor.LOADER_REGISTRY")
    def test_ingest_multiple_files(self, mock_registry):
        mock_loader = MagicMock()
        mock_loader.load.side_effect = [[MagicMock()], [MagicMock(), MagicMock()]]
        mock_registry.get.return_value = mock_loader

        with patch.object(Path, "rglob") as mock_rglob:
            f1 = MagicMock(spec=Path, suffix=".md")
            f2 = MagicMock(spec=Path, suffix=".md")
            mock_rglob.return_value = [f1, f2]

            result = ingest_directory("/fake/dir")

        self.assertEqual(len(result), 3)
        self.assertEqual(mock_loader.load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
