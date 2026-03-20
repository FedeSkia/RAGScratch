from pathlib import Path

from langchain_text_splitters import MarkdownTextSplitter

from rag_app.config import settings
from rag_app.ingestion.base import DocumentLoader
from rag_app.ingestion.model.models import Document


class MDIngestor(DocumentLoader):
    def __init__(self):
        self.markdown_splitter: MarkdownTextSplitter = MarkdownTextSplitter(chunk_size=settings.embedding_chunk_size, chunk_overlap=settings.embedding_chunk_overlap)

    def load(self, **kwargs) -> list[Document]:
        path_to_file: str = kwargs["path_to_file"]
        with open(path_to_file, "r", encoding="utf-8") as f:
            content = f.read()
            chunks: list[str] = self.markdown_splitter.split_text(content)
            path = Path(path_to_file)
            return [Document(content=chunk, metadata = {
                "source": path_to_file,
                "filename": path.name,
                "file_type": path.suffix,
                "chunk_index": i,
                "chunk_total": len(chunks),
            }) for i, chunk in enumerate(chunks)]
