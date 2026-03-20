from pathlib import Path

from rag_app.ingestion.base import DocumentLoader
from rag_app.ingestion.md_ingestor import MDIngestor
from rag_app.ingestion.model.models import Document

LOADER_REGISTRY: dict[str, DocumentLoader] = {
    ".md": MDIngestor(),
}

def ingest_directory(directory: str) -> list[Document]:
    documents = []
    for file_path in Path(directory).rglob("*"):
        loader = LOADER_REGISTRY.get(file_path.suffix)
        if loader:
            documents.extend(loader.load(path_to_file=str(file_path)))
    return documents