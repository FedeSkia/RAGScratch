from pydantic import BaseModel

class Document(BaseModel):
    content: str
    metadata: dict

class EmbeddedDocument(Document):
    embedding: list[float]

class RetrievedDocument(Document):
    score: float