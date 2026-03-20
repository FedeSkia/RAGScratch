from pydantic import BaseModel

class Document(BaseModel):
    content: str
    metadata: dict