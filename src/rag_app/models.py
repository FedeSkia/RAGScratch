from uuid import UUID
from pydantic import BaseModel


class InputData(BaseModel):
    query: str
    user_id: str
    thread_id: UUID | None = None