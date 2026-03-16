from pydantic import BaseModel


class InputData(BaseModel):
    content: str
    user_id: str
    conversation_id: str | None = None