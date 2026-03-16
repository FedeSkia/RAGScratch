from anthropic import Anthropic, types
from fastapi import FastAPI
from pydantic import BaseModel

from rag_app.db.DatabaseManager import DatabaseManager

app = FastAPI()
db = DatabaseManager()

client = Anthropic()

class InputData(BaseModel):
    role: str
    content: str
    user_id: str
    conversation_id: str

@app.post("/query")
def read_root(data: InputData) -> types.Message:
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {"role": data.role, "content": data.content}
        ]
    )
    return message

@app.get("/conversations/{user_id}")
def get_conversations(user_id: str):
    return db.get_all_conversations(user_id)


@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int, limit: int = 50):
    return db.get_conversation_history(conversation_id, limit)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()