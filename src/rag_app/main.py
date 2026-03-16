from anthropic import types
from fastapi import FastAPI

from rag_app.chat_service.ChatService import ChatService
from rag_app.db.DatabaseManager import DatabaseManager
from rag_app.models import InputData

app = FastAPI()
db = DatabaseManager()
chatService = ChatService()

@app.post("/query")
def read_root(data: InputData) -> types.Message:
    if data.conversation_id is None:
        return chatService.add_new_conversation(data)
    return []

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