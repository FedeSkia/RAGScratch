from anthropic import Anthropic, APIError
import traceback
from sqlalchemy.orm import Session
from rag_app.db.DatabaseManager import DatabaseManager
from rag_app.models import InputData

client = Anthropic()


class ChatService:

    def __init__(self, db: Session):
        self.database_manager = DatabaseManager(db)

    def add_new_conversation(self, user_input: InputData):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": user_input.content}
                ]
            )
            response_text = message.content[0].text
            conversation = self.database_manager.create_conversation(user_input.user_id)
            self.database_manager.save_message(conversation.thread_id, "user", user_input.content)
            self.database_manager.save_message(conversation.thread_id, message.role, response_text)

            return response_text
        except APIError as e:
            print(f"Anthropic API error: {e}", traceback.format_exc())
            raise
