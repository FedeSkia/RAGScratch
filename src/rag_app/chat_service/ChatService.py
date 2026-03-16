from anthropic import Anthropic
import traceback
import uuid
from rag_app.db.DatabaseManager import DatabaseManager
from rag_app.models import InputData

client = Anthropic()

class ChatService:

    def __init__(self):
        self.database_manager = DatabaseManager()

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
            conversation_id = self.database_manager.create_conversation(user_id=user_input.user_id, thread_id=uuid.uuid4())
            self.database_manager.save_message(conversation_id, user_input.role, user_input.content)
            self.database_manager.save_message(conversation_id, message.role, response_text)

            return response_text
        except Exception:
            print("Failed to add new conversation", traceback.format_exc())
            raise


