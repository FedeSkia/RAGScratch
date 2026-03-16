import traceback
from typing import List

from anthropic import Anthropic, APIError
from anthropic.types import MessageParam
from sqlalchemy import Column
from sqlalchemy.orm import Session

from rag_app.db.DatabaseManager import DatabaseManager
from rag_app.db.orm_models import Message
from rag_app.models import InputData

client = Anthropic()


class ChatService:

    def __init__(self, db: Session):
        self.database_manager = DatabaseManager(db)

    def add_new_conversation(self, user_input: InputData):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=[
                    MessageParam(role="user", content=user_input.query)
                ]
            )
            response_text = response.content[0].text
            conversation = self.database_manager.create_conversation(user_input.user_id)
            self.database_manager.save_message(conversation.thread_id, "user", user_input.query)
            self.database_manager.save_message(conversation.thread_id, response.role, response_text)

            return response_text
        except APIError as e:
            print(f"Anthropic API error: {e}", traceback.format_exc())
            raise

    def send_message_with_history(self, user_input: InputData):
        if user_input.thread_id is None:
            raise Exception("Conversation ID is required")
        formatted_messages = self._retrieve_and_format_past_chat(user_input)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=formatted_messages
        )
        self.database_manager.save_message(user_input.thread_id, "user", user_input.query)
        response_text = response.content[0].text
        self.database_manager.save_message(user_input.thread_id, response.role, response_text)
        return response_text

    def _retrieve_and_format_past_chat(self, user_input: InputData) -> list[MessageParam]:
        messages: List[Message] = self.database_manager.get_conversation_history(user_input.thread_id, 50)
        formatted_messages = [
            MessageParam(role=msg.role, content=msg.content)
            for msg in messages
        ]
        formatted_messages.append({
            "role": "user",
            "content": user_input.query
        })
        return formatted_messages

