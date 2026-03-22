import traceback
import uuid
from datetime import datetime, timedelta
from typing import List, TypedDict

from anthropic import Anthropic, APIError
from anthropic.types import MessageParam, ToolUseBlock, TextBlock, ToolParam
from sqlalchemy.orm import Session

from rag_app.db.database_manager import DatabaseManager
from rag_app.db.orm_models import Message
from rag_app.models import InputData
from rag_app.retrieval.base import Retriever

from rag_app.config import settings

client = Anthropic()

SEARCH_TOOL = ToolParam(
    name="search_documents",
    description="Search the knowledge base for relevant documents. "
                "Use this when the user asks a question that may require specific information from ingested files."
                "This tool is using cosine similarity on a PostgresDB to look for relevant information."
                "You can choose how many records to fetch."
                "You can add references to text and you decide if you want to perform a resume",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents",
            },
            "k": {
                "type": "integer",
                "description": "How many records to be fetched"
            }
        },
        "required": ["query", "k"],
    },
)


class ChatService:

    def __init__(self, db: Session, retriever: Retriever):
        self.database_manager = DatabaseManager(db)
        self.retriever = retriever

    def add_new_conversation(self, user_input: InputData):
        try:
            messages = [MessageParam(role="user", content=user_input.query)]
            response_text = self._call_with_tools(messages)
            conversation = self.database_manager.create_conversation(user_input.user_id)
            self.database_manager.save_message(conversation.thread_id, "user", user_input.query)
            self.database_manager.save_message(conversation.thread_id, "assistant", response_text)
            return response_text
        except APIError as e:
            print(f"Anthropic API error: {e}", traceback.format_exc())
            raise

    def send_message_with_history(self, user_input: InputData):
        thread_id = user_input.thread_id
        if user_input.thread_id is None:
            raise Exception("Conversation ID is required")
        messages = self._retrieve_and_format_past_chat(thread_id)
        messages.append({"role": "user", "content": user_input.query})
        response_text = self._call_with_tools(messages)
        self.database_manager.save_message(thread_id, "user", user_input.query)
        self.database_manager.save_message(thread_id, "assistant", response_text)

        if self._should_regenerate_summary(thread_id):
            self._generate_conversation_summary(thread_id)

        return response_text

    def _call_with_tools(self, messages: list) -> str:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            tools=[SEARCH_TOOL],
            messages=messages,
        )
        i = 0
        while response.stop_reason == "tool_use" and i <= 4:
            tool_block = next(b for b in response.content if isinstance(b, ToolUseBlock) and b.name == SEARCH_TOOL.get("name"))
            results = self.retriever.retrieve(tool_block.input["query"], tool_block.input["k"])
            tool_result_content = "\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')} | Score: {doc.score:.2f}]\n{doc.content}"
                for doc in results
            )
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_block.id, "content": tool_result_content},
            ]})
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                tools=[SEARCH_TOOL],
                messages=messages,
            )
            i += 1
        return next(b.text for b in response.content if isinstance(b, TextBlock))

    def _retrieve_and_format_past_chat(self, thread_id: uuid.UUID) -> list[MessageParam]:
        past_messages: List[Message] = self.database_manager.get_conversation_history(thread_id, 50)
        return [
            MessageParam(role=msg.role, content=msg.content)
            for msg in past_messages
        ]

    def _generate_conversation_summary(self, thread_id: uuid.UUID):
        past_messages = self._retrieve_and_format_past_chat(thread_id)
        past_messages.append(MessageParam(content="Create a summary of the conversation.", role="user"))
        summary_response = client.messages.create(
            model="claude-haiku-4-5",
            system="You are an assistant specialized in resuming conversation",
            max_tokens=1024,
            messages=past_messages
        )
        summary = summary_response.content[0].text
        self.database_manager.update_conversation_with_summary(thread_id=thread_id, summary=summary)

    def _should_regenerate_summary(self, thread_id: uuid.UUID) -> bool:
        conversation = self.database_manager.get_conversation(thread_id)
        if conversation is None:
            return False

        past_messages: List[Message] = self.database_manager.get_conversation_history(thread_id, 50)
        message_count = len(past_messages)

        if conversation.summary_generated_at is None:
            return message_count >= settings.summary_min_messages

        time_since = datetime.now() - conversation.summary_generated_at
        messages_since_summary = sum(
            1 for msg in past_messages if msg.created_at > conversation.summary_generated_at
        )

        return time_since > timedelta(hours=1) and messages_since_summary >= 5
