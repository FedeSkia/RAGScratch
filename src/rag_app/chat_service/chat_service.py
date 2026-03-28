import traceback
import uuid
from typing import List, Tuple

from anthropic import Anthropic, APIError
from anthropic.types import MessageParam, ToolUseBlock, TextBlock, ToolParam
from sqlalchemy.orm import Session

from rag_app.config import settings
from rag_app.db.database_manager import DatabaseManager
from rag_app.db.orm_models import Message
from rag_app.models import InputData
from rag_app.retrieval.base import Retriever


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
    """Orchestrates conversations with Claude, integrating tool-based document retrieval
    and conversation history management."""

    def __init__(self, db: Session, retriever: Retriever, client: Anthropic):
        self.database_manager = DatabaseManager(db)
        self.retriever = retriever
        self.client = client

    def add_new_conversation(self, user_input: InputData) -> Tuple[str, str]:
        """Start a new conversation. Sends the user query to Claude (with tool access),
        persists both user and assistant messages, and returns the response text.

        :param user_input: contains the query and user_id
        :returns: Claude's response text and thread_id
        :raises APIError: if the Anthropic API call fails
        """
        try:
            messages = [MessageParam(role="user", content=user_input.query)]
            response_text = self._call_with_tools(messages)
            conversation = self.database_manager.create_conversation(user_input.user_id)
            self.database_manager.save_message(conversation.thread_id, "user", user_input.query)
            self.database_manager.save_message(conversation.thread_id, "assistant", response_text)
            return response_text, conversation.thread_id
        except APIError as e:
            print(f"Anthropic API error: {e}", traceback.format_exc())
            raise

    def send_message_with_history(self, user_input: InputData) -> str:
        """Continue an existing conversation. Loads past messages (or summary + recent messages),
        appends the new query, sends everything to Claude with tool access, and persists the exchange.
        Triggers summary regeneration if conditions are met.

        :param user_input: contains the query, user_id, and thread_id
        :returns: Claude's response text
        :raises Exception: if thread_id is None
        """
        thread_id = user_input.thread_id
        if user_input.thread_id is None:
            raise Exception("Thread ID is required")
        summary, messages = self._retrieve_and_format_past_chat(thread_id)
        messages.append({"role": "user", "content": user_input.query})
        response_text = self._call_with_tools(messages, system=summary)
        self.database_manager.save_message(thread_id, "user", user_input.query)
        self.database_manager.save_message(thread_id, "assistant", response_text)

        if self._should_regenerate_summary(thread_id):
            self._generate_conversation_summary(thread_id)

        return response_text

    def _call_with_tools(self, messages: list, system: str | None = None) -> str:
        """Send messages to Claude with the search_documents tool available.
        If Claude invokes the tool, retrieves documents via the retriever and feeds
        the results back, looping up to 5 times. Returns the final text response.

        :param messages: conversation messages to send
        :param system: optional system prompt (e.g. conversation summary)
        :returns: Claude's final text response
        """
        kwargs = dict(model="claude-haiku-4-5", max_tokens=1024, tools=[SEARCH_TOOL], messages=messages)
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        i = 0
        while response.stop_reason == "tool_use" and i <= 4:
            tool_block = next(
                b for b in response.content if isinstance(b, ToolUseBlock) and b.name == SEARCH_TOOL.get("name"))
            results = self.retriever.retrieve(tool_block.input["query"], tool_block.input["k"])
            tool_result_content = "\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')} | Score: {doc.score:.2f}]\n{doc.content}"
                for doc in results
            )
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_block.id, "content": tool_result_content},
            ]})
            response = self.client.messages.create(**kwargs)
            i += 1
        return next(b.text for b in response.content if isinstance(b, TextBlock))

    def _retrieve_and_format_past_chat(self, thread_id: uuid.UUID) -> tuple[str | None, list[MessageParam]]:
        """Load conversation history for the given thread. If a summary exists, returns it
        as a system prompt string along with only the messages created after the summary.
        Otherwise, returns None and the last messages history from the last summary.

        :param thread_id: the conversation thread identifier
        :returns: tuple of (system_prompt_or_none, formatted_messages)
        """
        conversation = self.database_manager.get_conversation(thread_id=thread_id)
        past_messages: List[Message] = self.database_manager.get_conversation_history(thread_id,
                                                                                      settings.summary_min_messages)
        if conversation and conversation.summary:
            recent = [
                MessageParam(role=msg.role, content=msg.content)
                for msg in past_messages
                if msg.created_at > conversation.summary_generated_at
            ]
            return f"Summary of previous conversation: {conversation.summary}", recent
        return None, [
            MessageParam(role=msg.role, content=msg.content)
            for msg in past_messages
        ]

    def _retrieve_past_chat_for_conversation_summary(self, thread_id: uuid.UUID) -> tuple[
        str | None, list[MessageParam]]:
        """ retrieve the conversation summary (if present) and the last messages that have not been summarized"""
        conversation = self.database_manager.get_conversation(thread_id=thread_id)
        past_messages: List[Message] = self.database_manager.get_conversation_history(thread_id,
                                                                                      settings.summary_min_messages)
        past_messages_claude = [MessageParam(content=orm_msg.content, role=orm_msg.role) for orm_msg in past_messages]
        if conversation and conversation.summary:
            return conversation.summary, past_messages_claude
        return None, past_messages_claude

    def _generate_conversation_summary(self, thread_id: uuid.UUID) -> None:
        """Generate a summary of the conversation by asking Claude to summarize
        the message history, then persist it to the database.

        :param thread_id: the conversation thread identifier
        """
        summary, past_messages = self._retrieve_past_chat_for_conversation_summary(thread_id)
        f"Summary of previous conversation: {summary}"
        past_messages.append(MessageParam(content="Create a summary of the conversation.", role="user"))
        summary_response = self.client.messages.create(
            model="claude-haiku-4-5",
            system="You are an assistant specialized in resuming conversation. Your task is to add the last message to the summary",
            max_tokens=1024,
            messages=past_messages
        )
        summary = summary_response.content[0].text
        self.database_manager.update_conversation_with_summary(thread_id=thread_id, summary=summary)

    def _should_regenerate_summary(self, thread_id: uuid.UUID) -> bool:
        """Determine if the conversation summary should be regenerated.
        Returns True if message count since last summary >= summary_min_messages, or there are at least 5 msgs without
        a summary

        :param thread_id: the conversation thread identifier
        :returns: True if summary should be regenerated
        """
        conversation = self.database_manager.get_conversation(thread_id)
        if conversation is None:
            return False

        # If no conversation and more messages than "summary_min_messages" are present then generate it
        if conversation.summary_generated_at is None:
            return self.database_manager.count_messages_for_thread(thread_id) >= settings.summary_min_messages

        messages_with_no_summary_count = self.database_manager.count_messages_since(thread_id,
                                                                                    conversation.summary_generated_at)
        return messages_with_no_summary_count >= settings.summary_min_messages
