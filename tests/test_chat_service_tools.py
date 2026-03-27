import uuid
import unittest
from unittest.mock import patch, MagicMock

from anthropic.types import TextBlock, ToolUseBlock

from rag_app.chat_service.chat_service import ChatService
from rag_app.ingestion.model.models import RetrievedDocument
from rag_app.models import InputData


class TestChatServiceWithTools(unittest.TestCase):

    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_retriever = MagicMock()
        with patch("rag_app.chat_service.chat_service.DatabaseManager") as mock_dm_cls:
            self.mock_dm = mock_dm_cls.return_value
            self.service = ChatService(self.mock_db, self.mock_retriever)

    def _text_response(self, text="response"):
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [TextBlock(type="text", text=text)]
        return response

    def _tool_use_response(self, query="search query", k=3):
        response = MagicMock()
        response.stop_reason = "tool_use"
        tool_block = ToolUseBlock(type="tool_use", id="tool_123", name="search_documents", input={"query": query, "k": k})
        response.content = [tool_block]
        return response

    @patch("rag_app.chat_service.chat_service.client")
    def test_no_tool_use(self, mock_client):
        mock_client.messages.create.return_value = self._text_response("direct answer")

        result = self.service._call_with_tools([{"role": "user", "content": "hello"}])

        self.assertEqual(result, "direct answer")
        self.mock_retriever.retrieve.assert_not_called()

    @patch("rag_app.chat_service.chat_service.client")
    def test_single_tool_use(self, mock_client):
        mock_client.messages.create.side_effect = [
            self._tool_use_response("python basics", 5),
            self._text_response("here is the answer"),
        ]
        self.mock_retriever.retrieve.return_value = [
            RetrievedDocument(content="Python is a language", metadata={"source": "doc.md"}, score=0.9),
        ]

        result = self.service._call_with_tools([{"role": "user", "content": "what is python?"}])

        self.assertEqual(result, "here is the answer")
        self.mock_retriever.retrieve.assert_called_once_with("python basics", 5)

    @patch("rag_app.chat_service.chat_service.client")
    def test_tool_use_loop_limited_to_max_iterations(self, mock_client):
        # Always return tool_use to test the limiter
        mock_client.messages.create.return_value = self._tool_use_response()
        self.mock_retriever.retrieve.return_value = [
            RetrievedDocument(content="text", metadata={}, score=0.5),
        ]

        # After 5 iterations (0..4), loop exits and tries to get TextBlock — which won't exist
        with self.assertRaises(StopIteration):
            self.service._call_with_tools([{"role": "user", "content": "query"}])

        self.assertEqual(self.mock_retriever.retrieve.call_count, 5)

    @patch("rag_app.chat_service.chat_service.client")
    def test_add_new_conversation_saves_messages(self, mock_client):
        mock_client.messages.create.return_value = self._text_response("hi")
        data = InputData(query="hello", user_id="user-1")

        self.service.add_new_conversation(data)

        self.mock_dm.create_conversation.assert_called_once_with("user-1")
        self.assertEqual(self.mock_dm.save_message.call_count, 2)

    @patch("rag_app.chat_service.chat_service.client")
    def test_send_message_with_history_requires_thread_id(self, mock_client):
        data = InputData(query="hello", user_id="user-1", thread_id=None)

        with self.assertRaises(Exception, msg="Conversation ID is required"):
            self.service.send_message_with_history(data)

    @patch("rag_app.chat_service.chat_service.client")
    def test_send_message_with_history_saves_messages(self, mock_client):
        mock_client.messages.create.return_value = self._text_response("reply")
        thread_id = uuid.uuid4()
        self.mock_dm.get_conversation_history.return_value = []
        self.mock_dm.get_conversation.return_value = None
        data = InputData(query="msg", user_id="user-1", thread_id=thread_id)

        result = self.service.send_message_with_history(data)

        self.assertEqual(result, "reply")
        self.assertEqual(self.mock_dm.save_message.call_count, 2)

    @patch("rag_app.chat_service.chat_service.client")
    def test_tool_result_content_includes_source_and_score(self, mock_client):
        mock_client.messages.create.side_effect = [
            self._tool_use_response("query", 2),
            self._text_response("answer"),
        ]
        self.mock_retriever.retrieve.return_value = [
            RetrievedDocument(content="chunk text", metadata={"source": "readme.md"}, score=0.85),
        ]

        self.service._call_with_tools([{"role": "user", "content": "question"}])

        # Check the tool_result message sent back
        second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
        tool_result_msg = second_call_messages[-1]["content"][0]
        self.assertIn("readme.md", tool_result_msg["content"])
        self.assertIn("0.85", tool_result_msg["content"])


if __name__ == "__main__":
    unittest.main()
