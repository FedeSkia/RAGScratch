import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from rag_app.chat_service.ChatService import ChatService
from rag_app.db.DatabaseManager import DatabaseManager
from rag_app.models import InputData


class TestChatService:

    def _make_anthropic_response(self, text="response text", role="assistant"):
        response = MagicMock()
        response.content = [MagicMock(text=text)]
        response.role = role
        return response

    @patch("rag_app.chat_service.ChatService.client")
    def test_add_new_conversation(self, mock_client, db_session):
        mock_client.messages.create.return_value = self._make_anthropic_response("hi there")
        service = ChatService(db_session)
        data = InputData(query="hello", user_id="user-1")

        result = service.add_new_conversation(data)

        assert result == "hi there"
        mock_client.messages.create.assert_called_once()
        history = service.database_manager.get_conversation_history(
            service.database_manager.get_all_conversations("user-1")[0].thread_id
        )
        assert len(history) == 2

    @patch("rag_app.chat_service.ChatService.client")
    def test_send_message_with_history(self, mock_client, db_session):
        mock_client.messages.create.return_value = self._make_anthropic_response("reply")
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        dm.save_message(conv.thread_id, "user", "first msg")
        dm.save_message(conv.thread_id, "assistant", "first reply")

        service = ChatService(db_session)
        data = InputData(query="second msg", user_id="user-1", thread_id=conv.thread_id)

        result = service.send_message_with_history(data)

        assert result == "reply"
        history = dm.get_conversation_history(conv.thread_id)
        assert len(history) == 4  # 2 original + user msg + assistant reply

    @patch("rag_app.chat_service.ChatService.client")
    def test_send_message_raises_without_thread_id(self, mock_client, db_session):
        service = ChatService(db_session)
        data = InputData(query="hello", user_id="user-1", thread_id=None)
        import pytest
        with pytest.raises(Exception, match="Conversation ID is required"):
            service.send_message_with_history(data)

    def test_should_regenerate_summary_no_conversation(self, db_session):
        service = ChatService(db_session)
        assert service._should_regenerate_summary(uuid.uuid4()) is False

    def test_should_regenerate_summary_first_time(self, db_session):
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        # Set summary_generated_at to None
        conv.summary_generated_at = None
        db_session.commit()

        for i in range(10):
            dm.save_message(conv.thread_id, "user" if i % 2 == 0 else "assistant", f"msg{i}")

        service = ChatService(db_session)
        assert service._should_regenerate_summary(conv.thread_id) is True

    def test_should_regenerate_summary_not_enough_messages(self, db_session):
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        conv.summary_generated_at = None
        db_session.commit()

        dm.save_message(conv.thread_id, "user", "msg1")

        service = ChatService(db_session)
        assert service._should_regenerate_summary(conv.thread_id) is False
