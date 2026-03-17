import uuid
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from rag_app.main import app


def _mock_db():
    return MagicMock()


@patch("rag_app.main.get_db", _mock_db)
class TestAPI:

    @patch("rag_app.main.ChatService")
    def test_post_query_new_conversation(self, mock_cs_cls):
        mock_cs_cls.return_value.add_new_conversation.return_value = "response"
        client = TestClient(app)
        resp = client.post("/query", json={"query": "hello", "user_id": "u1"})
        assert resp.status_code == 200
        mock_cs_cls.return_value.add_new_conversation.assert_called_once()

    @patch("rag_app.main.ChatService")
    def test_post_query_with_thread_id(self, mock_cs_cls):
        tid = str(uuid.uuid4())
        mock_cs_cls.return_value.send_message_with_history.return_value = "reply"
        client = TestClient(app)
        resp = client.post("/query", json={"query": "hi", "user_id": "u1", "thread_id": tid})
        assert resp.status_code == 200
        mock_cs_cls.return_value.send_message_with_history.assert_called_once()

    @patch("rag_app.main.DatabaseManager")
    def test_get_conversations(self, mock_dm_cls):
        mock_dm_cls.return_value.get_all_conversations.return_value = []
        client = TestClient(app)
        resp = client.get("/conversations/user-1")
        assert resp.status_code == 200

    @patch("rag_app.main.DatabaseManager")
    def test_get_messages(self, mock_dm_cls):
        tid = str(uuid.uuid4())
        mock_dm_cls.return_value.get_conversation_history.return_value = []
        client = TestClient(app)
        resp = client.get(f"/conversations/{tid}/messages")
        assert resp.status_code == 200
