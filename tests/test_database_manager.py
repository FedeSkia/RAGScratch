import uuid
from rag_app.db.DatabaseManager import DatabaseManager


class TestDatabaseManager:

    def test_create_conversation(self, db_session):
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        assert conv.user_id == "user-1"
        assert conv.thread_id is not None

    def test_create_conversation_with_thread_id(self, db_session):
        dm = DatabaseManager(db_session)
        tid = str(uuid.uuid4())
        conv = dm.create_conversation("user-1", thread_id=tid)
        assert str(conv.thread_id) == tid

    def test_save_and_get_message(self, db_session):
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        msg = dm.save_message(conv.thread_id, "user", "hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.thread_id == conv.thread_id

    def test_get_conversation_history(self, db_session):
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        dm.save_message(conv.thread_id, "user", "msg1")
        dm.save_message(conv.thread_id, "assistant", "msg2")
        history = dm.get_conversation_history(conv.thread_id)
        assert len(history) == 2
        assert history[0].content == "msg1"
        assert history[1].content == "msg2"

    def test_get_conversation_history_respects_limit(self, db_session):
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        for i in range(5):
            dm.save_message(conv.thread_id, "user", f"msg{i}")
        history = dm.get_conversation_history(conv.thread_id, limit=3)
        assert len(history) == 3

    def test_get_all_conversations(self, db_session):
        dm = DatabaseManager(db_session)
        dm.create_conversation("user-1")
        dm.create_conversation("user-1")
        dm.create_conversation("user-2")
        convs = dm.get_all_conversations("user-1")
        assert len(convs) == 2

    def test_get_conversation(self, db_session):
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        fetched = dm.get_conversation(conv.thread_id)
        assert fetched.user_id == "user-1"

    def test_get_conversation_not_found(self, db_session):
        dm = DatabaseManager(db_session)
        result = dm.get_conversation(uuid.uuid4())
        assert result is None

    def test_update_conversation_with_summary(self, db_session):
        dm = DatabaseManager(db_session)
        conv = dm.create_conversation("user-1")
        dm.update_conversation_with_summary(conv.thread_id, "test summary")
        updated = dm.get_conversation(conv.thread_id)
        assert updated.summary == "test summary"
