from uuid import UUID
from pydantic import ValidationError
import pytest
from rag_app.models import InputData


class TestInputData:

    def test_valid_without_thread_id(self):
        data = InputData(query="hello", user_id="u1")
        assert data.thread_id is None

    def test_valid_with_thread_id(self):
        data = InputData(query="hello", user_id="u1", thread_id="12345678-1234-5678-1234-567812345678")
        assert isinstance(data.thread_id, UUID)

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            InputData(user_id="u1")

    def test_missing_user_id_raises(self):
        with pytest.raises(ValidationError):
            InputData(query="hello")
