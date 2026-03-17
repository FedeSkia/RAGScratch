import sqlite3
import uuid

import pytest
from sqlalchemy import create_engine, String, event
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import sessionmaker
from rag_app.db.database import Base

# Register UUID adapter for sqlite3 so it stores as string
sqlite3.register_adapter(uuid.UUID, lambda u: str(u))
sqlite3.register_converter("UUID", lambda b: uuid.UUID(b.decode()))

# Patch PostgreSQL UUID columns to String for SQLite compatibility
for table in Base.metadata.tables.values():
    for col in table.columns:
        if isinstance(col.type, PG_UUID):
            col.type = String(36)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
