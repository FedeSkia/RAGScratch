import sqlite3
import uuid

import pytest
from sqlalchemy import create_engine, String, JSON, event
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from rag_app.db.database import Base

# Register UUID adapter for sqlite3 so it stores as string
sqlite3.register_adapter(uuid.UUID, lambda u: str(u))
sqlite3.register_converter("UUID", lambda b: uuid.UUID(b.decode()))

# Patch PostgreSQL-specific column types for SQLite compatibility
for table in Base.metadata.tables.values():
    for col in table.columns:
        if isinstance(col.type, PG_UUID):
            col.type = String(36)
        elif isinstance(col.type, JSONB):
            col.type = JSON()
        elif isinstance(col.type, Vector):
            col.type = String()


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
