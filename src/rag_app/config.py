import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # PostgreSQL
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "conversations_db")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "")

    # API Keys
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dims: int = int(os.getenv("EMBEDDING_DIMS", "1536"))

    embedding_chunk_size: int = int(os.getenv("EMBEDDING_CHUNK_SIZE", "1200"))
    embedding_chunk_overlap: int = int(os.getenv("EMBEDDING_CHUNK_OVERLAP", "300"))
    open_ai_api_key: str = str(os.getenv("OPEN_AI_API_KEY"))
    # Summary
    summary_min_messages: int = int(os.getenv("SUMMARY_MIN_MESSAGES", "10"))

    path_to_files_to_be_ingested: str = str(os.getenv("PATH_TO_FILES", "/Users/federicoconoci/PycharmProjects/fastApiDocs"))

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


settings = Settings()