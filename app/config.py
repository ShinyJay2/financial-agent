from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    HYPERCLOVA_API_KEY: str = Field(..., env="HYPERCLOVA_API_KEY")
    DATABASE_URL: Optional[str] = Field(None, env="DATABASE_URL")
    CHROMA_DB_DIR: str = Field(".chroma_db", env="CHROMA_DB_DIR")
    # Sentence-Transformer model names
    EMBEDDING_MODEL_NAME: str = Field("all-mpnet-base-v2", env="EMBEDDING_MODEL_NAME")
    SPARSE_MODEL_NAME: str = Field("naver/splade-v3", env="SPARSE_MODEL_NAME")
    DART_API_KEY: str 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
