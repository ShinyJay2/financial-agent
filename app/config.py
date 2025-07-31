from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    HYPERCLOVA_API_KEY: str = Field(..., env="HYPERCLOVA_API_KEY")
    DATABASE_URL: Optional[str] = Field(None, env="DATABASE_URL")
    CHROMA_DB_DIR: str = Field("./chroma_db", env="CHROMA_DB_DIR")
    DART_API_KEY: str 
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    EMBEDDING_MODEL_NAME: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL_NAME")
    VISION_MODEL_NAME: str = Field("gpt-4o-mini", env="VISION_MODEL_NAME")
    CHROMA_COLLECTION_NAME: str = "finagent_collection"
    MEMORY_COLLECTION_NAME: str = "memory_collection"
    NAVER_CLIENT_ID:     str
    NAVER_CLIENT_SECRET: str
    #GOOGLE_APPLICATION_CREDENTIALS: str = Field(
        #"miraeasset-finagent.json",  # relative to your project root
        #env="GOOGLE_APPLICATION_CREDENTIALS")

      
    model_config = SettingsConfigDict(
        env_file = ".env",        # ← point at your .env here
        env_file_encoding = "utf-8",
        extra="ignore"
    )

settings = Settings()
