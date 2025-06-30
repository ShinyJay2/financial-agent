from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    HYPERCLOVA_API_KEY: str = Field(..., env="HYPERCLOVA_API_KEY")
    HYPERCLOVASTUDIO_REQUEST_ID: str = Field(..., env="HYPERCLOVASTUDIO_REQUEST_ID")
    DATABASE_URL: Optional[str] = Field(None, env="DATABASE_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
