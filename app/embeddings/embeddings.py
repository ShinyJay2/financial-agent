# app/embeddings/embeddings.py

from openai import OpenAI
from typing import Iterable
from ..config import settings

# Initialize the OpenAI v1 client with your API key
client = OpenAI(api_key=settings.OPENAI_API_KEY)

def get_embedding(text: str) -> list[float]:
    """
    Return the embedding vector for `text` using the configured OpenAI model.
    Uses the v1 syntax: client.embeddings.create(...)
    """
    resp = client.embeddings.create(
        model=settings.EMBEDDING_MODEL_NAME,
        input=text
    )
    # resp.data is a list of embedding objects; take the first
    return resp.data[0].embedding

def get_embeddings_batch(texts: Iterable[str]) -> list[list[float]]:
    """
    Embed a batch of texts in one API call.
    """
    resp = client.embeddings.create(
        model=settings.EMBEDDING_MODEL_NAME,
        input=list(texts),
    )
    return [item.embedding for item in resp.data]
