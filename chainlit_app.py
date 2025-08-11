import os
from pathlib import Path
import chainlit as cl
from dotenv import load_dotenv
from app.rag_pipeline import RAGPipeline

load_dotenv()
print(f"Current working directory: {os.getcwd()}")  # Debug print
BASE_DIR = Path(__file__).parent
DB_DIR = Path(os.getenv("CHROMA_DB_DIR", BASE_DIR / "chroma_db"))

pipeline = RAGPipeline(
    chunk_method="section",
    bm25_k=20,
    dense_k=20,
    final_k=10,
    max_tokens=4096,
    overlap=50,
)
pipeline.finalize()

@cl.on_message
async def main(message: cl.Message):
    user_text = message.content
    answer = pipeline.answer(user_text)[:200]
    print(f"Answer content: {answer}")
    await cl.Message(content=str(answer)).send()  # Force plain text