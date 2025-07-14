#!/usr/bin/env python3
import sys
from openai import OpenAI
from app.config import settings

def main():
    # 1) Ensure API key is available
    key = settings.OPENAI_API_KEY
    if not key:
        print("❌ Missing OPENAI_API_KEY in environment", file=sys.stderr)
        sys.exit(1)

    # 2) Instantiate the new v1 client
    client = OpenAI(api_key=key)

    # 3) Prepare test input
    model = settings.EMBEDDING_MODEL_NAME
    text = "안녕하세요, OpenAI 임베딩 테스트입니다."

    # 4) Call the embeddings endpoint
    resp = client.embeddings.create(
        model=model,
        input=text
    )

    # 5) Extract and display results
    embedding = resp.data[0].embedding
    print(f"✅ Model: {model}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

if __name__ == "__main__":
    main()
