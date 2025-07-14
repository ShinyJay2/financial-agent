#!/usr/bin/env python3
from pathlib import Path
from app.chunking.chunker import chunk_file
from app.embeddings.embeddings import get_embedding

def main():
    xml_path = Path("data/20250704800844.xml")
    if not xml_path.exists():
        print(f"❌ File not found: {xml_path}")
        return

    print(f"Loading and chunking {xml_path}…")
    chunks = chunk_file(str(xml_path), max_tokens=500, overlap=50)
    print(f"✅ {len(chunks)} chunks created.\n")

    # Embed the first few chunks as a smoke test
    for i, chunk in enumerate(chunks[:3]):
        vec = get_embedding(chunk)
        print(f"--- Chunk {i+1} ---")
        print(f"Text preview: {chunk.replace(chr(10), ' ')[:200]}…")
        print(f"Embedding length: {len(vec)}")
        print(f"First 5 dims: {vec[:5]}\n")

if __name__ == "__main__":
    main()
