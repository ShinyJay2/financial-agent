#!/usr/bin/env python3
import argparse
from pathlib import Path
from app.chunking.chunker import chunk_file

def main():
    parser = argparse.ArgumentParser(
        description="Test the chunker on one or more files and print a summary."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to the files you want to chunk (e.g. data/20250704800844.xml)"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="How many chunks to show a preview of (default: 3)"
    )
    args = parser.parse_args()

    for fp in args.paths:
        path = Path(fp)
        if not path.exists():
            print(f"⚠️  File not found: {fp}")
            continue

        print(f"\n=== Testing {fp} ===")
        chunks = chunk_file(str(path))
        print(f"Total chunks: {len(chunks)}\n")

        for i, chunk in enumerate(chunks[: args.preview]):
            preview = chunk.replace("\n", " ")[:200]
            print(f"--- Chunk {i+1} preview ---")
            print(preview)
            print()

        print("=" * 40)

if __name__ == "__main__":
    main()
