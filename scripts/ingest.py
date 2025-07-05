# scripts/ingest.py

import os
from langchain.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.vectorstore import upsert_document

DATA_DIR = "data"        # put your PDFs/MD/TXT files here
CHUNK_SIZE = 1000
OVERLAP    = 200

def load_documents():
    docs = []
    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)
        if fname.lower().endswith(".pdf"):
            docs.extend(UnstructuredPDFLoader(path).load())
        elif fname.lower().endswith((".md", ".txt")):
            docs.extend(TextLoader(path).load())
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP
    )
    return splitter.split_documents(docs)

def main():
    docs = load_documents()
    chunks = chunk_documents(docs)
    for i, chunk in enumerate(chunks):
        # use source filename + chunk index as ID
        src = chunk.metadata.get("source", "doc").split(os.sep)[-1]
        doc_id = f"{src}_{i}"
        upsert_document(doc_id, chunk.page_content)
    print(f"Ingested {len(chunks)} chunks into Chroma.")

if __name__ == "__main__":
    main()
