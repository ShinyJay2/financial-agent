# app/vectorstore.py

import chromadb
from chromadb.config import Settings as ChromaSettings
from .config import settings
from .embeddings import dense_model, sparse_model

# 1) Connect or create your Chroma database (DuckDB+Parquet backend)
client = chromadb.Client(
    Settings=ChromaSettings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=settings.CHROMA_DB_DIR
    )
)

# 2) Get or create two collections:
dense_collection  = client.get_or_create_collection(name="dense_documents")
sparse_collection = client.get_or_create_collection(name="sparse_documents")

def upsert_document(doc_id: str, text: str) -> None:
    """
    Embed a single document chunk and store both dense & sparse vectors.
    Call this in your ingestion script for each chunk.
    """
    # 2.1) Compute embeddings
    dvec = dense_model.encode(text).tolist()
    svec = sparse_model.encode_document(text).tolist()
    # 2.2) Upsert into Chroma
    dense_collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[dvec]
    )
    sparse_collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[svec]
    )

def hybrid_search(
    query: str,
    k: int = 5,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> list[str]:
    """
    Run separate vector queries against the dense and sparse collections,
    then merge their scores by the given weights and return the top-k doc IDs.
    """
    # 3.1) Encode the query
    q_dvec = dense_model.encode(query).tolist()
    q_svec = sparse_model.encode_query(query).tolist()

    # 3.2) Query each collection
    dres = dense_collection.query(
        query_embeddings=[q_dvec],
        n_results=k
    )
    sres = sparse_collection.query(
        query_embeddings=[q_svec],
        n_results=k
    )

    # 3.3) Extract IDs and scores
    d_ids, d_scores = dres["ids"][0], dres["distances"][0]
    s_ids, s_scores = sres["ids"][0], sres["distances"][0]

    # 3.4) Merge scores
    score_map: dict[str, float] = {}
    for _id, score in zip(d_ids, d_scores):
        score_map[_id] = score_map.get(_id, 0) + score * dense_weight
    for _id, score in zip(s_ids, s_scores):
        score_map[_id] = score_map.get(_id, 0) + score * sparse_weight

    # 3.5) Sort and return top-k
    ranked = sorted(
        score_map.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc_id for doc_id, _ in ranked[:k]]
