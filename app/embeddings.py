# app/embeddings.py

from sentence_transformers import SentenceTransformer, SparseEncoder
from sentence_transformers import Router
from .config import settings

# 1) Load your dense (semantic) model
dense_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

# 2) Load your sparse (lexical + semantic) encoder
sparse_model = SparseEncoder(settings.SPARSE_MODEL_NAME)

# 3) (Optional) Router to merge sparse + dense under the hood
hybrid_router = Router(
    {"dense": dense_model, "sparse": sparse_model},
    merge="sum"
)
