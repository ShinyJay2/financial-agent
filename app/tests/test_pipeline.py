from app.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(chunk_method="section", bm25_k=20, dense_k=20, final_k=10, max_tokens=4096, overlap=50)
pipeline.finalize()
answer = pipeline.answer("에코프로비엠 위험성을 알려줘")
print(f"Pipeline response: {answer}")