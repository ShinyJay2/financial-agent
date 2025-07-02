from fastapi import FastAPI, Query, HTTPException
from .hyperclova_client import ask_hyperclova

app = FastAPI(
    title="Financial AI Agent",
    description="미래에셋증권 AI 페스티벌 금융 에이전트 API",
    version="0.1.0"
)

@app.get("/agent", summary="금융 질의 응답", response_model=dict)
async def agent(
    question: str = Query(..., description="질문을 입력하세요"),
):
    """
    Ask HyperClova the given question and return its answer.
    """
    try:
        answer = ask_hyperclova(question)
        return {"answer": answer}
    except Exception as e:
        # send back a 500 if anything goes wrong
        raise HTTPException(status_code=500, detail=str(e))
