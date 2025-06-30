from fastapi import FastAPI, Query
from .hyperclova_client import ask_hyperclova

app = FastAPI()

@app.get("/agent")
async def agent(question: str = Query(..., description="질문")):
    return {"answer": ask_hyperclova(question)}
