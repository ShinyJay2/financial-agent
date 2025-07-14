# app/agent/server.py

import uvicorn
from fastapi import FastAPI
from app.agent.routes import router as agent_router

app = FastAPI(
    title="Financial AI Agent",
    description="미래에셋증권 AI 페스티벌 금융 에이전트 API",
    version="0.1.0"
)

# Mount all of the routes defined in routes.py
app.include_router(agent_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "app.agent.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
