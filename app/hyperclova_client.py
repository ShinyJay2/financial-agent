# app/hyperclova_client.py
import os
import requests
from .config import settings

# If your key starts with "dummy", run in mock mode
USE_MOCK = settings.HYPERCLOVA_API_KEY.startswith("dummy")

BASE_URL = "https://clovastudio.apigw.ntruss.com/testapp/v1/chat-completions/HCX-003"

def ask_hyperclova(question: str) -> str:
    if USE_MOCK:
        # Return a canned response so nothing actually calls out
        return "이 부분은 모의 응답입니다."

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {settings.HYPERCLOVA_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": settings.HYPERCLOVASTUDIO_REQUEST_ID,
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a financial assistant AI."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.2,
        "maxTokens": 200
    }
    resp = requests.post(BASE_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
