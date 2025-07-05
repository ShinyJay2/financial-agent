# FinAgent

A Retrieval-Augmented Financial AI Agent combining HyperClova, real-time Korean market data, LangChain entity extraction, FastAPI and Chainlit.

## Features

* **Natural-language input**: “오늘의 삼성전자 주가를 알려줘” without typing ticker codes
* **Real-time price lookup** for KOSPI/KOSDAQ/KONEX via PyKRX + FinanceDataReader
* **General financial Q\&A** through HyperClova augmented with Chroma RAG (dense & sparse ST v5.0)
* **Entity extraction** using LangChain to map company names to tickers dynamically
* **Interactive Chainlit chat UI** and FastAPI `/agent` endpoint with Swagger

## Requirements

* Python 3.9 or higher
* Git and Internet access for API calls
* POSIX-compatible shell (macOS, Linux, WSL)

## Quickstart

1. Clone the repo and enter the directory
   git clone `<your-repo-url>`
   cd finagent

2. Create and activate a virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

4. Configure environment
   cp .env.example .env
   Edit `.env` to add your `HYPERCLOVA_API_KEY`

## Testing FastAPI Agent

Start the server:
uvicorn app.main\:app --reload

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and try `/agent?question=` with:

* 삼성전자 주가 알려줘
* 2024년 코스피 전망

## Testing Chainlit UI

Launch the chat UI:
chainlit run chainlit\_app.py --watch

Visit the URL in your browser and ask:

* 오늘의 삼성전자 주가를 알려줘 → real-time price
* Any other financial question → HyperClova answer

## Project Structure

finagent/
├ .env.example
├ requirements.txt
├ scripts/ingest.py
├ app/
│ ├ config.py
│ ├ embeddings.py
│ ├ krx\_client.py
│ ├ ticker\_map.py
│ ├ hyperclova\_client.py
│ └ entity\_extraction.py
├ chainlit\_app.py
└ README.md

## File Overview

* config.py – Load `.env` variables (API keys, model names)
* embeddings.py – Initialize dense & sparse Sentence-Transformers models
* krx\_client.py – Real-time stock and index lookup via PyKRX + fallback
* ticker\_map.py – Dynamic company name to ticker mapping
* hyperclova\_client.py – Wrapper for HyperClova API calls
* entity\_extraction.py – LangChain chain to extract `company_name`
* scripts/ingest.py – Example of chunking and upserting docs into Chroma
* chainlit\_app.py – Chainlit-based chat UI combining price lookup and Q\&A

## Next Steps

* Extend `scripts/ingest.py` for PDF/Markdown loaders
* Tune RAG retrieval weights and few-shot prompts
* Add Redis caching for hot tickers and AI responses
* Dockerize FastAPI, Chroma and Chainlit for CI/CD deployment
