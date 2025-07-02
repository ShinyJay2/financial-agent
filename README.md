# FinAgent

A Retrieval-Augmented Financial AI Agent combining HyperClova, real-time Korean market data, LangChain entity extraction, FastAPI and Chainlit.

## Features

* **Natural-language input**: “오늘의 삼성전자 주가를 알려줘” without typing ticker codes
* **Real-time price lookup** for KOSPI/KOSDAQ/KONEX via PyKRX + FinanceDataReader
* **General financial Q\&A** through HyperClova augmented with Chroma RAG (dense & sparse ST v5.0)
* **Entity extraction** using LangChain to map company names to tickers dynamically
* **Two interfaces**:

  * FastAPI `/agent` endpoint with Swagger UI
  * Interactive Chainlit chat UI

## Requirements

* Python 3.9+
* Git, Internet access (for API calls)
* macOS/Linux/Windows with a POSIX-compatible shell

## Quickstart

1. **Clone & enter project**
   git clone `<your-repo-url>`
   cd finagent

2. **Create & activate virtualenv**
   python3 -m venv .venv
   source .venv/bin/activate

3. **Install dependencies**
   pip install --upgrade pip
   pip install -r requirements.txt

4. **Configure environment**
   cp .env.example .env
   Edit `.env` and fill in your `HYPERCLOVA_API_KEY`

## Testing FastAPI Agent

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and use “Try it out” for the `/agent` endpoint.

* Example: `/agent?question=삼성전자 주가 알려줘`
* Example: `/agent?question=2024년 코스피 전망`

## Testing Chainlit UI

Launch Chainlit:

```
chainlit run chainlit_app.py --watch
```

In your browser, go to the printed URL (e.g. [http://localhost:8000](http://localhost:8000)) and chat:

* “오늘의 삼성전자 주가를 알려줘” → real-time price
* Any other financial question → HyperClova answer

## Project Structure

finagent/
├ .env.example  Environment variable template
├ requirements.txt Project dependencies
├ scripts/
│ └ ingest.py  Document chunking & Chroma ingestion example
├ app/
│ ├ config.py Pydantic settings loader
│ ├ embeddings.py Sentence-Transformers & Router initialization
│ ├ vectorstore.py Chroma client (`upsert_document`, `hybrid_search`)
│ ├ krx\_client.py PyKRX + FDR real-time stock/index API
│ ├ ticker\_map.py Dynamic name→ticker mapping via PyKRX
│ ├ hyperclova\_client.py HyperClova HTTP wrapper
│ ├ entity\_extraction.py LangChain extraction chain for company names
│ ├ routers/
│ │ └ agent.py FastAPI GET `/agent` logic
│ └ main.py FastAPI app instantiation
├ chainlit\_app.py Chainlit chat UI entrypoint
└ README.md  This file

## File Overview

* **config.py**: Reads `.env` keys (`HYPERCLOVA_API_KEY`, model names, Chroma directory)
* **embeddings.py**: Loads dense & sparse models and optional hybrid Router
* **vectorstore.py**: Wraps Chroma with add/query and hybrid scoring
* **krx\_client.py**: Fetches real-time prices and indices with lookback + fallback
* **ticker\_map.py**: Builds a name→ticker dictionary at startup via PyKRX
* **hyperclova\_client.py**: Centralizes HyperClova API calls
* **entity\_extraction.py**: LangChain chain to parse out `company_name` from text
* **routers/agent.py**: FastAPI endpoint that branches on price vs. RAG intent
* **chainlit\_app.py**: Single-file Chainlit app combining both features
* **scripts/ingest.py**: Example for ingesting your own docs into Chroma

## Next Steps

* Extend the ingestion script for PDF/Markdown loaders
* Fine-tune RAG weights and few-shot prompt examples
* Add Redis caching for hot tickers and AI responses
* Dockerize FastAPI, Chroma, Chainlit and set up CI/CD for staging/deployment
