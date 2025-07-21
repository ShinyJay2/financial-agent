# Financial Agent

A Retrieval-Augmented Generation (RAG)–powered financial agent built with FastAPI and ChromaDB.

## File structure -app

```text
📦app
 ┣ 📂__pycache__
 ┃ ┣ 📜__init__.cpython-313.pyc
 ┃ ┣ 📜config.cpython-313.pyc
 ┃ ┣ 📜dart_fss_client.cpython-313.pyc
 ┃ ┣ 📜embeddings.cpython-313.pyc
 ┃ ┗ 📜rag_pipeline.cpython-313.pyc
 ┣ 📂agent
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜risk_agent.py
 ┃ ┣ 📜routes.py
 ┃ ┗ 📜server.py
 ┣ 📂chunking
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-313.pyc
 ┃ ┃ ┗ 📜chunker.cpython-313.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜chunker.py
 ┣ 📂clients
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜hyperclova_client.cpython-313.pyc
 ┃ ┗ 📜hyperclova_client.py
 ┣ 📂embeddings
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-313.pyc
 ┃ ┃ ┗ 📜embeddings.cpython-313.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜embeddings.py
 ┣ 📂ingestion
 ┃ ┣ 📜(test)search_api_news_client.py
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜dart_fss_client.py
 ┃ ┣ 📜desktop_research_client.py
 ┃ ┣ 📜hankyung_client.py
 ┃ ┣ 📜krx_client.py
 ┃ ┣ 📜mobile_research_client.py
 ┃ ┣ 📜news_client.py
 ┃ ┗ 📜yahoo_client.py
 ┣ 📂retrieval
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-313.pyc
 ┃ ┃ ┗ 📜vectorstore.cpython-313.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜vectorstore.py
 ┣ 📂risk
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜drawdown.py
 ┃ ┣ 📜implied_vol.py
 ┃ ┗ 📜volatility.py
 ┣ 📂router
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜clova_router.cpython-313.pyc
 ┃ ┗ 📜clova_router.py
 ┣ 📂scripts
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜test_pipeline.cpython-313.pyc
 ┃ ┣ 📜ingest_all.py
 ┃ ┗ 📜test_pipeline.py
 ┣ 📂tests
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-313.pyc
 ┃ ┃ ┣ 📜test_chunk_embed.cpython-313.pyc
 ┃ ┃ ┣ 📜test_rag.cpython-313.pyc
 ┃ ┃ ┗ 📜test_rag_mock.cpython-313.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜test_agent.py
 ┃ ┣ 📜test_chunk_embed.py
 ┃ ┣ 📜test_chunker.py
 ┃ ┣ 📜test_dart.py
 ┃ ┣ 📜test_openai.py
 ┃ ┣ 📜test_rag.py
 ┃ ┣ 📜test_rag_mock.py
 ┃ ┗ 📜test_yahoo_client.py
 ┣ 📂utils
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜entity_extraction.py
 ┃ ┣ 📜helper.py
 ┃ ┗ 📜ticker_map.py
 ┣ 📜.DS_Store
 ┣ 📜chainlit_app.py
 ┣ 📜config.py
 ┣ 📜dart_fss_client.py
 ┗ 📜rag_pipeline.py
```

## Setup

```bash
# 1. Create & activate your virtualenv
python3 -m venv finagent
```

```bash
source finagent/bin/activate
```

```bash
# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the RAG integration tests
python -m app.tests.test_rag
```

```bash
# Spin up a quick static file server (for local demos)
python -m http.server 8000
```

```bash
# Launch the FastAPI server (with hot-reload)
uvicorn app.agent.server:app --reload
```
