# Financial AI Agent

This agent answers stock-market queries by:
1. Fetching data (e.g. via yfinance)
2. Augmenting prompts with real data
3. Passing them to HyperCLOVA X
4. Returning JSON answers at `/agent?question=`

## Setup

예상 구조도:

financial-agent/
├── .github/                  # CI/CD workflows
│   └── workflows/ci.yml
├── .gitignore
├── .env.example              # template for your .env
├── Dockerfile
├── Makefile                  # convenience tasks: start, test, lint
├── README.md
├── pyproject.toml            # or requirements.txt if you stick with pip
├── poetry.lock               # if you use Poetry
├── docs/                     # design docs, architecture diagrams
│   └── design.md
├── app/                      # your application code
│   ├── main.py
│   ├── config.py
│   ├── hyperclova_client.py
│   ├── yahoo_client.py
│   ├── db.py                 # DB connection & session
│   ├── models.py             # ORM models (e.g. SQLModel/Pydantic)
│   └── routers/              # FastAPI routers
│       └── agent.py
├── migrations/               # if you use Alembic for PostgreSQL
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   └── test_yahoo_client.py
└── demo/                     # optional demo UI
    └── streamlit_app.py


