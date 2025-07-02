# chainlit_app.py

import re
import chainlit as cl

from app.krx_client import get_realtime_price
from app.hyperclova_client import ask_hyperclova
from app.ticker_map import COMPANY_TICKERS as TICKER_MAP

from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1) Define a Pydantic schema for extraction
class Company(BaseModel):
    company_name: str = Field(..., description="Korean company name")

# 2) Build the output parser
parser = PydanticOutputParser(pydantic_object=Company)

# 3) Prompt template to extract the company name
prompt_template = """
Extract the Korean company name from this query:
Query: "{query}"
Return JSON only, with a single key 'company_name'.
"""
prompt = PromptTemplate.from_template(prompt_template)

# 4) Wrap HyperClova as a LangChain LLM
class HyperClovaLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "hyperclova"
    def _call(self, prompt: str, stop=None) -> str:
        return ask_hyperclova(prompt)

llm = HyperClovaLLM()

# 5) Create the extraction chain
extract_chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)

# 6) Simple regex to detect price intent
PRICE_KEYWORDS = ("주가", "현재가", "시세")

@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()

    # 1) If the user asks for a price
    if any(k in query for k in PRICE_KEYWORDS):
        # Extract company name via LangChain
        result_json = extract_chain.run({"query": query})
        company = Company.parse_raw(result_json).company_name

        # Resolve to ticker code
        ticker = TICKER_MAP.get(company)
        if not ticker:
            await cl.Message(
                content=f"❓ Could not resolve company name: {company}"
            ).send()
            return

        # Fetch and reply with real-time price
        try:
            price = get_realtime_price(ticker)
            await cl.Message(
                content=f"💹 {company}({ticker}) current price: {price:,} KRW"
            ).send()
        except Exception as e:
            await cl.Message(
                content=f"❗️ Failed to fetch price for {company}({ticker}): {e}"
            ).send()

    # 2) Otherwise delegate to HyperClova
    else:
        try:
            answer = ask_hyperclova(query)
            await cl.Message(content=answer).send()
        except Exception as e:
            await cl.Message(
                content=f"❌ HyperClova error: {e}"
            ).send()
