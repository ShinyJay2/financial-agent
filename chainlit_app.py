# chainlit_app.py

import re
import chainlit as cl

from app.krx_client import get_realtime_price
from app.hyperclova_client import ask_hyperclova
from app.ticker_map import find_ticker

from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1) Pydantic model for extraction
class Company(BaseModel):
    company_name: str = Field(..., description="Korean company name")

# 2) Parser and prompt
parser = PydanticOutputParser(pydantic_object=Company)
prompt = PromptTemplate.from_template(
    """
Extract the Korean company name from this query:
Query: "{query}"
Return JSON only with a single key 'company_name'.
"""
)

# 3) Wrap HyperClova for LangChain
class HyperClovaLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "hyperclova"

    def _call(self, prompt: str, stop=None) -> str:
        return ask_hyperclova(prompt)

llm = HyperClovaLLM()

# 4) Build the extraction chain
extract_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_parser=parser
)

# 5) Keywords for price intent
PRICE_KEYWORDS = ("주가", "현재가", "시세")

@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()

    # 1) Price intent?
    if any(k in query for k in PRICE_KEYWORDS):
        # a) Extract company_name model
        company_model = extract_chain.predict_and_parse(query=query)
        company = company_model.company_name

        # b) Resolve to ticker
        ticker = find_ticker(company)
        if not ticker:
            await cl.Message(
                content=f"❓ Could not resolve company: {company}"
            ).send()
            return

        # c) Fetch price
        try:
            price = get_realtime_price(ticker)
            await cl.Message(
                content=f"💹 {company} ({ticker}) current price: {price:,} KRW"
            ).send()
        except Exception as e:
            await cl.Message(
                content=f"❗️ Failed fetching price for {company}({ticker}): {e}"
            ).send()

    # 2) General fallback to HyperClova
    else:
        try:
            answer = ask_hyperclova(query)
            await cl.Message(content=answer).send()
        except Exception as e:
            await cl.Message(
                content=f"❌ HyperClova error: {e}"
            ).send()
