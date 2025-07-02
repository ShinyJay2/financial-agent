# app/entity_extraction.py

from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms.base import LLM

from app.hyperclova_client import ask_hyperclova

class HyperClovaLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "hyperclova"

    def _call(self, prompt: str, stop=None) -> str:
        # Delegate to your existing HyperClova client
        return ask_hyperclova(prompt)

class Company(BaseModel):
    company_name: str = Field(..., description="Korean company name extracted from the user query")

# Build the output parser and prompt
parser = PydanticOutputParser(pydantic_object=Company)

prompt_template = """
Extract the Korean company name from this user query.
Query: "{query}"
Return a JSON object with a single field "company_name".
"""
prompt = PromptTemplate.from_template(prompt_template)

# Instantiate the chain
llm = HyperClovaLLM()
extract_chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)

def extract_company_name(query: str) -> str:
    """
    Run the extraction chain on the given query and return the company_name.
    """
    # This returns a JSON string like: {"company_name":"삼성전자"}
    result_json = extract_chain.run({"query": query})
    # Parse it into our Pydantic model
    company = Company.parse_raw(result_json).company_name
    return company
