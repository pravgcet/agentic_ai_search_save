import os
from tabnanny import verbose
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLEAI_STUDIO_API_KEY"), model="gemini-2.0-flash")

#response = llm.invoke("Explain how human brain works in few words")

parser = PydanticOutputParser(return_id=True, pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a helpful assistant that can answer questions about the world.
    Answer the query and use necessary tools.
    Wrap the output in this format and provide no other text\n{format_instructions}
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(llm = llm, prompt = prompt, tools = tools)

agent_executor = AgentExecutor(agent = agent, verbose=True ,tools = tools)

query = input("Enter your query: ")

raw_response = agent_executor.invoke({"query": query})

#print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output"))
 #   print(structured_response.summary)
except:
    print("Failed to parse response")