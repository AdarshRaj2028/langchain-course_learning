from dotenv import load_dotenv
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# Pydantic is Python's standard data validation and parsing library.
from pydantic import BaseModel, Field

# When we are creating an agent we need to give two mandatory things to the agent, LLM and Tools
from langchain.tools import tool

# Decorator that turns a regular Python function into a structured tool the LLM can understand and call.
from langchain_core.messages import HumanMessage

# Wraps a user's prompt/input into a standardized message object that the agent or LLM expects.
from tavily import TavilyClient

# Web search tool written by and wrapped inside a langchain tool by Tavily team
from langchain_tavily import TavilySearch

load_dotenv()


class JobPosting(BaseModel):
    """Schema for an individual job posting extracted by the agent"""
    company_name: str
    title: str = Field(description="Name of the job title")
    key_requirements: List[str] = Field(description="Top 3 key requirements skills required by the company like Langchain, Python etc.")
    url: str = Field(description="The URL of the source")
    # Field(description="...") is crucial for LLMs but optional,
    # Pydantic compiles those descriptions into a JSON Schema
    # that tells the LLM what to put inside each field.


class AgentResponse(BaseModel):
    """Schema for agent response with structured job posting details"""

    answer: str = Field(description="Detailed Markdown summary of the job search findings and application advice") 
    sources: List[JobPosting] = Field(
        default_factory=list,
        description="List of Job postings found.",
    )


# default_factory=list tells Pydantic: "If no value is provided for jobs, call list() to create a brand-new empty list [] for each instance."
tavily = TavilyClient()


@tool
def search_web(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for in web
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)  # type: ignore


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
tools = [search_web]  # Using our custom made tool
official_tools = [TavilySearch()]
# agent = create_agent(model=llm, tools=tools)  # When using the custom made tool as web search tool
agent = create_agent(model=llm, tools=official_tools, response_format=AgentResponse)

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Search for 3 job posting for an ai/ml intern in kolkata using langchain on linked in and list their details."
            )
        ]
    }
)
structured: AgentResponse = response["structured_response"]
# IMPORTANT:
# Constrained JSON schemas cause some LLMs to treat generic 'answer' fields as brief summaries
# and offload details into nested objects(sources here), explicitly define every field we want extracted.

print("Answer:\n", structured.answer)
print("\nSources:")
for source in structured.sources:
    print("-> ", source.url)

# When response_format=Schema is passed to create_agent,
# LangChain automatically validates the LLM's final output into a ready-to-use
# Pydantic object accessible directly via
# response["structured_response"] using dot notation, eliminating manual JSON string parsing.


# Manual way of extracting the agents output:

# content = response["messages"][-1].content
# # Direct access (when content is a list of blocks)
# # response                        # 1. dict  -> {"messages": [...]}
# #   ["messages"]                  # 2. list  -> [HumanMessage, AIMessage, ToolMessage, AIMessage]
# #     [-1]                        # 3. AIMessage -> the final reply object
# #       .content                  # 4. list  -> [{'type': 'text', 'text': '...', 'extras': {...}}]
# #         [0]                     # 5. dict  -> {'type': 'text', 'text': '...', 'extras': {...}}
# #           ["text"]              # 6. str   -> 'The weather in Tokyo is currently sunny.'
# print(content[0]["text"])
