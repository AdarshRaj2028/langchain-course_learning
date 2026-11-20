from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# When we are creating an agent we need to give two mandatory things to the agent, LLM and Tools
from langchain.tools import tool

# Decorator that turns a regular Python function into a structured tool the LLM can understand and call.
from langchain_core.messages import HumanMessage

# Wraps a user's prompt/input into a standardized message object that the agent or LLM expects.
from tavily import TavilyClient
# Web search tool written by and wrapped inside a langchain tool by Tavily team 
from langchain_tavily import TavilySearch

load_dotenv()

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
agent = create_agent(model=llm, tools=official_tools)

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Search for 3 job posting for an ai/ml intern in kolkata using langchain on linked in and list their details."
            )
        ]
    }
)

content = response["messages"][-1].content
# Direct access (when content is a list of blocks)
# response                        # 1. dict  -> {"messages": [...]}
#   ["messages"]                  # 2. list  -> [HumanMessage, AIMessage, ToolMessage, AIMessage]
#     [-1]                        # 3. AIMessage -> the final reply object
#       .content                  # 4. list  -> [{'type': 'text', 'text': '...', 'extras': {...}}]
#         [0]                     # 5. dict  -> {'type': 'text', 'text': '...', 'extras': {...}}
#           ["text"]              # 6. str   -> 'The weather in Tokyo is currently sunny.'
print(content[0]["text"])
