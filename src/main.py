# # Calling an LLM app usually takes 3 steps:

# Prompt: fill a template with your input - a formatted prompt
# Model: send the prompt to the LLM - a response object
# Parser: pull the plain text out of that response - a string


# A Runnable is any LangChain component that follows one standard contract:
# it takes an input, does something, and returns an output, using the same method names.

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
)  # PromptTemplate is a reusable blueprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

information = """Mukesh Dhirubhai Ambani (born 19 April 1957) is an Indian businessman. 
He is the chairman and managing director of Reliance Industries, the largest public company in India by market capitalisation in 2025. 
As of June 2026, he is the richest person in Asia and the 22nd richest in the world, with a net worth of US$91.8 billion.[5][6] 
He has attracted fame due to his growth and wealth, and criticism for being a plutocrat,[7] and reports of market manipulation, political corruption, cronyism, and exploitation.[8][9][10][11]

Born in 1957 to Dhirubhai Ambani, the founder of Reliance Industries, and Kokilaben in Aden, 
Mukesh Ambani completed his studies at St. Xavier's College, Mumbai and Institute of Chemical Technology. 
He dropped out of Stanford University in 1980 to join Reliance Industries. He expanded the energy ventures of the company, and directed the set up of its largest petroleum refinery at Jamnagar. 
He took an increasing role in the company after his father developed health issues in the late 1980s. After the death of his father in 2002, 
the ownership of the companies in the group was divided between him and his younger brother Anil Ambani.[12] 
Later, he expanded the portfolio of the group and launched or acquired several ventures in energy, finance, healthcare, media, telecommunications, and sports."""

# An f-string fills the value immediately, at the moment the line runs.
# That produces a plain str, which is not a Runnable, so it can’t be piped with |

summary_template = """
Given the information {information} about a person, I want you to create:
1. Short summary
2. Two interesting facts about them.
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)
# input_variables=["information"] declares which slots exist.

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0
)  # if using gemini

# llm = ChatOllama(model="gemma3:270m", temperature=0.5)  # if using Ollama

# Every LangChain component is a Runnable, meaning it has the same methods: invoke, stream, batch.
# When you join them with |, the combined chain is also a Runnable, so it has those methods too.
chain = summary_prompt_template | llm | StrOutputParser()

response = chain.invoke({"information": information})

print(response)
