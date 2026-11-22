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
        query: The query to search for in webLangGraph
ID
Feedback
Input
Output
Attributes
Feedback
Add feedback
Input
Markdown
User
Search for 3 job posting for an ai/ml intern in kolkata using langchain on linked in and list their details.
Output
Markdown
User
Search for 3 job posting for an ai/ml intern in kolkata using langchain on linked in and list their details.
AI
tavily_search
ID
queryAI/ML intern job postings in Kolkata LinkedIn LangChain
tavily_search
ID
{"query": "AI/ML intern job postings in Kolkata LinkedIn LangChain", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://in.linkedin.com/jobs/view/ai-agent-developer-intern-langchain-deep-agents-healthcare-unpaid-at-aarya-foundation-4325219085", "title": "AI Agent Developer Intern (LangChain Deep Agents / Healthcare) - Unpaid at Aarya Foundation — India | LinkedIn Jobs", "content": "### Applied AI Engineer\n\n BJAK\n\n Greater Kolkata Area\n Artificial Intelligence Engineer \n\n ### Artificial Intelligence Engineer\n\n viamagus\n\n India\n AI Engineer \n\n ### AI Engineer\n\n Papigen\n\n India\n AI Engineer Intern - AI/ML/GenAI Design & Development \n\n ### AI Engineer Intern - AI/ML/GenAI Design & Development\n\n Product Manager Accelerator\n\n India\n AI Developer Jobs in India \n\n ### AI Developer Jobs in India\n\n India $30.00 - $50.00\n\nKnow when new jobs open up [...] Greater Kolkata Area ₹500,000.00 - ₹700,000.00\n\n## People also viewed\n\n Gen AI \n\n ### Gen AI\n\n HCLTech\n\n Chennai, Tamil Nadu, India\n AI Solutions Engineer \n\n ### AI Solutions Engineer\n\n Sleek\n\n India\n AI Platform Engineer \n\n ### AI Platform Engineer\n\n PairSoft\n\n Ahmedabad, Gujarat, India\n GenAI Developer \n\n ### GenAI Developer\n\n Infosys\n\n Bengaluru East, Karnataka, India\n Generative AI Engineer \n\n ### Generative AI Engineer\n\n SentrAIx\n\n India\n Applied AI Engineer [...] ### AI Engineer\n\n #### India Health Link\n\n Chennai, Tamil Nadu, India\n Generative AI Engineer \n\n ### Generative AI Engineer\n\n #### iShare Inc.\n\n India ₹2,500,000.00 - ₹3,000,000.00\n AI Engineer (Remote) \n\n ### AI Engineer (Remote)\n\n #### Joblet-AI\n\n India\n Full Stack AI Engineer WFH \n\n ### Full Stack AI Engineer WFH\n\n #### Qubrid AI\n\n Kolkata, West Bengal, India\n AI Engineer \n\n ### AI Engineer\n\n #### Ayurak AI\n\n India\n Associate AI Engineer (GenAI / Agentic AI)", "score": 0.7285552, "raw_content": null, "id": "139ec3-00"}, {"url": "https://in.indeed.com/q-ai-intern-l-kolkata,-west-bengal-jobs.html", "title": "25 Ai Intern Job Vacancies in Kolkata, West Bengal", "content": "29 Ai Intern jobs available in Kolkata, West Bengal. Has worked with LangChain for prompt orchestration, tool calling, and simple chain-based workflows.", "score": 0.6724393, "raw_content": null, "id": "b6424c-01"}, {"url": "https://in.linkedin.com/in/kishanraj58", "title": "Kishan Raj - Aspiring AI/ML Engineer | Generative AI | RAG | LangChain", "content": "# Kishan Raj\n\nHeadline: Aspiring AI/ML Engineer | Generative AI | RAG | LangChain | FastAPI | Python | IIT Patna Intern\nLocation: Greater Kolkata Area India\nExperience: 2 months\nLinkedIn: · 157 followers\nDescription: Final-year Computer Science undergraduate passionate about Artificial Intelligence…\n\n## Experience\n- Summer Intern at Indian Institute of Technology, Patna (May 2025 to Jul 2025)\n\n## Education\n- Sikkim Manipal Institute of Technology, Under graduate, Ai&ml [...] ## Skills\noptimization, evaluation, pipelines, frameworks, ethical conduct, deep learning frameworks, computer science, preprocessing, research, models, python, learning frameworks, generative models, deepfakes, media\n\n## Certifications\n- Deepfake Generator, Indian Institute of Technology, Patna (2025) [...] - Liked: Excited to share that I've started my 1-year internship at Keysight Technologies I'm truly grateful to Keysight Technologies for this...\n- Liked: Every journey starts with curiosity. Over the past few months, Punit Gautam has embraced every opportunity to learn, collaborate, and grow...\n- Liked: Excited to share that I've successfully completed my Summer Internship in the IT FHQ Department at Oil India Limited, Duliajan, Assam ,...", "score": 0.6617043, "raw_content": null, "id": "697d10-02"}, {"url": "https://in.linkedin.com/jobs/view/artificial-intelligence-intern-at-hivyr-ai-4403930323", "title": "Artificial Intelligence Intern at Hivyr.ai — Greater Kolkata Area", "content": "This role offers hands-on exposure to real-world AI and machine learning projects, using frameworks like LangChain and LlamaIndex", "score": 0.65201986, "raw_content": null, "id": "7d3930-03"}, {"url": "https://in.linkedin.com/jobs/view/machine-learning-intern-at-core-platform-4409381043", "title": "Machine Learning Intern at CORE Platform — Greater Kolkata Area | LinkedIn Jobs", "content": "- AI Data Intern\n \n ### AI Data Intern\n- ML Engineer\n \n ### ML Engineer\n- AI/ML Engineer - Production AI & LLM Systems (India)\n \n ### AI/ML Engineer - Production AI & LLM Systems (India)\n- Junior Lead ML Engineer - Computer Vision\n \n ### Junior Lead ML Engineer - Computer Vision\n- Modelling Resident\n \n ### Modelling Resident\n- AI Engineer\n \n ### AI Engineer\n- AI Engineer\n \n ### AI Engineer\n- AI Engineer\n \n ### AI Engineer\n- AI Engineer\n \n ### AI Engineer [...] ### AI Engineer\n- Artificial Intelligence Engineer\n \n ### Artificial Intelligence Engineer [...] ### Machine Learning Intern\n\n# Machine Learning Intern\n\n#### CORE Platform Greater Kolkata Area See who CORE Platform has hired for this role\n\nSee who CORE Platform has hired for this role\n\n- ### Seniority level\n \n Internship\n- ### Employment type\n \n Internship\n- ### Industries\n \n Information Services\n\nReferrals increase your chances of interviewing at CORE Platform by 2x\n\nSee who you know\n\nGet notified when a new job is posted.\n\n## Similar jobs\n\n## People also viewed", "score": 0.5848048, "raw_content": null, "id": "a1663d-04"}, {"url": "https://www.linkedin.com/posts/e-solve-group_hiring-aiml-engineer-intern-nlp-generative-activity-7445828554472407040-g2fT", "title": "We're looking for an AI/ML Engineer Intern to work on… | e-Solve Group", "content": "Hiring: AI/ML Engineer Intern (NLP, Generative AI, AI Workflow) Location: Lake Gardens, Kolkata (WFO) Role: Internship About the Role:", "score": 0.5371628, "raw_content": null, "id": "15e0f8-05"}, {"url": "https://in.linkedin.com/jobs/view/machine-learning-engineering-intern-at-routed-ai-4257889776", "title": "Machine Learning Engineering Intern at Routed AI — India | LinkedIn Jobs", "content": "•⁠ ⁠You think like a product engineer - not just writing code, but testing it across real-world use cases, especially with a student end-user in mind. You’re quick to debug, validate edge cases, and ensure APIs actually work in the wild\n\n•⁠ ⁠Bonus: Experience building \\personalized recommendation systems\\ or ranking models\n\n•⁠ ⁠Bonus: Familiarity with \\Azure\\, LangChain, or career-related data (roles, jobs, skills)\n\n•⁠ ⁠Bonus: Based in or near Ahmedabad (hybrid preferred) [...] •⁠ ⁠Build and tune \\AI agents\\ (e.g., retrieval + reasoning flows)\n\n•⁠ ⁠Optimize \\LLM prompts, responses, memory, and evals\\ to improve agent accuracy and speed\n\n•⁠ ⁠Speed up \\massive data querying and embeddings pipelines\\ using SQL/Snowflake + vector DBs\n\n•⁠ ⁠Implement \\custom evals and test harnesses\\ to ensure reliability across agent use cases\n\n•⁠ ⁠Integrate and test new tools to power our APIs or autonomous agents\n\n•⁠ ⁠Deploy your work end-to-end: from local test → API → live user flow [...] •⁠ ⁠\\Join early\\: Help define the backbone of a product already in real use by students, professionals, and institutions\n\nHow to Apply\n\nEmail founders@routedai.com with:\n\n•⁠ ⁠Your resume and LinkedIn\n\n•⁠ ⁠A short note on why this excites you\n\n•⁠ ⁠GitHub or examples of past backend/ML projects\n\n•⁠ ⁠(Optional bonus): Links, screenshots, or code showing agents, LLM projects, or infra you’ve worked on\n\n ### Seniority level\n\n Internship\n ### Employment type\n\n Internship\n ### Industries", "score": 0.46493194, "raw_content": null, "id": "1f4105-06"}, {"url": "https://in.linkedin.com/jobs/view/ai-ml-developer-intern-at-legalxonline-4426093690", "title": "AI/ML Developer Intern at LegalXOnline", "content": "Apply for AI/ML Developer Intern at LegalXOnline in Kolkata, West Bengal, India. Stipend: ₹4,000/month + Goodies + Performance-Based Incentives", "score": 0.46320146, "raw_content": null, "id": "1a2ba1-07"}, {"url": "https://in.linkedin.com/jobs/technology-intern-%28ai-ml%29-jobs", "title": "Technology Intern (ai Ml) Jobs in India (519 Open Roles) | LinkedIn", "content": "Get notified when a new job is posted.\n\n# 519 Technology Intern (ai Ml) Jobs in India [...] - ### Artificial Intelligence Intern\n \n #### Innovfix Private Limited\n- Ph D/Research Intern - AI Research Lab\n \n ### Ph D/Research Intern - AI Research Lab\n \n #### Hewlett Packard Enterprise\n- PhD Intern, AI ML in Wireless L1/L2 - Fall 2026\n \n ### PhD Intern, AI ML in Wireless L1/L2 - Fall 2026\n \n #### NVIDIA [...] #### Sankar Group\n- Moonshot Internship - AI Engineer, Decision Engine (Fast-Track to FT Role)\n \n ### Moonshot Internship - AI Engineer, Decision Engine (Fast-Track to FT Role)\n \n #### Unloq®\n- AI/ML Intern – Agentic AI (Autonomous Agents)\n \n ### AI/ML Intern – Agentic AI (Autonomous Agents)\n \n #### Stackular\n- AI Agent Development Internship in Odisha, Bhubaneswar (Hybrid)\n \n ### AI Agent Development Internship in Odisha, Bhubaneswar (Hybrid)\n \n #### Sankar Group", "score": 0.36658973, "raw_content": null, "id": "6425b6-08"}, {"url": "https://tavily.com/knowledge/company/pjYU3mXgJHWz", "title": "LangChain", "content": "# LangChain\nID: pjYU3mXgJHWz\nWebsite: www.langchain.com\nLegal name: LangChain, Inc.\nHeadquarters: San Francisco, CA, United States\nLinkedIn: www.linkedin.com/company/langchain · 551,850 followers\nIndustry: Technology, Information and Internet\nCompany type: Privately Held\nBusiness model: B2B\nSaaS: yes [...] ## Technology\n- Detected: 116\n- kubernetes · last seen 2026-09-21\n- playwright · last seen 2026-09-21\n- postgresql · last seen 2026-09-21\n- microsoft azure · last seen 2026-09-21\n- google analytics · last seen 2026-09-21\n- docker · last seen 2026-09-21\n- microsoft · last seen 2026-09-21\n- google cloud platform · last seen 2026-09-21\n\n## Reputation\n- Open roles: 150\n\n## Similar companies\n- langdock\n- llamaindex\n- flowiseai\n- flowise\n- arize ai\n- weights & biases\n- trulens\n- isotropic networks [...] ## Products and services\nBusiness areas: Agent Development Platform, AI Agent Infrastructure, LLM Observability and Monitoring, AI Model Governance\nProducts: Agent Development Platform, Agent Observability Software, Agent Evaluation Software, Agent Deployment Software, Code Sandbox for Agent Execution, LLM Gateway for Model Call Control, No-Code Agent Builder, Open Source Agent Frameworks", "score": 0.18832399, "raw_content": null, "id": "6038ba-09"}], "response_time": 0.78, "request_id": "4c7258e1-307a-4e87-b5ce-b692341885ae"}
AI
Based on your request, here are three AI/ML internship opportunities identified in the Kolkata area that involve or value experience with LangChain.
Note: Job listings change frequently. Please verify these links directly on LinkedIn for current status.
1. Artificial Intelligence Intern at Hivyr.ai

    Role Details: This internship provides hands-on exposure to real-world AI and Machine Learning projects.
    Key Requirement: The role specifically mentions working with frameworks like LangChain and LlamaIndex to build and manage AI applications.
    Location: Greater Kolkata Area.
    Link: View on Link
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
