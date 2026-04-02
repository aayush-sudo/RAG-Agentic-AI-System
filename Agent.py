# Import the Libraries

import os
from datetime import datetime

# !pip install -U langchain-google-genai
# !pip install -U langchain
# !pip install -U langchain-community
# !pip install faiss-cpu
# !pip install -U ddgs

# LangChain and Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


# Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAAKdt1vjwfiVQ93V3lvqEw80TLJ0brrBY"


# We have Implemented 4 TOOLS :
# 1. 
#
#
#


# Tool 1: RAG Knowledge Base

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# RAG Data

college_data = [
"K. J. Somaiya School of Engineering (KJSSE) is a private engineering institute located in Vidyavihar, Mumbai, Maharashtra, India. It is a constituent college of Somaiya Vidyavihar University.",
"KJSSE was formerly known as K. J. Somaiya College of Engineering and transitioned into a constituent school under Somaiya Vidyavihar University after the university was established.",
"The KJSSE campus is part of the Somaiya Vidyavihar campus, which spans approximately 65 acres in Mumbai and includes multiple institutions and research centers.",
"KJSSE offers undergraduate B.Tech programs in Computer Engineering, Information Technology, Artificial Intelligence and Data Science, Electronics Engineering, Mechanical Engineering, and Robotics and Artificial Intelligence.",
"The institute offers postgraduate M.Tech programs in specializations such as Computer Engineering, Artificial Intelligence and Data Science, Information Technology, Electronics Engineering (VLSI), and Robotics & Automation.",
"KJSSE also offers PhD programs in various engineering disciplines including Computer Engineering, Information Technology, Electronics, and Mechanical Engineering.",
"The academic curriculum at KJSSE is flexible and includes core subjects, professional electives, open electives, and project-based learning components.",
"KJSSE encourages interdisciplinary learning by offering minors in domains like Artificial Intelligence, Data Science, Cyber Security, Internet of Things (IoT), and Robotics.",
"The institute has active student chapters and clubs such as CSI, ACM, CodeCell, robotics clubs, and technical communities that promote innovation and coding culture.",
"KJSSE hosts major student project teams including Orion Racing India (Formula Student), Redshift Racing (BAJA SAE), Team Eta (Shell Eco-Marathon), and Robocon team.",
"The campus infrastructure includes modern laboratories, Wi-Fi enabled classrooms, a central library, seminar halls, auditorium, sports facilities, and hostel accommodation.",
"KJSSE has a strong placement record with top recruiters such as Microsoft, Accenture, Oracle, and other leading tech companies visiting the campus.",
"The institute promotes entrepreneurship and innovation through riidl (Research Innovation Incubation Design Lab), which supports startups and student-led ventures.",
"KJSSE is approved by AICTE and recognized by UGC, and its programs are accredited by NBA and NAAC with a high grade.",
"KJSSE emphasizes practical and industry-oriented learning through internships, live projects, hackathons, and research initiatives."
]

#vector store and retriever tool
vectorstore = FAISS.from_texts(college_data, embeddings)
retriever = vectorstore.as_retriever()

# Tool 1 : RAG
rag_tool = create_retriever_tool(
    retriever,
    "college_knowledge_base",
    "Searches internal college documents. Use this to find information about the College."
)

# Tool 2: Web Search using DuckDuckGo(Free)
web_search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="Useful for finding current events, live information, or things not found in the company database."
)

# Tool 3: Calculator
@tool
def calculator_tool(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result."""
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error calculating: {e}"

# Tool 4: Current Time
@tool
def time_tool(query: str = "") -> str:
    """Returns the current date and time."""
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


tools = [rag_tool, web_search_tool, calculator_tool, time_tool]


# Gemini 2.5 Flash Agent is used
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Defined agent behaviour
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly capable assistant. You have access to a knowledge base, web search, a calculator, and a clock. Always rely on your tools to answer factual, mathematical, or temporal questions. If searching the knowledge base yields no results, fall back to web search."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent and Executor Tool Created
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# User Prompt
if __name__ == "__main__":
    print("Agent Is Ready, (Type 'exit' to Quit anytime.)\n")
    print("Enter Your Prompt")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Agent stopped")
            break

        try:
            response = agent_executor.invoke({"input": user_input})
            print("Agent:", response["output"])
        except Exception as e:
            print("Error:", str(e))
