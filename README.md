# Multi-Tool Agentic AI System (Gemini + RAG)

## Overview

This project is a simple implementation of an Agentic AI system built using LangChain and Google Gemini API. The idea behind this project is to create an AI agent that does not just answer questions directly, but can think step-by-step and use different tools depending on the user query.

The agent can:
- Search information from a knowledge base (RAG)
- Perform calculations
- Get current time
- Search the web for latest information

This project was done as part of our Artificial Intelligence Lab.

---

## What makes it “Agentic”?

Unlike normal AI programs which just return answers, this system:
- Decides which tool to use
- Uses multiple tools in one query if needed
- Follows a reasoning process (like thinking → acting → observing)

---

## Tools Used

1. **College Knowledge Base (RAG)**
   - Stores information about KJSSE
   - Uses FAISS for retrieval

2. **Web Search**
   - Uses DuckDuckGo for free search

3. **Calculator**
   - Solves basic math expressions

4. **Time Tool**
   - Gives current date and time

---

## Tech Stack

- Python  
- LangChain  
- Google Gemini API  
- FAISS  
- DuckDuckGo Search  

---

## How It Works

1. User enters a query  
2. Agent understands the question  
3. Agent decides which tool to use  
4. Tool is executed  
5. Final answer is generated  

Sometimes multiple tools are used in one query.

---

## How to Run

### Step 1: Install libraries
pip install -U langchain langchain-community langchain-google-genai faiss-cpu pandas ddgs

### Step 2: Add your API key
import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"

### Step 3: Run the code
python Code.py
