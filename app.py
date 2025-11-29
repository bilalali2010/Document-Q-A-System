import os
import streamlit as st
from llm import get_llm
from doc_chat import run_doc_chat
from research_agent import run_research_agent
from sql_generator import run_sql_generator

st.set_page_config(page_title="Month2 — Orchestration & Data", layout="wide")
st.title("Month 2 — LangChain & Agents — Tools Hub")

# Ensure user set OPENROUTER_API_KEY in Streamlit secrets or env
if not (os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", None)):
    st.error("Please set OPENROUTER_API_KEY in Streamlit secrets or environment before running.")
    st.stop()

# Initialize LLM once and reuse
llm = get_llm()

st.sidebar.title("Tools")
tool = st.sidebar.radio("Choose a tool", ["Document Q&A / Chat", "Research Assistant", "SQL Query Generator", "About"])

if tool == "Document Q&A / Chat":
    run_doc_chat(llm)
elif tool == "Research Assistant":
    run_research_agent(llm)
elif tool == "SQL Query Generator":
    run_sql_generator(llm)
else:
    st.header("About")
    st.write("""
    Tools included:
    - Document Q&A & Chat (upload PDF/DOCX/TXT, build embeddings with sentence-transformers, query with RetrievalQA)
    - Research Assistant (DuckDuckGo free search + LLM summarization)
    - SQL Query Generator (convert NL to SQL — executes only SELECT statements for safety)
    
    Make sure you added the secret `OPENROUTER_API_KEY` in Streamlit secrets.
    """)
