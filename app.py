import os
import streamlit as st
from llm import get_openrouter_llm
from doc_chat import run_doc_chat
from research_agent import run_research_agent
from sql_generator import run_sql_generator

st.set_page_config(page_title="DocuChat AI — Tools Hub", layout="wide")
st.title("DocuChat AI — LangChain & OpenRouter Agents Hub")

# Ensure API key is set
if not (os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", None)):
    st.error("Please set OPENROUTER_API_KEY in Streamlit secrets or environment before running.")
    st.stop()

# Initialize OpenRouter LLM
llm = get_openrouter_llm()

st.sidebar.title("Tools")
tool = st.sidebar.radio("Choose a tool", ["Document Q&A / Chat", "Research Assistant", "SQL Query Generator", "About"])

if tool == "Document Q&A / Chat":
    run_doc_chat(llm)
elif tool == "Research Assistant":
    run_research_agent(llm)
elif tool == "SQL Query Generator":
    run_sql_generator(llm)
else:
    st.header("About DocuChat AI")
    st.write("""
    DocuChat AI includes:
    - Document Q&A & Chat (PDF/DOCX/TXT, embeddings, FAISS, OpenRouter LLM)
    - Research Assistant (DuckDuckGo + OpenRouter summarization)
    - SQL Query Generator (NL → SELECT SQL, read-only)
    
    Make sure OPENROUTER_API_KEY is set in your Streamlit secrets.
    """)
