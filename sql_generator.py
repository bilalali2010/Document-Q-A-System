import streamlit as st
from llm import get_llm

def run_sql_generator():
    st.header("🗄️ SQL Query Generator")
    question = st.text_input("Describe your SQL query in plain English:")
    if question:
        llm = get_llm()
        prompt = f"Generate a safe SELECT-only SQL query for the following question: {question}"
        response = llm.invoke(prompt)
        st.write("### Generated SQL")
        st.code(response.content, language="sql")
