import streamlit as st
from langchain import LLMChain
from langchain.prompts import PromptTemplate
import sqlite3
import pandas as pd
import re
import tempfile
from typing import Tuple

SELECT_ONLY_RE = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)

def generate_sql(llm, schema_text: str, question: str) -> str:
    prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=(
            "You are an assistant that converts natural language to SQL (SQLite dialect). "
            "Only output a single SQL SELECT statement and nothing else. The database schema is:\n\n{schema}\n\n"
            "Question: {question}\n\nSQL:"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    sql = chain.run(schema=schema_text, question=question).strip()
    # Sometimes LLMs return backticks or triple quotes — clean common wrappers
    if sql.startswith("```"):
        sql = "\n".join(sql.splitlines()[1:-1]).strip()
    # Try to isolate the first statement
    sql = sql.split(";")[0].strip()
    return sql

def safe_execute_sql(db_path: str, sql: str) -> Tuple[str, pd.DataFrame]:
    """
    Only allow SELECT queries. Returns (sql_used, dataframe or error message)
    """
    if not SELECT_ONLY_RE.match(sql):
        return sql, pd.DataFrame({"error": ["Only SELECT queries are allowed for safety."]})

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return sql, df
    except Exception as e:
        return sql, pd.DataFrame({"error": [str(e)]})

def run_sql_generator(llm):
    st.header("SQL Query Generator")
    st.write("Upload a SQLite .db file OR use the demo schema. The generator will create a SELECT SQL and execute it (read-only).")

    db_file = st.file_uploader("Upload SQLite .db file (optional)", type=["db", "sqlite"])
    if db_file:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        t.write(db_file.read())
        t.flush()
        db_path = t.name
        st.success("Database uploaded.")
    else:
        st.info("Using demo in-memory DB (sales).")
        # create a temporary sqlite file for demo
        demo_df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=6, freq="M").astype(str),
            "sales": [100,200,150,300,250,400],
            "region": ["A","B","A","B","A","B"]
        })
        db_path = tempfile.NamedTemporaryFile(delete=False, suffix=".db").name
        conn = sqlite3.connect(db_path)
        demo_df.to_sql("sales", conn, index=False, if_exists="replace")
        conn.close()

    st.markdown("**(Optional)** Provide short schema description for better SQL outputs (e.g., `sales(date TEXT, sales INTEGER, region TEXT)`).")
    schema_text = st.text_area("Schema (optional)", height=80, value="sales(date TEXT, sales INTEGER, region TEXT)")
    question = st.text_input("Ask your data question (e.g., 'total sales by region')", key="sql_q")
    if st.button("Generate & Run") and question:
        with st.spinner("Generating SQL..."):
            sql = generate_sql(llm, schema_text, question)
        st.write("**Generated SQL (raw)**")
        st.code(sql)
        with st.spinner("Running SQL (read-only)..."):
            used_sql, result = safe_execute_sql(db_path, sql)
        if "error" in result.columns:
            st.error("Error running SQL:")
            st.table(result)
        else:
            st.success("Query succeeded — showing results")
            st.dataframe(result)
