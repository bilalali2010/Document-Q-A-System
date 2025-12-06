import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from llm import get_llm
from duckduckgo_search import ddg

def run_research_agent():
    st.header("🌐 Research Assistant")
    query = st.text_input("Enter your research query:")
    if query:
        results = ddg(query, max_results=3)
        context = "\n".join([f"- {r['title']}: {r['href']}" for r in results])
        llm = get_llm()
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="Use the context below to summarize the query.\nContext:\n{context}\nQuery:\n{query}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        summary = chain.run(context=context, query=query)
        st.write("### Summary")
        st.write(summary)
