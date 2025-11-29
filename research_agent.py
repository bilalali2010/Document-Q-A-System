import streamlit as st
from duckduckgo_search import ddg
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from typing import List

def web_search_duckduckgo(query: str, max_results: int = 5) -> List[str]:
    """
    Simple free search using duckduckgo_search.ddg.
    Returns list of "title — snippet (url)" strings.
    """
    results = ddg(query, max_results=max_results) or []
    out = []
    for r in results:
        title = r.get("title") or ""
        body = r.get("body") or ""
        href = r.get("href") or ""
        out.append(f"{title} — {body} ({href})")
    return out

def run_research_agent(llm):
    st.header("Research Assistant (DuckDuckGo + LLM)")
    st.write("This will perform a free web search (DuckDuckGo) and ask the LLM to synthesize results with short citations.")

    q = st.text_input("Enter research query", key="research_q")
    max_results = st.number_input("Max search results", min_value=1, max_value=10, value=5)
    if st.button("Search") and q:
        with st.spinner("Searching the web..."):
            results = web_search_duckduckgo(q, max_results=int(max_results))
        st.subheader("Search results")
        for r in results:
            st.write("-", r)

        prompt = PromptTemplate(
            input_variables=["question", "search_results"],
            template=(
                "You are a research assistant. Given the following search results, write a concise, well-structured answer "
                "to the user's question. Include inline citations referencing the results by index (e.g., [1], [2]).\n\n"
                "Search results:\n{search_results}\n\nQuestion: {question}\n\nAnswer:"
            )
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        with st.spinner("Synthesizing answer..."):
            answer = chain.run(question=q, search_results="\n".join([f"[{i+1}] {r}" for i, r in enumerate(results)]))
        st.subheader("Answer (synthesized)")
        st.write(answer)
        st.subheader("Sources")
        for i, r in enumerate(results):
            st.write(f"[{i+1}] {r}")
