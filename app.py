import streamlit as st
from llm import get_llm
from doc_chat import load_document, build_vectorstore

st.set_page_config(page_title="DocuChat AI", layout="wide")

st.title("📄 DocuChat AI")
st.write("Upload a document and chat with it using OpenRouter models.")

uploaded_file = st.file_uploader("Upload PDF or TEXT", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("Reading & indexing document..."):
        text = load_document(uploaded_file)
        vectorstore = build_vectorstore(text)
        retriever = vectorstore.as_retriever()

    st.success("Document processed!")

    question = st.text_input("Ask something about your document:")

    if question:
        llm = get_llm()

        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
Use ONLY the context below to answer.
If the answer is not in the document, reply "Not found in the document."

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        st.write("### Answer:")
        st.write(response.content)
