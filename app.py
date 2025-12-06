import streamlit as st
from doc_chat import load_document, build_vectorstore
from llm import answer_question

st.set_page_config(page_title="Document-Q-A-System", layout="wide")
st.title("📄 Document Q&A System")
st.write("Upload a PDF or TXT file and ask questions about it.")

# -------------------------
# Session state
# -------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Upload document
# -------------------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
if uploaded_file:
    with st.spinner("Processing document..."):
        text = load_document(uploaded_file)
        st.session_state.vectorstore = build_vectorstore(text)
    st.success("✅ Document processed! You can now ask questions.")

# -------------------------
# Ask questions
# -------------------------
if st.session_state.vectorstore:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("🤖 Thinking..."):
            # Retrieve relevant chunks
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([d.page_content for d in docs])

            # Ask AI to generate answer
            answer = answer_question(context, question)

            # Update chat history
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": answer})

# -------------------------
# Render chat history
# -------------------------
if st.session_state.history:
    st.write("---")
    st.write("### Chat History")
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"👤 **You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"🤖 **AI:** {msg['content']}")
