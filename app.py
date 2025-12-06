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
# Placeholder for chat messages
# -------------------------
chat_placeholder = st.empty()

def render_chat():
    with chat_placeholder.container():
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.markdown(f"👤 **You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"🤖 **AI:** {msg['content']}")

# Render chat first
render_chat()

# -------------------------
# Input form at the bottom
# -------------------------
with st.form("chat_input", clear_on_submit=True):
    user_msg = st.text_area("Type your question here...", height=60)
    send_btn = st.form_submit_button("Send")

    if send_btn and user_msg.strip():
        if not st.session_state.vectorstore:
            st.warning("❌ Please upload and process a document first.")
        else:
            # Add user message
            st.session_state.history.append({"role": "user", "content": user_msg.strip()})
            render_chat()

            # Retrieve relevant chunks and ask AI
            with st.spinner("🤖 Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(user_msg.strip())
                context = "\n\n".join([d.page_content for d in docs])

                answer = answer_question(context, user_msg.strip())
                st.session_state.history.append({"role": "assistant", "content": answer})

            render_chat()
