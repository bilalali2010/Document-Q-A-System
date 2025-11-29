import io
import streamlit as st
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pypdf import PdfReader
import docx

EMBED_MODEL = "all-MiniLM-L6-v2"

# --------- Loaders ----------
def load_pdf_bytes(file_bytes: bytes, name="uploaded_pdf"):
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text()
        if txt and txt.strip():
            docs.append(Document(page_content=txt, metadata={"source": name, "page": i+1}))
    return docs

def load_docx_bytes(file_bytes: bytes, name="uploaded_docx"):
    docs = []
    doc = docx.Document(io.BytesIO(file_bytes))
    for i, para in enumerate(doc.paragraphs):
        t = para.text.strip()
        if t:
            docs.append(Document(page_content=t, metadata={"source": name, "para": i+1}))
    return docs

def load_txt_bytes(file_bytes: bytes, name="uploaded_txt"):
    try:
        s = file_bytes.decode("utf-8")
    except:
        s = file_bytes.decode("latin-1")
    return [Document(page_content=s, metadata={"source": name})]

def load_documents_from_upload(uploaded_files):
    docs = []
    for f in uploaded_files:
        name = getattr(f, "name", "uploaded")
        data = f.read()
        if name.lower().endswith(".pdf"):
            docs.extend(load_pdf_bytes(data, name=name))
        elif name.lower().endswith(".docx") or name.lower().endswith(".doc"):
            docs.extend(load_docx_bytes(data, name=name))
        elif name.lower().endswith(".txt"):
            docs.extend(load_txt_bytes(data, name=name))
        else:
            try:
                docs.extend(load_pdf_bytes(data, name=name))
            except Exception:
                docs.extend(load_txt_bytes(data, name=name))
    return docs

# --------- UI / Runner ----------
def run_doc_chat(llm):
    st.header("Document Q&A / Chat with Documents")
    st.write("Upload PDF / DOCX / TXT. The app will extract text, embed locally with SentenceTransformer, index with Chroma, and answer questions using OpenRouter LLM.")

    uploaded = st.file_uploader("Upload documents (multiple)", accept_multiple_files=True, key="docchat_upload")
    if not uploaded:
        st.info("Upload one or more documents to get started.")
        return

    with st.spinner("Loading documents..."):
        docs = load_documents_from_upload(uploaded)
    st.success(f"Loaded {len(docs)} text chunks from uploads.")

    # Prepare texts and embeddings
    texts = [d.page_content for d in docs]
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_texts(texts, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    if "doc_chat_history" not in st.session_state:
        st.session_state.doc_chat_history = []

    question = st.text_input("Ask a question about uploaded documents", key="doc_q")
    if st.button("Ask") and question:
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        result = chain(question)
        answer = result.get("result") or result.get("answer") or str(result)
        src_docs = result.get("source_documents", [])
        st.session_state.doc_chat_history.append({"q": question, "a": answer, "sources": src_docs})

    st.subheader("Chat History")
    for item in reversed(st.session_state.doc_chat_history[-20:]):
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(f"**A:** {item['a']}")
        if item.get("sources"):
            st.write("**Sources:**")
            for s in item["sources"]:
                meta = getattr(s, "metadata", {}) or {}
                st.write(f"- {meta.get('source','unknown')} (meta: {meta})")
        st.markdown("---")
