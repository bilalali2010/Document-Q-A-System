import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings

def load_document(file):
    """
    Load text from PDF or TXT file.
    """
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    # For TXT files
    return file.read().decode("utf-8").strip()


def build_vectorstore(text):
    """
    Split the document text into chunks and create a FAISS vector store
    using OpenRouter-compatible embeddings.
    """
    # 1️⃣ Split text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # 2️⃣ Create embeddings using OpenRouter
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_API_BASE") or "https://api.openrouter.ai/v1"
    )

    # 3️⃣ Build FAISS vector store
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore
