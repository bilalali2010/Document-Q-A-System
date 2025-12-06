from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
import os

def load_document(file):
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    return file.read().decode("utf-8").strip()


def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # 👈 Updated embeddings to use OpenRouter
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),  # Your OpenRouter key
        base_url=os.getenv("OPENROUTER_API_BASE") or "https://api.openrouter.ai/v1",  # Point to OpenRouter
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore
