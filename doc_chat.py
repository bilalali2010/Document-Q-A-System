import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI

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

    # OpenRouter client
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base=os.getenv("OPENROUTER_API_BASE") or "https://api.openrouter.ai/v1"
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        client=client
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore
