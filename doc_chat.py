import os
import requests
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

class OpenRouterEmbeddings(Embeddings):
    """Embeddings wrapper for OpenRouter API."""
    def __init__(self, model="text-embedding-3-small", api_key=None, base_url=None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_API_BASE") or "https://api.openrouter.ai/v1"

    def embed_documents(self, texts):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/embeddings"
        data = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(url, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


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
    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Use OpenRouter embeddings directly
    embeddings = OpenRouterEmbeddings(
        model="text-embedding-3-small"
    )

    # Build FAISS
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore
