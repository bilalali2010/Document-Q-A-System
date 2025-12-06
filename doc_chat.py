from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# -------------------------
# Load document text
# -------------------------
def load_document(file):
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    # TXT files
    return file.read().decode("utf-8").strip()


# -------------------------
# Build FAISS vectorstore
# -------------------------
def build_vectorstore(text):
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build vectorstore
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore
