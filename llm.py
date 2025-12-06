import os
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model="google/gemini-2.0-pro-exp:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )
