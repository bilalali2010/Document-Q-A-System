import os
from openai import OpenAI
from langchain.chat_models import ChatOpenAI

def get_llm():
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base=os.getenv("OPENROUTER_API_BASE") or "https://api.openrouter.ai/v1"
    )
    return ChatOpenAI(
        client=client,
        model_name="x-ai/grok-4.1-fast:free",
        temperature=0
    )
