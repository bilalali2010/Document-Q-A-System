import os
from langchain.llms import OpenAI

def get_openrouter_llm(model_name: str | None = None, temperature: float = 0.0):
    """
    Initialize a LangChain LLM that talks explicitly to OpenRouter using your OPENROUTER_API_KEY.
    """

    # 1️⃣ Map your OpenRouter key to OPENAI_API_KEY internally (required by LangChain)
    router_key = os.getenv("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not router_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment or Streamlit secrets!")
    os.environ["OPENAI_API_KEY"] = router_key

    # 2️⃣ Set OpenRouter API base explicitly
    base = os.getenv("OPENROUTER_API_BASE") or "https://api.openrouter.ai/v1"
    os.environ["OPENAI_API_BASE"] = base

    # 3️⃣ Select model
    model = model_name or os.getenv("OPENAI_MODEL") or "x-ai/grok-4.1-fast:free"

    # 4️⃣ Return LLM instance (LangChain uses OpenAI wrapper, but now points to OpenRouter)
    llm = OpenAI(model_name=model, temperature=temperature)
    return llm
