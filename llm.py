import os
import requests
import streamlit as st

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "arcee-ai/trinity-mini:free"

if not API_KEY:
    st.error("❌ OPENROUTER_API_KEY missing.")
    st.stop()


def ask_ai(messages):
    """
    Send messages to OpenRouter and get the AI response.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": messages
    }

    try:
        r = requests.post(API_URL, headers=headers, json=data, timeout=180)
        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"]["content"].strip()
            return reply or "⚠️ Could not generate a response."
        else:
            return f"⚠️ API Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"⚠️ Network error: {e}"


def answer_question(context, question):
    """
    Build a prompt for the AI using the retrieved document context.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions from documents."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    return ask_ai(messages)
