from langchain_ollama import ChatOllama

try:
    from .config import LLM_MODEL
except ImportError:
    from config import LLM_MODEL


def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )
