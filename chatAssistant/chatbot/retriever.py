import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

try:
    from .config import (
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        QDRANT_URL,
        TOP_K,
    )
except ImportError:
    from config import (
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        QDRANT_URL,
        TOP_K,
    )


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
        },
    )


def get_retriever():
    embedding = get_embedding_model()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
    )

    return vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K,
        },
    )
