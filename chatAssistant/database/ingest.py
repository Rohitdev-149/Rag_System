import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from chatAssistant.chatbot.config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        PDF_PATH,
        QDRANT_URL,
    )
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from chatbot.config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        PDF_PATH,
        QDRANT_URL,
    )


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    reader = PdfReader(PDF_PATH)
    docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": str(PDF_PATH), "page": page_number},
        )
        for page_number, page in enumerate(reader.pages)
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    texts = text_splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
        },
    )

    QdrantVectorStore.from_documents(
        documents=texts,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        force_recreate=True,
    )

    print(f"Stored {len(texts)} resume chunks in Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
