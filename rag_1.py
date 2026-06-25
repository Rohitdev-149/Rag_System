import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama

pdf_path = Path(__file__).parent / "RohitResume.pdf"

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()




text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(documents=docs)

embedded = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu",
        "local_files_only": True
    }
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=texts,
#     embedding=embedded,
#     collection_name="resume",
#     url="http://localhost:6333"
# )

retriver = QdrantVectorStore.from_existing_collection(
    embedding=embedded, 
    collection_name="resume",
    url="http://localhost:6333"
)

llm = ChatOllama(
    model="llama3.2"
   )

print("=" * 60)
print("🤖 Resume AI Assistant")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    query = input("\n🧑 You : ")

    if query.lower() in ["exit", "quit", "bye"]:
        print("\n👋 Bye!")
        break
    revelent_chunks = retriver.similarity_search(
    query=query,
    k=3

    )

    context = "\n\n".join(
    doc.page_content for doc in revelent_chunks
    )

    SYSTEM_PROMPT = f"""
    You are a helpful assistant.

    answer the question based on the context provided below. If the answer is not contained within the text below, say "I don't know".

    you guide the candidate to improve his resume for the position of AI/ML engineer based on the context provided below.
    "

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

   

    response = llm.invoke(SYSTEM_PROMPT)

    print("\n===== QUESTION =====\n")
    print(query)

    print("\n===== ANSWER =====\n")
    print(response.content)