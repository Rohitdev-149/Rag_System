from langchain_core.output_parsers import StrOutputParser

try:
    from .llm import get_llm
    from .prompt import get_prompt
    from .retriever import get_retriever
except ImportError:
    from llm import get_llm
    from prompt import get_prompt
    from retriever import get_retriever


def start_chat():
    retriever = get_retriever()
    llm = get_llm()
    prompt = get_prompt()
    chain = prompt | llm | StrOutputParser()

    print("=" * 60)
    print("Resume AI Assistant")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        answer = chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )

        print()
        print("Assistant")
        print("-" * 60)
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    start_chat()
