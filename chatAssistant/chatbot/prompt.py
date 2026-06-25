from langchain_core.prompts import ChatPromptTemplate


def get_prompt():

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert AI Resume Assistant.

Your responsibilities are:

1. Answer ONLY using the provided context.
2. Never make up information.
3. If information is missing, reply:
"I don't know."
4. Help the candidate improve the resume for AI/ML Engineer roles.
5. Give clear, structured and professional answers.

Context:
{context}
                """
            ),
            (
                "human",
                """
Question:

{question}
                """
            )
        ]
    )

    return prompt