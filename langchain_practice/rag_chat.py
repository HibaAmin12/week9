# rag_chat.py
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rag_rerank import retrieve, format_docs, prompt, llm   # reuse what we built

# Prompt that rewrites a follow-up into a standalone question
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's last message as a standalone question, using the chat history "
     "to resolve words like 'it', 'that', 'the first one'. "
     "If it is already standalone, return it unchanged. Return ONLY the question."),
    ("human", "Chat history:\n{history}\n\nLast message: {question}"),
])
rewriter = rewrite_prompt | llm | StrOutputParser()

history = []   # list of (question, answer) pairs


def ask(question):
    # 1. Make the question standalone (skip on the first turn)
    if history:
        h = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])  # last 3 turns only
        standalone = rewriter.invoke({"history": h, "question": question})
    else:
        standalone = question

    # 2. Retrieve with the standalone question
    docs = retrieve(standalone)

    # 3. Answer
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(docs), "question": standalone}
    )

    history.append((question, answer))
    return standalone, answer, docs


if __name__ == "__main__":
    for q in ["What are AI winters?",
              "When did the first one happen?",
              "Who won the 2022 FIFA World Cup?"]:
        standalone, answer, docs = ask(q)
        print("\nUser      :", q)
        print("Rewritten :", standalone)
        print("Answer    :", answer)
        print("Pages     :", [d.metadata["page"] for d in docs])