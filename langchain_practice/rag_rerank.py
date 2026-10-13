from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from build_index import get_vectorstore

vs = get_vectorstore()                                           # loads index from disk
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # small local reranker


def retrieve(question, fetch_k=10, top_n=4, debug=False):
    # 1. Wide net: 10 candidates from FAISS
    candidates = vs.similarity_search(question, k=fetch_k)

    # 2. Score each (question, chunk) pair together
    scores = reranker.predict([(question, d.page_content) for d in candidates])

    # 3. Sort by score, higher is better
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    if debug:
        print("  FAISS order  :", [d.metadata["page"] for d in candidates])
        print("  Reranked     :", [(d.metadata["page"], round(float(s), 2)) for d, s in ranked[:top_n]])

    # 4. Keep only the best top_n
    return [d for d, _ in ranked[:top_n]]


def format_docs(docs):
    return "\n\n".join(
        f"[{i}] ({d.metadata['source']}, p.{d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY from the passages below. Add a citation like [1] after every claim. "
     "If the answer is not in the passages, say exactly: 'This information is not in the documents.'\n\n"
     "PASSAGES:\n{context}"),
    ("human", "{question}"),
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag = (
    {"docs": RunnableLambda(retrieve), "question": lambda q: q}   # same shape as before
    | RunnableLambda(lambda x: {**x, "context": format_docs(x["docs"])})
    | RunnableLambda(lambda x: {**x, "answer": (prompt | llm | StrOutputParser()).invoke(x)})
)

if __name__ == "__main__":
    for q in ["When was AI founded as an academic discipline?",
              "What are AI winters?",
              "What caused the growth of deep learning after 2012?",
              "Who won the 2022 FIFA World Cup?"]:
        print("\nQuestion:", q)
        retrieve(q, debug=True)               # shows before/after order
        out = rag.invoke(q)
        print("Answer:", out["answer"])
        print("Sources:", [(d.metadata["page"]) for d in out["docs"]])