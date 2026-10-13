from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Build the index ----------
docs = [
    Document(
        page_content=(
            "Gravity is the force that pulls objects toward the earth. "
            "Isaac Newton explained that every object attracts every other object. "
            "The strength of this force depends on the mass of the objects and the distance between them. "
            "Because of gravity on Earth, things fall downward."
        ),
        metadata={"source": "physics.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "In photosynthesis, plants make food from sunlight, water and carbon dioxide. "
            "This process takes place inside the chloroplasts in the leaves. "
            "As a result, plants produce glucose and oxygen."
        ),
        metadata={"source": "biology.pdf", "page": 4},
    ),
]
chunks = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30).split_documents(docs)
vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vs.as_retriever(search_kwargs={"k": 3})

# ---------- Chain ----------
def format_docs(docs):
    # "[1] (file, p.N)\ntext" for each chunk, joined with blank lines
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
    RunnableParallel(docs=retriever, question=RunnablePassthrough())  # {"docs", "question"}
    .assign(context=lambda x: format_docs(x["docs"]))                 # + "context"
    .assign(answer=prompt | llm | StrOutputParser())                  # + "answer"
)

# ---------- Test ----------
for q in ["How do plants make food?",
          "What did Newton say about gravity?",
          "What is the capital of Pakistan?"]:   # not in the documents on purpose
    out = rag.invoke(q)
    print("\nQuestion:", q)
    print("Answer:", out["answer"])
    print("Retrieved:", [(d.metadata["source"], d.metadata["page"]) for d in out["docs"]])

print("\n--- Context sent to the LLM for the last question ---")
print(out["context"])