#  RAG on the real PDF
from dotenv import load_dotenv
load_dotenv()

import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- 1. Load ----------
pages = PyMuPDFLoader("Artificial_intelligence.pdf").load()

# ---------- 2. Clean ----------
for p in pages:
    # Remove Wikipedia reference markers: [1], [23], [a], [b]
    p.page_content = re.sub(r"\[(?:\d+|[a-z])\]", "", p.page_content)
    # Make page numbers human-friendly (1-based)
    p.metadata["page"] = p.metadata["page"] + 1

# ---------- 3. Split ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(pages)
print(f"{len(pages)} pages -> {len(chunks)} chunks")

# ---------- 4. Embed + store (this makes ~382 embedding calls' worth, costs a few cents at most) ----------
vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vs.as_retriever(search_kwargs={"k": 4})

# ---------- 5. Chain ----------
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
    RunnableParallel(docs=retriever, question=RunnablePassthrough())
    .assign(context=lambda x: format_docs(x["docs"]))
    .assign(answer=prompt | llm | StrOutputParser())
)

# ---------- 6. Test ----------
for q in ["When was AI founded as an academic discipline?",
          "What are AI winters?",
          "What caused the growth of deep learning after 2012?",
          "Who won the 2022 FIFA World Cup?"]:      # not in the PDF on purpose
    out = rag.invoke(q)
    print("\nQuestion:", q)
    print("Answer:", out["answer"])
    print("Sources:", [(d.metadata["source"], d.metadata["page"]) for d in out["docs"]])