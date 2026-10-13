# ---------- Load secrets ----------
from dotenv import load_dotenv
load_dotenv()  # Reads OPENAI_API_KEY from the .env file into the environment

# ---------- Imports ----------
from langchain_core.documents import Document                  # Container: text + metadata
from langchain_core.prompts import ChatPromptTemplate          # Builds chat messages with {variables}
from langchain_core.output_parsers import StrOutputParser      # Turns the LLM message into a plain string
from langchain_core.runnables import RunnablePassthrough, RunnableParallel  # Chain helpers
from langchain_text_splitters import RecursiveCharacterTextSplitter         # Cuts long text into chunks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings      # LLM and embedding model
from langchain_community.vectorstores import FAISS             # Vector store (source of the DeprecationWarning)

# ---------- Step 1: Build the index ----------
docs = [
    Document(
        page_content="Gravity wo force hai jo cheezon ko zameen ki taraf kheenchti hai. ...",
        metadata={"source": "physics.pdf", "page": 1},   # Used later for citations
    ),
    Document(
        page_content="Photosynthesis mein paudhe sooraj ki roshni, ...",
        metadata={"source": "biology.pdf", "page": 4},
    ),
]

# Cut each document into ~150-character chunks; consecutive chunks share ~30 characters
chunks = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30).split_documents(docs)

# Embed every chunk and store the vectors in FAISS (chunk text + metadata is kept alongside)
vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))

# Wrap the store as a retriever: string in -> list of top-3 Documents out
retriever = vs.as_retriever(search_kwargs={"k": 3})

# ---------- Step 2: Helper to format chunks for the prompt ----------
def format_docs(docs):
    # Produces: "[1] (file, p.N)\ntext\n\n[2] ..."
    # The numbers let the LLM cite [1], [2], and let us map back to docs[i-1]
    return "\n\n".join(
        f"[{i}] ({d.metadata['source']}, p.{d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)   # start numbering at 1
    )

# ---------- Step 3: Prompt ----------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY from the passages below. Add a citation like [1] after every claim. "
     "If the answer is not in the passages, say exactly: 'Documents mein ye maloomat nahi hai.'\n\n"
     "PASSAGES:\n{context}"),     # {context} is filled by format_docs output
    ("human", "{question}"),      # {question} is the user's question
])

# ---------- Step 4: LLM ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # temperature=0 -> stable, non-creative answers

# ---------- Step 5: The chain ----------
rag = (
    # Send the same input (the question) down two branches in parallel:
    #   docs     -> retriever finds chunks
    #   question -> passed through unchanged
    # Result: {"docs": [...], "question": "..."}
    RunnableParallel(docs=retriever, question=RunnablePassthrough())

    # Add a new key "context" built from docs (existing keys are kept)
    .assign(context=lambda x: format_docs(x["docs"]))

    # Add a new key "answer": prompt -> LLM -> string
    .assign(answer=prompt | llm | StrOutputParser())
)

# ---------- Step 6: Test ----------
for q in ["Paudhe khana kaise banate hain?",
          "Newton ne gravity ke bare mein kya bataya?",
          "Pakistan ka capital kya hai?"]:      # Not in the documents on purpose
    out = rag.invoke(q)                          # Runs the full chain for one question
    print("\nSawal:", q)
    print("Jawab:", out["answer"])
    print("Retrieved:", [(d.metadata["source"], d.metadata["page"]) for d in out["docs"]])

print("\n--- Context sent to the LLM for the last question ---")
print(out["context"])