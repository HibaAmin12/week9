from dotenv import load_dotenv
load_dotenv()  # Reads OPENAI_API_KEY from .env

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Documents: text + metadata (where the text came from)
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

# 2. Split: small chunks on purpose so the split is visible
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
chunks = splitter.split_documents(docs)

print(f"{len(docs)} documents -> {len(chunks)} chunks\n")
for i, c in enumerate(chunks):
    print(i, c.metadata, "|", c.page_content)

# 3. Embed + store
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.from_documents(chunks, emb)

# 4. Search (questions are in English now)
print("\n--- Search ---")
for q in ["How do plants make food?", "Why do things fall down?"]:
    print("\nQuestion:", q)
    for d, score in vs.similarity_search_with_score(q, k=2):
        print(f"  score={score:.3f} {d.metadata} | {d.page_content[:70]}")