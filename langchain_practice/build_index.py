from dotenv import load_dotenv
load_dotenv()

import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "Artificial_intelligence.pdf"
INDEX_DIR = "faiss_index"


def get_vectorstore():
    """Load the FAISS index from disk, or build and save it on the first run."""
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

    pages = PyMuPDFLoader(PDF_PATH).load()
    for p in pages:
        p.page_content = re.sub(r"\[(?:\d+|[a-z])\]", "", p.page_content)
        p.metadata["page"] = p.metadata["page"] + 1
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(pages)
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(INDEX_DIR)
    return vs


if __name__ == "__main__":
    vs = get_vectorstore()
    for d, score in vs.similarity_search_with_score("What are AI winters?", k=3):
        print(f"score={score:.3f} page={d.metadata['page']} | {d.page_content[:80]!r}")