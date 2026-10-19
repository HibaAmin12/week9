import argparse
import logging
import shutil

from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, POLICIES_FILE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)   # hide noisy HTTP logs
logger = logging.getLogger(__name__)


def load_and_split(path) -> list[Document]:
    """Load the markdown file and split it into chunks.

    Step 1: split by markdown headers, so each policy is its own chunk.
    Step 2: split very long sections by size (safety net).
    """
    text = path.read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "section")])
    sections = header_splitter.split_text(text)

    size_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = size_splitter.split_documents(sections)

    logger.info("Loaded %s -> %d sections -> %d chunks", path.name, len(sections), len(chunks))
    return chunks


def get_embeddings() -> Embeddings:
    """Provider-agnostic: EMBED_MODEL looks like 'openai:text-embedding-3-small'."""
    return init_embeddings(EMBED_MODEL)


def build_vectorstore(rebuild: bool = False) -> Chroma:
    """Create the vector store, or load it if it already exists.

    rebuild=True deletes the old index first, so re-running never creates duplicates.
    """
    if rebuild and CHROMA_DIR.exists():
        logger.info("Removing old index at %s", CHROMA_DIR)
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    # Index only if the collection is empty (first run or after rebuild)
    existing = len(vectorstore.get()["ids"])
    if existing == 0:
        chunks = load_and_split(POLICIES_FILE)
        vectorstore.add_documents(chunks)
        logger.info("Indexed %d chunks", len(chunks))
    else:
        logger.info("Using existing index (%d chunks)", existing)

    return vectorstore


def get_retriever(k: int = 2) -> VectorStoreRetriever:
    """Public entry point: other modules (answer.py, the graph) import this."""
    return build_vectorstore().as_retriever(search_kwargs={"k": k})


def evaluate_retrieval(vectorstore: Chroma) -> None:
    """Manual retrieval check, without any LLM.

    Each case is (question, expected section). In Chroma, LOWER distance = MORE similar.
    """
    cases = [
        ("How long do refunds take?", "Refund Policy"),
        ("Do you ship to India?", "Shipping Policy"),
        ("Can I get my money back?", "Refund Policy"),       # no word 'refund'
        ("My package came broken", "Damaged Items"),          # no word 'damaged'
        ("Can I cancel what I just bought?", "Order Cancellation"),
        ("When can I call you?", "Contact Hours"),
        ("paisay kab wapas milengay?", "Refund Policy"),      # Roman Urdu
        ("Do you accept Bitcoin?", None),                     # NOT in the KB: look at distance
    ]

    correct, scored = 0, 0
    for question, expected in cases:
        results = vectorstore.similarity_search_with_score(question, k=2)
        top_section, top_score = results[0][0].metadata["section"], results[0][1]

        if expected is None:
            status = "INFO"
        else:
            scored += 1
            ok = top_section == expected
            correct += ok
            status = "PASS" if ok else "FAIL"

        print(f"\n[{status}] {question}")
        print(f"  expected: {expected or 'nothing (out of scope)'}")
        for doc, score in results:
            print(f"  -> {doc.metadata['section']:<20} distance={score:.3f}")

    print(f"\nRetrieval accuracy (top-1): {correct}/{scored}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and test the FAQ index")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the index")
    args = parser.parse_args()

    vectorstore = build_vectorstore(rebuild=args.rebuild)
    evaluate_retrieval(vectorstore)


if __name__ == "__main__":
    main()