# LangChain RAG Practice

A practical learning project exploring **Large Language Models (LLMs), prompt templates, PDF processing, vector search, and Retrieval-Augmented Generation (RAG)** using LangChain.

The project progresses from simple LLM calls and prompt chains to a PDF-based retrieval workflow, with optional reranking and conversational question handling.

## Project Structure

```text
langchain_practice/
├── Artificial_intelligence.pdf   # PDF document used for RAG practice
├── Basic_LLM_call.py             # Basic LLM call
├── Prompt_template_chain.py      # Prompt templates and LCEL chain practice
├── Pdf_upload.py                 # PDF upload/loading workflow
├── chunk_VS.py                   # Chunking and vector-store practice
├── build_index.py                # Build or load the FAISS index
├── Full_RAG_chain.py             # End-to-end RAG chain
├── rag_rerank.py                 # Retrieval with cross-encoder reranking
├── rag_chat.py                   # Conversational RAG and question rewriting
├── faiss_index/                  # Persisted FAISS index files
├── requirements.txt              # Python dependencies
└── .venv/                        # Local virtual environment (do not commit)
```

## Key Features

- **Basic LLM calls:** Connect to an LLM and inspect its response.
- **Prompt templates:** Build reusable prompts and connect steps with LangChain Expression Language (LCEL).
- **PDF processing:** Load a PDF and prepare its text for retrieval.
- **Chunking:** Split document text into smaller sections suitable for embedding and search.
- **FAISS vector search:** Retrieve candidate chunks using semantic similarity.
- **End-to-end RAG:** Pass retrieved context to an LLM to generate a grounded answer.
- **Cross-encoder reranking:** Re-score retrieved question–chunk pairs to improve relevance ordering.
- **Conversational retrieval:** Use chat history to rewrite follow-up questions before retrieving context.
- **Source metadata:** Keep source and page information available for citations, where the implementation supplies it.

## RAG Pipeline

```text
Question
   |
   v
Question embedding
   |
   v
FAISS vector search
   |
   v
Relevant candidate chunks
   |
   v
Cross-encoder reranking (rag_rerank.py)
   |
   v
Format retrieved context
   |
   v
Prompt template + LLM
   |
   v
Answer
```

For conversational questions, `rag_chat.py` can rewrite a follow-up into a standalone question before retrieval.

## Requirements

- Python 3.10 or newer recommended
- An OpenAI API key for scripts that use OpenAI models or embeddings
- Internet access for API calls and for downloading the reranker model on its first run
- Sufficient memory to load the local cross-encoder model

## Setup

### 1. Open the project directory

```bash
cd ~/Documents/week9/langchain_practice
```

### 2. Create and activate a virtual environment

If `.venv` already exists and is active, skip this step.

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root if the scripts require OpenAI access:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Keep `.env` private and make sure it is listed in `.gitignore`. Never commit real API keys.

## Running the Scripts

Run the scripts from the project root, with the virtual environment activated.

```bash
python Basic_LLM_call.py
python Prompt_template_chain.py
python Pdf_upload.py
python chunk_VS.py
python build_index.py
python Full_RAG_chain.py
python rag_rerank.py
python rag_chat.py
```

Some scripts may depend on the API key, the PDF file, or a previously built FAISS index. Follow the configuration and index paths defined inside each script. If `build_index.py` creates the index, run it before scripts that expect the index to exist.

## File Overview

| File | Responsibility |
|---|---|
| `Basic_LLM_call.py` | Practice making a basic LLM request. |
| `Prompt_template_chain.py` | Practice prompt templates and chaining components using LCEL. |
| `Pdf_upload.py` | Practice loading or preparing a PDF for processing. |
| `chunk_VS.py` | Practice chunking and vector-store operations. |
| `build_index.py` | Create or load the FAISS vector index used for retrieval. |
| `Full_RAG_chain.py` | Connect retrieval, context formatting, prompts, and an LLM into a RAG workflow. |
| `rag_rerank.py` | Retrieve candidate chunks and rerank them using a cross-encoder. |
| `rag_chat.py` | Use recent chat history to rewrite follow-up questions before retrieval and answer generation. |

Descriptions above are based on the filenames and the practice workflow; exact behavior depends on each script's implementation.

## Important Concepts

### Chunking

Chunking divides long documents into smaller text sections. Overlap can preserve context across chunk boundaries. Chunk size and overlap should be adjusted to suit the document and retrieval task.

### Embeddings and FAISS

An embedding model converts text into vectors. FAISS searches for vectors close to the question vector and returns candidate chunks. Similarity search is fast, but the returned chunks may be only partially relevant.

### Cross-Encoder Reranking

A cross-encoder evaluates a question and a candidate chunk together, then assigns a relevance score. A common two-stage approach is:

1. Use FAISS to retrieve a shortlist quickly.
2. Rerank that shortlist with the cross-encoder.
3. Pass the highest-ranked chunks to the LLM.

Reranking can improve relevance but adds processing time. Cross-encoder scores should not be compared directly with FAISS distance scores.

### RAG Answer Generation

The retrieved chunks are formatted and passed to the LLM as context. The prompt can instruct the model to answer from that context and to acknowledge when the information is missing. Prompt instructions help guide behavior but do not guarantee that every answer is correct.

### Conversational RAG

Follow-up questions often depend on earlier messages. Rewriting a follow-up into a standalone question helps the retriever search for the intended topic.

## Troubleshooting

- **API key error:** Check that `.env` exists in the project root and contains a valid `OPENAI_API_KEY`.
- **Module not found:** Activate `.venv`, then run `pip install -r requirements.txt`.
- **FAISS index missing:** Run `build_index.py` and confirm the index directory configured in the code.
- **Reranker first-run delay:** The cross-encoder may need to download model files and cache them locally.
- **Poor retrieval results:** Inspect the returned chunks first; then check chunk boundaries, embedding model language support, the number of candidates retrieved, and reranker ordering.
- **Deprecation warning:** A warning does not necessarily mean the script failed. Check current package documentation before changing imports or dependencies.

## Security and Git

Add these to `.gitignore` if they are not already present:

```gitignore
.venv/
.env
__pycache__/
*.py[cod]
```

Do not commit API keys, virtual-environment files, or private credentials. If a key is accidentally pushed to GitHub, revoke it and generate a replacement.

## Learning Outcomes

By working through this project, you can practice:

- calling LLMs and designing prompt templates;
- connecting LangChain components with LCEL;
- loading and chunking PDF content;
- creating embeddings and searching with FAISS;
- constructing an end-to-end RAG pipeline;
- improving retrieval with cross-encoder reranking; and
- handling follow-up questions with conversational history.


