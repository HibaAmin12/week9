"""Document store backed by a LangChain FAISS vector store (one FAISS index per notebook).

Layout on disk (inside the notebook's docs folder):
    <doc_id>/doc.json        units + chunks + metadata      (the source of truth, also used to re-index)
    <doc_id>/original.<ext>  the uploaded file
    _faiss_index/index.faiss + index.pkl + index.json       (FAISS vectors, LangChain docstore, embedding id)

Every FAISS entry is a LangChain `Document`: page_content = heading + chunk text (what gets embedded),
metadata = doc_id, doc, idx, unit, label, text, start. Search is filtered by the selected sources.
"""
import json
import os
import shutil
import threading
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from . import chunking, config, embeddings
from .costs import usd

INDEX_DIRNAME = "_faiss_index"
_FAISS_KW = dict(distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)  # vectors are unit length -> cosine similarity


class NotebookRetriever(BaseRetriever):
    """Standard LangChain retriever: top-k chunks from FAISS, limited to the selected sources."""
    store: Any
    doc_ids: Any = None
    k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> list:
        return self.store.search_docs(query, self.doc_ids, self.k)


class Store:
    def __init__(self, root: str = None):
        self.root = root or config.DATA_DIR
        os.makedirs(self.root, exist_ok=True)
        self.index_dir = os.path.join(self.root, INDEX_DIRNAME)
        self.lock = threading.RLock()
        self.embedder = embeddings.LazyEmbeddings()
        self.vs = None      # FAISS vector store (None while empty)
        self.stale = {}     # doc_id -> name, saved on disk but not in the current FAISS index (re-index fixes it)
        self.docs = {}      # id -> {"meta": {...}, "units": [...], "chunks": [...]}
        self._load_all()

    # ---------- persistence ----------
    def _dir(self, doc_id):
        return os.path.join(self.root, doc_id)

    def folder(self, doc_id):
        return self._dir(doc_id)

    @staticmethod
    def _chunk_id(doc_id, i):
        return f"{doc_id}:{i}"

    def _load_index(self):
        meta = os.path.join(self.index_dir, "index.json")
        if not os.path.isfile(meta):
            return None
        with open(meta) as f:
            if json.load(f).get("embed_id") != embeddings.embed_id():
                return None  # built with another embedding model: unusable, re-index needed
        try:  # our own file, written by save_local, so the pickle is trusted
            return FAISS.load_local(self.index_dir, self.embedder, allow_dangerous_deserialization=True, **_FAISS_KW)
        except Exception as exc:
            print(f"[faiss] could not load index, re-index needed: {exc}")
            return None

    def _save_index(self):
        if self.vs is None or self.vs.index.ntotal == 0:
            self.vs = None
            shutil.rmtree(self.index_dir, ignore_errors=True)
            return
        self.vs.save_local(self.index_dir)
        with open(os.path.join(self.index_dir, "index.json"), "w") as f:
            json.dump({"embed_id": embeddings.embed_id()}, f)

    def _indexed_doc_ids(self):
        if self.vs is None:
            return set()
        return {d.metadata.get("doc_id") for d in self.vs.docstore._dict.values()}

    def _load_all(self):
        on_disk = {}
        for doc_id in sorted(os.listdir(self.root)):
            path = os.path.join(self._dir(doc_id), "doc.json")
            if os.path.isfile(path):
                with open(path) as f:
                    on_disk[doc_id] = json.load(f)
        self.vs = self._load_index()
        indexed = self._indexed_doc_ids()
        orphans = [i for i, d in self.vs.docstore._dict.items() if d.metadata.get("doc_id") not in on_disk] if self.vs else []
        if orphans:  # a source folder was removed by hand
            self.vs.delete(orphans)
            self._save_index()
        for doc_id, d in on_disk.items():
            if doc_id in indexed:
                self.docs[doc_id] = d
            else:
                self.stale[doc_id] = d["meta"]["name"]

    # ---------- indexing ----------
    @staticmethod
    def _to_documents(doc_id, name, chunks):
        docs = [Document(page_content=c["ctx"], metadata={"doc_id": doc_id, "doc": name, "idx": i, "unit": c["unit"],
                                                          "label": c["label"], "text": c["text"], "start": c["start"]})
                for i, c in enumerate(chunks)]
        return docs, [Store._chunk_id(doc_id, i) for i in range(len(chunks))]

    def _index(self, doc_id, name, chunks):
        docs, ids = self._to_documents(doc_id, name, chunks)
        if self.vs is None:
            self.vs = FAISS.from_documents(docs, self.embedder, ids=ids, **_FAISS_KW)
        else:
            self.vs.add_documents(docs, ids=ids)

    def add(self, doc_id, name, units, chunks, ext, embed_tokens=0, embed_cost=0.0):
        """Embed the chunks, add them to FAISS and save the document. `units` are LangChain Documents."""
        unit_dicts = chunking.units_to_dicts(units)
        meta = {"id": doc_id, "name": name, "ext": ext, "n_units": len(unit_dicts), "n_chunks": len(chunks),
                "chars": sum(len(u["text"]) for u in unit_dicts), "embed_id": embeddings.embed_id(),
                "embed_tokens": embed_tokens, "embed_cost": embed_cost}
        with self.lock:
            self._index(doc_id, name, chunks)          # raises if embedding fails -> nothing is saved
            os.makedirs(self._dir(doc_id), exist_ok=True)
            with open(os.path.join(self._dir(doc_id), "doc.json"), "w") as f:
                json.dump({"meta": meta, "units": unit_dicts, "chunks": chunks}, f)
            self._save_index()
            self.docs[doc_id] = {"meta": meta, "units": unit_dicts, "chunks": chunks}
        return meta

    def reindex(self):
        """Embed sources that are not in the FAISS index (other embedding model, or copied in), using their stored
        chunks. No re-upload needed. Returns (documents, tokens, cost)."""
        done, tokens, cost = 0, 0, 0.0
        with self.lock:
            for doc_id in list(self.stale):
                with open(os.path.join(self._dir(doc_id), "doc.json")) as f:
                    d = json.load(f)
                t = embeddings.count_tokens([c["ctx"] for c in d["chunks"]])
                c = usd(t, config.EMBED_PRICE_PER_M)
                self._index(doc_id, d["meta"]["name"], d["chunks"])
                d["meta"].update(embed_id=embeddings.embed_id(), embed_tokens=t, embed_cost=c)
                with open(os.path.join(self._dir(doc_id), "doc.json"), "w") as f:
                    json.dump(d, f)
                self.docs[doc_id] = d
                del self.stale[doc_id]
                done, tokens, cost = done + 1, tokens + t, cost + c
            if done:
                self._save_index()
        return done, tokens, cost

    def delete(self, doc_id) -> bool:
        with self.lock:
            d = self.docs.pop(doc_id, None)
            self.stale.pop(doc_id, None)
            if d is not None and self.vs is not None:
                present = set(self.vs.docstore._dict)
                ids = [i for i in (self._chunk_id(doc_id, n) for n in range(len(d["chunks"]))) if i in present]
                if ids:
                    self.vs.delete(ids)
                self._save_index()
        shutil.rmtree(self._dir(doc_id), ignore_errors=True)
        return d is not None

    def original_path(self, doc_id):
        d = self._dir(doc_id)
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if f.startswith("original."):
                return os.path.join(d, f)
        return None

    # ---------- queries ----------
    def list(self):
        with self.lock:
            return [d["meta"] for d in self.docs.values()]

    def units(self, doc_id):
        d = self.docs.get(doc_id)
        return None if d is None else d["units"]

    def _selected(self, doc_ids):
        with self.lock:
            return [d for i, d in self.docs.items() if not doc_ids or i in doc_ids]

    @staticmethod
    def _hit(d, idx, score=0.0):
        c = d["chunks"][idx]
        return {"doc_id": d["meta"]["id"], "doc": d["meta"]["name"], "idx": idx, "label": c["label"],
                "unit": c["unit"], "text": c["text"], "score": float(score)}

    @staticmethod
    def hit(doc: Document) -> dict:
        """LangChain Document (from FAISS / reranker) -> the hit dict used by rag.py."""
        m = doc.metadata
        return {"doc_id": m["doc_id"], "doc": m["doc"], "idx": m["idx"], "label": m["label"], "unit": m["unit"],
                "text": m["text"], "score": float(m.get("relevance_score", m.get("score", 0.0)))}

    def as_retriever(self, doc_ids=None, k: int = None) -> NotebookRetriever:
        return NotebookRetriever(store=self, doc_ids=set(doc_ids or ()), k=k or config.RETRIEVE_K)

    def search_docs(self, query: str, doc_ids, k: int) -> list:
        """FAISS similarity search (cosine) limited to the selected sources. Each result carries metadata['score']."""
        with self.lock:
            if self.vs is None or not self.docs:
                return []
            wanted = set(doc_ids or ()) or set(self.docs)
            pairs = self.vs.similarity_search_with_score(
                query, k=k, filter=lambda m: m.get("doc_id") in wanted and m.get("doc_id") in self.docs,
                fetch_k=self.vs.index.ntotal)
        return [Document(page_content=d.page_content, metadata={**d.metadata, "score": float(s)}) for d, s in pairs]

    def expand(self, hit, radius):
        """The hit plus `radius` chunks on each side, as segments [{unit, label, text}].

        Neighbours in the same page/section are merged into one segment; a neighbour on the next or
        previous page becomes its own segment, so an explanation split across pages stays complete.
        """
        d = self.docs[hit["doc_id"]]
        lo, hi = max(0, hit["idx"] - radius), min(len(d["chunks"]) - 1, hit["idx"] + radius)
        segs = []
        for j in range(lo, hi + 1):
            c = d["chunks"][j]
            a, b = self._span(d["units"][c["unit"]]["text"], c)
            if segs and segs[-1]["unit"] == c["unit"]:
                segs[-1]["b"] = max(segs[-1]["b"], b)
            else:
                segs.append({"unit": c["unit"], "label": c["label"], "a": a, "b": b})
        for sg in segs:
            sg["text"] = d["units"][sg["unit"]]["text"][sg.pop("a"):sg.pop("b")]
        return segs

    @staticmethod
    def _span(unit_text, c):
        """Where a chunk sits inside its unit's text."""
        i = unit_text.find(c["text"], max(0, c["start"] - 10))
        if i < 0:
            i = max(unit_text.find(c["text"]), 0)
        return i, i + len(c["text"])

    def sample(self, doc_ids, k):
        """Evenly spaced chunks across the selected documents, in reading order (for summaries)."""
        docs = self._selected(doc_ids)
        total = sum(len(d["chunks"]) for d in docs)
        if total == 0:
            return []
        out = []
        for d in docs:
            n = len(d["chunks"])
            share = max(1, round(k * n / total))
            step = n / min(share, n)
            out += [self._hit(d, int(i * step)) for i in range(min(share, n))]
        return out[:k + len(docs)]
