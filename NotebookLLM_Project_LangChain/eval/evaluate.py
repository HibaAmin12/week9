"""Retrieval evaluation. Measures whether the right passage is found, with and without the reranker.

Usage:
  python -m eval.evaluate --docs path/to/file1.pdf path/to/file2.docx --dataset eval/sample_dataset.jsonl

Dataset: one JSON per line: {"question": "...", "answer_contains": "text that must appear in a retrieved chunk"}
Metrics: Hit@k (right passage in top k) and MRR (how high it ranks). Retrieval only, so it costs nothing
when EMBED_PROVIDER=local.
"""
import argparse
import json
import os
import tempfile
import uuid

from app import chunking, config, loaders, rag
from app.store import Store


def build_store(paths):
    store = Store(tempfile.mkdtemp(prefix="eval_"))
    for p in paths:
        units = chunking.merge_small(loaders.load(p))
        chunks = chunking.make_chunks(units)
        store.add(uuid.uuid4().hex[:8], os.path.basename(p), units, chunks, os.path.splitext(p)[1])
        print(f"indexed {p}: {len(chunks)} chunks")
    return store


def evaluate(store, rows, use_rerank, k):
    config.RERANK_ENABLED = use_rerank
    hits, rr = 0, 0.0
    for row in rows:
        found, _, _ = rag.retrieve(store, row["question"], set())
        needle = row["answer_contains"].lower()
        rank = next((i for i, h in enumerate(found[:k], 1) if needle in h["text"].lower()), None)
        if rank:
            hits += 1
            rr += 1 / rank
    n = max(len(rows), 1)
    return hits / n, rr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=config.TOP_K)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.dataset, encoding="utf-8") if l.strip()]
    store = build_store(a.docs)
    print(f"\n{len(rows)} questions, k={a.k}")
    for name, flag in (("vector only", False), ("vector + rerank", True)):
        hit, mrr = evaluate(store, rows, flag, a.k)
        print(f"{name:18s} Hit@{a.k}: {hit:.2%}   MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()
