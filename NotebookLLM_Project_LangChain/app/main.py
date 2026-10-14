"""Flask API + static UI (RAG pipeline built with LangChain + FAISS).

GET    /api/notebooks                              list notebooks
POST   /api/notebooks                              create {name}
GET    /api/notebooks/<nid>                        notebook, sources, saved chat, cost totals
PATCH  /api/notebooks/<nid>                        rename {name}
DELETE /api/notebooks/<nid>                        delete notebook and its sources
POST   /api/notebooks/<nid>/upload                 multipart file -> indexed source + embedding cost
GET    /api/notebooks/<nid>/docs/<id>/units        pages/sections (source viewer)
DELETE /api/notebooks/<nid>/docs/<id>              remove a source
POST   /api/notebooks/<nid>/chat                   {message, doc_ids} -> answer + citations + cost
POST   /api/notebooks/<nid>/suggest                {message, answer, doc_ids, passages} -> 3 follow-up questions
POST   /api/notebooks/<nid>/reindex                re-embed sources saved with another embedding model
DELETE /api/notebooks/<nid>/messages               clear the chat
POST   /api/notebooks/<nid>/usage/reset            zero this notebook's cost totals
GET    /health
"""
import os
import uuid
from functools import wraps

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from . import chunking, config, embeddings, guardrails, llm, loaders, notebooks, rag
from .costs import usd

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
notebooks.init()


def with_nb(fn):
    """Look up the notebook from the URL and pass it to the view."""
    @wraps(fn)
    def wrapper(nid, **kwargs):
        nb = notebooks.get(nid)
        if nb is None:
            return jsonify(error="Notebook not found."), 404
        return fn(nb, **kwargs)
    return wrapper


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "notebooks": len(notebooks.all_notebooks()), "embed": embeddings.embed_id(),
            "model": config.OPENAI_MODEL, "vector_store": "faiss", "key_set": bool(config.OPENAI_API_KEY)}


@app.get("/api/notebooks")
def list_notebooks():
    return jsonify([n.summary() for n in notebooks.all_notebooks()])


@app.post("/api/notebooks")
def create_notebook():
    name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
    return jsonify(notebooks.create(name or "Untitled notebook").summary())


@app.get("/api/notebooks/<nid>")
@with_nb
def get_notebook(nb):
    return jsonify(**nb.summary(), docs=nb.store.list(), messages=nb.messages(),
                   totals=nb.ledger.snapshot(), stale=list(nb.store.stale.values()),
                   embed=embeddings.embed_id(), embed_price=config.EMBED_PRICE_PER_M)


@app.patch("/api/notebooks/<nid>")
@with_nb
def rename_notebook(nb):
    nb.rename((request.get_json(silent=True) or {}).get("name") or "")
    return jsonify(nb.summary())


@app.delete("/api/notebooks/<nid>")
def delete_notebook(nid):
    return jsonify(deleted=notebooks.delete(nid))


@app.get("/api/notebooks/<nid>/docs/<doc_id>/units")
@with_nb
def units(nb, doc_id):
    u = nb.store.units(doc_id)
    return (jsonify(u), 200) if u is not None else (jsonify(error="Document not found"), 404)


@app.delete("/api/notebooks/<nid>/docs/<doc_id>")
@with_nb
def delete_doc(nb, doc_id):
    return jsonify(deleted=nb.store.delete(doc_id))


@app.post("/api/notebooks/<nid>/upload")
@with_nb
def upload(nb):
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="No file received."), 400
    name = secure_filename(f.filename) or "file"
    ext = os.path.splitext(name)[1].lower()
    if ext not in config.ALLOWED_EXT:
        return jsonify(error=f"{ext or 'This'} files are not supported. Use: {', '.join(sorted(config.ALLOWED_EXT))}"), 400

    doc_id = uuid.uuid4().hex[:8]
    folder = nb.store.folder(doc_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "original" + ext)
    f.save(path)
    try:
        units_ = chunking.merge_small(loaders.load(path))
        chunks = chunking.make_chunks(units_)
        if not chunks:
            raise ValueError("No readable text found in this file (scanned file with OCR disabled or unavailable?).")
        tokens = embeddings.count_tokens([c["ctx"] for c in chunks])
        cost = usd(tokens, config.EMBED_PRICE_PER_M)
        meta = nb.store.add(doc_id, name, units_, chunks, ext, tokens, cost)   # embeds + adds to FAISS
        totals = nb.ledger.add(embed_tokens=tokens, embed_cost=cost)
    except Exception as exc:
        nb.store.delete(doc_id)
        return jsonify(error=f"Could not index {name}: {exc}"), 422
    return jsonify(doc=meta, usage={"embed_tokens": tokens, "embed_cost": cost,
                                    "provider": embeddings.embed_id()}, totals=totals)


@app.post("/api/notebooks/<nid>/reindex")
@with_nb
def reindex(nb):
    try:
        n, tokens, cost = nb.store.reindex()
    except Exception as exc:
        return jsonify(error=f"Re-index failed: {exc}"), 502
    totals = nb.ledger.add(embed_tokens=tokens, embed_cost=cost)
    return jsonify(reindexed=n, embed_tokens=tokens, embed_cost=cost, totals=totals)


@app.post("/api/notebooks/<nid>/chat")
@with_nb
def chat(nb):
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify(error="Type a question first."), 400

    # ---- Guardrail: runs BEFORE retrieval and the LLM, so a blocked message costs $0 ----
    blocked = guardrails.check(message)
    if blocked:
        zero = {"embed_tokens": 0, "embed_cost": 0.0, "rerank_tokens": 0, "rerank_cost": 0.0,
                "input_tokens": 0, "input_cost": 0.0, "output_tokens": 0, "output_cost": 0.0,
                "total_cost": 0.0, "seconds": 0, "model": "guardrail"}
        out = {"answer": blocked["answer"], "citations": [], "usage": zero, "mode": "guardrail",
               "retrieved": [], "guardrail": blocked["category"], "totals": nb.ledger.snapshot()}
        nb.append({"role": "user", "content": message},
                  {"role": "assistant", "content": out["answer"],
                   **{k: out[k] for k in ("citations", "usage", "mode", "retrieved")}})
        return jsonify(out)

    try:
        out = rag.answer(nb.store, message, set(body.get("doc_ids") or []), nb.history(config.HISTORY_TURNS * 2))
    except llm.LLMError as exc:
        return jsonify(error=str(exc)), 502
    except Exception as exc:
        return jsonify(error=f"Unexpected error: {exc}"), 500
    u = out["usage"]
    out["totals"] = nb.ledger.add(embed_tokens=u["embed_tokens"], embed_cost=u["embed_cost"],
                                  rerank_tokens=u["rerank_tokens"], rerank_cost=u["rerank_cost"],
                                  input_tokens=u["input_tokens"], input_cost=u["input_cost"],
                                  output_tokens=u["output_tokens"], output_cost=u["output_cost"], requests=1)
    nb.append({"role": "user", "content": message},
              {"role": "assistant", "content": out["answer"], **{k: out[k] for k in ("citations", "usage", "mode", "retrieved")}})
    return jsonify(out)


@app.post("/api/notebooks/<nid>/suggest")
@with_nb
def suggest(nb):
    body = request.get_json(silent=True) or {}
    try:
        out = rag.suggest(nb.store, (body.get("message") or "").strip(), body.get("answer") or "",
                          set(body.get("doc_ids") or []), body.get("passages") or [])
    except llm.LLMError as exc:
        return jsonify(error=str(exc)), 502
    except Exception as exc:
        return jsonify(error=f"Unexpected error: {exc}"), 500
    u = out.pop("usage")
    if u:  # the small extra call is billed to the notebook total (not counted as a question)
        out["totals"] = nb.ledger.add(input_tokens=u["input_tokens"], input_cost=u["input_cost"],
                                      output_tokens=u["output_tokens"], output_cost=u["output_cost"])
    return jsonify(out)


@app.delete("/api/notebooks/<nid>/messages")
@with_nb
def clear_chat(nb):
    nb.clear_messages()
    return jsonify(ok=True)


@app.post("/api/notebooks/<nid>/usage/reset")
@with_nb
def usage_reset(nb):
    return jsonify(nb.ledger.reset())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))