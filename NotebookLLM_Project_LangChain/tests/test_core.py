"""Run with: EMBED_PROVIDER=fake pytest -q   (fake embedder and fake LLM, no network or models needed)"""
import io
import json
import os
import tempfile

os.environ["EMBED_PROVIDER"] = "fake"
os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["EMBED_PRICE_PER_M"] = "0.02"   # pretend embeddings are billed, to test the maths
os.environ["OPENAI_API_KEY"] = "test"
os.environ["ALLOWED_EXT"] = ".pdf,.txt,.csv"

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from app import chunking, config, costs, llm, loaders
from app.main import app
from app.store import Store


def _units(*pairs):
    return [Document(page_content=t, metadata={"label": l, "heading": "", "mergeable": False}) for l, t in pairs]


# ---------------- chunking (LangChain splitter) ----------------
def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(300))
    chunks = chunking.make_chunks(_units(("Page 1", text)), 1200, 200)
    assert len(chunks) > 3
    assert all(len(c["text"]) <= 1200 for c in chunks)
    assert chunks[1]["start"] < chunks[0]["start"] + len(chunks[0]["text"])   # overlap
    assert chunks[-1]["start"] + len(chunks[-1]["text"]) == len(text)         # nothing lost


def test_chunk_text_is_substring_of_unit():
    u = [Document(page_content="Alpha beta. " * 300, metadata={"label": "Page 1", "heading": "H", "mergeable": False})]
    for c in chunking.make_chunks(u):
        assert c["text"] in u[0].page_content and c["ctx"].startswith("H\n")


def test_small_sections_merge():
    u = [Document(page_content="short", metadata={"label": "Section: A", "heading": "A", "mergeable": True}),
         Document(page_content="x" * 500, metadata={"label": "Section: B", "heading": "B", "mergeable": True})]
    assert len(chunking.merge_small(u)) == 1


def test_page_markers_and_csv(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("[[PAGE 1]] first page text [[PAGE 2]] second page text")
    assert [u.metadata["label"] for u in loaders.load(str(p))] == ["Page 1", "Page 2"]
    c = tmp_path / "b.csv"
    c.write_text("name,age\nAnn,30\nBob,41\n")
    assert "name: Ann; age: 30" in loaders.load(str(c))[0].page_content


def test_cost_math():
    assert costs.usd(1_000_000, 0.25) == 0.25
    assert round(costs.usd(2000, 2.0), 6) == 0.004


# ---------------- FAISS store ----------------
def _store(tmp_path, *docs):
    s = Store(str(tmp_path))
    for doc_id, name, pairs in docs:
        units = _units(*pairs)
        s.add(doc_id, name, units, chunking.make_chunks(units), ".txt")
    return s


def test_store_uses_langchain_faiss_and_retriever(tmp_path):
    s = _store(tmp_path, ("d1", "a.txt", [("Page 1", "Antiretroviral therapy should start early for all patients.")]),
               ("d2", "b.txt", [("Page 1", "Banana bread needs ripe bananas and flour.")]))
    assert isinstance(s.vs, FAISS) and s.vs.index.ntotal == 2
    r = s.as_retriever({"d1", "d2"}, 2)
    assert isinstance(r, BaseRetriever)
    docs = r.invoke("antiretroviral therapy early")
    assert docs[0].metadata["doc_id"] == "d1" and isinstance(docs[0], Document)
    # selecting only d2 must never return d1 chunks
    only = s.as_retriever({"d2"}, 5).invoke("antiretroviral therapy early")
    assert {d.metadata["doc_id"] for d in only} == {"d2"}


def test_faiss_index_persists_and_delete_removes_vectors(tmp_path):
    s = _store(tmp_path, ("d1", "a.txt", [("Page 1", "Alpha beta gamma delta.")]),
               ("d2", "b.txt", [("Page 1", "Epsilon zeta eta theta.")]))
    assert os.path.isfile(tmp_path / "_faiss_index" / "index.faiss")
    s2 = Store(str(tmp_path))                       # restart
    assert len(s2.list()) == 2 and not s2.stale and s2.vs.index.ntotal == 2
    assert s2.delete("d1") is True
    assert s2.vs.index.ntotal == 1
    assert Store(str(tmp_path)).vs.index.ntotal == 1
    assert s2.delete("d2") is True and s2.vs is None
    assert not (tmp_path / "_faiss_index").exists()


def test_reindex_after_embedding_change(tmp_path):
    _store(tmp_path, ("abc12345", "x.txt", [("Page 1", "Hello world. " * 20)]))
    meta = tmp_path / "_faiss_index" / "index.json"
    meta.write_text(json.dumps({"embed_id": "local:other-model"}))     # index built with another model
    s2 = Store(str(tmp_path))
    assert s2.stale == {"abc12345": "x.txt"} and s2.list() == []
    n, tokens, cost = s2.reindex()
    assert n == 1 and len(s2.list()) == 1 and not s2.stale and tokens > 0
    assert len(s2.as_retriever(set(), 3).invoke("hello world")) > 0


def test_old_numpy_sources_are_picked_up_and_reindexed(tmp_path):
    """Sources saved by the pre-LangChain version (doc.json + vectors.npy, no FAISS index) are migrated by re-index."""
    d = tmp_path / "old00001"
    d.mkdir()
    units = [{"label": "Page 1", "text": "Old document about malaria treatment.", "heading": "", "mergeable": False}]
    chunks = chunking.make_chunks(_units(("Page 1", units[0]["text"])))
    (d / "doc.json").write_text(json.dumps({"meta": {"id": "old00001", "name": "old.pdf", "embed_id": "local:bge"},
                                            "units": units, "chunks": chunks}))
    s = Store(str(tmp_path))
    assert s.stale == {"old00001": "old.pdf"}
    assert s.reindex()[0] == 1
    assert s.as_retriever(set(), 3).invoke("malaria")[0].metadata["doc"] == "old.pdf"


def test_expand_includes_neighbour_on_next_page(tmp_path):
    s = _store(tmp_path, ("d1", "x.pdf", [("Page 7", "Alpha beta gamma. " * 5), ("Page 8", "Delta epsilon zeta. " * 5)]))
    hit = s.sample(set(), 10)[0]            # first chunk, on page 7
    segs = s.expand(hit, 1)
    assert [x["label"] for x in segs] == ["Page 7", "Page 8"]
    assert s.expand(hit, 0)[0]["label"] == "Page 7"


# ---------------- API with a fake LangChain chat model ----------------
def _fake_llm(text, i=1000, o=200):
    return RunnableLambda(lambda _prompt: AIMessage(content=text, usage_metadata={
        "input_tokens": i, "output_tokens": o, "total_tokens": i + o}, response_metadata={"model_name": "fake"}))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(llm, "get_llm", lambda: _fake_llm("ART is started early [1]."))
    return app.test_client()


def _upload(client, nid, body, name="doc.txt"):
    return client.post(f"/api/notebooks/{nid}/upload", data={"file": (io.BytesIO(body.encode()), name)},
                       content_type="multipart/form-data")


DOC = "[[PAGE 1]] Antiretroviral therapy should start early for all patients. [[PAGE 2]] Unrelated cooking notes."


def test_end_to_end_with_costs(client):
    nid = client.post("/api/notebooks", json={"name": "Test"}).json["id"]
    r = _upload(client, nid, DOC)
    assert r.status_code == 200, r.json
    assert r.json["usage"]["embed_tokens"] > 0 and r.json["usage"]["embed_cost"] > 0
    doc_id = r.json["doc"]["id"]

    r = client.post(f"/api/notebooks/{nid}/chat", json={"message": "When to start antiretroviral therapy?", "doc_ids": [doc_id]})
    assert r.status_code == 200, r.json
    u = r.json["usage"]
    assert u["input_tokens"] == 1000 and u["output_tokens"] == 200
    assert round(u["input_cost"], 8) == round(1000 / 1e6 * config.PRICE_IN, 8)
    assert round(u["output_cost"], 8) == round(200 / 1e6 * config.PRICE_OUT, 8)
    assert u["total_cost"] >= u["input_cost"] + u["output_cost"]
    assert r.json["citations"][0]["label"].startswith("Page 1")
    assert r.json["citations"][0]["segments"][0]["text"]
    assert r.json["totals"]["requests"] == 1

    assert client.get(f"/api/notebooks/{nid}/docs/{doc_id}/units").json[0]["label"] == "Page 1"
    assert client.delete(f"/api/notebooks/{nid}/docs/{doc_id}").json["deleted"] is True


def test_chat_history_is_sent_to_the_model(client, monkeypatch):
    seen = {}

    def capture(prompt_value):
        seen["messages"] = prompt_value.to_messages()
        return AIMessage(content="ok [1]", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    monkeypatch.setattr(llm, "get_llm", lambda: RunnableLambda(capture))
    nid = client.post("/api/notebooks", json={"name": "H"}).json["id"]
    doc = _upload(client, nid, DOC).json["doc"]["id"]
    for q in ("When to start antiretroviral therapy?", "And for children?"):
        client.post(f"/api/notebooks/{nid}/chat", json={"message": q, "doc_ids": [doc]})
    kinds = [type(m).__name__ for m in seen["messages"]]
    assert kinds == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]


def test_notebooks_are_separate_and_chat_is_saved(client):
    a = client.post("/api/notebooks", json={"name": "A"}).json["id"]
    b = client.post("/api/notebooks", json={"name": "B"}).json["id"]
    doc = _upload(client, a, DOC).json["doc"]["id"]
    client.post(f"/api/notebooks/{a}/chat", json={"message": "When to start therapy?", "doc_ids": [doc]})
    na, nb_ = client.get(f"/api/notebooks/{a}").json, client.get(f"/api/notebooks/{b}").json
    assert len(na["docs"]) == 1 and len(nb_["docs"]) == 0
    assert [m["role"] for m in na["messages"]] == ["user", "assistant"] and nb_["messages"] == []
    assert na["totals"]["total_cost"] > 0 and nb_["totals"]["total_cost"] == 0
    client.patch(f"/api/notebooks/{a}", json={"name": "Renamed"})
    assert client.get(f"/api/notebooks/{a}").json["name"] == "Renamed"
    assert client.delete(f"/api/notebooks/{b}").json["deleted"] is True
    assert client.get(f"/api/notebooks/{b}").status_code == 404


def test_bad_extension(client):
    nid = client.post("/api/notebooks", json={}).json["id"]
    assert _upload(client, nid, "x", "evil.exe").status_code == 400


def test_parse_questions_handles_json_fences_and_lines():
    from app import rag
    assert rag._parse_questions('```json\n["What is A?", "How does B work?", "Why use C?", "Extra one here?"]\n```') == \
        ["What is A?", "How does B work?", "Why use C?"]
    assert rag._parse_questions("1. What is A?\n2. How does B work?\n3. Why use C?") == \
        ["What is A?", "How does B work?", "Why use C?"]


def test_suggest_endpoint_returns_questions_and_bills_notebook(client, monkeypatch):
    monkeypatch.setattr(llm, "get_llm", lambda: _fake_llm('["What is ART?", "When is ART started?", "Who needs ART?"]', 500, 40))
    nid = client.post("/api/notebooks", json={"name": "S"}).json["id"]
    r = client.post(f"/api/notebooks/{nid}/suggest", json={"message": "x", "answer": "y", "doc_ids": [],
                                                           "passages": ["ART is antiretroviral therapy."]})
    assert r.status_code == 200
    assert r.json["questions"] == ["What is ART?", "When is ART started?", "Who needs ART?"]
    assert r.json["totals"]["input_tokens"] == 500
    assert r.json["totals"]["requests"] == 0          # not counted as a question


def test_llm_errors_become_friendly_messages(client, monkeypatch):
    import httpx, openai
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def boom(_):
        raise openai.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
    monkeypatch.setattr(llm, "get_llm", lambda: RunnableLambda(boom))
    nid = client.post("/api/notebooks", json={"name": "E"}).json["id"]
    doc = _upload(client, nid, DOC).json["doc"]["id"]
    r = client.post(f"/api/notebooks/{nid}/chat", json={"message": "When to start therapy?", "doc_ids": [doc]})
    assert r.status_code == 502 and "401" in r.json["error"]


def test_faiss_scores_are_cosine_similarities(tmp_path):
    s = _store(tmp_path, ("d1", "a.txt", [("Page 1", "alpha beta gamma")]))
    doc = s.search_docs("alpha beta gamma", set(), 1)[0]
    assert abs(doc.metadata["score"] - 1.0) < 1e-3          # identical text -> cosine 1.0
