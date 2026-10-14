"""Notebooks: every notebook has its own sources, chat history and cost totals.

data/notebooks/<id>/notebook.json   name + creation time
                    messages.json   saved chat (questions, answers, citations, costs)
                    usage.json      cost totals for this notebook
                    docs/<doc_id>/  sources (see store.py)
"""
import json
import os
import shutil
import threading
import time
import uuid

from . import config
from .costs import Ledger
from .store import Store

_lock = threading.RLock()
_books = {}


def _root():
    return os.path.join(config.DATA_DIR, "notebooks")


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _read(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Notebook:
    def __init__(self, nid):
        self.id = nid
        self.dir = os.path.join(_root(), nid)
        self.meta_path = os.path.join(self.dir, "notebook.json")
        self.msg_path = os.path.join(self.dir, "messages.json")
        self.lock = threading.Lock()
        self.meta = _read(self.meta_path, {"name": "Untitled notebook", "created": time.time()})
        self.store = Store(os.path.join(self.dir, "docs"))
        self.ledger = Ledger(os.path.join(self.dir, "usage.json"))

    def rename(self, name):
        self.meta["name"] = name.strip() or self.meta["name"]
        _write(self.meta_path, self.meta)

    def messages(self):
        with self.lock:
            return _read(self.msg_path, [])

    def append(self, *msgs):
        with self.lock:
            data = _read(self.msg_path, [])
            data.extend(msgs)
            _write(self.msg_path, data)

    def clear_messages(self):
        with self.lock:
            _write(self.msg_path, [])

    def history(self, n):
        """Plain role/content pairs for the LLM prompt."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages()[-n:]]

    def summary(self):
        return {"id": self.id, "name": self.meta["name"], "created": self.meta["created"],
                "n_docs": len(self.store.list()), "total_cost": self.ledger.snapshot()["total_cost"]}


def create(name="Untitled notebook", adopt=()):
    nid = uuid.uuid4().hex[:8]
    d = os.path.join(_root(), nid)
    os.makedirs(os.path.join(d, "docs"))
    _write(os.path.join(d, "notebook.json"), {"name": name.strip() or "Untitled notebook", "created": time.time()})
    for doc in adopt:  # sources from older versions that had no notebooks
        shutil.move(os.path.join(config.DATA_DIR, doc), os.path.join(d, "docs", doc))
    legacy_usage = os.path.join(config.DATA_DIR, "usage.json")
    if adopt and os.path.isfile(legacy_usage):
        shutil.move(legacy_usage, os.path.join(d, "usage.json"))
    nb = Notebook(nid)
    with _lock:
        _books[nid] = nb
    return nb


def init():
    os.makedirs(_root(), exist_ok=True)
    with _lock:
        for nid in sorted(os.listdir(_root())):
            if os.path.isfile(os.path.join(_root(), nid, "notebook.json")):
                _books[nid] = Notebook(nid)
    old = [d for d in os.listdir(config.DATA_DIR) if os.path.isfile(os.path.join(config.DATA_DIR, d, "doc.json"))]
    if old:
        create("My first notebook", adopt=old)


def get(nid):
    return _books.get(nid)


def all_notebooks():
    with _lock:
        return sorted(_books.values(), key=lambda n: -n.meta["created"])


def delete(nid):
    with _lock:
        existed = _books.pop(nid, None) is not None
    shutil.rmtree(os.path.join(_root(), nid), ignore_errors=True)
    return existed
