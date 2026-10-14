"""Token estimation, price calculation and a persistent usage ledger."""
import json
import os
import threading


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token). Used where the provider returns no count."""
    return max(1, len(text) // 4) if text else 0


def usd(tokens: int, price_per_million: float) -> float:
    return tokens / 1_000_000 * price_per_million


class Ledger:
    """Running totals of every operation, saved to data/usage.json so they survive restarts."""

    FIELDS = ("embed_tokens", "embed_cost", "rerank_tokens", "rerank_cost",
              "input_tokens", "input_cost", "output_tokens", "output_cost",
              "total_cost", "requests")

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.totals = {k: 0 for k in self.FIELDS}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.totals.update(json.load(f))
            except (OSError, ValueError):
                pass

    def add(self, **delta) -> dict:
        with self.lock:
            for k, v in delta.items():
                self.totals[k] = self.totals.get(k, 0) + v
            self.totals["total_cost"] = round(
                self.totals["embed_cost"] + self.totals["rerank_cost"]
                + self.totals["input_cost"] + self.totals["output_cost"], 8)
            self._save()
            return dict(self.totals)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.totals)

    def reset(self) -> dict:
        with self.lock:
            self.totals = {k: 0 for k in self.FIELDS}
            self._save()
            return dict(self.totals)

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.totals, f)
