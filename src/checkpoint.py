"""
checkpoint.py — Checkpoint manager. Tracks completed work so steps can resume safely.
Each step gets its own checkpoint file: data/checkpoints/{step_name}.json
"""

import json
from pathlib import Path
from typing import Any, Optional


class Checkpoint:
    def __init__(self, step_name: str, checkpoint_dir: Path):
        self.step_name = step_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.checkpoint_dir / f"{step_name}.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"completed": [], "meta": {}}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_done(self, key: Any) -> bool:
        """Return True if this key has been marked complete."""
        return str(key) in self._data["completed"]

    def mark_done(self, key: Any):
        """Mark a key as complete and persist immediately."""
        k = str(key)
        if k not in self._data["completed"]:
            self._data["completed"].append(k)
            self._save()

    def set_meta(self, key: str, value: Any):
        """Store arbitrary metadata in the checkpoint."""
        self._data["meta"][key] = value
        self._save()

    def get_meta(self, key: str, default: Any = None) -> Any:
        return self._data["meta"].get(key, default)

    def reset(self):
        """Wipe the checkpoint file."""
        self._data = {"completed": [], "meta": {}}
        self._save()

    def count_done(self) -> int:
        return len(self._data["completed"])

    def __repr__(self):
        return f"Checkpoint(step={self.step_name}, done={self.count_done()})"
