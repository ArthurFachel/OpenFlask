"""
db.py - JSON file persistence layer for API keys.

Storage format  (api_keys.json)
--------------------------------
{
  "keys": [
    {
      "id": 1,
      "user_id": "unisinos",
      "key_hash": "<sha256 hex>",
      "key_prefix": "malta_Xk",
      "created_at": "2025-01-01T00:00:00+00:00",
      "last_used_at": null,
      "active": true
    },
    ...
  ]
}

All reads and writes go through _load() / _save(), which hold a
threading.Lock so concurrent Flask threads don't corrupt the file.
"""

import json
import os
import threading
from typing import Any

DB_PATH = os.getenv("API_KEYS_DB_PATH", "./api_keys.json")

_lock = threading.Lock()


def _load() -> dict:
    """Read and return the full JSON store, creating it if absent."""
    if not os.path.exists(DB_PATH):
        return {"keys": []}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    """Atomically write *data* to the JSON file."""
    tmp = DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, DB_PATH) 


def init_db() -> None:
    """Create an empty key store if the file does not exist yet."""
    with _lock:
        if not os.path.exists(DB_PATH):
            _save({"keys": []})



def insert_key(user_id: str, key_hash: str, key_prefix: str, created_at: str) -> int:
    """Append a new key entry and return its auto-incremented id."""
    with _lock:
        data = _load()
        existing_ids = [k["id"] for k in data["keys"]]
        new_id = max(existing_ids, default=0) + 1
        data["keys"].append({
            "id": new_id,
            "user_id": user_id,
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "created_at": created_at,
            "last_used_at": None,
            "active": True,
        })
        _save(data)
        return new_id


def touch_last_used(key_hash: str, timestamp: str) -> None:
    """Update last_used_at for the entry matching *key_hash*."""
    with _lock:
        data = _load()
        for key in data["keys"]:
            if key["key_hash"] == key_hash:
                key["last_used_at"] = timestamp
                break
        _save(data)


def set_active(key_prefix: str, active: bool) -> int:
    """
    Enable or revoke all keys whose prefix matches *key_prefix*.
    Returns the number of entries updated.
    """
    with _lock:
        data = _load()
        count = 0
        for key in data["keys"]:
            if key["key_prefix"] == key_prefix:
                key["active"] = active
                count += 1
        _save(data)
        return count



def get_active_key(key_hash: str) -> dict | None:
    """Return the entry dict for an active key matching *key_hash*, or None."""
    with _lock:
        data = _load()
    for key in data["keys"]:
        if key["key_hash"] == key_hash and key["active"]:
            return key
    return None


def list_keys() -> list[dict]:
    """Return all key entries ordered by creation date."""
    with _lock:
        data = _load()
    return sorted(data["keys"], key=lambda k: k["created_at"])