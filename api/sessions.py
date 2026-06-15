"""
sessions.py - JSON persistence for chat session histories.

Storage format  (sessions.json)
--------------------------------
{
  "sessions": {
    "<session_id>": {
      "session_id": "sess_Xk9mB2...",
      "user_id": "unisinos",
      "created_at": "2025-06-08T12:00:00+00:00",
      "updated_at": "2025-06-08T12:05:00+00:00",
      "history": [
        {"role": "user",      "content": "Onde fica a Bacia do Araripe?"},
        {"role": "assistant", "content": "A Bacia do Araripe está no Nordeste..."}
      ]
    },
    ...
  }
}
"""

import json
import os
import secrets
import threading
from datetime import datetime, timezone

SESSIONS_PATH = os.getenv("SESSIONS_PATH", "./sessions.json")

_lock = threading.Lock()



def _load() -> dict:
    if not os.path.exists(SESSIONS_PATH):
        return {"sessions": {}}
    with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    tmp = SESSIONS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, SESSIONS_PATH)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_sessions() -> None:
    with _lock:
        if not os.path.exists(SESSIONS_PATH):
            _save({"sessions": {}})



def create_session(user_id: str) -> str:
    """Create a new empty session and return its session_id."""
    session_id = "sess_" + secrets.token_urlsafe(16)
    now = _now_utc()
    with _lock:
        data = _load()
        data["sessions"][session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "history": [],
        }
        _save(data)
    return session_id


def get_session(session_id: str) -> dict | None:
    """Return the session dict or None if not found."""
    with _lock:
        data = _load()
    return data["sessions"].get(session_id)


def append_turn(session_id: str, user_content: str, assistant_content: str) -> list:
    """
    Append a user+assistant turn to the session history.
    Returns the updated history list.
    """
    with _lock:
        data = _load()
        session = data["sessions"].get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        session["history"].append({"role": "user",      "content": user_content})
        session["history"].append({"role": "assistant", "content": assistant_content})
        session["updated_at"] = _now_utc()
        _save(data)
        return session["history"]


def list_sessions(user_id: str | None = None) -> list[dict]:
    """
    Return all sessions, optionally filtered by user_id.
    History is excluded to keep the response light.
    """
    with _lock:
        data = _load()
    sessions = [
        {k: v for k, v in s.items() if k != "history"}
        for s in data["sessions"].values()
    ]
    if user_id:
        sessions = [s for s in sessions if s["user_id"] == user_id]
    return sorted(sessions, key=lambda s: s["created_at"])