"""
auth.py - API key lifecycle helpers and Flask request guard.

Key format
----------
  malta_<32 random url-safe base64 chars>

  Example:  malta_Xk9mB2vQpL...
  Prefix stored in DB: first 8 chars of the full raw key  →  "malta_Xk"

Hashing
-------
  SHA-256 of the *entire* raw key string (UTF-8 encoded).
  The digest is stored as a lower-case hex string.
  `secrets.compare_digest` is used for constant-time comparison.
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone
from functools import wraps

from flask import request, jsonify

import db as db



PREFIX_LEN = 8  


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of *raw_key*."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_raw_key() -> str:
    """Return a fresh, cryptographically random API key string."""
    random_part = secrets.token_urlsafe(24)  # 32 url-safe chars
    return f"malta_{random_part}"



def create_key(user_id: str) -> str:
    """
    Generate a new API key for *user_id*, persist its hash, and return the
    **raw key** (shown once – never stored in plain text).
    """
    raw_key = generate_raw_key()
    key_hash = hash_key(raw_key)
    key_prefix = raw_key[:PREFIX_LEN]
    row_id = db.insert_key(
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        created_at=_now_utc(),
    )
    return raw_key


def validate_key(raw_key: str) -> str | None:
    """
    Validate *raw_key* against stored hashes using constant-time comparison.

    Returns the *user_id* on success, or None if the key is invalid/revoked.
    """
    candidate_hash = hash_key(raw_key)
    row = db.get_active_key(candidate_hash)
    if row is None:
        return None

    stored_hash: str = row["key_hash"]
    if not hmac.compare_digest(candidate_hash, stored_hash):
        return None

    db.touch_last_used(candidate_hash, _now_utc())
    return row["user_id"]


def revoke_key(key_prefix: str) -> int:
    """Revoke all keys matching *key_prefix*. Returns rows affected."""
    return db.set_active(key_prefix, active=False)



_HEADER_NAME = "X-API-Key"


def require_api_key(f):
    """
    Flask route decorator that enforces API key authentication.

    Reads the key from the ``X-API-Key`` HTTP header.
    Attaches the resolved *user_id* to the Flask ``g`` object so downstream
    handlers can use it for logging / rate-limiting if needed.

    Usage::

        @app.route("/complete", methods=["POST"])
        @require_api_key
        def complete():
            ...
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        raw_key = request.headers.get(_HEADER_NAME, "")
        if not raw_key:
            return jsonify({"error": f"Missing '{_HEADER_NAME}' header"}), 401

        user_id = validate_key(raw_key)
        if user_id is None:
            return jsonify({"error": "Invalid or revoked API key"}), 401

        from flask import g
        g.api_user = user_id

        return f(*args, **kwargs)
    return wrapper