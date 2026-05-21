from flask import Flask, jsonify, request, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from tavily import TavilyClient
from dotenv import load_dotenv
import subprocess
import json
import os
import re
import time
import uuid
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path


try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80 MB, CPU-friendly
except ImportError:
    _embed_model = None

load_dotenv()



TAVILY_API_KEY   = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

MAX_RETRIES      = int(os.environ.get("OPENCLAW_MAX_RETRIES", 3))
RETRY_BACKOFF    = float(os.environ.get("OPENCLAW_RETRY_BACKOFF", 2.0))
OPENCLAW_TIMEOUT = int(os.environ.get("OPENCLAW_TIMEOUT", 60))
RATE_LIMIT       = os.environ.get("RATE_LIMIT", "30 per minute")

# Folder with PDFs — only the server process can read this path.
# Override via env:  RAG_DOCS_DIR=/home/openclaw/docs
RAG_DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", "./docs")).expanduser()



CHUNK_SIZE    = int(os.environ.get("RAG_CHUNK_SIZE", 800))   # words per chunk (was 400)
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 100)) # word overlap   (was 80)
TOP_K         = int(os.environ.get("RAG_TOP_K", 4))           # chunks per query (was 5)


EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", 32))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)



app    = Flask(__name__)
tavily = TavilyClient(TAVILY_API_KEY)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
    storage_uri="memory://",
)



# Conversation history  { session_id: [{"role": ..., "content": ...}] }
sessions: dict[str, list[dict]] = defaultdict(list)

# Single unified RAG index — built at startup, invisible to users
_rag_index: dict = {}   # {"chunks": [...], "embeddings": np.ndarray, "sources": [...]}



def _extract_pdf(path: Path) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed — run: pip install pdfplumber")
    with pdfplumber.open(path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def _extract_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk(text: str) -> list[str]:
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + CHUNK_SIZE]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]





def _embed(texts: list[str]) -> np.ndarray:
    if _embed_model is None:
        raise RuntimeError("sentence-transformers not installed — run: pip install sentence-transformers")

    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vecs  = _embed_model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=EMBED_BATCH_SIZE,
        )
        all_vecs.append(vecs)
        log.info("  Embedded batch %d/%d", min(i + EMBED_BATCH_SIZE, len(texts)), len(texts))

    vecs  = np.concatenate(all_vecs, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.maximum(norms, 1e-10)).astype(np.float32)


def build_rag_index() -> None:
    """
    Scan RAG_DOCS_DIR for PDF files, extract text, chunk, embed,
    and store everything in the private _rag_index.
    Called once at startup (and can be re-triggered via internal reload).
    """
    global _rag_index

    if not RAG_DOCS_DIR.exists():
        log.warning("RAG_DOCS_DIR does not exist: %s — RAG disabled.", RAG_DOCS_DIR)
        return

    docs = sorted([
        *RAG_DOCS_DIR.glob("**/*.pdf"),
        *RAG_DOCS_DIR.glob("**/*.md"),
        *RAG_DOCS_DIR.glob("**/*.markdown"),
    ])
    if not docs:
        log.warning("No PDF or Markdown files found in %s — RAG disabled.", RAG_DOCS_DIR)
        return

    all_chunks:  list[str]  = []
    all_sources: list[str]  = []   # filename per chunk, for internal debug only

    for doc_path in docs:
        try:
            log.info("Indexing %s …", doc_path.name)
            if doc_path.suffix.lower() == ".pdf":
                text = _extract_pdf(doc_path)
            else:
                text = _extract_markdown(doc_path)
            chunks = _chunk(text)
            all_chunks.extend(chunks)
            all_sources.extend([doc_path.name] * len(chunks))
            log.info("  → %d chunks", len(chunks))
        except Exception as exc:
            log.error("Failed to index %s: %s", doc_path.name, exc)

    if not all_chunks:
        log.warning("No text extracted from any PDF — RAG disabled.")
        return

    log.info("Embedding %d total chunks (batch_size=%d) …", len(all_chunks), EMBED_BATCH_SIZE)
    embeddings = _embed(all_chunks)

    _rag_index = {
        "chunks":     all_chunks,
        "embeddings": embeddings,
        "sources":    all_sources,
    }
    log.info("RAG index ready: %d chunks from %d files.", len(all_chunks), len(docs))


def _retrieve(query: str) -> list[str]:
    """Return the TOP_K most relevant chunks for the query, silently."""
    if not _rag_index:
        return []
    q_vec   = _embed([query])[0]
    scores  = _rag_index["embeddings"] @ q_vec
    indices = np.argsort(scores)[::-1][:TOP_K]
    return [_rag_index["chunks"][i] for i in indices]


def _rag_context_block(chunks: list[str]) -> str | None:
    if not chunks:
        return None
    parts = ["[Internal knowledge]"]
    for chunk in chunks:
        parts.append(chunk)
    parts.append("[End of internal knowledge]")
    return "\n\n".join(parts)



def coerce_bool(value, default: bool) -> bool:
    """Accept True/False, 1/0, or strings 'true'/'false'/'1'/'0'."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return default


SEARCH_KEYWORDS = re.compile(
    r"\b(who|what|when|where|which|latest|current|today|news|weather|price|"
    r"score|result|update|release|version|how much|how many|define|meaning|"
    r"capital|president|ceo|stock|rate|forecast)\b",
    re.IGNORECASE,
)


def needs_search(text: str) -> bool:
    if len(text.split()) <= 3:
        return False
    return bool(SEARCH_KEYWORDS.search(text))


def search_web(query: str) -> tuple[str, list[dict]]:
    try:
        response = tavily.search(query=query, include_answer="basic", search_depth="advanced")
        lines, sources = [], []
        if response.get("answer"):
            lines.append(f"Answer: {response['answer']}\n")
        for i, result in enumerate(response.get("results", [])[:5], 1):
            title, url, content = result.get("title",""), result.get("url",""), result.get("content","")
            lines.append(f"[{i}] {title}\nURL: {url}\n{content}\n")
            sources.append({"index": i, "title": title, "url": url})
        return "\n".join(lines), sources
    except Exception as e:
        return f"(Web search failed: {e})", []


def build_prompt(
    history: list[dict],
    user_message: str,
    web_context: str | None,
    rag_context: str | None,
) -> str:
    parts = []


    if rag_context:
        parts.append(
            "You have access to the following internal knowledge. "
            "Use it to answer the user's question when relevant. "
            "Do not reveal the source or that internal documents exist.\n\n"
            + rag_context
        )

    if web_context:
        parts.append(
            "Use the following web search results to help answer. "
            "If you reference a source, mention its [index number].\n\n"
            f"--- Web Search Results ---\n{web_context}\n--- End of Results ---\n"
        )

    if history:
        parts.append("--- Conversation so far ---")
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{role}: {turn['content']}")
        parts.append("--- End of conversation ---\n")

    parts.append(f"User: {user_message}")
    return "\n\n".join(parts)


def run_openclaw(prompt: str) -> str:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            proc = subprocess.run(
                ["openclaw", "infer", "model", "run", "--prompt", prompt, "--json"],
                capture_output=True, text=True, timeout=OPENCLAW_TIMEOUT,
            )
            stdout = proc.stdout.strip()
            try:
                result  = json.loads(stdout)
                outputs = result.get("outputs", [])
                if outputs:
                    return outputs[0].get("text") or outputs[0].get("content") or str(outputs[0])
                return stdout
            except json.JSONDecodeError:
                return stdout or proc.stderr.strip()
        except subprocess.TimeoutExpired:
            last_error = "timeout"
        except FileNotFoundError:
            raise RuntimeError("openclaw CLI not found. Is it installed?")
        except Exception as e:
            last_error = str(e)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)
    raise RuntimeError(f"OpenClaw failed after {MAX_RETRIES} attempts. Last error: {last_error}")




@app.route("/api/message", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def message():
    """
    Send a message.

    Request body:
      {
        "message":    "...",
        "session_id": "abc123",              // optional
        "search":     true | false | "auto"  // default: "auto"
      }

    Response body:
      {
        "session_id": "...",
        "message":    "...",
        "response":   "...",
        "sources":    [...],
        "searched":   true | false
      }
    """
    data       = request.get_json(force=True)
    prompt     = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    search_raw = data.get("search", "auto")
    rag_opt    = coerce_bool(data.get("rag"), default=True)

    if not prompt:
        return jsonify({"error": "No message provided."}), 400

    history = sessions[session_id]

    if isinstance(search_raw, str) and search_raw.strip().lower() == "auto":
        do_search = needs_search(prompt)
    else:
        do_search = coerce_bool(search_raw, default=True)

    web_context, sources = (None, [])
    if do_search:
        web_context, sources = search_web(prompt)

    rag_context = _rag_context_block(_retrieve(prompt)) if rag_opt else None

    final_prompt = build_prompt(history, prompt, web_context, rag_context)

    try:
        response_text = run_openclaw(final_prompt)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    history.append({"role": "user",      "content": prompt})
    history.append({"role": "assistant", "content": response_text})

    return jsonify({
        "session_id": session_id,
        "message":    prompt,
        "response":   response_text,
        "sources":    sources,
        "searched":   do_search,
        "rag":        bool(rag_opt),
    })


@app.route("/api/message/stream", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def message_stream():
    """
    Streaming version of /api/message via Server-Sent Events.

    Events:
      data: {"token": "..."}
      data: {"done": true, "session_id": "...", "sources": [...]}
    """
    data       = request.get_json(force=True)
    prompt     = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    search_raw = data.get("search", "auto")
    rag_opt    = coerce_bool(data.get("rag"), default=True)

    if not prompt:
        return jsonify({"error": "No message provided."}), 400

    history = sessions[session_id]

    if isinstance(search_raw, str) and search_raw.strip().lower() == "auto":
        do_search = needs_search(prompt)
    else:
        do_search = coerce_bool(search_raw, default=True)

    web_context, sources = (None, [])
    if do_search:
        web_context, sources = search_web(prompt)

    rag_context  = _rag_context_block(_retrieve(prompt)) if rag_opt else None
    final_prompt = build_prompt(history, prompt, web_context, rag_context)

    def generate():
        collected = []
        try:
            proc = subprocess.Popen(
                ["openclaw", "infer", "model", "run", "--prompt", final_prompt, "--stream"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            for line in proc.stdout:
                token = line.rstrip("\n")
                collected.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
            proc.wait()
            full_response = "".join(collected)
            history.append({"role": "user",      "content": prompt})
            history.append({"role": "assistant",  "content": full_response})
            yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'sources': sources, 'rag': bool(rag_opt)})}\n\n"
        except FileNotFoundError:
            yield f"data: {json.dumps({'error': 'openclaw CLI not found'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/history/<session_id>", methods=["GET"])
def history(session_id: str):
    turns = sessions.get(session_id, [])
    return jsonify({
        "session_id": session_id,
        "turns": [{"index": i, **t} for i, t in enumerate(turns)],
    })


@app.route("/api/history/<session_id>", methods=["DELETE"])
def clear_history(session_id: str):
    sessions.pop(session_id, None)
    return jsonify({"session_id": session_id, "cleared": True})


@app.route("/api/search", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def search():
    data  = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided."}), 400
    try:
        response = tavily.search(query=query, include_answer="basic", search_depth="advanced")
        return jsonify({"query": query, "answer": response.get("answer"), "results": response.get("results", [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    proc = subprocess.run(
        ["openclaw", "gateway", "status"],
        capture_output=True, text=True, timeout=10,
    )
    return jsonify({
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "openclaw":  (proc.stdout + proc.stderr).strip(),
    })


@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Slow down a bit."}), 429




build_rag_index()   # index all PDFs before accepting requests

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Flask running on http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
