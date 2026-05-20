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
from collections import defaultdict
from datetime import datetime

load_dotenv()


TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

MAX_RETRIES     = int(os.environ.get("OPENCLAW_MAX_RETRIES", 3))
RETRY_BACKOFF   = float(os.environ.get("OPENCLAW_RETRY_BACKOFF", 2.0))  # seconds
OPENCLAW_TIMEOUT = int(os.environ.get("OPENCLAW_TIMEOUT", 60))
RATE_LIMIT      = os.environ.get("RATE_LIMIT", "30 per minute")

# ─── App & extensions ─────────────────────────────────────────────────────────

app = Flask(__name__)
tavily = TavilyClient(TAVILY_API_KEY)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
    storage_uri="memory://",
)

# ─── In-memory session store  { session_id: [{"role": ..., "content": ...}] }
sessions: dict[str, list[dict]] = defaultdict(list)

# ─── Keywords that suggest a web search is needed ─────────────────────────────
SEARCH_KEYWORDS = re.compile(
    r"\b(who|what|when|where|which|latest|current|today|news|weather|price|"
    r"score|result|update|release|version|how much|how many|define|meaning|"
    r"capital|president|ceo|stock|rate|forecast)\b",
    re.IGNORECASE,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def needs_search(text: str) -> bool:
    """Heuristic: does this message likely need fresh web data?"""
    if len(text.split()) <= 3:
        return False
    return bool(SEARCH_KEYWORDS.search(text))


def search_web(query: str) -> tuple[str, list[dict]]:
    """
    Run a Tavily search.
    Returns (context_string, sources_list).
    """
    try:
        response = tavily.search(
            query=query,
            include_answer="basic",
            search_depth="advanced",
        )
        lines = []
        sources = []

        if response.get("answer"):
            lines.append(f"Answer: {response['answer']}\n")

        for i, result in enumerate(response.get("results", [])[:5], 1):
            title   = result.get("title", "")
            url     = result.get("url", "")
            content = result.get("content", "")
            lines.append(f"[{i}] {title}\nURL: {url}\n{content}\n")
            sources.append({"index": i, "title": title, "url": url})

        return "\n".join(lines), sources
    except Exception as e:
        return f"(Web search failed: {e})", []


def build_prompt(history: list[dict], user_message: str, web_context: str | None) -> str:
    """
    Build a full prompt that includes conversation history and optional
    web context, formatted for OpenClaw's --prompt flag.
    """
    parts = []

    if web_context:
        parts.append(
            "Use the following web search results to help answer.\n"
            "If you reference a source, mention its [index number].\n\n"
            f"--- Web Search Results ---\n{web_context}\n--- End of Results ---\n"
        )

    # Append conversation history
    if history:
        parts.append("--- Conversation so far ---")
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{role}: {turn['content']}")
        parts.append("--- End of conversation ---\n")

    parts.append(f"User: {user_message}")
    return "\n\n".join(parts)


def run_openclaw(prompt: str) -> str:
    """
    Call OpenClaw CLI with retry + exponential backoff.
    Raises RuntimeError after all retries are exhausted.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            proc = subprocess.run(
                ["openclaw", "infer", "model", "run", "--prompt", prompt, "--json"],
                capture_output=True,
                text=True,
                timeout=OPENCLAW_TIMEOUT,
            )
            stdout = proc.stdout.strip()
            try:
                result = json.loads(stdout)
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
    Send a message. Supports sessions, smart search triggering, and citations.

    Request body:
      {
        "message":    "What is the capital of Colombia?",
        "session_id": "abc123",          // optional — omit to start a new session
        "search":     true | false | "auto"  // default: "auto"
      }

    Response body:
      {
        "session_id": "abc123",
        "message":    "...",
        "response":   "...",
        "sources":    [{"index": 1, "title": "...", "url": "..."}],
        "searched":   true | false
      }
    """
    data       = request.get_json(force=True)
    prompt     = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    search_opt = data.get("search", "auto")  # true | false | "auto"

    if not prompt:
        return jsonify({"error": "No message provided."}), 400

    history = sessions[session_id]

    # Decide whether to search
    if search_opt == "auto":
        do_search = needs_search(prompt)
    else:
        do_search = bool(search_opt)

    web_context, sources = (None, [])
    if do_search:
        web_context, sources = search_web(prompt)

    final_prompt = build_prompt(history, prompt, web_context)

    try:
        response_text = run_openclaw(final_prompt)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    # Save turn to session history
    history.append({"role": "user",      "content": prompt})
    history.append({"role": "assistant", "content": response_text})

    return jsonify({
        "session_id": session_id,
        "message":    prompt,
        "response":   response_text,
        "sources":    sources,
        "searched":   do_search,
    })


@app.route("/api/message/stream", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def message_stream():
    """
    Same as /api/message but streams the response token-by-token via
    Server-Sent Events (SSE).

    The client receives events like:
      data: {"token": "Bras"}
      data: {"token": "ília"}
      data: {"done": true, "session_id": "...", "sources": [...]}
    """
    data       = request.get_json(force=True)
    prompt     = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    search_opt = data.get("search", "auto")

    if not prompt:
        return jsonify({"error": "No message provided."}), 400

    history = sessions[session_id]

    if search_opt == "auto":
        do_search = needs_search(prompt)
    else:
        do_search = bool(search_opt)

    web_context, sources = (None, [])
    if do_search:
        web_context, sources = search_web(prompt)

    final_prompt = build_prompt(history, prompt, web_context)

    def generate():
        collected = []
        try:
            proc = subprocess.Popen(
                ["openclaw", "infer", "model", "run", "--prompt", final_prompt, "--stream"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in proc.stdout:
                token = line.rstrip("\n")
                collected.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            proc.wait()
            full_response = "".join(collected)
            history.append({"role": "user",      "content": prompt})
            history.append({"role": "assistant",  "content": full_response})
            yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'sources': sources})}\n\n"

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
    """
    Return the full conversation history for a session.

    Response body:
      {
        "session_id": "abc123",
        "turns": [
          {"role": "user",      "content": "...", "index": 0},
          {"role": "assistant", "content": "...", "index": 1}
        ]
      }
    """
    turns = sessions.get(session_id, [])
    return jsonify({
        "session_id": session_id,
        "turns": [{"index": i, **t} for i, t in enumerate(turns)],
    })


@app.route("/api/history/<session_id>", methods=["DELETE"])
def clear_history(session_id: str):
    """Clear the conversation history for a session."""
    sessions.pop(session_id, None)
    return jsonify({"session_id": session_id, "cleared": True})


@app.route("/api/search", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def search():
    """
    Standalone Tavily search (no OpenClaw).

    Request body:  { "query": "latest AI news" }
    Response body: { "query": "...", "answer": "...", "results": [...] }
    """
    data  = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided."}), 400

    try:
        response = tavily.search(
            query=query,
            include_answer="basic",
            search_depth="advanced",
        )
        return jsonify({
            "query":   query,
            "answer":  response.get("answer"),
            "results": response.get("results", []),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Check OpenClaw gateway status."""
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Flask running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
