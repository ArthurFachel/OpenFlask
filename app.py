#MALTA-GEO 
#@ArthurFachel
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
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2") 
except ImportError:
    _embed_model = None

load_dotenv()


SYSTEM_PROMPT = """
{
  "role": "Assistente",
  "task": "Responder perguntas, em português, sobre geologia.",
  "content": {
    "description": "Você é um Assistente de IA especializado em geologia, com foco no Puesto Seguel.",
    "objectives": [
      "Responder com precisão e clareza sobre rochas, sedimentos e minerais (tipos, formação, distribuição).",
      "Explicar formações geológicas (estruturas, estratigrafia, história).",
      "Apresentar informações sobre fósseis (descobertas, relevância científica)."
    ],
    "restrictions": [
      "Não faça perguntas diretas ao usuário.",
      "Respostas devem ser concisas, lógicas e baseadas em evidências científicas.",
      "Manter tom profissional, sem mencionar regras internas ou linguagem inadequada.",
      "Responda apenas e unicamente em Português, Brasil!"
    ],
    "language": "Português, Brasil",
    "examples": [
      {
        "question": "Onde fica o Puesto Seguel?",
        "answer": "O Puesto Seguel é uma localidade geológica situada na Bacia de Neuquén (Cuenca Neuquina), na província de Neuquén, na Argentina."
      },
      {
        "question": "Me fale sobre esse afloramento.",
        "answer": "Este afloramento é conhecido como Puesto Seguel. Ele possui ~3 km de extensão e ~60 m de altura, sendo localizado próximo à cidade de Zapala, Argentina. O Puesto Seguel consiste em uma grande área de depósitos lateralmente contínuos, que registram, na base, os sedimentos da Formação Los Molles, considerados os primeiros depósitos marinhos da bacia. É importante comentar que a proximidade entre a plataforma e o arco de ilhas na região era favorável à ocorrência de deslizamentos subaquosos de grande escala, gerando depósitos turbidíticos que evidenciam eventos de alta energia. O afloramento conta também com forte presença de elementos arquiteturais oriundos de sistemas flúvio-deltaicos da Formação Lajas, depositada logo acima, com transição gradual. Por fim, este afloramento está inserido em uma região de dobras, dentro do contexto do Subciclo Cuyano: Hettangiano ao Caloviano médio (Jurássico Inferior médio). O Puesto Seguel está, majoritariamente, geneticamente relacionado a ambientes flúvio-deltaicos da Formação Lajas."
      },
      {
        "question": "Descreva a amostra PSL-25.",
        "answer": "A amostra PSL-25 corresponde a um arenito médio, maciço, moderadamente selecionado, composto por quartzo, feldspato, plagioclásio, fragmentos de rochas vulcânicas, metamórficas e sedimentares (como ardósia e filito). Ela foi coletada na Porção Norte do afloramento Puesto Seguel durante a campanha de campo realizada em Abril/2024."
      }
    ]
  }
}
"""


#PROMPT
def build_input_data(
    text: str,
    inner_contexts: list[str],
    outer_contexts: list[str],
    web_context: str | None = None,
    rag_context: str | None = None,
    history: list[dict] | None = None,
) -> str:
    """
    Assemble the full prompt string that will be sent to OpenClaw.

    Context hierarchy (outermost → innermost):
      outer_contexts  – broad background / domain knowledge supplied by the caller
      rag_context     – chunks retrieved from the local PDF index
      web_context     – live web-search results
      inner_contexts  – tightly scoped context snippets (e.g. sample descriptions)
      user question   – the actual query
    """
    parts: list[str] = []

    # --- outer context (broad background) ---
    if outer_contexts:
        parts.append("=== Contexto Externo ===")
        for ctx in outer_contexts:
            parts.append(ctx)
        parts.append("=== Fim do Contexto Externo ===\n")

    # --- RAG (internal documents) ---
    if rag_context:
        parts.append(rag_context)

    # --- web search results ---
    if web_context:
        parts.append(
            "Use the following web search results to help answer. "
            "If you reference a source, mention its [index number].\n\n"
            f"--- Web Search Results ---\n{web_context}\n--- End of Results ---\n"
        )

    # --- conversation history ---
    if history:
        parts.append("--- Histórico da Conversa ---")
        for turn in history:
            role = "Usuário" if turn["role"] == "user" else "Assistente"
            parts.append(f"{role}: {turn['content']}")
        parts.append("--- Fim do Histórico ---\n")

    # --- inner context (tightly scoped snippets) ---
    if inner_contexts:
        parts.append("=== Contexto Interno ===")
        for ctx in inner_contexts:
            parts.append(ctx)
        parts.append("=== Fim do Contexto Interno ===\n")

    # --- user question ---
    parts.append(f"pergunta: {text}")

    combined_context = "\n\n".join(parts)

    # Wrap in OpenClaw-style header/footer with the MALTA-GEO system prompt
    input_data = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        f"{SYSTEM_PROMPT}"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{combined_context}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return input_data


#START
TAVILY_API_KEY   = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

MAX_RETRIES      = int(os.environ.get("OPENCLAW_MAX_RETRIES", 3))
RETRY_BACKOFF    = float(os.environ.get("OPENCLAW_RETRY_BACKOFF", 2.0))
OPENCLAW_TIMEOUT = int(os.environ.get("OPENCLAW_TIMEOUT", 60))
RATE_LIMIT       = os.environ.get("RATE_LIMIT", "30 per minute")

RAG_DOCS_DIR  = Path(os.environ.get("RAG_DOCS_DIR", "./docs")).expanduser()
CHUNK_SIZE    = int(os.environ.get("RAG_CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 100))
TOP_K         = int(os.environ.get("RAG_TOP_K", 4))
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", 32))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


#start
app    = Flask(__name__)
tavily = TavilyClient(TAVILY_API_KEY)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
    storage_uri="memory://",
)

#SESSIONS
sessions: dict[str, list[dict]] = defaultdict(list)

_rag_index: dict = {}

#RAG
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

    all_chunks:  list[str] = []
    all_sources: list[str] = []

    for doc_path in docs:
        try:
            log.info("Indexing %s …", doc_path.name)
            text   = _extract_pdf(doc_path) if doc_path.suffix.lower() == ".pdf" else _extract_markdown(doc_path)
            chunks = _chunk(text)
            all_chunks.extend(chunks)
            all_sources.extend([doc_path.name] * len(chunks))
            log.info("  → %d chunks", len(chunks))
        except Exception as exc:
            log.error("Failed to index %s: %s", doc_path.name, exc)

    if not all_chunks:
        log.warning("No text extracted from any document — RAG disabled.")
        return

    log.info("Embedding %d total chunks …", len(all_chunks))
    embeddings = _embed(all_chunks)
    _rag_index = {"chunks": all_chunks, "embeddings": embeddings, "sources": all_sources}
    log.info("RAG index ready: %d chunks from %d files.", len(all_chunks), len(docs))


def _retrieve(query: str) -> list[str]:
    if not _rag_index:
        return []
    q_vec   = _embed([query])[0]
    scores  = _rag_index["embeddings"] @ q_vec
    indices = np.argsort(scores)[::-1][:TOP_K]
    return [_rag_index["chunks"][i] for i in indices]


def _rag_context_block(chunks: list[str]) -> str | None:
    if not chunks:
        return None
    parts = ["[Internal knowledge]"] + chunks + ["[End of internal knowledge]"]
    return "\n\n".join(parts)


#UTIL
def coerce_bool(value, default: bool) -> bool:
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
            title   = result.get("title", "")
            url     = result.get("url", "")
            content = result.get("content", "")
            lines.append(f"[{i}] {title}\nURL: {url}\n{content}\n")
            sources.append({"index": i, "title": title, "url": url})
        return "\n".join(lines), sources
    except Exception as e:
        return f"(Web search failed: {e})", []


#OPENCLAW
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


#FLASK
@app.route("/")
def index():
    return "MALTA-GEO API"


@app.route("/complete", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def complete():
    """
    Legacy-compatible endpoint that mirrors the original MALTA-GEO /complete route.

    Request body:
      {
        "query":          "...",
        "inner_contexts": ["..."],   // optional
        "outer_contexts": ["..."]    // optional
      }

    Response body:
      { "response": "..." }
    """
    data            = request.get_json(force=True)
    query           = data.get("query", "").strip()
    inner_contexts  = data.get("inner_contexts", [])
    outer_contexts  = data.get("outer_contexts", [])

    if not query:
        return jsonify({"error": "Campo 'query' é obrigatório"}), 400

    input_data = build_input_data(query, inner_contexts, outer_contexts)

    try:
        response_text = run_openclaw(input_data)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": response_text})


@app.route("/api/message", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def message():
    """
    Full-featured chat endpoint with session history, RAG, web search,
    and inner/outer context support.

    Request body:
      {
        "message":        "...",
        "session_id":     "abc123",              // optional
        "search":         true | false | "auto", // default: "auto"
        "rag":            true | false,           // default: true
        "inner_contexts": ["..."],               // optional
        "outer_contexts": ["..."]                // optional
      }

    Response body:
      {
        "session_id": "...",
        "message":    "...",
        "response":   "...",
        "sources":    [...],
        "searched":   true | false,
        "rag":        true | false
      }
    """
    data            = request.get_json(force=True)
    prompt          = data.get("message", "").strip()
    session_id      = data.get("session_id") or str(uuid.uuid4())
    search_raw      = data.get("search", "auto")
    rag_opt         = coerce_bool(data.get("rag"), default=True)
    inner_contexts  = data.get("inner_contexts", [])
    outer_contexts  = data.get("outer_contexts", [])

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

    final_prompt = build_input_data(
        prompt,
        inner_contexts,
        outer_contexts,
        web_context=web_context,
        rag_context=rag_context,
        history=history,
    )

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



build_rag_index()   # index all PDFs/Markdown before accepting requests

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Flask running on http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
