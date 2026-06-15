from flask import Flask, request, jsonify, send_file, g
from langchain_chroma import Chroma
import subprocess
import time
import re
from tavily import TavilyClient
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
import chromadb
import requests
import os
import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import db
import auth
import sessions
from auth import require_api_key

load_dotenv()

db.init_db()
sessions.init_sessions()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MAX_RETRIES = int(os.getenv("OPENCLAW_MAX_RETRIES", 3))
RETRY_BACKOFF = float(os.getenv("OPENCLAW_RETRY_BACKOFF", 2.0))
OPENCLAW_TIMEOUT = int(os.getenv("OPENCLAW_TIMEOUT", 60))

tavily = TavilyClient(TAVILY_API_KEY)

SYSTEM_PROMPT = """
{
  "role": "Assistente", 
  "task": "Responder perguntas sobre geologia na Bacia do Araripe",
  "content": {
    "description": "Você é um Assistente de IA especializado em geologia, com foco exclusivo na Bacia do Araripe (Nordeste do Brasil).",
    "objectives": [
      "Responder com precisão e clareza sobre rochas, sedimentos e minerais (tipos, formação, distribuição).",
      "Explicar formações geológicas (estruturas, estratigrafia, história).",
      "Apresentar informações sobre fósseis (descobertas, relevância científica)."
    ],
    "restrictions": [
      "Só responda sobre a Bacia do Araripe. Caso contrário, informe que seu escopo é restrito a esse tema.",
      "Não faça perguntas diretas ao usuário.",
      "Respostas devem ser concisas, lógicas e baseadas em evidências científicas.",
      "Manter tom profissional, sem mencionar regras internas ou linguagem inadequada."
    ],
    "language": "Português, Brasil",
    "example": {
      "question": "Quais são as formações geológicas da Bacia do Araripe?",
      "answer": "As principais são a Formação Santana (fósseis preservados) e a Formação Crato (calcário laminado)."
    }
  }
}"""

CSV_PATH = os.getenv("CSV_PATH", "./queries_log.csv")
API_URL = os.getenv("AWS_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "book_collection")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", 5))
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek.r1-v1:0")

embed_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embed_model,
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)



def retrieve_context(query: str) -> tuple[str, list[dict]]:
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    return context, sources


def to_csv(
    user_id: str,
    session_id: str,
    query: str,
    inner_contexts: list,
    outer_contexts: list,
    history_turns: int,
) -> dict:
    try:
        new_record = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "inner_contexts": "; ".join(inner_contexts) if inner_contexts else "",
            "outer_contexts": "; ".join(outer_contexts) if outer_contexts else "",
            "num_inner_contexts": len(inner_contexts),
            "num_outer_contexts": len(outer_contexts),
            "history_turns": history_turns,
        }

        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        else:
            df = pd.DataFrame([new_record])

        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        return new_record

    except Exception as e:
        return {"error": str(e)}

SEARCH_KEYWORDS = re.compile(
    r"\b("
    r"who|what|when|where|which|latest|current|today|"
    r"news|weather|price|score|result|update|release|"
    r"version|how much|how many|define|meaning|"
    r"capital|president|ceo|stock|rate|forecast"
    r")\b",
    re.IGNORECASE,
)

def needs_search(text: str) -> bool:
    if len(text.split()) <= 3:
        return False

    return bool(SEARCH_KEYWORDS.search(text))

def build_prompt(
    text: str,
    rag_context: str,
    inner_contexts: list,
    outer_contexts: list,
    history: list,
    web_context: str = "",
    schema: Optional[type[BaseModel]] = None,
) -> str:
    combined_context = ""

    if rag_context:
        combined_context += f"contexto relevante:\n{rag_context}\n\n"
    if inner_contexts:
        combined_context += f"inner_contexts: {', '.join(inner_contexts)}\n"
    if outer_contexts:
        combined_context += f"outer_contexts: {', '.join(outer_contexts)}\n"
    if web_context:
        combined_context += (
            f"resultados da busca web:\n"
            f"{web_context}\n\n"
        )

    history_block = ""
    if history:
        turns = []
        for turn in history:
            label = "Usuário" if turn.get("role") == "user" else "Assistente"
            turns.append(f"{label}: {turn.get('content', '')}")
        history_block = "histórico da conversa:\n" + "\n".join(turns) + "\n\n"

    prompt = f"{SYSTEM_PROMPT}\n{combined_context}{history_block}pergunta: {text}"

    if schema is not None:
        json_schema_str = json.dumps(schema.model_json_schema(), indent=2)
        prompt += f"\n\nRespond ONLY with a JSON object that follows this schema:\n{json_schema_str}"

    return prompt

def search_web(query: str) -> tuple[str, list[dict]]:
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

        for i, result in enumerate(response.get("results", [])[:5], start=1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")

            lines.append(
                f"[{i}] {title}\n"
                f"URL: {url}\n"
                f"{content}\n"
            )

            sources.append({
                "index": i,
                "title": title,
                "url": url,
            })

        return "\n".join(lines), sources

    except Exception as e:
        return f"(Web search failed: {e})", []
    

def call_aws_llm(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.9,
    timeout: int = 60,
) -> str:
    body = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if model:
        body["model"] = model

    response = requests.post(
        API_URL,
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    return data["body"]["response"]["output"]["message"]["content"][0]["text"]

def call_openclaw(prompt: str) -> str:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            proc = subprocess.run(
                [
                    "openclaw",
                    "infer",
                    "model",
                    "run",
                    "--prompt",
                    prompt,
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=OPENCLAW_TIMEOUT,
            )

            stdout = proc.stdout.strip()

            try:
                result = json.loads(stdout)

                outputs = result.get("outputs", [])

                if outputs:
                    return (
                        outputs[0].get("text")
                        or outputs[0].get("content")
                        or str(outputs[0])
                    )

                return stdout

            except json.JSONDecodeError:
                return stdout or proc.stderr.strip()

        except subprocess.TimeoutExpired:
            last_error = "timeout"

        except FileNotFoundError:
            raise RuntimeError("OpenClaw CLI não encontrado")

        except Exception as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(
        f"OpenClaw falhou após {MAX_RETRIES} tentativas. Último erro: {last_error}"
    )

app = Flask(__name__)


@app.route("/")
def index():
    return "MALTA-GEO API"


@app.route("/complete", methods=["POST"])
@require_api_key
def complete():
    """
    POST /complete

    Requires ``X-API-Key`` header.

    First request (no session_id):
    {
        "query": "Onde fica a Bacia do Araripe?",
        "inner_contexts": [],
        "outer_contexts": [],
        "rag": false
    }
    → response includes "session_id"; store it for subsequent requests.

    Subsequent requests (with session_id):
    {
        "query": "Quais fósseis existem lá?",
        "session_id": "sess_Xk9mB2...",
        "rag": false
    }
    """
    data = request.get_json()

    query = data.get("query")
    if not query:
        return jsonify({"error": "Campo 'query' é obrigatório"}), 400

    inner_contexts = data.get("inner_contexts", [])
    outer_contexts = data.get("outer_contexts", [])
    rag_enable = data.get("rag", False)
    session_id = data.get("session_id")
    inference = data.get("inference_type","aws")
    search_mode = data.get("search", "auto")
    
    if session_id:
        session = sessions.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session '{session_id}' not found"}), 404
        if session["user_id"] != g.api_user:
            return jsonify({"error": "Session does not belong to this API key"}), 403
        history = session["history"]
    else:
        session_id = sessions.create_session(g.api_user)
        history = []

    rag_context = ""
    if rag_enable:
        rag_context, _ = retrieve_context(query)
    if search_mode == "auto":
        do_search = needs_search(query)
    else:
        do_search = bool(search_mode)

    web_context = ""
    sources = []

    if do_search:
      web_context, sources = search_web(query)


    to_csv(g.api_user, session_id, query, inner_contexts, outer_contexts, len(history) // 2)

    prompt = build_prompt(
    query,
    rag_context,
    inner_contexts,
    outer_contexts,
    history,
    web_context=web_context,
)
   
    if inference == "aws":
        try:
            text_output = call_aws_llm(
                prompt=prompt,
                model=data.get("model", DEFAULT_MODEL),
                max_tokens=data.get("max_tokens", 4096),
                temperature=data.get("temperature", 0.1),
                top_p=data.get("top_p", 0.9),
            )
        except requests.HTTPError as e:
            return jsonify({"error": str(e), "status_code": e.response.status_code}), 502
        except requests.Timeout:
            return jsonify({"error": "LLM request timed out"}), 504
        except (KeyError, IndexError) as e:
            return jsonify({"error": f"Unexpected response structure: {e}"}), 502
    elif inference == "openclaw":

        text_output = call_openclaw(prompt)

    else:
        return jsonify({
            "error": "inference_type deve ser 'aws' ou 'openclaw'"
        }), 400
    
    sessions.append_turn(session_id, query, text_output)

    return jsonify({
    "generated_text": text_output,
    "session_id": session_id,
    "inference_type": inference,
    "searched": do_search,
    "sources": sources,
})


@app.route("/sessions", methods=["GET"])
@require_api_key
def get_sessions():
    """
    GET /sessions
    Returns all sessions belonging to the authenticated user.
    """
    user_sessions = sessions.list_sessions(user_id=g.api_user)
    return jsonify({
        "total": len(user_sessions),
        "sessions": user_sessions,
    })


@app.route("/sessions/<session_id>", methods=["GET"])
@require_api_key
def get_session_history(session_id):
    """
    GET /sessions/<session_id>
    Returns the full history of a session.
    """
    session = sessions.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    if session["user_id"] != g.api_user:
        return jsonify({"error": "Session does not belong to this API key"}), 403
    return jsonify(session)


@app.route("/logs", methods=["GET"])
@require_api_key
def get_logs():
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify({"message": "Nenhum log disponível ainda"}), 404

        df = pd.read_csv(CSV_PATH)
        return jsonify({
            "total_queries": len(df),
            "logs": df.to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs/export", methods=["GET"])
@require_api_key
def export_logs():
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify({"message": "Nenhum log disponível"}), 404

        return send_file(CSV_PATH, as_attachment=True, download_name="queries_log.csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)