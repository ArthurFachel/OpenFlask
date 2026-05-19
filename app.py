from flask import Flask, jsonify, request
import subprocess
import json
import os

app = Flask(__name__)


@app.route("/api/message", methods=["POST"])
def message():
    """
    Send a prompt to OpenClaw and get the AI response back.

    Request body:  { "message": "What is the capital of Brazil?" }
    Response body: { "message": "...", "response": "..." }
    """
    data = request.get_json(force=True)
    prompt = data.get("message", "").strip()

    if not prompt:
        return jsonify({"error": "No message provided."}), 400

    try:
        proc = subprocess.run(
            ["openclaw", "infer", "model", "run", "--prompt", prompt, "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        raw = (proc.stdout + proc.stderr).strip()

        try:
            result = json.loads(raw)
            # Extract the text output from the response
            outputs = result.get("outputs", [])
            if outputs:
                text = outputs[0].get("text") or outputs[0].get("content") or str(outputs[0])
            else:
                text = raw
            return jsonify({"message": prompt, "response": text})
        except json.JSONDecodeError:
            # Return raw output if not JSON
            return jsonify({"message": prompt, "response": raw})

    except FileNotFoundError:
        return jsonify({"error": "openclaw CLI not found. Is it installed?"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Request timed out after 60 seconds."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Check OpenClaw gateway status."""
    proc = subprocess.run(
        ["openclaw", "gateway", "status"],
        capture_output=True, text=True, timeout=10
    )
    return jsonify({"output": (proc.stdout + proc.stderr).strip()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Flask running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)