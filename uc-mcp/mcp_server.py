"""
UC-MCP — mcp_server.py
Plain HTTP MCP Server — JSON-RPC 2.0 over HTTP POST

Implements:
  - tools/list: returns the query_policy_documents tool definition
  - tools/call: executes query_policy_documents via RAG server

Run:
  python mcp_server.py --port 8765

Test:
  python test_client.py --port 8765 --run-all
"""

import json
import argparse
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler


# ── Load .env ─────────────────────────────────────────────────────────────────
def load_dotenv():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(4):
        env_path = os.path.join(cur, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
            break
        cur = os.path.dirname(cur)


load_dotenv()

# ── Import RAG — uses participant's rag_server, falls back to stub ─────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../uc-rag"))
try:
    from rag_server import query as rag_query
    print("[mcp_server] Using participant rag_server.py")
except (ImportError, AttributeError, Exception) as e:
    print(f"[mcp_server] rag_server.py unavailable ({e}), trying stub_rag.py")
    try:
        from stub_rag import query as rag_query
        print("[mcp_server] Using stub_rag.py (fallback)")
    except Exception as e2:
        print(f"[mcp_server] ERROR: Could not load any RAG module: {e2}")
        def rag_query(question, llm_call=None):
            return {
                "answer": "[ERROR] No RAG module available. Run --build-index first.",
                "cited_chunks": [],
                "refused": True
            }

# ── Import LLM adapter ────────────────────────────────────────────────────────
from llm_adapter import call_llm


# ── TOOL DEFINITION ───────────────────────────────────────────────────────────
TOOL_DEFINITION = {
    "name": "query_policy_documents",
    "description": (
        "Answers questions strictly about the City Municipal Corporation (CMC) internal policies: "
        "(1) CMC HR Leave Policy — leave entitlements, approvals, leave without pay, encashment; "
        "(2) CMC IT Acceptable Use Policy — device usage, personal phone restrictions, remote access, data handling; "
        "(3) CMC Finance Reimbursement Policy — travel claims, equipment allowances, home office reimbursements. "
        "Returns cited answers grounded in retrieved document chunks. "
        "For questions outside these three policy documents (e.g., budget forecasts, procurement, legal matters), "
        "this tool returns a refusal with isError: true. Do NOT call this tool for general knowledge queries."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A non-empty policy question about CMC HR Leave Policy, "
                    "IT Acceptable Use Policy, or Finance Reimbursement Policy."
                ),
                "minLength": 1,
            }
        },
        "required": ["question"],
    },
}


# ── SKILL: query_policy_documents ─────────────────────────────────────────────
def query_policy_documents(question: str) -> dict:
    """
    Call the RAG server with the question.
    Returns MCP content format: {"content": [...], "isError": bool}

    Error handling:
    - Empty question → isError: True
    - RAG refuses (no chunks above threshold) → isError: True
    - RAG raises exception → isError: True with error message
    """
    if not question or not question.strip():
        return {
            "content": [{"type": "text", "text": "Error: question must be a non-empty string."}],
            "isError": True
        }

    try:
        result = rag_query(question, llm_call=call_llm)
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"[RAG ERROR] {str(e)}"}],
            "isError": True
        }

    refused = result.get("refused", False)
    answer = result.get("answer", "")
    cited = result.get("cited_chunks", [])

    if refused or not answer:
        return {
            "content": [{"type": "text", "text": answer or "No answer could be retrieved."}],
            "isError": True
        }

    # Build response with citations appended
    cited_str = ""
    if cited:
        cited_str = "\n\nSources:\n" + "\n".join(
            f"  - {c['doc_name']} chunk {c['chunk_index']} (score: {c.get('score', 'n/a')})"
            for c in cited
        )

    return {
        "content": [{"type": "text", "text": answer + cited_str}],
        "isError": False
    }


# ── SKILL: serve_mcp — JSON-RPC 2.0 handler ───────────────────────────────────
class MCPHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler implementing JSON-RPC 2.0.
    Handles POST requests to / with JSON-RPC body.
    Returns HTTP 200 for all JSON-RPC responses (including application errors).
    """

    def do_POST(self):
        # Read and parse body
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            rpc_request = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_jsonrpc_error(
                id=None, code=-32700, message="Parse error"
            )
            return

        rpc_id = rpc_request.get("id")
        method = rpc_request.get("method", "")
        params = rpc_request.get("params", {})

        # ── tools/list ────────────────────────────────────────────────────────
        if method == "tools/list":
            self._send_jsonrpc_result(rpc_id, {"tools": [TOOL_DEFINITION]})

        # ── tools/call ────────────────────────────────────────────────────────
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name != "query_policy_documents":
                self._send_jsonrpc_error(
                    id=rpc_id, code=-32601,
                    message=f"Tool not found: {tool_name}"
                )
                return

            question = arguments.get("question", "")
            result = query_policy_documents(question)
            self._send_jsonrpc_result(rpc_id, result)

        # ── unknown method ────────────────────────────────────────────────────
        else:
            self._send_jsonrpc_error(
                id=rpc_id, code=-32601,
                message=f"Method not found: {method}"
            )

    def _send_jsonrpc_result(self, id, result):
        response = {
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        }
        self._write_json(response)

    def _send_jsonrpc_error(self, id, code, message):
        response = {
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message}
        }
        self._write_json(response)

    def _write_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[mcp_server] {self.command} {self.path} → {args[1] if len(args) > 1 else ''}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UC-MCP Plain HTTP MCP Server")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to listen on (default: 8765)")
    args = parser.parse_args()

    # Warn if RAG index is missing
    db_path = os.path.join(os.path.dirname(__file__), "../uc-rag/chroma_db")
    stub_db_path = os.path.join(os.path.dirname(__file__), "../uc-rag/stub_chroma_db")
    if not os.path.exists(db_path) and not os.path.exists(stub_db_path):
        print("[mcp_server] WARNING: No RAG index found.")
        print("[mcp_server] Build with: python ../uc-rag/rag_server.py --build-index")
        print("[mcp_server]         or: python ../uc-rag/stub_rag.py --build-index")

    server = HTTPServer(("localhost", args.port), MCPHandler)
    print(f"[mcp_server] MCP server running on http://localhost:{args.port}")
    print(f"[mcp_server] Test with: python test_client.py --port {args.port} --run-all")
    print(f"[mcp_server] Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mcp_server] Stopped.")


if __name__ == "__main__":
    main()
