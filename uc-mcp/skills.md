skills:
  - name: query_policy_documents
    description: "Accepts a question string, forwards it to the RAG server (rag_server.py or stub_rag.py fallback), and returns the answer in MCP content format."
    input: "question (str) — a policy question about CMC HR, IT, or Finance policies."
    output: "A dict with keys: content (list of {type: text, text: str}), isError (bool)."
    error_handling: "If the RAG server returns refused=True, return isError: true with the refusal message as content. If the RAG server raises an exception, return isError: true with the error message. Never return an empty content array on failure."

  - name: serve_mcp
    description: "Starts an HTTP server on a configurable port (default 8765) and dispatches incoming JSON-RPC 2.0 POST requests to tools/list or tools/call handlers."
    input: "port (int, default 8765)."
    output: "Runs the server indefinitely until interrupted. Returns JSON-RPC 2.0 compliant responses (HTTP 200 for all application-level responses)."
    error_handling: "Unknown JSON-RPC methods return error code -32601 (Method not found). Malformed JSON bodies return error code -32700 (Parse error). All error responses use HTTP 200 with isError in the result payload."
