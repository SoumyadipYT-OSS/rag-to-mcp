skills:
  - name: chunk_documents
    description: "Loads all policy .txt files from the data/policy-documents directory, splits each document into sentence-aware chunks of at most 400 tokens, and returns a list of chunk objects with metadata."
    input: "Path to the policy documents directory (string)."
    output: "A list of dicts, each with keys: doc_name (str), chunk_index (int), text (str)."
    error_handling: "Raises FileNotFoundError if the directory does not exist. Raises ValueError if no .txt files are found. Never splits mid-sentence; always completes the current sentence before splitting."

  - name: retrieve_and_answer
    description: "Embeds the query using all-MiniLM-L6-v2, retrieves the top-3 chunks from ChromaDB by cosine similarity, filters out chunks below the 0.6 threshold, then calls the LLM with retrieved chunks as context only to generate a grounded answer."
    input: "query (str), collection (ChromaDB collection), embedder (SentenceTransformer), llm_call (callable), top_k (int, default 3), threshold (float, default 0.6)."
    output: "A dict with keys: answer (str), cited_chunks (list of {doc_name, chunk_index, score}), refused (bool)."
    error_handling: "If no chunks score above the 0.6 threshold, returns the refusal template with a list of the sources that were checked. Never generates an answer from general knowledge when chunks are below threshold."
