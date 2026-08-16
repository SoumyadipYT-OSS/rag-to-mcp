"""
UC-RAG — RAG Server
rag_server.py — Full implementation

Implements:
  1. chunk_documents — sentence-aware chunking (max 400 tokens)
  2. build_index      — embed chunks and store in ChromaDB
  3. retrieve_and_answer — embed query, retrieve top-k, enforce threshold, call LLM
  4. naive_query      — load all docs without retrieval (show failure mode)

Run:
  python rag_server.py --build-index
  python rag_server.py --query "Who approves leave without pay?"
  python rag_server.py --naive --query "Can I use my personal phone for work files?"
"""

import argparse
import os
import re
import sys


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

# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[LLM NOT CONFIGURED] Set GEMINI_API_KEY in .env or environment."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        return response.text
    except ImportError:
        try:
            import google.generativeai as old_genai  # type: ignore
            old_genai.configure(api_key=api_key)
            model = old_genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            return f"[LLM ERROR] {str(e2)}"
    except Exception as e:
        return f"[LLM ERROR] {e}"


# ── SKILL: chunk_documents ────────────────────────────────────────────────────
def chunk_documents(docs_dir: str, max_tokens: int = 400) -> list:
    """
    Load all .txt files from docs_dir.
    Split each into sentence-aware chunks of max_tokens (approximated as words).
    Returns list of: {doc_name, chunk_index, text}

    Enforcement:
    - Never split mid-sentence (sentence-boundary aware)
    - Never exceed max_tokens per chunk
    """
    chunks = []
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Policy documents directory not found: {docs_dir}")

    # Sentence splitter: split on . ! ? followed by whitespace or end
    sentence_end_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    txt_files = sorted([f for f in os.listdir(docs_dir) if f.endswith(".txt")])
    if not txt_files:
        raise ValueError(f"No .txt files found in {docs_dir}")

    for filename in txt_files:
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Split text into sentences
        sentences = sentence_end_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunk_idx = 0
        current_chunk_sentences = []
        current_token_count = 0

        for sentence in sentences:
            # Approximate tokens as words
            token_count = len(sentence.split())

            if current_token_count + token_count > max_tokens and current_chunk_sentences:
                # Store the current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "doc_name": filename,
                    "chunk_index": chunk_idx,
                    "text": chunk_text
                })
                chunk_idx += 1
                current_chunk_sentences = []
                current_token_count = 0

            current_chunk_sentences.append(sentence)
            current_token_count += token_count

        # Flush remaining sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "doc_name": filename,
                "chunk_index": chunk_idx,
                "text": chunk_text
            })

    return chunks


# ── INDEX BUILDER ─────────────────────────────────────────────────────────────
def build_index(docs_dir: str, db_path: str = "./chroma_db"):
    """
    Chunk all documents and store embeddings in ChromaDB.
    Called once before querying.
    """
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install with: pip install sentence-transformers chromadb")
        sys.exit(1)

    print("[build_index] Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[build_index] Chunking documents from: {docs_dir}")
    chunks = chunk_documents(docs_dir)
    print(f"[build_index] Total chunks: {len(chunks)}")

    print(f"[build_index] Creating ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    # Delete collection if it exists, to rebuild from scratch
    try:
        client.delete_collection("policy_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="policy_docs",
        metadata={"hnsw:space": "cosine"}
    )

    # Embed and upsert in batches
    texts = [c["text"] for c in chunks]
    ids = [f"{c['doc_name']}__chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [{"doc_name": c["doc_name"], "chunk_index": c["chunk_index"]} for c in chunks]

    print("[build_index] Embedding chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"[build_index] Done. {len(chunks)} chunks stored in ChromaDB.")


# ── SKILL: retrieve_and_answer ────────────────────────────────────────────────
def retrieve_and_answer(
    query: str,
    collection,
    embedder,
    llm_call=None,
    top_k: int = 3,
    threshold: float = 0.6,
) -> dict:
    """
    Embed query, retrieve top_k chunks from ChromaDB.
    Filter chunks below threshold (cosine distance → similarity = 1 - distance).
    If no chunks pass threshold, return refusal template.
    Otherwise call LLM with retrieved chunks as context only.

    Returns: {answer, cited_chunks: [{doc_name, chunk_index, score}]}
    """
    if llm_call is None:
        llm_call = call_llm

    query_embedding = embedder.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine distance (0=identical, 2=opposite)

    # Convert cosine distance to similarity: similarity = 1 - distance
    # For cosine metric in ChromaDB, distance = 1 - cosine_similarity
    # So similarity = 1 - distance
    scored_chunks = []
    for doc_text, meta, dist in zip(docs, metas, distances):
        similarity = 1.0 - dist
        if similarity >= threshold:
            scored_chunks.append({
                "text": doc_text,
                "doc_name": meta["doc_name"],
                "chunk_index": meta["chunk_index"],
                "score": round(similarity, 4)
            })

    # Refusal template if no chunks pass threshold
    if not scored_chunks:
        all_sources = [m["doc_name"] for m in metas]
        refusal = (
            "This question is not covered in the retrieved policy documents.\n"
            f"Retrieved chunks: {all_sources}. Please contact the relevant "
            "department for guidance."
        )
        return {"answer": refusal, "cited_chunks": [], "refused": True}

    # Build context-only prompt
    context_blocks = []
    for i, chunk in enumerate(scored_chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['doc_name']}, chunk {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a policy assistant for the City Municipal Corporation.

ENFORCEMENT RULES:
- Answer ONLY using information present in the retrieved chunks below.
- Never add context from outside the retrieved set.
- Cite the source document name and chunk index for every claim.
- If the retrieved chunks do not contain sufficient information, state that explicitly.

Retrieved Policy Chunks:
{context}

Question: {query}

Answer (cite sources):"""

    answer = llm_call(prompt)
    cited_chunks = [
        {"doc_name": c["doc_name"], "chunk_index": c["chunk_index"], "score": c["score"]}
        for c in scored_chunks
    ]

    return {"answer": answer, "cited_chunks": cited_chunks, "refused": False}


# ── expose a query() function for mcp_server to import ───────────────────────
def query(question: str, llm_call=None) -> dict:
    """
    Public entry-point used by mcp_server.py.
    Loads the ChromaDB collection, embedder, and calls retrieve_and_answer.
    """
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        return {"answer": f"[ERROR] Missing dependency: {e}", "cited_chunks": [], "refused": True}

    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(db_path):
        return {
            "answer": "[ERROR] ChromaDB index not found. Run: python rag_server.py --build-index",
            "cited_chunks": [],
            "refused": True
        }

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("policy_docs")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if llm_call is None:
        llm_call = call_llm

    return retrieve_and_answer(question, collection, embedder, llm_call)


# ── NAIVE MODE ────────────────────────────────────────────────────────────────
def naive_query(query_str: str, docs_dir: str, llm_call=None):
    """
    Load all documents into context without retrieval (shows failure modes).
    """
    if llm_call is None:
        llm_call = call_llm

    all_text = ""
    if not os.path.exists(docs_dir):
        return f"[ERROR] Docs directory not found: {docs_dir}"

    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                all_text += f"\n\n=== {filename} ===\n" + f.read()

    prompt = f"""Answer the following question based on these policy documents:

{all_text}

Question: {query_str}
Answer:"""

    return llm_call(prompt)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="UC-RAG RAG Server")
    parser.add_argument("--build-index", action="store_true",
                        help="Build ChromaDB index from policy documents")
    parser.add_argument("--query", type=str,
                        help="Query the RAG server")
    parser.add_argument("--naive", action="store_true",
                        help="Run naive (no retrieval) mode to see failure modes")
    parser.add_argument("--docs-dir", type=str,
                        default=os.path.join(_script_dir, "../data/policy-documents"),
                        help="Path to policy documents directory")
    parser.add_argument("--db-path", type=str,
                        default=os.path.join(_script_dir, "chroma_db"),
                        help="Path to ChromaDB storage directory")
    args = parser.parse_args()

    if not args.build_index and not args.query:
        parser.print_help()
        sys.exit(1)

    if args.build_index:
        print("Building index...")
        build_index(args.docs_dir, args.db_path)
        print("Index built. Run with --query to test.")

    if args.query:
        if args.naive:
            print(f"\n[Naive mode] Query: {args.query}")
            result = naive_query(args.query, args.docs_dir, call_llm)
            print(f"\nNaive answer:\n{result}")
        else:
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                print(f"[ERROR] Missing dependency: {e}")
                sys.exit(1)

            db_path = args.db_path
            if not os.path.exists(db_path):
                print(f"[ERROR] Index not found at {db_path}. Run --build-index first.")
                sys.exit(1)

            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection("policy_docs")
            embedder = SentenceTransformer("all-MiniLM-L6-v2")

            print(f"\n[RAG mode] Query: {args.query}")
            result = retrieve_and_answer(args.query, collection, embedder, call_llm)
            print(f"\nAnswer:\n{result['answer']}")
            if result["cited_chunks"]:
                print("\nCited chunks:")
                for c in result["cited_chunks"]:
                    print(f"  - {c['doc_name']} chunk {c['chunk_index']} (score: {c['score']})")
            else:
                print("\n[No chunks retrieved above threshold — refusal issued]")


if __name__ == "__main__":
    main()
