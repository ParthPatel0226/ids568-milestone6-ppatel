# RAG Pipeline Architecture Diagram
## Milestone 6 — MLOps Course Module 7

---

## ASCII Pipeline Diagram

┌─────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                                  │
│                  (Milestone 6 — MLOps Module 7)                     │
└─────────────────────────────────────────────────────────────────────┘
INGESTION PHASE (One-time, Offline)
─────────────────────────────────────────────────────────────────────
┌──────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  Raw MLOps   │     │      CHUNKER         │     │    EMBEDDER       │
│  Documents   │────▶│ RecursiveCharacter   │────▶│ all-MiniLM-L6-v2 │
│  (10 docs)   │     │ TextSplitter         │     │  (384-dim)        │
│              │     │ chunk_size=512        │     │  HuggingFace      │
│  ~1000 words │     │ overlap=64            │     │  Sentence         │
│  per doc     │     │                      │     │  Transformers     │
└──────────────┘     └─────────────────────┘     └────────┬──────────┘
│
┌─────────────────────────────▼──────────┐
│           VECTOR STORE                  │
│        FAISS (IndexFlatL2)              │
│                                         │
│  • ~50 chunks indexed                   │
│  • 384-dim embedding vectors            │
│  • Cosine similarity search             │
│  • Exact nearest neighbor (small corpus)│
└─────────────────────────────────────────┘
QUERY PHASE (Per-request, Online)
─────────────────────────────────────────────────────────────────────
┌──────────────┐     ┌─────────────────────┐
│  User Query  │     │      EMBEDDER        │
│              │────▶│  all-MiniLM-L6-v2   │
│ "What is     │     │  (same model as      │
│  RAG?"       │     │   ingestion)         │
└──────────────┘     └──────────┬──────────┘
│
│  query vector (384-dim)
▼
┌─────────────────────┐
│      RETRIEVER       │
│   FAISS k-NN Search  │
│                     │
│  k=3 top chunks      │
│  ~0.10s latency      │
└──────────┬──────────┘
│
┌────────────┴────────────┐
│  Retrieved Chunks (k=3)  │
│                         │
│  chunk_1: [Source: X]   │
│  chunk_2: [Source: Y]   │
│  chunk_3: [Source: Z]   │
└────────────┬────────────┘
│
▼
┌─────────────────────┐
│   PROMPT BUILDER     │
│                     │
│  System instruction  │
│  + Retrieved context │
│  + User query        │
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│     GENERATOR        │
│  qwen2.5:7b-instruct │
│  via Ollama (local)  │
│                     │
│  ~53s CPU latency    │
│  Grounded generation │
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│   FINAL ANSWER       │
│  Grounded response   │
│  with source context │
└─────────────────────┘
DATA TRANSFORMATIONS SUMMARY
─────────────────────────────────────────────────────────────────────
Raw Text  ──▶  LangChain Documents  ──▶  512-char Chunks
──▶  384-dim Embedding Vectors  ──▶  FAISS Index
──▶  Top-k Retrieved Chunks  ──▶  Structured Prompt
──▶  LLM Response  ──▶  Grounded Answer
KEY DESIGN DECISIONS
─────────────────────────────────────────────────────────────────────
Component         Choice                  Rationale
─────────         ──────                  ─────────
Chunker           RecursiveCharacter      Preserves semantic units
Chunk size        512 chars               Balances context vs precision
Overlap           64 chars (12.5%)        Prevents boundary info loss
Embedder          all-MiniLM-L6-v2        Fast, CPU-friendly, 384-dim
Vector Store      FAISS IndexFlatL2       Open-source, exact search
Retrieval k       3                       Enough context, not too noisy
Generator         qwen2.5:7b-instruct     Open-weight, runs on 8GB RAM
Serving           Ollama                  Simplest local serving stack

---

## Component Descriptions

### 1. Chunker
Splits raw documents into overlapping fixed-size chunks using
`RecursiveCharacterTextSplitter`. Tries paragraph → sentence → word
boundaries in order to preserve semantic coherence.

### 2. Embedder
Converts text chunks and queries into 384-dimensional dense vectors
using `sentence-transformers/all-MiniLM-L6-v2`. Same model used for
both indexing and query encoding to ensure vector space consistency.

### 3. Vector Store (FAISS)
Stores all chunk embeddings and enables fast similarity search.
`IndexFlatL2` performs exact L2 distance search — appropriate for
small corpora (~50 chunks). Returns top-k most similar chunks.

### 4. Retriever
Takes encoded query vector, searches FAISS index, returns top-k=3
chunks with similarity scores. Average latency ~0.10s.

### 5. Prompt Builder
Assembles retrieved chunks with source metadata into a structured
prompt that instructs the LLM to answer using only provided context.

### 6. Generator
`qwen2.5:7b-instruct` served via Ollama locally. Receives structured
prompt and produces grounded natural language answer. Average latency
~53s on CPU (8GB RAM, no GPU).