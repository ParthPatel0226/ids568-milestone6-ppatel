# Milestone 6: RAG Pipeline & Agentic System
## IDS568 MLOps Course — Module 7

![Python](https://img.shields.io/badge/python-3.11-blue)
![Model](https://img.shields.io/badge/model-qwen2.5%3A7b--instruct-green)
![Vector Store](https://img.shields.io/badge/vector--store-FAISS-orange)
![Serving](https://img.shields.io/badge/serving-Ollama-purple)

---

## Overview

This repository implements two core MLOps AI systems:

1. **RAG Pipeline** (`rag_pipeline.py`) — A complete retrieval-augmented
   generation system over a 10-document MLOps knowledge corpus using FAISS
   vector search and `qwen2.5:7b-instruct` for grounded generation.

2. **Agent Controller** (`agent_controller.py`) — A multi-tool agent that
   intelligently selects between a `RetrieverTool` and `SummarizerTool`
   to solve 10 diverse multi-step MLOps tasks with full decision traces.

---

## Model & Serving Details

| Property | Value |
|----------|-------|
| **Model Name** | qwen2.5:7b-instruct |
| **Parameter Class** | 7B |
| **Serving Stack** | Ollama (local) |
| **Quantization** | Q4_K_M (4-bit, Ollama default) |
| **Hardware** | CPU-only, 8GB RAM, Windows |
| **RAG Generation Latency** | ~53s per query (CPU) |
| **Agent Task Latency** | ~128s per task (CPU) |
| **Embedding Model** | all-MiniLM-L6-v2 (384-dim) |
| **Vector Store** | FAISS IndexFlatL2 |

---

## Repository Structure

ids568-milestone6-ppatel/
├── rag_pipeline.py            # Part 1: RAG implementation
├── agent_controller.py        # Part 2: Agent implementation
├── rag_evaluation_report.md   # Part 1: Evaluation report
├── rag_pipeline_diagram.md    # Part 1: Pipeline architecture
├── agent_report.md            # Part 2: Agent analysis
├── agent_traces/              # Part 2: 10 task traces (JSON)
│   ├── task_01.json
│   ├── task_02.json
│   ├── ...
│   ├── task_10.json
│   └── all_traces.json
├── rag_results.json           # Part 1: Evaluation results
├── requirements.txt           # Pinned dependencies
└── README.md                  # This file

---

## Setup Instructions

### Prerequisites
- Python 3.11
- Windows / macOS / Linux
- 8GB RAM minimum
- Ollama installed ([https://ollama.com/download](https://ollama.com/download))

### Step 1: Clone the repository
```bash
git clone https://github.com/ParthPatel0226/ids568-milestone6-ppatel.git
cd ids568-milestone6-ppatel
```

### Step 2: Create virtual environment
```bash
python -m venv venv611
# Windows:
venv611\Scripts\activate
# macOS/Linux:
source venv611/bin/activate
```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Pull the LLM model via Ollama
```bash
ollama pull qwen2.5:7b-instruct
```
> This downloads ~4.7GB. Run once, then the model is cached locally.

### Step 5: Start Ollama server (if not auto-started)
```bash
ollama serve
```
> On Windows, Ollama starts automatically after install.
> If generation fails with a connection error, open a separate terminal
> and run `ollama serve` to start the server manually.

### Step 6: Verify model is available
```bash
ollama run qwen2.5:7b-instruct "Hello, are you ready?"
```
> You should get a short response. If you do, the model is ready.

### Step 7: Verify all Python dependencies
```bash
python -c "import langchain; import faiss; import sentence_transformers; import ollama; print('All imports OK!')"
```

---

## Usage

### Run RAG Pipeline
```bash
python rag_pipeline.py
```
This will:
- Load and chunk 10 MLOps documents
- Build FAISS vector index
- Run 10 evaluation queries with retrieval metrics
- Run grounded generation for all 10 queries
- Save results to `rag_results.json`

**Expected runtime: ~10 minutes on CPU**

### Run Agent Controller
```bash
python agent_controller.py
```
This will:
- Build shared FAISS vector store
- Initialize RetrieverTool + SummarizerTool
- Run 10 multi-step evaluation tasks
- Save individual traces to `agent_traces/task_XX.json`
- Save combined traces to `agent_traces/all_traces.json`

**Expected runtime: ~20-25 minutes on CPU**

---

## Architecture Overview

### RAG Pipeline

Documents → Chunker (512 chars, 64 overlap)
→ Embedder (all-MiniLM-L6-v2)
→ FAISS Index
→ Retriever (k=3)
→ Prompt Builder
→ qwen2.5:7b-instruct (Ollama)
→ Grounded Answer

### Agent Controller

Task → Tool Selection (LLM decides)
→ RETRIEVE_ONLY:            Retriever → Generator
→ RETRIEVE_THEN_SUMMARIZE:  Retriever → Summarizer → Generator
→ MULTI_STEP:               Retriever x2 → Dedup → Generator
→ SUMMARIZE_ONLY:           Summarizer → Generator
→ Final Answer + Full Trace Logged

---

## Results Summary

### RAG Pipeline (Part 1)

| Metric | Value |
|--------|-------|
| Avg Precision@3 | 0.700 |
| Source Match Rate | 1.000 |
| Avg Retrieval Latency | ~0.10s |
| Avg Generation Latency | 53.00s |
| Avg End-to-End Latency | 53.10s |

### Agent Controller (Part 2)

| Metric | Value |
|--------|-------|
| Tasks Completed | 10/10 |
| Success Rate | 100% |
| Avg Task Latency | 128.6s |

---

## Evaluation Files

| File | Description |
|------|-------------|
| `rag_results.json` | Full retrieval and generation metrics for all 10 queries |
| `agent_traces/task_XX.json` | Individual agent trace per task with step-by-step decisions |
| `agent_traces/all_traces.json` | Combined traces for all 10 tasks |
| `rag_evaluation_report.md` | Detailed retrieval accuracy and grounding analysis |
| `agent_report.md` | Agent performance, failure analysis, model tradeoffs |

---

## Known Limitations

1. **CPU-only inference** — Generation latency (~53s/query) is high due
   to CPU-only inference. GPU deployment would reduce this to ~2–3s.

2. **Small corpus** — The knowledge base contains 10 documents. Retrieval
   precision would improve with a larger, more diverse corpus.

3. **7B model grounding** — The 7B model occasionally supplements answers
   with minor details not present in retrieved context. A 14B model would
   show stricter grounding adherence.

4. **Naive sub-query splitting** — MULTI_STEP tasks use word-count based
   sub-query generation. LLM-based sub-query decomposition would improve
   quality on complex comparative tasks.

5. **No persistent index** — The FAISS index is rebuilt on every run.
   Saving/loading the index would reduce startup time significantly.

6. **No GPU support** — Current setup runs on CPU only. For faster
   inference, a GPU with at least 8GB VRAM and vLLM or Transformers
   serving would be needed.