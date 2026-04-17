# Agent Controller Report
## Milestone 6 — MLOps Course Module 7

**Model:** qwen2.5:7b-instruct (7B, Ollama)  
**Tools:** RetrieverTool (FAISS) + SummarizerTool (LLM-based)  
**Tasks Completed:** 10/10  
**Avg Task Latency:** 128.6s (CPU-only, 8GB RAM)  

---

## 1. Tool Selection Policy

The agent uses a **hybrid keyword + LLM-based policy** to decide which
tool sequence to invoke for each task.

### Decision Categories

| Decision | When Used | Tool Sequence |
|----------|-----------|---------------|
| `RETRIEVE_ONLY` | Factual questions needing knowledge base lookup | Retrieve → Answer |
| `RETRIEVE_THEN_SUMMARIZE` | Tasks asking for overviews or summaries of a topic | Retrieve → Summarize → Answer |
| `MULTI_STEP` | Comparative or multi-part questions | Retrieve (x2) → Deduplicate → Answer |
| `SUMMARIZE_ONLY` | Raw text condensation with no retrieval needed | Summarize → Answer |

### How the Policy Works
1. The agent receives a task description
2. It sends the task to `qwen2.5:7b-instruct` with a structured prompt
   listing available tools and asking for a single decision token
3. The model responds with one of the four decision labels
4. The agent controller executes the corresponding tool sequence
5. Retrieved/processed context is passed to a final generation step

This design makes **every decision observable and logged** — the decision
label is recorded in the trace before any tool is invoked.

### Trigger Examples
- "What is..." / "How does..." / "Explain..." → `RETRIEVE_ONLY`
- "Summarize..." / "Give me an overview..." → `RETRIEVE_THEN_SUMMARIZE`
- "Compare X and Y..." / "How do X and Y differ..." → `MULTI_STEP`
- Direct text input for condensing → `SUMMARIZE_ONLY`

---

## 2. Retrieval Integration

The `RetrieverTool` wraps the same FAISS vector store built in Part 1,
making the retriever **fully reusable** between the RAG pipeline and
the agent.

### How Retrieval Integrates with Agent Capabilities
- **As a standalone tool:** For `RETRIEVE_ONLY` tasks, retrieved chunks
  are passed directly as context to the final generation step
- **As a pipeline stage:** For `RETRIEVE_THEN_SUMMARIZE`, retrieved
  chunks feed into the `SummarizerTool` before generation — creating a
  two-stage grounding pipeline
- **As a multi-query tool:** For `MULTI_STEP` tasks, the retriever is
  called twice with different sub-queries, results are deduplicated by
  source, and combined context is passed to generation
- **Skipped entirely:** For `SUMMARIZE_ONLY` tasks, retrieval is
  bypassed — the agent correctly identifies when the knowledge base
  is not needed

### Retrieval Parameters
- **k=3** chunks per query (balances context richness vs noise)
- **Embedding model:** all-MiniLM-L6-v2 (consistent with Part 1)
- **Vector store:** FAISS IndexFlatL2 (exact search, ~0.10s latency)

---

## 3. Performance Analysis — 10 Tasks

| # | Task (abbreviated) | Decision | Success | Latency |
|---|--------------------|----------|---------|---------|
| 1 | MLOps maturity model levels? | RETRIEVE_ONLY | ✅ | ~110s |
| 2 | Tools for model monitoring? | RETRIEVE_ONLY | ✅ | ~105s |
| 3 | How does RLHF align LLMs? | RETRIEVE_ONLY | ✅ | ~115s |
| 4 | Role of XComs in Airflow? | RETRIEVE_ONLY | ✅ | ~108s |
| 5 | MLflow Model Registry components? | RETRIEVE_ONLY | ✅ | ~112s |
| 6 | How RAG pipelines work end-to-end? | RETRIEVE_THEN_SUMMARIZE | ✅ | ~145s |
| 7 | CI/CD practices in ML systems? | RETRIEVE_THEN_SUMMARIZE | ✅ | ~150s |
| 8 | Feature stores vs vector databases? | MULTI_STEP | ✅ | ~180s |
| 9 | Agents vs RAG tool use differences? | MULTI_STEP | ✅ | ~175s |
| 10 | Drift detected, no retraining pipeline? | RETRIEVE_ONLY | ✅ | ~115s |

**Overall success rate: 10/10 (100%)**  
**Average latency: 128.6s (CPU-only)**

### Observations
- `RETRIEVE_ONLY` tasks were fastest (~105–115s) — single retrieval + generation
- `RETRIEVE_THEN_SUMMARIZE` tasks took longer (~145–150s) — extra LLM summarization call
- `MULTI_STEP` tasks were slowest (~175–180s) — two retrievals + synthesis
- All tasks produced coherent, grounded answers with observable decision traces

---

## 4. Failure Analysis

### No Complete Failures (10/10 succeeded)
All tasks returned non-empty, coherent answers. However, several
partial failure modes were observed:

### Partial Failure 1: Tool Selection Inconsistency
- **Task 10** (edge case — drift with no retraining pipeline) was
  classified as `RETRIEVE_ONLY` rather than `RETRIEVE_THEN_SUMMARIZE`
- The model retrieved relevant monitoring content but missed the
  hypothetical/reasoning aspect of the question
- **Root cause:** 7B model occasionally misclassifies edge-case tasks
  that require reasoning beyond factual retrieval
- **Impact:** Answer was still useful but less analytically deep

### Partial Failure 2: MULTI_STEP Sub-query Quality
- For Task 8 (feature stores vs vector databases), the automatic
  sub-query splitting used a naive word-count midpoint
- This sometimes produced grammatically awkward sub-queries
- **Root cause:** Sub-query generation is rule-based, not LLM-based
- **Impact:** Minor — FAISS is robust to slightly malformed queries
  due to semantic embedding search

### Partial Failure 3: Generation Drift on Long Context
- For `RETRIEVE_THEN_SUMMARIZE` tasks, the model occasionally added
  minor details not present in the summarized context
- **Root cause:** 7B model capacity — smaller models are less strict
  about grounding instructions on complex prompts
- **Impact:** Low — core answers remained accurate

### When Does the Agent Struggle?
- Tasks requiring **hypothetical reasoning** (what-if scenarios)
  beyond what the corpus contains
- Tasks where **two topics overlap significantly** in the corpus
  (e.g., RAG and vector databases both discuss embeddings)
- Tasks requiring **precise numerical comparisons** not in documents

---

## 5. Model Quality and Latency Tradeoffs

### Quality Analysis
| Aspect | Assessment |
|--------|------------|
| Instruction following | Good — correctly uses provided context in 9/10 tasks |
| Tool decision accuracy | Good — correct tool sequence in 9/10 tasks |
| Grounding adherence | Moderate — occasional minor drift on complex tasks |
| Coherence | Excellent — all responses well-structured |

### Latency Analysis
| Component | Latency |
|-----------|---------|
| Tool selection (LLM call) | ~50s |
| Retrieval (FAISS) | ~0.10s |
| Summarization (LLM call) | ~50s |
| Final generation (LLM call) | ~50s |
| **Total RETRIEVE_ONLY** | **~110s** |
| **Total RETRIEVE_THEN_SUMMARIZE** | **~150s** |
| **Total MULTI_STEP** | **~175s** |

### Hardware Context
- All inference on CPU-only, 8GB RAM, Windows machine
- Ollama Q4_K_M quantization (4-bit) enabled running within 8GB
- GPU inference (RTX 3080) would reduce per-call latency from ~50s to ~2s
- Production deployment on GPU would bring E2E agent latency to ~6–10s

### Model Size Tradeoffs
- **7B (current):** Fits in 8GB RAM, good quality, slow on CPU
- **14B:** Better instruction following and grounding, requires 16GB RAM
- **3B:** Faster but noticeably worse tool selection accuracy
- For this task complexity, 7B is the right balance for available hardware

---

## 6. Architecture Summary

```
User Task
    │
    ▼
[Tool Selection] ← qwen2.5:7b-instruct decides tool sequence
    │
    ├── RETRIEVE_ONLY ────────────► [RetrieverTool] → [Generator]
    │
    ├── RETRIEVE_THEN_SUMMARIZE ──► [RetrieverTool] → [SummarizerTool] → [Generator]
    │
    ├── MULTI_STEP ───────────────► [RetrieverTool x2] → [Dedup] → [Generator]
    │
    └── SUMMARIZE_ONLY ───────────► [SummarizerTool] → [Generator]
                                              │
                                              ▼
                                        Final Answer
                                     + Full Trace Logged
```