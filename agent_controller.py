"""
Multi-Tool Agent Controller - Milestone 6
MLOps Course - Module 7

Agent coordinates two tools:
  1. Retriever Tool  — searches FAISS vector store for relevant MLOps knowledge
  2. Summarizer Tool — summarizes long text into concise bullet points

Decision policy:
  - Use Retriever when query needs factual knowledge from the corpus
  - Use Summarizer when task involves condensing or structuring retrieved content
  - Use both (Retrieve → Summarize) for multi-step tasks requiring grounded summaries
  - Use reasoning-only for simple logical/comparative tasks after retrieval

Model: qwen2.5:7b-instruct via Ollama
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ─────────────────────────────────────────────
# DOCUMENT CORPUS (reused from rag_pipeline.py)
# ─────────────────────────────────────────────

DOCUMENTS = [
    {
        "title": "Introduction to MLOps",
        "content": """
MLOps, short for Machine Learning Operations, is a set of practices that combines Machine Learning, 
DevOps, and Data Engineering. The goal of MLOps is to deploy and maintain ML models in production 
reliably and efficiently. MLOps addresses the full lifecycle of ML models including data preparation, 
model training, model evaluation, model deployment, and model monitoring. Key principles of MLOps 
include automation, reproducibility, versioning, and continuous monitoring. MLOps teams typically 
use tools like MLflow for experiment tracking, Airflow for pipeline orchestration, Docker for 
containerization, and Kubernetes for scaling. The MLOps maturity model ranges from manual processes 
at level 0 to fully automated CI/CD pipelines at level 3. Organizations adopt MLOps to reduce the 
time from model development to production deployment and to ensure model reliability over time.
        """
    },
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": """
Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by providing 
them with relevant external knowledge at inference time. Instead of relying solely on parametric 
knowledge stored in model weights, RAG retrieves relevant documents from an external knowledge base 
and uses them as context for generation. The RAG pipeline consists of two main components: a retriever 
and a generator. The retriever uses dense vector search to find semantically similar documents to the 
query. The generator, typically a large language model, uses the retrieved documents as context to 
produce grounded responses. RAG reduces hallucinations because the model can reference actual documents 
rather than fabricating information. Common retrieval methods include BM25 for sparse retrieval and 
FAISS for dense retrieval using embedding vectors. Hybrid search combines both sparse and dense methods 
for better recall. RAG is widely used in question answering systems, chatbots, and knowledge management 
applications. Chunking strategy significantly affects RAG performance — smaller chunks improve precision 
while larger chunks preserve more context.
        """
    },
    {
        "title": "Vector Databases and Embeddings",
        "content": """
Vector databases are specialized storage systems designed to store and query high-dimensional embedding 
vectors efficiently. Unlike traditional databases that use exact matching, vector databases use 
approximate nearest neighbor (ANN) algorithms to find semantically similar vectors. Popular vector 
databases include FAISS, Chroma, Pinecone, Weaviate, and Qdrant. FAISS (Facebook AI Similarity Search) 
is an open-source library developed by Meta that provides efficient similarity search for dense vectors. 
Embeddings are dense numerical representations of text that capture semantic meaning. Sentence 
transformers like all-MiniLM-L6-v2 produce 384-dimensional embeddings suitable for semantic search. 
The quality of embeddings directly affects retrieval accuracy — better embeddings lead to more relevant 
document retrieval. Indexing strategies like IVF (Inverted File Index) and HNSW (Hierarchical Navigable 
Small World) enable fast approximate search at scale. Cosine similarity and dot product are common 
distance metrics used to compare embedding vectors. Vector databases are a critical component of modern 
RAG systems and semantic search applications.
        """
    },
    {
        "title": "Model Monitoring in Production",
        "content": """
Model monitoring is the practice of tracking ML model performance after deployment to detect degradation, 
drift, and failures. There are several types of drift that can affect model performance: data drift occurs 
when the statistical distribution of input features changes over time, concept drift occurs when the 
relationship between inputs and outputs changes, and prediction drift occurs when model output 
distributions shift. Key monitoring metrics include prediction confidence scores, feature distribution 
statistics, model accuracy on labeled samples, and latency measurements. Tools used for model monitoring 
include Evidently AI, WhyLabs, Arize, and custom dashboards built with Grafana and Prometheus. Alerting 
systems notify teams when metrics exceed defined thresholds. A/B testing and shadow deployment are 
strategies used to safely evaluate new model versions. Model retraining pipelines are triggered 
automatically when drift is detected beyond acceptable thresholds. Monitoring is essential for 
maintaining model reliability and business value in production environments.
        """
    },
    {
        "title": "CI/CD for Machine Learning",
        "content": """
Continuous Integration and Continuous Deployment (CI/CD) for machine learning extends traditional 
software CI/CD practices to handle the unique challenges of ML systems. Unlike software, ML systems 
have three components that need versioning: code, data, and models. A typical ML CI/CD pipeline includes 
stages for data validation, model training, model evaluation, model registration, and deployment. 
GitHub Actions, Jenkins, and GitLab CI are popular tools for implementing ML CI/CD pipelines. Data 
validation steps check for schema changes, missing values, and distribution shifts before training. 
Model evaluation gates ensure only models that meet quality thresholds are deployed to production. 
MLflow Model Registry is used to manage model versions and stage transitions from Staging to Production. 
Automated testing for ML includes unit tests for data preprocessing functions, integration tests for 
pipeline components, and performance tests comparing new models against baselines. Canary deployments 
gradually shift traffic to new model versions to minimize risk. Blue-green deployments maintain two 
identical production environments for zero-downtime model updates.
        """
    },
    {
        "title": "Feature Engineering and Feature Stores",
        "content": """
Feature engineering is the process of transforming raw data into meaningful input representations 
for machine learning models. Good feature engineering can significantly improve model performance 
by capturing domain knowledge and relevant patterns. Common feature engineering techniques include 
normalization and standardization for numerical features, one-hot encoding and target encoding for 
categorical features, and TF-IDF or word embeddings for text features. Feature stores are centralized 
repositories that store, manage, and serve features for ML training and inference. Popular feature 
stores include Feast, Tecton, and Hopsworks. Feature stores solve the training-serving skew problem 
by ensuring that the same feature computation logic is used in both training and production. Offline 
feature stores serve historical features for model training while online feature stores serve 
low-latency features for real-time inference. Feature versioning allows teams to track changes to 
feature definitions over time. Point-in-time correct joins prevent data leakage by ensuring that 
only features available at prediction time are used during training.
        """
    },
    {
        "title": "LLM Fine-tuning and Alignment",
        "content": """
Fine-tuning large language models (LLMs) adapts pre-trained models to specific tasks or domains using 
supervised learning on task-specific datasets. Full fine-tuning updates all model parameters but 
requires significant compute resources. Parameter-efficient fine-tuning (PEFT) methods like LoRA 
(Low-Rank Adaptation) and QLoRA reduce memory requirements by only updating a small number of 
additional parameters. LoRA works by decomposing weight updates into low-rank matrices, reducing 
the number of trainable parameters by orders of magnitude. Instruction fine-tuning teaches models 
to follow natural language instructions using datasets of instruction-response pairs. Reinforcement 
Learning from Human Feedback (RLHF) aligns model outputs with human preferences by training a reward 
model on human comparisons and using it to guide policy optimization. Direct Preference Optimization 
(DPO) is a simpler alternative to RLHF that directly optimizes for human preferences without a 
separate reward model. Catastrophic forgetting is a challenge where fine-tuning causes models to 
lose previously learned knowledge. Regularization techniques and careful learning rate scheduling 
help mitigate catastrophic forgetting during fine-tuning.
        """
    },
    {
        "title": "Data Pipeline Orchestration with Apache Airflow",
        "content": """
Apache Airflow is an open-source workflow orchestration platform used to programmatically author, 
schedule, and monitor data pipelines. Airflow represents workflows as Directed Acyclic Graphs (DAGs) 
where nodes are tasks and edges define dependencies between tasks. Operators in Airflow define the 
type of work to be done — PythonOperator executes Python functions, BashOperator runs shell commands, 
and DockerOperator runs tasks inside Docker containers. The Airflow scheduler monitors DAGs and 
triggers task instances when their dependencies are satisfied. Airflow's web UI provides a visual 
interface for monitoring DAG runs, viewing task logs, and managing pipeline execution. XComs (cross 
communications) allow tasks to exchange small amounts of data within a DAG run. Sensors are special 
operators that wait for a condition to be true before proceeding, such as waiting for a file to 
appear in S3. Airflow is commonly used in MLOps pipelines for scheduling data preprocessing, model 
training, evaluation, and deployment tasks. Best practices include keeping tasks idempotent so they 
can be safely retried, using connections and variables for configuration management, and implementing 
proper error handling and alerting.
        """
    },
    {
        "title": "Agentic AI Systems and Tool Use",
        "content": """
Agentic AI systems are autonomous AI agents that can plan, reason, and execute multi-step tasks by 
coordinating the use of various tools. Unlike simple question-answering systems, agents maintain state 
across multiple steps and make decisions about which tools to invoke based on the current task context. 
The ReAct (Reasoning and Acting) framework combines chain-of-thought reasoning with action execution, 
allowing agents to interleave reasoning steps with tool calls. Common tools available to AI agents 
include web search, code execution, database queries, API calls, and document retrieval. LangChain and 
LlamaIndex are popular frameworks for building agentic systems with tool use capabilities. Tool 
selection policies determine how agents decide which tool to use — rule-based policies use predefined 
conditions while LLM-based policies use the language model itself to reason about tool selection. 
Memory systems in agents include short-term working memory stored in the context window and long-term 
memory stored in external databases. Agent evaluation focuses on task completion rate, tool selection 
accuracy, reasoning quality, and efficiency in terms of number of steps taken. Failure modes include 
tool hallucination where agents fabricate tool outputs and reasoning loops where agents get stuck 
repeating the same actions.
        """
    },
    {
        "title": "Experiment Tracking with MLflow",
        "content": """
MLflow is an open-source platform for managing the complete machine learning lifecycle including 
experiment tracking, model packaging, model registry, and model deployment. The MLflow Tracking 
component logs parameters, metrics, artifacts, and metadata for each training run. Experiments in 
MLflow group related runs together, making it easy to compare different model configurations. The 
MLflow Model Registry provides a centralized hub for managing model versions and their lifecycle 
stages: None, Staging, Production, and Archived. Model signatures define the expected input and 
output schema for a model, enabling automatic input validation during serving. MLflow Projects 
package ML code in a reusable and reproducible format using a MLproject file that specifies 
dependencies and entry points. MLflow Models support multiple flavors including scikit-learn, 
PyTorch, TensorFlow, and generic Python functions. The mlflow.autolog() function automatically 
captures parameters, metrics, and artifacts for supported frameworks without explicit logging calls. 
MLflow integrates with popular orchestration tools like Airflow and Kubernetes for production 
deployment. Model lineage tracking in MLflow connects deployed models back to the specific data 
and code used to train them, supporting auditability and compliance requirements.
        """
    }
]


# ─────────────────────────────────────────────
# TOOL 1: RETRIEVER
# ─────────────────────────────────────────────

class RetrieverTool:
    """
    Searches the FAISS vector store for relevant MLOps knowledge chunks.
    Returns top-k chunks with source metadata and similarity scores.
    """
    name = "retriever"
    description = "Search the MLOps knowledge base for relevant information on a topic."

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def run(self, query: str, k: int = 3) -> dict:
        t0 = time.time()
        results = self.vector_store.similarity_search_with_score(query, k=k)
        latency = time.time() - t0

        chunks = []
        for doc, score in results:
            chunks.append({
                "source": doc.metadata.get("title", "Unknown"),
                "content": doc.page_content.strip(),
                "score": round(float(score), 4)
            })

        return {
            "tool": "retriever",
            "query": query,
            "chunks": chunks,
            "latency_s": round(latency, 4)
        }


# ─────────────────────────────────────────────
# TOOL 2: SUMMARIZER
# ─────────────────────────────────────────────

class SummarizerTool:
    """
    Summarizes text content into concise bullet-point summaries.
    Uses the LLM to condense information without adding external knowledge.
    """
    name = "summarizer"
    description = "Summarize a piece of text into concise bullet points."

    def __init__(self, model: str = "qwen2.5:7b-instruct"):
        self.model = model

    def run(self, text: str, focus: str = "") -> dict:
        focus_instruction = f"Focus specifically on: {focus}." if focus else ""
        prompt = f"""Summarize the following text into 3-5 concise bullet points.
{focus_instruction}
Only use information present in the text. Do not add external knowledge.

Text:
{text}

Summary (bullet points):"""

        t0 = time.time()
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - t0
        summary = response["message"]["content"].strip()

        return {
            "tool": "summarizer",
            "summary": summary,
            "latency_s": round(latency, 4)
        }


# ─────────────────────────────────────────────
# AGENT CONTROLLER
# ─────────────────────────────────────────────

class AgentController:
    """
    Multi-tool agent that coordinates retrieval and summarization.

    Tool Selection Policy:
    ─────────────────────
    The agent uses keyword-based + LLM-based hybrid policy:

    1. RETRIEVE_ONLY: Query contains factual knowledge-seeking keywords
       (what, how, explain, describe, define, list) → retrieve then answer

    2. RETRIEVE_THEN_SUMMARIZE: Query asks for summary, overview, or
       comparison of a topic → retrieve then summarize retrieved content

    3. MULTI_STEP: Query requires multiple sub-questions → retrieve for
       each sub-question then synthesize

    4. SUMMARIZE_ONLY: Input is raw text to be condensed (no retrieval needed)

    The LLM makes the final tool selection decision based on the task description.
    """

    def __init__(self, retriever_tool, summarizer_tool,
                 model: str = "qwen2.5:7b-instruct"):
        self.retriever = retriever_tool
        self.summarizer = summarizer_tool
        self.model = model
        self.tools = {
            "retriever": retriever_tool,
            "summarizer": summarizer_tool
        }

    def decide_tools(self, task: str) -> str:
        """
        LLM-based tool selection: asks the model which tools to use.
        Returns one of: RETRIEVE_ONLY, RETRIEVE_THEN_SUMMARIZE,
                        MULTI_STEP, SUMMARIZE_ONLY
        """
        prompt = f"""You are an AI agent with two tools:
1. retriever: searches a knowledge base for relevant information
2. summarizer: summarizes text into bullet points

Given this task, decide which tool sequence to use.
Reply with EXACTLY one of these options (nothing else):
- RETRIEVE_ONLY          (just retrieve and answer)
- RETRIEVE_THEN_SUMMARIZE (retrieve then summarize the result)
- MULTI_STEP             (retrieve multiple times for sub-questions)
- SUMMARIZE_ONLY         (no retrieval needed, just summarize given text)

Task: {task}

Your decision:"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        decision = response["message"]["content"].strip().upper()

        # Normalize to valid options
        for valid in ["RETRIEVE_ONLY", "RETRIEVE_THEN_SUMMARIZE",
                      "MULTI_STEP", "SUMMARIZE_ONLY"]:
            if valid in decision:
                return valid
        return "RETRIEVE_ONLY"  # safe default

    def generate_answer(self, task: str, context: str) -> tuple:
        """Generate final answer from task + retrieved/processed context."""
        prompt = f"""You are a helpful MLOps assistant.
Use the provided context to answer the task completely.
Only use information from the context. Do not fabricate information.

Context:
{context}

Task: {task}

Answer:"""

        t0 = time.time()
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - t0
        return response["message"]["content"].strip(), round(latency, 4)

    def run(self, task: str, task_id: int = 0) -> dict:
        """
        Execute a task using the agent.
        Returns full trace with all intermediate steps.
        """
        trace = {
            "task_id": task_id,
            "task": task,
            "steps": [],
            "final_answer": "",
            "total_latency_s": 0,
            "success": False
        }

        t_start = time.time()
        print(f"\n{'='*60}")
        print(f"TASK {task_id}: {task}")
        print(f"{'='*60}")

        # Step 1: Tool selection
        print("  [Step 1] Deciding tool sequence...")
        decision = self.decide_tools(task)
        print(f"  [Decision] → {decision}")

        trace["steps"].append({
            "step": 1,
            "action": "tool_selection",
            "decision": decision,
            "reasoning": f"Agent decided: {decision}"
        })

        # Step 2: Execute based on decision
        context = ""

        if decision == "RETRIEVE_ONLY":
            print("  [Step 2] Retrieving relevant knowledge...")
            result = self.retriever.run(task)
            context = "\n\n".join([
                f"[Source: {c['source']}]\n{c['content']}"
                for c in result["chunks"]
            ])
            trace["steps"].append({
                "step": 2,
                "action": "retrieve",
                "query": task,
                "retrieved_sources": [c["source"] for c in result["chunks"]],
                "latency_s": result["latency_s"]
            })
            print(f"  [Retrieved] {[c['source'] for c in result['chunks']]}")

        elif decision == "RETRIEVE_THEN_SUMMARIZE":
            print("  [Step 2] Retrieving relevant knowledge...")
            result = self.retriever.run(task)
            raw_context = "\n\n".join([c["content"] for c in result["chunks"]])
            sources = [c["source"] for c in result["chunks"]]
            trace["steps"].append({
                "step": 2,
                "action": "retrieve",
                "query": task,
                "retrieved_sources": sources,
                "latency_s": result["latency_s"]
            })
            print(f"  [Retrieved] {sources}")

            print("  [Step 3] Summarizing retrieved content...")
            sum_result = self.summarizer.run(raw_context, focus=task)
            context = sum_result["summary"]
            trace["steps"].append({
                "step": 3,
                "action": "summarize",
                "focus": task,
                "latency_s": sum_result["latency_s"]
            })
            print(f"  [Summary] {context[:150]}...")

        elif decision == "MULTI_STEP":
            print("  [Step 2] Breaking task into sub-queries...")
            # Split into two sub-queries based on task keywords
            words = task.split()
            mid = len(words) // 2
            sub_q1 = " ".join(words[:mid + 3])
            sub_q2 = " ".join(words[mid:])

            all_chunks = []
            for i, sub_q in enumerate([sub_q1, sub_q2], start=2):
                print(f"  [Step {i}] Retrieving for: '{sub_q[:50]}...'")
                result = self.retriever.run(sub_q)
                all_chunks.extend(result["chunks"])
                trace["steps"].append({
                    "step": i,
                    "action": "retrieve",
                    "query": sub_q,
                    "retrieved_sources": [c["source"] for c in result["chunks"]],
                    "latency_s": result["latency_s"]
                })

            # Deduplicate chunks by source
            seen = set()
            unique_chunks = []
            for c in all_chunks:
                if c["source"] not in seen:
                    unique_chunks.append(c)
                    seen.add(c["source"])

            context = "\n\n".join([
                f"[Source: {c['source']}]\n{c['content']}"
                for c in unique_chunks[:4]
            ])
            print(f"  [Multi-step retrieved] {list(seen)}")

        elif decision == "SUMMARIZE_ONLY":
            print("  [Step 2] Summarizing task directly...")
            sum_result = self.summarizer.run(task)
            context = sum_result["summary"]
            trace["steps"].append({
                "step": 2,
                "action": "summarize",
                "latency_s": sum_result["latency_s"]
            })

        # Step 3: Generate final answer
        print("  [Step Final] Generating answer...")
        answer, gen_latency = self.generate_answer(task, context)
        trace["final_answer"] = answer
        trace["total_latency_s"] = round(time.time() - t_start, 2)
        trace["success"] = len(answer) > 20
        trace["steps"].append({
            "step": len(trace["steps"]) + 1,
            "action": "generate_answer",
            "latency_s": gen_latency
        })

        print(f"  [Answer] {answer[:200]}...")
        print(f"  [Total Latency] {trace['total_latency_s']}s")

        return trace


# ─────────────────────────────────────────────
# 10 EVALUATION TASKS
# ─────────────────────────────────────────────

EVAL_TASKS = [
    # RETRIEVE_ONLY tasks (factual knowledge)
    "What is the MLOps maturity model and what are its levels?",
    "What tools are commonly used for model monitoring in production?",
    "How does RLHF align language model outputs with human preferences?",
    "What is the role of XComs in Apache Airflow DAGs?",
    "What are the key components of the MLflow Model Registry?",

    # RETRIEVE_THEN_SUMMARIZE tasks
    "Give me a brief summary of how RAG pipelines work end-to-end.",
    "Summarize the key practices for implementing CI/CD in machine learning systems.",

    # MULTI_STEP tasks
    "Compare how feature stores and vector databases both address data serving challenges in ML systems.",
    "How do agentic AI systems use tools differently compared to standard RAG pipelines?",

    # Edge case task
    "What happens when model drift is detected and no retraining pipeline exists?"
]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_vector_store():
    """Build FAISS vector store from document corpus."""
    print("Building vector store...")
    docs = [
        Document(
            page_content=d["content"].strip(),
            metadata={"title": d["title"]}
        )
        for d in DOCUMENTS
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"✅ Vector store ready — {vector_store.index.ntotal} vectors")
    return vector_store


def main():
    print("=" * 60)
    print("MILESTONE 6 — AGENT CONTROLLER")
    print("Model: qwen2.5:7b-instruct | Tools: retriever + summarizer")
    print("=" * 60)

    # Build shared vector store
    vector_store = build_vector_store()

    # Initialize tools
    retriever_tool = RetrieverTool(vector_store)
    summarizer_tool = SummarizerTool(model="qwen2.5:7b-instruct")

    # Initialize agent
    agent = AgentController(
        retriever_tool=retriever_tool,
        summarizer_tool=summarizer_tool,
        model="qwen2.5:7b-instruct"
    )

    # Run all 10 evaluation tasks
    all_traces = []
    os.makedirs("agent_traces", exist_ok=True)

    for i, task in enumerate(EVAL_TASKS, start=1):
        trace = agent.run(task, task_id=i)
        all_traces.append(trace)

        # Save individual trace
        trace_path = f"agent_traces/task_{i:02d}.json"
        with open(trace_path, "w") as f:
            json.dump(trace, f, indent=2)
        print(f"  ✅ Trace saved: {trace_path}")

    # Save all traces combined
    with open("agent_traces/all_traces.json", "w") as f:
        json.dump(all_traces, f, indent=2)

    # Summary
    successes = sum(1 for t in all_traces if t["success"])
    avg_latency = sum(t["total_latency_s"] for t in all_traces) / len(all_traces)

    print("\n" + "=" * 60)
    print("AGENT EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Tasks completed:  {successes}/{len(all_traces)}")
    print(f"  Avg latency:      {avg_latency:.1f}s")
    print(f"  Traces saved:     agent_traces/")
    print("\n🎉 Agent Controller Complete!")


if __name__ == "__main__":
    main()