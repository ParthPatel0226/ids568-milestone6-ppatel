"""
RAG Pipeline - Milestone 6
MLOps Course - Module 7
Model: qwen2.5:7b-instruct via Ollama
Vector Store: FAISS
Embeddings: sentence-transformers/all-MiniLM-L6-v2
"""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Ollama
import ollama

# ─────────────────────────────────────────────
# 1. DOCUMENT CORPUS
# 10 MLOps / ML articles as raw text
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
# 2. CHUNKING & INDEXING
# ─────────────────────────────────────────────

def create_documents(raw_docs):
    """Convert raw text documents into LangChain Document objects."""
    docs = []
    for d in raw_docs:
        doc = Document(
            page_content=d["content"].strip(),
            metadata={"title": d["title"], "source": d["title"]}
        )
        docs.append(doc)
    print(f"✅ Loaded {len(docs)} documents")
    return docs


def chunk_documents(docs, chunk_size=512, chunk_overlap=64):
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    
    Design decisions:
    - chunk_size=512: Balances context preservation vs retrieval precision.
      Large enough to contain complete thoughts, small enough for precise retrieval.
    - chunk_overlap=64: ~12.5% overlap prevents information loss at chunk boundaries.
    - RecursiveCharacterTextSplitter: Tries to split on paragraphs, sentences, words
      in order — preserves semantic coherence better than fixed-size splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"   Chunk size: {chunk_size} chars | Overlap: {chunk_overlap} chars")
    return chunks


def build_vector_store(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings and build FAISS vector index.
    
    Design decisions:
    - all-MiniLM-L6-v2: Fast, lightweight 384-dim embeddings. Good balance of
      speed and quality for semantic search. Works well on CPU.
    - FAISS: Open-source, runs locally, no external services needed.
      IndexFlatL2 used internally for exact search (corpus is small enough).
    """
    print(f"\n📦 Building vector store with {model_name}...")
    t0 = time.time()
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - t0
    
    print(f"✅ Vector store built in {elapsed:.2f}s")
    print(f"   Embedding model: {model_name}")
    print(f"   Vectors indexed: {vector_store.index.ntotal}")
    
    return vector_store, embeddings, elapsed


# ─────────────────────────────────────────────
# 3. RETRIEVAL
# ─────────────────────────────────────────────

def retrieve(vector_store, query, k=3):
    """Retrieve top-k most relevant chunks for a query."""
    t0 = time.time()
    results = vector_store.similarity_search_with_score(query, k=k)
    latency = time.time() - t0
    return results, latency


# ─────────────────────────────────────────────
# 4. GROUNDED GENERATION
# ─────────────────────────────────────────────

def generate_answer(query, retrieved_docs, model="qwen2.5:7b-instruct"):
    """
    Generate a grounded answer using retrieved context and Ollama LLM.
    The prompt explicitly instructs the model to use only the provided context.
    """
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
        for doc, _ in retrieved_docs
    ])
    
    prompt = f"""You are a helpful MLOps assistant. Answer the question using ONLY the provided context below.
If the answer is not in the context, say "I don't have enough information to answer this."
Do not use any knowledge outside of the provided context.

Context:
{context}

Question: {query}

Answer:"""

    t0 = time.time()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - t0
    answer = response["message"]["content"].strip()
    
    return answer, latency


# ─────────────────────────────────────────────
# 5. EVALUATION — 10 HANDCRAFTED QUERIES
# ─────────────────────────────────────────────

EVAL_QUERIES = [
    {
        "id": 1,
        "query": "What is MLOps and what are its key principles?",
        "expected_sources": ["Introduction to MLOps"],
        "expected_keywords": ["automation", "reproducibility", "versioning", "monitoring"]
    },
    {
        "id": 2,
        "query": "How does RAG reduce hallucinations in language models?",
        "expected_sources": ["Retrieval-Augmented Generation (RAG)"],
        "expected_keywords": ["documents", "context", "retrieval", "grounded"]
    },
    {
        "id": 3,
        "query": "What is FAISS and how does it work?",
        "expected_sources": ["Vector Databases and Embeddings"],
        "expected_keywords": ["similarity", "vectors", "Meta", "search"]
    },
    {
        "id": 4,
        "query": "What types of drift affect machine learning models in production?",
        "expected_sources": ["Model Monitoring in Production"],
        "expected_keywords": ["data drift", "concept drift", "prediction drift"]
    },
    {
        "id": 5,
        "query": "How does LoRA reduce memory requirements during LLM fine-tuning?",
        "expected_sources": ["LLM Fine-tuning and Alignment"],
        "expected_keywords": ["low-rank", "parameters", "matrices"]
    },
    {
        "id": 6,
        "query": "What is the difference between offline and online feature stores?",
        "expected_sources": ["Feature Engineering and Feature Stores"],
        "expected_keywords": ["training", "inference", "real-time", "historical"]
    },
    {
        "id": 7,
        "query": "What are DAGs in Apache Airflow and how are tasks defined?",
        "expected_sources": ["Data Pipeline Orchestration with Apache Airflow"],
        "expected_keywords": ["Directed Acyclic Graphs", "operators", "tasks", "dependencies"]
    },
    {
        "id": 8,
        "query": "How does an AI agent decide which tool to use?",
        "expected_sources": ["Agentic AI Systems and Tool Use"],
        "expected_keywords": ["tool selection", "policy", "reasoning", "ReAct"]
    },
    {
        "id": 9,
        "query": "What are the lifecycle stages of a model in MLflow Model Registry?",
        "expected_sources": ["Experiment Tracking with MLflow"],
        "expected_keywords": ["Staging", "Production", "Archived", "versions"]
    },
    {
        "id": 10,
        "query": "What is canary deployment and why is it used in ML systems?",
        "expected_sources": ["CI/CD for Machine Learning"],
        "expected_keywords": ["traffic", "deployment", "risk", "versions"]
    }
]


def evaluate_retrieval(vector_store, queries, k=3):
    """
    Evaluate retrieval accuracy on handcrafted queries.
    Metrics: Precision@k, keyword hit rate, source match rate.
    """
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION — 10 QUERIES")
    print("="*60)
    
    results = []
    
    for q in tqdm(queries, desc="Evaluating"):
        retrieved, retrieval_latency = retrieve(vector_store, q["query"], k=k)
        
        # Get retrieved source titles
        retrieved_sources = [doc.metadata.get("title", "") for doc, _ in retrieved]
        retrieved_text = " ".join([doc.page_content for doc, _ in retrieved]).lower()
        
        # Precision@k: fraction of retrieved docs from expected sources
        expected = set(q["expected_sources"])
        hits = sum(1 for s in retrieved_sources if s in expected)
        precision_at_k = hits / k
        
        # Keyword hit rate: fraction of expected keywords found in retrieved text
        kw_hits = sum(1 for kw in q["expected_keywords"] if kw.lower() in retrieved_text)
        keyword_hit_rate = kw_hits / len(q["expected_keywords"])
        
        # Source match: did we get at least one correct source?
        source_match = any(s in expected for s in retrieved_sources)
        
        results.append({
            "id": q["id"],
            "query": q["query"],
            "retrieved_sources": retrieved_sources,
            "expected_sources": q["expected_sources"],
            "precision_at_k": precision_at_k,
            "keyword_hit_rate": keyword_hit_rate,
            "source_match": source_match,
            "retrieval_latency_s": round(retrieval_latency, 4)
        })
        
        status = "✅" if source_match else "❌"
        print(f"\nQ{q['id']}: {q['query'][:60]}...")
        print(f"  {status} Retrieved: {retrieved_sources}")
        print(f"  Precision@{k}: {precision_at_k:.2f} | Keywords: {keyword_hit_rate:.2f} | Latency: {retrieval_latency:.3f}s")
    
    # Aggregate metrics
    avg_precision = np.mean([r["precision_at_k"] for r in results])
    avg_keyword = np.mean([r["keyword_hit_rate"] for r in results])
    source_match_rate = np.mean([r["source_match"] for r in results])
    avg_latency = np.mean([r["retrieval_latency_s"] for r in results])
    
    print("\n" + "="*60)
    print("AGGREGATE RETRIEVAL METRICS")
    print("="*60)
    print(f"  Avg Precision@{k}:     {avg_precision:.3f}")
    print(f"  Avg Keyword Hit Rate:  {avg_keyword:.3f}")
    print(f"  Source Match Rate:     {source_match_rate:.3f}")
    print(f"  Avg Retrieval Latency: {avg_latency:.4f}s")
    
    return results, {
        "avg_precision_at_k": avg_precision,
        "avg_keyword_hit_rate": avg_keyword,
        "source_match_rate": source_match_rate,
        "avg_retrieval_latency_s": avg_latency
    }


def run_grounded_generation_eval(vector_store, queries, k=3):
    """Run full RAG pipeline on all queries and measure generation latency."""
    print("\n" + "="*60)
    print("GROUNDED GENERATION EVALUATION")
    print("="*60)
    
    gen_results = []
    
    for q in queries:
        print(f"\nQ{q['id']}: {q['query']}")
        retrieved, retrieval_latency = retrieve(vector_store, q["query"], k=k)
        answer, gen_latency = generate_answer(q["query"], retrieved)
        e2e_latency = retrieval_latency + gen_latency
        
        print(f"  Answer: {answer[:200]}...")
        print(f"  Retrieval: {retrieval_latency:.3f}s | Generation: {gen_latency:.2f}s | E2E: {e2e_latency:.2f}s")
        
        gen_results.append({
            "id": q["id"],
            "query": q["query"],
            "answer": answer,
            "retrieval_latency_s": round(retrieval_latency, 4),
            "generation_latency_s": round(gen_latency, 4),
            "e2e_latency_s": round(e2e_latency, 4)
        })
    
    avg_gen = np.mean([r["generation_latency_s"] for r in gen_results])
    avg_e2e = np.mean([r["e2e_latency_s"] for r in gen_results])
    
    print("\n" + "="*60)
    print("LATENCY SUMMARY")
    print("="*60)
    print(f"  Avg Generation Latency: {avg_gen:.2f}s")
    print(f"  Avg E2E Latency:        {avg_e2e:.2f}s")
    
    return gen_results, {"avg_generation_latency_s": avg_gen, "avg_e2e_latency_s": avg_e2e}


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MILESTONE 6 — RAG PIPELINE")
    print("Model: qwen2.5:7b-instruct | Store: FAISS")
    print("=" * 60)

    # Step 1: Load and chunk documents
    docs = create_documents(DOCUMENTS)
    chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)

    # Step 2: Build vector store
    vector_store, embeddings, index_time = build_vector_store(chunks)

    # Step 3: Quick retrieval demo
    print("\n📋 Quick Retrieval Demo:")
    demo_results, _ = retrieve(vector_store, "What is RAG?", k=3)
    for doc, score in demo_results:
        print(f"  Score: {score:.4f} | Source: {doc.metadata['title']}")
        print(f"  Preview: {doc.page_content[:100]}...")

    # Step 4: Retrieval evaluation
    retrieval_results, retrieval_metrics = evaluate_retrieval(
        vector_store, EVAL_QUERIES, k=3
    )

    # Step 5: Generation evaluation (runs all 10 queries through LLM)
    print("\n⚠️  Running generation eval — this will take ~5-10 min on CPU...")
    gen_results, gen_metrics = run_grounded_generation_eval(
        vector_store, EVAL_QUERIES, k=3
    )

    # Step 6: Save results to JSON for report writing
    all_results = {
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": gen_metrics,
        "retrieval_results": retrieval_results,
        "generation_results": gen_results,
        "index_build_time_s": round(index_time, 4),
        "model": "qwen2.5:7b-instruct",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "k": 3
    }
    
    with open("rag_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✅ Results saved to rag_results.json")
    print("\n🎉 RAG Pipeline Complete!")
    print(f"   Avg Precision@3:    {retrieval_metrics['avg_precision_at_k']:.3f}")
    print(f"   Source Match Rate:  {retrieval_metrics['source_match_rate']:.3f}")
    print(f"   Avg E2E Latency:    {gen_metrics['avg_e2e_latency_s']:.2f}s")


if __name__ == "__main__":
    main()