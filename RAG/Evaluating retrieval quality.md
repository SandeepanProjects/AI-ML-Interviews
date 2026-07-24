Evaluating retrieval quality is one of the **most overlooked parts of building a RAG system**. Many developers focus on the LLM, but if the retriever returns the wrong chunks, even the best model cannot generate a correct answer.

In production, companies continuously evaluate retrieval quality before evaluating the LLM.

---

# The RAG Evaluation Pipeline

A production RAG pipeline is typically evaluated in stages.

```text
                   Documents
                       │
                       ▼
                  Chunking
                       │
                       ▼
                 Vector Database
                       │
                       ▼
                  Retriever
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 Retrieval Evaluation           Reranker Evaluation
        │                             │
        └──────────────┬──────────────┘
                       ▼
                LLM Evaluation
                       │
                       ▼
              End-to-End Evaluation
```

If retrieval fails, the LLM never sees the correct information.

---

# Creating a Retrieval Evaluation Dataset

Unlike model training data, retrieval evaluation needs **queries and their relevant documents**.

Example:

```python
evaluation_data = [
    {
        "query": "How do I authenticate with JWT?",
        "relevant_docs": [5]
    },
    {
        "query": "How do I deploy FastAPI?",
        "relevant_docs": [2]
    },
    {
        "query": "Explain Docker Compose",
        "relevant_docs": [7]
    }
]
```

Suppose our knowledge base is

```text
Doc 1 → Python Basics

Doc 2 → Deploy FastAPI

Doc 3 → Neural Networks

Doc 4 → Kubernetes

Doc 5 → JWT Authentication

Doc 6 → Redis

Doc 7 → Docker Compose
```

The evaluator knows the correct answers.

---

# Example Retrieval Function

Imagine the retriever returns

```python
retrieved = {
    "How do I authenticate with JWT?": [5, 1, 6],
    "How do I deploy FastAPI?": [2, 6, 7],
    "Explain Docker Compose": [4, 7, 3]
}
```

Now we measure quality.

---

# 1. Precision@K

Precision answers

> Of the retrieved chunks, how many are actually relevant?

Formula

```text
Precision@K =
Relevant Retrieved
-------------------
Retrieved K
```

Example

```text
Top 5 retrieved

JWT ✓

Redis ✗

Python ✗

OAuth ✓

Docker ✗
```

Relevant

```text
2
```

Retrieved

```text
5
```

Precision

```text
2 / 5 = 0.40
```

---

## Code

```python
def precision_at_k(retrieved, relevant, k):

    retrieved = retrieved[:k]

    hits = len(
        set(retrieved) &
        set(relevant)
    )

    return hits / k
```

Example

```python
retrieved = [5,1,6,3,7]
relevant = [5,3]

print(
    precision_at_k(
        retrieved,
        relevant,
        5
    )
)
```

Output

```text
0.4
```

---

# 2. Recall@K

Recall answers

> Did we retrieve all relevant chunks?

Formula

```text
Recall@K

Relevant Retrieved
------------------

Total Relevant
```

Suppose

Relevant docs

```text
JWT

OAuth

Security Guide
```

Retriever returns

```text
JWT

Redis

Python
```

Recall

```text
1 / 3 = 0.33
```

---

## Code

```python
def recall_at_k(
    retrieved,
    relevant,
    k
):

    retrieved = retrieved[:k]

    hits = len(
        set(retrieved) &
        set(relevant)
    )

    return hits / len(relevant)
```

---

# Precision vs Recall

Suppose

```text
Relevant docs

A

B

C
```

Retriever 1

```text
A
```

Precision

```text
1/1 = 1.0
```

Recall

```text
1/3 = 0.33
```

Perfect precision

Poor recall.

---

Retriever 2

```text
A

B

C

D

E
```

Precision

```text
3/5 = 0.60
```

Recall

```text
3/3 = 1.0
```

Production systems usually prefer **higher recall** in the first retrieval stage because rerankers can remove irrelevant results later.

---

# 3. Hit Rate@K

The simplest metric.

Question

> Did we retrieve at least one correct chunk?

Formula

```text
Hit Rate

1 if any relevant chunk appears

0 otherwise
```

Code

```python
def hit_rate(retrieved, relevant, k):

    retrieved = retrieved[:k]

    return int(
        len(set(retrieved) &
            set(relevant)) > 0
    )
```

Example

```text
Relevant

JWT

OAuth

Retrieved

Redis

JWT

Docker
```

Output

```text
1
```

---

# 4. Mean Reciprocal Rank (MRR)

MRR rewards retrieving the correct document early.

Suppose

```text
Rank 1

Redis

Rank 2

JWT

Rank 3

Docker
```

Correct answer

```text
Rank 2
```

Reciprocal Rank

```text
1 / 2 = 0.5
```

Another query

Correct answer

```text
Rank 1

↓

1 / 1 = 1
```

Average these scores across all queries.

---

## Code

```python
def reciprocal_rank(
    retrieved,
    relevant
):

    for rank, doc in enumerate(retrieved,1):

        if doc in relevant:

            return 1 / rank

    return 0
```

---

# 5. NDCG (Normalized Discounted Cumulative Gain)

Not every document has equal value.

Example

```text
Document A

Perfect answer

Score 3

-----------------

Document B

Useful

Score 2

-----------------

Document C

Slightly relevant

Score 1
```

NDCG rewards

* relevant documents
* appearing earlier
* higher graded relevance

Search engines like Google commonly use NDCG.

---

# 6. Retrieval Latency

Quality isn't enough.

Measure

```text
Average latency

P95 latency

P99 latency
```

Python

```python
import time

start = time.time()

results = retriever.search(query)

latency = time.time() - start

print(latency)
```

---

# 7. Recall vs Top-K

Suppose

```text
Top1

Recall

40%
```

Top5

```text
70%
```

Top10

```text
88%
```

Top20

```text
94%
```

Choosing `top_k` is a trade-off:

* Larger `top_k` improves recall.
* Larger `top_k` increases reranking cost and LLM context size.

---

# End-to-End Evaluation Example

```python
evaluation = [
    {
        "query": "JWT Authentication",
        "relevant": [5]
    },
    {
        "query": "Docker Compose",
        "relevant": [7]
    },
    {
        "query": "FastAPI Deployment",
        "relevant": [2]
    }
]

retrieved_results = {
    "JWT Authentication": [5,1,6],
    "Docker Compose": [7,4,2],
    "FastAPI Deployment": [6,2,3]
}
```

Evaluation

```python
precisions = []

for sample in evaluation:

    query = sample["query"]

    relevant = sample["relevant"]

    retrieved = retrieved_results[query]

    precisions.append(
        precision_at_k(
            retrieved,
            relevant,
            3
        )
    )

print(
    sum(precisions) /
    len(precisions)
)
```

---

# Real Production Evaluation Pipeline

```text
Ground Truth Dataset
        │
        ▼
+----------------------------+
| Query                      |
| Relevant Document IDs      |
+----------------------------+
        │
        ▼
Retriever
        │
        ▼
Top-K Candidate Chunks
        │
        ▼
Metrics
 ├── Precision@K
 ├── Recall@K
 ├── Hit Rate@K
 ├── MRR
 ├── NDCG
 └── Latency
        │
        ▼
Dashboard (Grafana/MLflow)
```

---

# Advanced Enterprise Metrics

Modern RAG systems often track additional retrieval-specific metrics:

| Metric              | What it Measures                                                  | Why it Matters                                  |
| ------------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| Precision@K         | Fraction of retrieved chunks that are relevant                    | Reduces irrelevant context sent to the LLM      |
| Recall@K            | Fraction of relevant chunks retrieved                             | Ensures the answer is available to the LLM      |
| Hit Rate@K          | Whether at least one relevant chunk was found                     | Good high-level success metric                  |
| MRR                 | Rank of the first relevant chunk                                  | Rewards putting the best chunk first            |
| NDCG                | Ranking quality with graded relevance                             | Useful when documents have different importance |
| Retrieval Latency   | Time taken to retrieve candidates                                 | Critical for user experience                    |
| Context Utilization | Percentage of retrieved context actually cited or used by the LLM | Detects wasted prompt tokens                    |
| Citation Accuracy   | Whether generated answers cite the correct retrieved chunks       | Measures grounding and faithfulness             |

---

# How Senior AI Engineers Evaluate Retrieval

A production evaluation loop typically looks like this:

```text
                 Ground Truth Queries
                         │
                         ▼
               Hybrid Retriever (BM25 + Dense)
                         │
                         ▼
                 Top 100 Candidate Chunks
                         │
                         ▼
                  Cross-Encoder Reranker
                         │
                         ▼
                  Top 10 Final Chunks
                         │
                         ▼
         Precision@K, Recall@K, MRR, NDCG
                         │
                         ▼
      Compare Against Previous Model Version
                         │
                         ▼
        Deploy Only If Retrieval Improves
```

This evaluation is usually integrated into CI/CD so that changes to chunking, embedding models, retrieval parameters, or rerankers are automatically tested against a fixed benchmark. Teams often set minimum thresholds (for example, Recall@10 ≥ 0.90 and MRR ≥ 0.80) before allowing a new retrieval pipeline to be deployed.
