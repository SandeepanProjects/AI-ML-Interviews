# How to Evaluate Retrieval Quality (Complete Production Guide with Code)

Evaluating retrieval quality is one of the **most important aspects of a production RAG system**.

Many teams only evaluate the final LLM answer.

That is a mistake.

A RAG system has **two separate stages**:

```text
                User Question
                      │
                      ▼
              Retrieval System
        (Vector DB / Hybrid Search)
                      │
            Retrieved Documents
                      │
                      ▼
                    LLM
                      │
                 Final Answer
```

If retrieval is poor, even the best LLM cannot produce a correct answer.

> **Garbage Retrieval → Garbage Context → Garbage Answer**

---

# Two Things Must Be Evaluated

```text
RAG System

├── Retrieval Quality
│     ├── Are the right documents retrieved?
│     ├── Are important documents missing?
│     └── Is the ranking correct?
│
└── Generation Quality
      ├── Faithfulness
      ├── Correctness
      ├── Hallucination
      └── Answer Quality
```

This chapter focuses on **retrieval evaluation**.

---

# Why Evaluate Retrieval?

Suppose your company has:

```
5 Million Documents
```

User asks:

> "How many maternity leave days are available?"

Retriever returns:

```
1. VPN Policy
2. Office Timing
3. Leave Policy
```

Although the correct document is present, it is ranked **third**.

If your prompt only includes the top 2 documents, the LLM never sees the correct answer.

Retrieval metrics help detect problems like this.

---

# Gold Dataset

To evaluate retrieval, you need a benchmark.

Example:

| Question          | Relevant Document  |
| ----------------- | ------------------ |
| Maternity leave?  | Leave Policy       |
| VPN reset?        | VPN Guide          |
| Health insurance? | Insurance Handbook |

This is called the **ground truth** or **gold dataset**.

---

# Retrieval Pipeline

```text
Question
     │
     ▼
Retriever
     │
     ▼
Top K Documents
     │
     ▼
Compare with Gold Documents
     │
     ▼
Metrics
```

---

# Important Retrieval Metrics

Production RAG systems commonly measure:

1. Recall@K
2. Precision@K
3. Mean Reciprocal Rank (MRR)
4. Hit Rate
5. nDCG
6. Context Precision
7. Context Recall

Let's examine each.

---

# 1. Recall@K

Recall answers:

> **Did we retrieve all relevant documents?**

Example

Relevant document:

```
Leave Policy
```

Retriever returns:

```
Leave Policy
VPN Guide
Travel Policy
```

Recall@3

```
1 / 1 = 100%
```

Now suppose there are two relevant documents:

```
Leave Policy
HR FAQ
```

Retriever returns only:

```
Leave Policy
VPN Guide
Travel Policy
```

Recall

```
1 / 2 = 50%
```

---

## Python

```python
def recall_at_k(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)

    return len(retrieved & relevant) / len(relevant)


retrieved = [
    "Leave Policy",
    "VPN Guide",
    "Travel Policy"
]

relevant = [
    "Leave Policy",
    "HR FAQ"
]

print(recall_at_k(retrieved, relevant))
```

Output

```
0.5
```

---

# 2. Precision@K

Precision asks:

> **How many retrieved documents are actually useful?**

Retrieved

```
Leave Policy
VPN
Travel
Office Rules
```

Relevant

```
Leave Policy
```

Precision

```
1 / 4 = 25%
```

---

## Python

```python
def precision_at_k(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)

    return len(retrieved & relevant) / len(retrieved)


print(
    precision_at_k(
        retrieved,
        relevant
    )
)
```

Output

```
0.25
```

---

# Precision vs Recall

High Recall

```text
Retrieve 100 Documents

↓

Correct Document Included
```

Good.

High Precision

```text
Retrieve 5 Documents

↓

All 5 Relevant
```

Even better.

Production systems aim for **high recall first**, then improve precision with reranking.

---

# 3. Hit Rate

Question:

> Was at least one correct document retrieved?

Example

Relevant

```
Leave Policy
```

Retrieved

```
VPN
Leave Policy
Travel
```

Hit

```
Yes
```

Score

```
1
```

---

Python

```python
def hit_rate(retrieved, relevant):
    return int(
        any(
            doc in relevant
            for doc in retrieved
        )
    )
```

---

# 4. Mean Reciprocal Rank (MRR)

MRR measures:

> **How early is the first correct document?**

Example

```
Rank 1

VPN

Rank 2

Travel

Rank 3

Leave Policy
```

Reciprocal Rank

```
1 / 3

=

0.333
```

Higher is better.

---

Python

```python
def reciprocal_rank(retrieved, relevant):

    for i, doc in enumerate(retrieved, start=1):

        if doc in relevant:

            return 1 / i

    return 0
```

---

# Why MRR Matters

Suppose

Top 5

```
1 VPN

2 Office

3 Travel

4 Leave Policy

5 Insurance
```

If your prompt includes only the top 3 documents, the answer will likely be wrong.

MRR captures ranking quality.

---

# 5. nDCG

nDCG (Normalized Discounted Cumulative Gain) rewards:

* Relevant documents
* Higher ranking positions

A relevant document at rank 1 gets more credit than the same document at rank 10.

This is commonly used in search systems.

---

# Context Precision

Context Precision answers:

> **How much of the retrieved context is actually useful?**

Suppose:

Retriever returns

```
10 Documents
```

Only

```
3
```

contain information needed for the answer.

Context Precision

```
3 / 10 = 30%
```

Low precision wastes LLM context window and increases cost.

---

# Context Recall

Context Recall answers:

> **Did retrieval include all information needed to answer correctly?**

Suppose the answer requires:

```
Leave Policy
HR FAQ
```

Retriever only returns:

```
Leave Policy
```

Context Recall is low because important evidence is missing.

---

# Example Using LangChain Retriever

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

docs = retriever.invoke(
    "How many maternity leave days?"
)

for doc in docs:
    print(doc.page_content)
```

Now compare these results with your gold dataset.

---

# Using Ragas

One of the most popular evaluation libraries is **Ragas**.

Install

```bash
pip install ragas
```

Example

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall
)

results = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall
    ]
)

print(results)
```

Example output

```
Context Precision

0.84

Context Recall

0.91
```

---

# Evaluating Many Queries

```python
questions = [

    "Leave policy",

    "VPN reset",

    "Insurance",

    "Travel reimbursement"
]

for q in questions:

    docs = retriever.invoke(q)

    score = recall_at_k(
        docs,
        gold[q]
    )

    print(q, score)
```

This creates a benchmark for your retriever.

---

# Enterprise Retrieval Evaluation Pipeline

```text
                 Gold Dataset
                      │
                      ▼
                User Question
                      │
                      ▼
             Hybrid Retriever
                      │
                      ▼
               Top 50 Results
                      │
                      ▼
                 Reranker
                      │
                      ▼
                 Top 5 Results
                      │
                      ▼
          Compute Metrics
      ├── Recall@K
      ├── Precision@K
      ├── MRR
      ├── Hit Rate
      ├── Context Recall
      └── Context Precision
```

---

# How to Improve Retrieval

If Recall is low:

* Better chunking
* Better embeddings
* Increase `k`
* Hybrid search
* Query rewriting

If Precision is low:

* Add reranking
* Better metadata filtering
* Better chunk boundaries
* Improve query rewriting

If MRR is low:

* Cross-encoder reranker
* Reciprocal Rank Fusion (RRF)
* Better embedding model

---

# Production Monitoring

Track retrieval metrics continuously.

```text
Prometheus

↓

Grafana Dashboard

↓

Recall@10

Precision@10

MRR

Latency

Token Cost

Failure Rate
```

A drop in recall often indicates changes in documents, chunking, embeddings, or indexing.

---

# Common Interview Questions

### Why evaluate retrieval separately?

Because the LLM can only answer from the retrieved context. Poor retrieval leads to poor answers regardless of model quality.

---

### Why isn't accuracy enough?

Final-answer accuracy mixes retrieval quality and generation quality. Separate retrieval metrics help identify whether the problem is in retrieval or generation.

---

### Which metric is most important?

There is no single best metric.

* **Recall@K** ensures relevant documents are found.
* **Precision@K** measures retrieval efficiency.
* **MRR** measures ranking quality.
* **Context Precision/Recall** evaluate whether the retrieved context is sufficient and relevant for generation.

They complement one another.

---

# Summary

| Metric            | What It Measures                                | Goal     |
| ----------------- | ----------------------------------------------- | -------- |
| Recall@K          | Were relevant documents retrieved?              | Maximize |
| Precision@K       | How many retrieved documents are relevant?      | Maximize |
| Hit Rate          | Was at least one correct document found?        | Maximize |
| MRR               | How early is the first relevant document?       | Maximize |
| nDCG              | Ranking quality with graded relevance           | Maximize |
| Context Precision | Fraction of retrieved context that is useful    | Maximize |
| Context Recall    | Fraction of required context that was retrieved | Maximize |

---

# Senior AI Engineer Interview Answer

> **In production RAG systems, I evaluate retrieval independently from generation because the LLM is only as good as the context it receives. I build a gold evaluation dataset containing questions and their relevant documents, then measure metrics such as Recall@K, Precision@K, Hit Rate, MRR, and nDCG to assess retrieval performance. I also use context-based metrics like Context Precision and Context Recall—often with libraries such as Ragas—to determine whether the retrieved context is both relevant and sufficient for answering the question. If recall is low, I improve chunking, embeddings, hybrid search, or query rewriting. If precision is low, I introduce reranking, metadata filtering, or better chunking. Continuous monitoring of these metrics in production helps maintain retrieval quality as documents and user behavior evolve.
