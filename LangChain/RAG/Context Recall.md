# Context Recall Explained End-to-End (with Production Code)

**Context Recall** is one of the **most important retrieval evaluation metrics** in a RAG system.

It tells us:

> **Did the retriever retrieve all the information required to answer the user's question?**

Context Recall is commonly used in production RAG evaluation frameworks like **Ragas**, together with Context Precision.

---

# Where Context Recall Fits

A RAG pipeline has two independent stages:

```text
                User Question
                      │
                      ▼
              Retriever
        (Vector DB / Hybrid Search)
                      │
              Retrieved Context
                      │
                      ▼
                    LLM
                      │
                 Final Answer
```

Context Recall evaluates the **retriever**, not the LLM.

---

# Why Context Recall Matters

Suppose the user asks:

> **How many maternity leave days does the company provide?**

The answer is spread across two documents:

```text
Document A
------------
Employees are eligible for maternity leave.

Document B
------------
Employees receive 180 days of maternity leave.
```

Suppose the retriever only returns:

```text
Document A
```

The LLM sees:

```text
Employees are eligible for maternity leave.
```

Can it answer:

> "180 days"

❌ No.

Although the retriever found a relevant document, it **missed essential information**.

This is exactly what Context Recall measures.

---

# Definition

Context Recall is:

```text
Useful Information Retrieved
────────────────────────────────────
Total Information Needed
```

Notice:

It is **not** about the number of retrieved documents.

It is about whether **all required evidence** is available.

---

# Example

Ground truth requires:

```text
1. Leave Policy
2. HR FAQ
```

Retriever returns:

```text
Leave Policy
VPN Guide
Travel Policy
```

Context Recall

```text
Retrieved Required Context = 1

Total Required Context = 2

Recall = 1 / 2 = 50%
```

Half of the required evidence is missing.

---

# Visual Example

Question

```text
What is the maternity leave policy?
```

Ground Truth

```text
Leave Policy

HR FAQ
```

Retriever

```text
Leave Policy

VPN Guide

Office Rules
```

Result

```text
Leave Policy ✓

HR FAQ ✗
```

Context Recall

```text
1 / 2 = 50%
```

---

# Why Low Context Recall Is Dangerous

Suppose the answer requires:

```text
Policy A

Policy B

Policy C
```

Retriever returns

```text
Policy A
```

The LLM never sees:

```text
Policy B

Policy C
```

Even the best GPT model cannot generate missing facts reliably.

This is why:

> **Missing context leads to hallucinations or incomplete answers.**

---

# Python Implementation

Suppose:

```python
retrieved_docs = [
    "Leave Policy",
    "VPN Guide",
    "Travel Policy"
]

required_docs = [
    "Leave Policy",
    "HR FAQ"
]
```

Compute Context Recall.

```python
def context_recall(retrieved, required):

    retrieved = set(retrieved)
    required = set(required)

    found = len(
        retrieved.intersection(required)
    )

    return found / len(required)
```

Test it.

```python
score = context_recall(
    retrieved_docs,
    required_docs
)

print(score)
```

Output

```text
0.5
```

50% Context Recall.

---

# Another Example

Retriever

```text
Leave Policy

HR FAQ

Insurance
```

Required

```text
Leave Policy

HR FAQ
```

Python

```python
retrieved = [
    "Leave Policy",
    "HR FAQ",
    "Insurance"
]

required = [
    "Leave Policy",
    "HR FAQ"
]

print(
    context_recall(
        retrieved,
        required
    )
)
```

Output

```text
1.0
```

Perfect recall.

---

# Context Recall vs Recall@K

This is a common interview question.

## Recall@K

Question:

> Did we retrieve the relevant documents?

Suppose

Relevant documents

```text
Leave Policy

HR FAQ
```

Retrieved

```text
Leave Policy

VPN Guide

Travel
```

Recall@3

```text
1 / 2 = 50%
```

---

## Context Recall

Question:

> Did we retrieve all the information required to answer the question?

Suppose

The answer requires:

```text
Leave Policy

HR FAQ

Legal Appendix
```

Retriever returns

```text
Leave Policy

HR FAQ
```

Recall@K

```text
2 / 2 = 100%
```

if only two documents were labeled relevant.

But Context Recall

```text
2 / 3 = 66%
```

because one necessary piece of evidence is still missing.

In practice, Context Recall focuses on **answer completeness**, not merely document relevance.

---

# Context Precision vs Context Recall

Suppose

Retriever returns

```text
Leave Policy

VPN Guide

Office Rules

Insurance
```

Required

```text
Leave Policy

HR FAQ
```

Context Precision

```text
Useful

1

Retrieved

4

=

25%
```

Context Recall

```text
Retrieved

1

Required

2

=

50%
```

Precision asks:

> How much retrieved context is useful?

Recall asks:

> Did we retrieve everything necessary?

---

# Production Example

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k":5}
)

docs = retriever.invoke(
    "How many maternity leave days?"
)

for doc in docs:
    print(doc.page_content)
```

Example output

```text
Leave Policy

Travel Policy

Office Rules

VPN

Insurance
```

Your evaluation process compares these retrieved chunks against the gold evidence needed to answer the question.

---

# Using Ragas

Install

```bash
pip install ragas
```

Evaluate

```python
from ragas import evaluate
from ragas.metrics import (
    context_recall
)

results = evaluate(
    dataset,
    metrics=[
        context_recall
    ]
)

print(results)
```

Example

```text
Context Recall

0.91
```

Meaning:

91% of the information needed to answer questions is being retrieved.

---

# Example Dataset

```python
from datasets import Dataset

dataset = Dataset.from_dict({

    "question":[
        "How many maternity leave days?"
    ],

    "contexts":[[
        "Employees receive 180 days.",
        "Travel reimbursement.",
        "VPN setup."
    ]],

    "ground_truth":[
        "Employees receive 180 days."
    ],

    "answer":[
        "Employees receive 180 days."
    ]
})
```

Ragas compares the retrieved contexts against the reference answer (or reference contexts, depending on the evaluation setup) to estimate whether the required evidence is present.

---

# How to Improve Context Recall

## Better Chunking

Instead of

```text
Entire Chapter
```

Use

```text
Smaller Semantic Chunks
```

This increases the chance of retrieving the exact information.

---

## Increase K

Instead of

```python
k = 3
```

Use

```python
k = 20
```

Retrieve more candidate documents before reranking.

---

## Hybrid Search

```text
Vector Search

+

BM25
```

Improves coverage for both semantic concepts and exact keywords.

---

## Query Rewriting

Instead of

```text
Vacation Policy
```

Rewrite as

```text
Employee Leave Policy
```

This often retrieves additional relevant evidence.

---

## Better Embeddings

Higher-quality embedding models typically improve retrieval recall by representing semantic relationships more accurately.

---

## Metadata Filtering

Restrict retrieval to the appropriate department or document type when applicable, reducing missed matches due to noisy search spaces.

---

# Enterprise Retrieval Pipeline

```text
                  User Question

                        │

                        ▼

                 Query Rewriter

                        ▼

                Hybrid Retrieval

             ┌──────────┴──────────┐

             ▼                     ▼

       Vector Search            BM25

             └──────────┬──────────┘

                        ▼

                  Top 50 Docs

                        ▼

             Cross Encoder Reranker

                        ▼

                   Top 5 Docs

                        ▼

            Compute Context Recall

                        ▼

                       LLM
```

---

# Common Interview Questions

### Why is Context Recall important?

Because if required evidence is missing, the LLM cannot reliably generate a complete and correct answer.

---

### Can Context Recall be 100% while Precision is low?

Yes.

Example:

Retrieved

```text
Leave Policy

HR FAQ

VPN

Travel

Insurance
```

All required information is present, so Context Recall is **100%**.

However, much of the retrieved context is irrelevant, so Context Precision is low.

---

### Can Precision be high while Recall is low?

Yes.

Suppose the retriever returns:

```text
Leave Policy
```

Only one document.

It is perfectly relevant.

Precision

```text
100%
```

But if the answer also requires:

```text
HR FAQ
```

Context Recall is only:

```text
50%
```

The retrieved context is clean but incomplete.

---

# Precision vs Recall

| Scenario                                                | Context Precision | Context Recall |
| ------------------------------------------------------- | ----------------- | -------------- |
| Few relevant chunks, many irrelevant chunks             | Low               | Can be high    |
| Only one relevant chunk retrieved, but answer needs two | High              | Low            |
| All required chunks retrieved, little noise             | High              | High           |
| Mostly irrelevant chunks, missing evidence              | Low               | Low            |

---

# Production Dashboard

Teams often monitor retrieval metrics over time.

```text
Prometheus
      │
      ▼
Grafana

Context Recall
Context Precision
MRR
Recall@K
Latency
Token Cost
Failure Rate
```

A sudden drop in Context Recall may indicate problems with document ingestion, chunking, embeddings, indexing, or query rewriting.

---

# Senior AI Engineer Interview Answer

> **Context Recall measures whether a retriever has retrieved all the information required to answer a user's question. Unlike Context Precision, which measures how much retrieved context is useful, Context Recall measures completeness. High Context Recall ensures the LLM receives all necessary evidence, reducing incomplete answers and hallucinations caused by missing context. In production RAG systems, I improve Context Recall through better chunking, stronger embedding models, hybrid retrieval, higher candidate retrieval (`k`), query rewriting, and reranking. I typically evaluate Context Recall with frameworks such as Ragas alongside Context Precision, MRR, and Recall@K to monitor retrieval quality independently from answer generation.**
