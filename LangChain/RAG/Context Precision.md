# Context Precision Explained End-to-End (with Production Code)

**Context Precision** is one of the most important metrics for evaluating **RAG (Retrieval-Augmented Generation)** systems.

It is widely used in evaluation frameworks such as **Ragas** because it measures the quality of the **retrieved context**, not the quality of the final answer.

Many engineers confuse **Context Precision** with **Precision@K**.

They are related, but **they measure different things**.

---

# Where Context Precision Fits

A RAG pipeline has two stages:

```text
                    User Question
                          │
                          ▼
                   Retriever
                (Vector / Hybrid)
                          │
                Retrieved Context
                          │
                          ▼
                        LLM
                          │
                     Final Answer
```

Context Precision evaluates:

> **How much of the retrieved context is actually useful for answering the question?**

It does **not** evaluate whether the LLM answered correctly.

---

# Why Do We Need Context Precision?

Suppose the user asks:

> **How many maternity leave days are available?**

Your retriever returns:

```text
1. Leave Policy
2. VPN Setup Guide
3. Cafeteria Menu
4. Health Insurance
5. Travel Policy
```

Only one document actually helps answer the question.

Useful context:

```text
Leave Policy
```

Irrelevant context:

```text
VPN Setup Guide
Cafeteria Menu
Health Insurance
Travel Policy
```

Although five documents were retrieved, only one contributes to the answer.

---

# Definition

Context Precision is:

```text
Useful Retrieved Context
──────────────────────────────
Total Retrieved Context
```

---

Example:

Retrieved:

```text
Leave Policy
VPN Guide
Travel Policy
Office Rules
```

Relevant:

```text
Leave Policy
```

Context Precision:

```text
1 / 4 = 0.25
```

Only 25% of the retrieved context is useful.

---

# Why Does This Matter?

LLMs have limited context windows.

Suppose your model can process:

```text
8,000 tokens
```

If you send:

```text
7,500 irrelevant tokens
500 useful tokens
```

the model wastes most of its attention on irrelevant information.

Better retrieval:

```text
6,500 useful tokens
1,500 supporting tokens
```

produces more reliable answers while reducing token costs.

---

# High Context Precision

```text
User Question
      │
      ▼
Retriever
      │
      ▼
Leave Policy
Leave FAQ
HR Handbook
      │
      ▼
LLM
```

Nearly everything is relevant.

---

# Low Context Precision

```text
User Question
      │
      ▼
Retriever
      │
      ▼
Leave Policy
VPN Guide
Travel Policy
Marketing Handbook
Server Manual
      │
      ▼
LLM
```

Most of the context is noise.

---

# Python Example

Suppose we know which documents are relevant.

```python
retrieved_docs = [
    "Leave Policy",
    "VPN Guide",
    "Travel Policy",
    "Office Rules"
]

relevant_docs = [
    "Leave Policy"
]
```

Compute Context Precision.

```python
def context_precision(retrieved, relevant):

    relevant = set(relevant)

    useful = sum(
        1
        for doc in retrieved
        if doc in relevant
    )

    return useful / len(retrieved)


score = context_precision(
    retrieved_docs,
    relevant_docs
)

print(score)
```

Output

```text
0.25
```

---

# Another Example

Retrieved:

```text
Leave Policy
Leave FAQ
HR Manual
VPN Guide
```

Relevant:

```text
Leave Policy
Leave FAQ
HR Manual
```

Python

```python
retrieved = [
    "Leave Policy",
    "Leave FAQ",
    "HR Manual",
    "VPN Guide"
]

relevant = [
    "Leave Policy",
    "Leave FAQ",
    "HR Manual"
]

print(
    context_precision(
        retrieved,
        relevant
    )
)
```

Output

```text
0.75
```

Now 75% of the retrieved context is useful.

---

# Visual Comparison

High Context Precision

```text
Retrieved

✓ Leave Policy
✓ HR FAQ
✓ Insurance
✗ VPN Guide
```

Score

```text
3 / 4 = 75%
```

---

Low Context Precision

```text
Retrieved

✓ Leave Policy
✗ VPN Guide
✗ Marketing
✗ Office Rules
```

Score

```text
1 / 4 = 25%
```

---

# Context Precision vs Precision@K

Many interviews ask this.

## Precision@K

Measures:

> How many retrieved documents are relevant?

Example

Top 5

```text
Leave Policy
VPN
Travel
HR FAQ
Insurance
```

Relevant

```text
Leave Policy
HR FAQ
```

Precision@5

```text
2 / 5 = 40%
```

---

## Context Precision

Measures:

> How much of the context actually supports answering the question?

Suppose:

```text
Leave Policy
```

contains the answer.

```text
HR FAQ
```

mentions leave but doesn't answer the question.

Then:

```text
Precision@5

2 / 5
```

but

```text
Context Precision

1 / 5
```

Context Precision is stricter because it focuses on usefulness for answering, not just topical relevance.

---

# Production Example with LangChain

Retrieve documents.

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

docs = retriever.invoke(
    "How many maternity leave days?"
)
```

Suppose:

```python
for d in docs:
    print(d.page_content)
```

Returns:

```text
Leave Policy
Travel Policy
VPN Setup
Health Insurance
Office Rules
```

Now compare against your evaluation dataset to determine which retrieved chunks actually contain the information needed for the answer.

---

# Using Ragas

Ragas can compute Context Precision automatically.

Install

```bash
pip install ragas
```

Example

```python
from ragas import evaluate
from ragas.metrics import context_precision

results = evaluate(
    dataset,
    metrics=[
        context_precision
    ]
)

print(results)
```

Example output

```text
Context Precision

0.82
```

Meaning:

Approximately 82% of retrieved context is useful.

---

# Example Dataset

```python
from datasets import Dataset

dataset = Dataset.from_dict({

    "question":[
        "How many maternity leave days?"
    ],

    "contexts":[[
        "Leave policy allows 180 days.",
        "VPN Guide",
        "Office Rules"
    ]],

    "ground_truth":[
        "Employees receive 180 days."
    ],

    "answer":[
        "Employees receive 180 days."
    ]
})
```

Ragas uses the contexts together with the ground truth to estimate context quality.

---

# How to Improve Context Precision

## Better Chunking

Instead of:

```text
Entire Chapter
```

Use:

```text
Smaller Semantic Chunks
```

---

## Metadata Filtering

Instead of searching every document:

```text
All Departments
```

Search only:

```text
HR
```

Example

```python
results = vector_store.similarity_search(
    query,
    filter={
        "department":"HR"
    }
)
```

---

## Reranking

Retrieve

```text
Top 50
```

Rerank

```text
Top 5
```

Only highly relevant chunks are passed to the LLM.

---

## Hybrid Search

Combine:

```text
Vector Search

+

BM25
```

to improve candidate retrieval before reranking.

---

# Production Architecture

```text
                 User Question

                       │

                       ▼

                Query Rewriter

                       ▼

              Hybrid Retrieval

          ┌───────────┴───────────┐

          ▼                       ▼

   Vector Search              BM25

          └───────────┬───────────┘

                      ▼

                 Top 50 Docs

                      ▼

                 Reranker

                      ▼

                 Top 5 Docs

                      ▼

        Compute Context Precision

                      ▼

                     LLM
```

---

# Common Interview Questions

### Why is Context Precision important?

It measures how much of the retrieved context actually helps answer the question. High Context Precision improves answer quality and reduces unnecessary token usage.

---

### Does high Context Precision guarantee a correct answer?

No.

The LLM can still misunderstand the context or generate an incorrect response.

Retrieval quality and generation quality are evaluated separately.

---

### Can Context Precision be 100%?

Yes.

If every retrieved chunk is useful for answering the question.

Example:

```text
Retrieved

Leave Policy
Leave FAQ
HR Handbook
```

All three contribute to the answer.

---

### How do you improve Context Precision?

Typical approaches include:

* Better chunking
* Metadata filtering
* Hybrid retrieval
* Cross-encoder reranking
* Query rewriting
* Removing duplicate or low-quality chunks

---

# Context Precision vs Context Recall

| Metric            | Question It Answers                                               |
| ----------------- | ----------------------------------------------------------------- |
| Context Precision | How much of the retrieved context is useful?                      |
| Context Recall    | Did retrieval include all information needed to answer correctly? |

Example:

Retrieve:

```text
Leave Policy
VPN Guide
Travel Policy
```

The answer requires:

```text
Leave Policy
HR FAQ
```

* **Context Precision** may be moderate because only one of three retrieved chunks is useful.
* **Context Recall** is low because the required **HR FAQ** chunk is missing.

---

# Senior AI Engineer Interview Answer

> **Context Precision measures the proportion of retrieved context that is actually useful for answering a user's question. Unlike Precision@K, which measures topical relevance, Context Precision focuses on whether the retrieved chunks contain information that contributes to the final answer. In production RAG systems, low Context Precision increases token costs, wastes the LLM's context window, and can lead to hallucinations because irrelevant information is included in the prompt. I improve Context Precision using semantic chunking, metadata filtering, hybrid retrieval, query rewriting, and cross-encoder reranking. I typically evaluate it with frameworks such as Ragas alongside Context Recall to monitor retrieval quality independently from generation quality.
