# How to Debug Poor RAG Performance (Complete Production Guide)

One of the most common interview questions for **Senior AI Engineers** is:

> **"Your RAG system is giving poor answers. How do you debug it?"**

Many engineers immediately start changing the prompt.

That is usually **the wrong first step**.

A RAG system has multiple stages:

```text
User Query
      │
      ▼
Query Understanding
      │
      ▼
Retriever
      │
      ▼
Retrieved Documents
      │
      ▼
Reranker
      │
      ▼
Context Builder
      │
      ▼
LLM
      │
      ▼
Generated Answer
```

Poor answers can come from **any stage**.

The key is to debug **stage by stage**, not the entire system at once.

---

# Step 1: Identify Where the Failure Is

Never ask:

> "Why is my answer wrong?"

Instead ask:

```text
Did retrieval fail?

↓

Did reranking fail?

↓

Did prompt fail?

↓

Did generation fail?
```

This systematic approach is much faster.

---

# Production Debugging Workflow

```text
User Question
       │
       ▼
Log Query
       │
       ▼
Log Retrieved Chunks
       │
       ▼
Log Similarity Scores
       │
       ▼
Log Reranker Scores
       │
       ▼
Log Final Prompt
       │
       ▼
Log LLM Response
       │
       ▼
Compute Evaluation Metrics
```

Without these logs, debugging becomes guesswork.

---

# Example Problem

Question:

> What is the maternity leave policy?

LLM Answer:

```text
Employees receive 90 days.
```

Actual answer:

```text
180 days.
```

Where did it fail?

---

# Step 2: Inspect Retrieved Documents

First inspect what retrieval returned.

```python
docs = retriever.invoke(
    "What is the maternity leave policy?"
)

for i, doc in enumerate(docs, start=1):
    print("=" * 40)
    print(f"Rank {i}")
    print(doc.page_content)
```

Suppose output:

```text
Rank 1
VPN Guide

Rank 2
Office Rules

Rank 3
Travel Policy
```

Immediately you know:

> **Retrieval failed.**

Don't touch the prompt.

---

# Step 3: Compute Retrieval Metrics

Evaluate retrieval separately.

Example:

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

Example:

```text
Context Precision = 0.21

Context Recall = 0.35
```

Interpretation:

Low recall means required evidence is missing.

Low precision means too much irrelevant context.

---

# Step 4: Check Similarity Scores

Inspect similarity scores.

```python
results = vector_store.similarity_search_with_score(
    query,
    k=5
)

for doc, score in results:
    print(score)
    print(doc.page_content)
```

Output:

```text
0.81 VPN Guide

0.80 Office Rules

0.79 Leave Policy
```

Notice:

The correct document exists but is ranked too low.

This suggests:

* poor embeddings
* missing reranker
* poor chunking

---

# Step 5: Check Chunking

Bad chunking is one of the biggest causes of poor RAG.

Bad chunk:

```text
Employees receive

---------------------------------

180 days of maternity leave.
```

The sentence is split across chunks.

Better:

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
```

Verify chunks:

```python
for chunk in chunks[:5]:
    print(chunk.page_content)
```

Ask:

* Does each chunk represent one topic?
* Are sentences split?
* Is important information separated?

---

# Step 6: Check Embeddings

Test whether semantically similar queries retrieve the same document.

```python
queries = [
    "Maternity leave",
    "Pregnancy leave",
    "Leave for mothers"
]

for q in queries:
    docs = retriever.invoke(q)

    print(q)

    print(docs[0].page_content)
```

If each query retrieves different unrelated documents:

The embedding model may not be suitable.

---

# Step 7: Check Metadata Filters

Suppose documents contain metadata:

```python
Document(
    page_content="Leave Policy",
    metadata={
        "department":"HR"
    }
)
```

If your filter is:

```python
filter={
    "department":"Finance"
}
```

Nothing useful will ever be retrieved.

Always log filters.

```python
print(search_filter)
```

---

# Step 8: Inspect Reranker

Before reranking:

```text
1 VPN

2 Travel

3 Leave Policy
```

After reranking:

```text
1 Leave Policy

2 Travel

3 VPN
```

If reranking doesn't improve ranking:

* wrong reranker model
* wrong query
* retrieval candidates too poor

Log reranker scores.

```python
for doc in reranked_docs:
    print(doc.metadata["score"])
```

---

# Step 9: Inspect Final Prompt

Many teams retrieve good documents but build a poor prompt.

Bad:

```text
Question

Documents

Answer
```

Better:

```text
You are an HR assistant.

Use ONLY the supplied context.

If the answer is missing,
say:

"I don't know."

Context:
...

Question:
...
```

Always print the final prompt.

```python
print(prompt)
```

---

# Step 10: Check Hallucinations

Context:

```text
Employees receive 180 days.
```

Answer:

```text
Employees receive 180 days
plus 30 optional days.
```

The model invented:

```text
30 optional days
```

Evaluate faithfulness.

```python
from ragas.metrics import faithfulness
```

Example:

```text
Faithfulness

0.52
```

The answer is poorly grounded.

---

# Step 11: Evaluate Answer Correctness

Use answer correctness metrics.

Example:

Ground Truth

```text
180 days.
```

Answer

```text
90 days.
```

Clearly incorrect.

Evaluate:

```python
from ragas.metrics import answer_correctness
```

---

# Step 12: Debug Latency

Break latency into stages.

```text
Embedding

15 ms

↓

Retriever

18 ms

↓

Reranker

120 ms

↓

LLM

900 ms

↓

Total

1053 ms
```

Instrument each step.

```python
import time

start = time.time()

docs = retriever.invoke(query)

print(time.time() - start)
```

Repeat for reranking and generation.

---

# Step 13: Track Token Usage

Sometimes the model never sees the relevant chunk because the prompt is too large.

```python
from tiktoken import encoding_for_model

enc = encoding_for_model("gpt-4")

tokens = len(
    enc.encode(prompt)
)

print(tokens)
```

If the prompt exceeds the model's effective context, trim or compress retrieved documents.

---

# Step 14: Evaluate Every Query

Build an evaluation set.

```python
questions = [
    "Leave policy",
    "VPN reset",
    "Insurance",
    "Travel reimbursement"
]
```

Run evaluation.

```python
for q in questions:

    docs = retriever.invoke(q)

    answer = chain.invoke(q)

    print(q)

    print(answer)
```

Never rely on a single query.

---

# Production Logging

Log everything.

```python
logger.info({
    "query": query,
    "retrieved_docs": [
        d.page_content
        for d in docs
    ],
    "reranked_docs": [
        d.page_content
        for d in reranked
    ],
    "answer": response
})
```

This makes failures reproducible.

---

# Monitoring Dashboard

Track these metrics continuously.

```text
Prometheus

↓

Grafana

↓

Context Recall

↓

Context Precision

↓

Faithfulness

↓

Answer Correctness

↓

Latency

↓

Token Usage

↓

Cost
```

---

# Common Root Causes

| Symptom                            | Likely Cause                                | Fix                                                   |
| ---------------------------------- | ------------------------------------------- | ----------------------------------------------------- |
| Wrong documents retrieved          | Poor embeddings, chunking, filters          | Better embeddings, semantic chunking, verify filters  |
| Correct document ranked low        | Missing reranker                            | Add cross-encoder reranking                           |
| Missing answer information         | Low Context Recall                          | Increase `k`, hybrid search, query rewriting          |
| Too much irrelevant context        | Low Context Precision                       | Metadata filtering, reranking                         |
| Hallucinations                     | Low Faithfulness                            | Ground prompts, improve retrieval, lower temperature  |
| Correct retrieval but wrong answer | Prompt or LLM issue                         | Inspect prompt, adjust instructions, evaluate model   |
| Slow responses                     | Large `k`, expensive reranker, long prompts | Profile latency, reduce candidate size, cache results |

---

# End-to-End Debugging Checklist

```text
User Question
      │
      ▼
✓ Verify query rewriting
      │
      ▼
✓ Inspect retrieved chunks
      │
      ▼
✓ Measure Context Recall
      │
      ▼
✓ Measure Context Precision
      │
      ▼
✓ Inspect reranking
      │
      ▼
✓ Inspect final prompt
      │
      ▼
✓ Measure Faithfulness
      │
      ▼
✓ Measure Answer Correctness
      │
      ▼
✓ Measure Latency
      │
      ▼
✓ Measure Token Usage
```

---

# Real Enterprise Debugging Example

Suppose users complain that the HR assistant gives incorrect leave policy answers.

1. **Inspect retrieval**: The HR policy chunk is retrieved at rank 18 because chunking split the policy across multiple chunks.
2. **Check Context Recall**: 0.42 — required evidence is often missing from the top results.
3. **Increase retrieval depth**: Change `k` from 5 to 30.
4. **Add a cross-encoder reranker**: Keep the best 5 documents after reranking.
5. **Improve chunking**: Switch to recursive token-based chunking with overlap.
6. **Re-evaluate**:

   * Context Recall: **0.91**
   * Context Precision: **0.84**
   * Faithfulness: **0.97**
   * Answer Correctness: **0.95**

The issue was **retrieval quality**, not the LLM itself.

---

# Senior AI Engineer Interview Answer

> **I debug RAG systems by treating retrieval and generation as separate stages. First, I inspect the retrieved documents and measure retrieval metrics such as Context Recall, Context Precision, Recall@K, MRR, and Hit Rate. If retrieval is weak, I investigate chunking, embeddings, metadata filters, query rewriting, retrieval depth (`k`), and hybrid search. Next, I evaluate reranking quality and verify that the most relevant documents are passed to the LLM. If retrieval is good but answers are still poor, I inspect the final prompt, evaluate Faithfulness and Answer Correctness, and check for hallucinations or prompt issues. I also profile latency, token usage, and cost for each pipeline stage and maintain comprehensive logs and dashboards with tools such as LangSmith, Prometheus, and Grafana. This stage-by-stage debugging approach isolates failures quickly and leads to targeted fixes instead of trial-and-error prompt tuning.
