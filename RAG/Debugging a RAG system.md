Debugging a RAG system is one of the most important skills for a Senior AI Engineer. In production, when users say:

> "The chatbot gives wrong answers."

The first question is **not**:

> "Is the LLM bad?"

Instead, it's:

> "Which stage of the RAG pipeline is failing?"

A RAG system has multiple stages, and each stage can independently introduce errors.

---

# RAG Debugging Pipeline

```text
                    User Query
                         │
                         ▼
                  Query Processing
                         │
                         ▼
                    Retrieval
                         │
                         ▼
                     Reranking
                         │
                         ▼
                Context Building
                         │
                         ▼
                 Prompt Generation
                         │
                         ▼
                        LLM
                         │
                         ▼
                     Evaluation
```

We debug each stage separately.

---

# Step 1: Check the Retrieval

Suppose the user asks

```
How does JWT authentication work?
```

Retriever returns

```
1. Docker Compose

2. Kubernetes

3. Redis Cache
```

Immediately we know

```
Retriever failed.
```

The LLM never had a chance.

---

## Debug Code

```python
query = "How does JWT authentication work?"

results = retriever.search(
    query=query,
    top_k=5
)

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc['title']}")
```

Output

```
1 Docker Compose
2 Redis
3 Kubernetes
```

Problem found.

---

# Step 2: Inspect Similarity Scores

Many people ignore similarity scores.

```python
results = retriever.search(query)

for doc in results:
    print(
        doc["score"],
        doc["title"]
    )
```

Output

```
0.81 Docker

0.80 Kubernetes

0.79 JWT

0.78 Redis
```

Notice

JWT should probably be first.

This often indicates

* poor embeddings
* poor chunking
* no reranker

---

# Step 3: Inspect the Chunks

Example chunk

```
Chunk

FastAPI supports...

...

...

...

JWT Authentication starts here
```

Only the last sentence is relevant.

Embedding represents the whole chunk.

The important information gets diluted.

---

Better chunk

```
JWT Authentication

Access token

Refresh token

Verification

Expiration
```

Now retrieval improves.

---

## Debug Chunk Size

```python
for chunk in chunks:

    print("=" * 50)
    print(len(chunk.split()))
    print(chunk[:300])
```

Questions to ask:

* Is the chunk too large?
* Is the chunk cut in the middle of a paragraph?
* Does it mix multiple topics?

---

# Step 4: Verify Embeddings

Sometimes the embedding model is unsuitable.

Example

```
Query

Secure API
```

Document

```
JWT Authentication
```

Embedding similarity

```
0.22
```

Another model

```
0.93
```

Huge difference.

---

Debug

```python
query_vector = embedding_model.encode(query)

doc_vector = embedding_model.encode(document)

score = cosine_similarity(
    [query_vector],
    [doc_vector]
)

print(score)
```

---

# Step 5: Check Retrieval Recall

Suppose

Ground truth

```
JWT Guide
```

Retriever

```
Docker

Redis

FastAPI
```

Recall

```
0%
```

The correct document never appeared.

Increase

```
top_k
```

Example

```python
for k in [5,10,20,50]:

    docs = retriever.search(query, top_k=k)

    print(k, contains_answer(docs))
```

Output

```
5 False

10 False

20 True

50 True
```

Maybe

```
top_k=5
```

is too small.

---

# Step 6: Inspect the Reranker

Retriever

```
1 JWT

2 OAuth

3 Docker
```

Reranker

```
1 Docker

2 JWT

3 OAuth
```

The reranker made things worse.

Debug

```python
before = retriever.search(query)

after = reranker.rerank(query, before)

print("Retriever")

for d in before:
    print(d["title"])

print()

print("Reranker")

for d in after:
    print(d["title"])
```

---

# Step 7: Inspect the Prompt

Poor prompt

```
Answer briefly.
```

Good prompt

```
Answer ONLY using the context.

If the answer is unavailable,
say "I don't know."
```

Debug

```python
print(prompt)
```

Example

```
Question

...

Context

...

Instructions

...
```

Always inspect the final prompt.

---

# Step 8: Inspect Context Length

Many chunks

```
Chunk1

Chunk2

Chunk3

...

Chunk30
```

LLM

```
Context window exceeded
```

The useful chunk may be truncated.

Debug

```python
tokens = tokenizer.encode(context)

print(len(tokens))
```

Output

```
18500 tokens
```

Model limit

```
16000
```

Problem found.

---

# Step 9: Check Hallucinations

Retrieved context

```
JWT uses access tokens.
```

Answer

```
JWT uses blockchain.
```

Hallucination.

Debug

```
Every sentence in answer

↓

Does it exist in context?
```

Libraries

```
Ragas

DeepEval

TruLens
```

measure

```
Faithfulness
```

---

# Step 10: Inspect Logs

A production system logs every retrieval.

Example

```python
logger.info(
    {
        "query": query,
        "retrieved_docs": ids,
        "scores": scores,
        "latency": latency
    }
)
```

Example log

```json
{
    "query":"JWT",

    "scores":[
        0.91,
        0.89,
        0.83
    ],

    "latency":34
}
```

Without logs,

debugging becomes nearly impossible.

---

# Step 11: Compare Retrieval Before and After Changes

Never rely on intuition.

Example

Old embedding

```
Recall@10

0.82
```

New embedding

```
Recall@10

0.91
```

Deploy

Otherwise

Don't.

---

# Step 12: Visualize the Entire Pipeline

```python
def debug_pipeline(query):

    print("Query")
    print(query)

    docs = retriever.search(query)

    print("\nRetrieved")

    for d in docs:
        print(
            d["score"],
            d["title"]
        )

    reranked = reranker.rerank(
        query,
        docs
    )

    print("\nReranked")

    for d in reranked:
        print(
            d["score"],
            d["title"]
        )

    context = "\n".join(
        x["text"]
        for x in reranked[:5]
    )

    print("\nContext")
    print(context[:1000])

    answer = llm.generate(
        query=query,
        context=context
    )

    print("\nAnswer")
    print(answer)
```

One function lets you inspect every stage.

---

# Production Logging Architecture

```text
User Query
      │
      ▼
+------------------+
| Query Logger     |
+------------------+
      │
      ▼
Retriever
      │
      ▼
+------------------+
| Retrieved IDs    |
| Scores           |
| Latency          |
+------------------+
      │
      ▼
Reranker
      │
      ▼
+------------------+
| Reranked IDs     |
| Scores           |
+------------------+
      │
      ▼
Prompt Builder
      │
      ▼
+------------------+
| Prompt Tokens    |
| Context Tokens   |
+------------------+
      │
      ▼
LLM
      │
      ▼
+------------------+
| Answer           |
| Output Tokens    |
| Cost             |
| Latency          |
+------------------+
```

---

# Real-World Debugging Checklist

| Symptom                         | Likely Cause                                    | How to Debug                                     | Possible Fix                                                  |
| ------------------------------- | ----------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| Wrong documents retrieved       | Poor embeddings, chunking, or query formulation | Print retrieved chunks and similarity scores     | Improve chunking, switch embedding model, use query rewriting |
| Correct document missing        | Low recall                                      | Measure Recall@K and inspect `top_k`             | Increase `top_k`, improve hybrid search                       |
| Relevant chunk ranked too low   | Weak ranking                                    | Compare retriever output with reranker output    | Tune or replace the reranker                                  |
| Irrelevant context sent to LLM  | Low context precision                           | Inspect final context before prompting           | Reduce `top_k`, improve reranking, deduplicate chunks         |
| Hallucinated answers            | Low faithfulness                                | Compare each answer claim with retrieved context | Strengthen prompt grounding, improve context quality          |
| Truncated or incomplete answers | Context window overflow                         | Count prompt tokens                              | Compress context, reduce retrieved chunks                     |
| Slow responses                  | Large vector index or expensive reranker        | Measure latency at each stage                    | Cache embeddings, optimize ANN index, rerank fewer candidates |

---

# Production-Grade Debugging Architecture

Senior AI engineers typically instrument every stage of the pipeline so they can answer questions like:

* **Was the right document retrieved?**
* **Was it ranked correctly?**
* **Did the LLM actually receive it?**
* **Did the LLM use it?**
* **If not, which stage failed?**

A common production architecture looks like this:

```text
User Query
      │
      ▼
Query Analyzer
      │
      ▼
Hybrid Retrieval
      │
      ├── Log retrieved IDs
      ├── Log similarity scores
      └── Measure Recall@K
      │
      ▼
Reranker
      │
      ├── Log ranking changes
      └── Measure MRR/NDCG
      │
      ▼
Context Builder
      │
      ├── Log selected chunks
      ├── Count prompt tokens
      └── Remove duplicates
      │
      ▼
LLM
      │
      ├── Log latency
      ├── Log token usage
      └── Store generated answer
      │
      ▼
Evaluation
      ├── Context Precision
      ├── Context Recall
      ├── Faithfulness
      └── Answer Correctness
```

The key principle is to **treat RAG as a pipeline of independently testable components**. By logging inputs and outputs at each stage and evaluating them with appropriate metrics, you can pinpoint whether poor performance originates from retrieval, reranking, context construction, prompting, or generation instead of guessing based on the final answer alone.
