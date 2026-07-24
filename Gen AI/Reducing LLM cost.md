Reducing LLM cost by 50% (or more) is one of the most common responsibilities of Senior AI Engineers. In production systems serving thousands or millions of requests, token costs often dominate infrastructure costs.

The biggest cost reductions usually come from **architecture changes**, not from prompt tweaks. Many production systems achieve **50–90% savings** by combining several techniques.

---

# Where LLM Costs Come From

```text
                User Request
                      │
                      ▼
               Prompt Tokens
                      │
                      ▼
             Retrieved Documents
                      │
                      ▼
              Conversation History
                      │
                      ▼
                LLM Generation
                      │
                      ▼
             Completion Tokens
```

Total Cost =

```
Input Tokens
+
Output Tokens
+
Number of Requests
```

---

# 1. Route Requests to Smaller Models (Highest ROI)

Don't use your most expensive model for every request.

Example architecture:

```text
                 User
                   │
                   ▼
            Intent Classifier
                   │
      ┌────────────┼─────────────┐
      ▼            ▼             ▼
 Simple FAQ    RAG Query    Complex Reasoning
      │            │             │
      ▼            ▼             ▼
 Small Model  Medium Model  Large Model
```

Python example:

```python
def choose_model(query: str):
    if len(query) < 30:
        return "small-model"

    if "summarize" in query.lower():
        return "medium-model"

    return "large-model"
```

Typical savings:

```
20–60%
```

---

# 2. Cache Responses

Many users ask the same questions repeatedly.

Without cache

```text
User A

↓

LLM

↓

Answer

User B

↓

LLM Again
```

With cache

```text
User A

↓

LLM

↓

Redis Cache

↓

User B

↓

Cache Hit

↓

No LLM Call
```

Python

```python
import hashlib

cache = {}

def ask_llm(prompt):

    key = hashlib.md5(
        prompt.encode()
    ).hexdigest()

    if key in cache:
        return cache[key]

    answer = llm(prompt)

    cache[key] = answer

    return answer
```

Savings

```
10–70%
```

---

# 3. Reduce Prompt Size

Bad prompt

```text
20 pages

↓

LLM
```

Good prompt

```text
Relevant 2 pages

↓

LLM
```

Instead of

```
15,000 tokens
```

Use

```
2,000 tokens
```

Techniques:

* Better retrieval
* Smaller chunks
* Better reranking
* Remove duplicate context

---

# 4. Better RAG Retrieval

Bad retrieval

```text
Top 20 Documents

↓

18 Irrelevant

↓

Huge Prompt
```

Good retrieval

```text
Top 3 Relevant Docs

↓

Small Prompt

↓

Same Accuracy
```

Example

```python
retriever.search(
    query,
    k=3
)
```

Instead of

```python
k=20
```

Savings

```
30–80%
```

---

# 5. Summarize Conversation History

Instead of sending the full chat every time:

```text
Message 1

Message 2

...

Message 100
```

Summarize older messages:

```text
Conversation Summary

+

Recent 5 Messages
```

Python

```python
if len(history) > 20:
    history = summarize(history)
```

Savings

```
40–80%
```

---

# 6. Compress Retrieved Documents

Instead of sending

```text
Entire PDF
```

Send only the relevant passages.

Example:

```python
compressed = compressor.compress(
    documents
)
```

Typical reduction:

```
60%
```

---

# 7. Limit Output Tokens

Bad

```text
max_tokens = 4000
```

Good

```python
max_tokens = 300
```

Prompt:

```
Answer in under 150 words.
```

Many APIs stop early if the answer is complete, but setting realistic limits prevents unnecessarily long generations.

---

# 8. Use Function Calling Instead of Text

Without tools

```text
LLM

↓

Writes 300 Tokens

↓

Application Parses Text
```

With structured tool calls

```text
LLM

↓

JSON

↓

Application
```

Example

```json
{
  "city":"London"
}
```

Instead of

```
The city the user requested is London.
```

Fewer output tokens and more reliable parsing.

---

# 9. Avoid Repeated Agent Loops

Bad

```text
Search

↓

Search

↓

Search

↓

Search

↓

Answer
```

Good

```
max_iterations = 3
```

Example

```python
Agent(
    max_iterations=3
)
```

Savings

```
20–50%
```

---

# 10. Batch Embeddings

Bad

```python
for text in docs:
    embed(text)
```

Good

```python
embed_batch(docs)
```

One request is usually much cheaper than many small ones.

---

# 11. Cache Embeddings

Bad

```text
Embed

↓

Store

↓

Embed Again
```

Good

```text
Hash Document

↓

Already Exists?

↓

Reuse Vector
```

Example

```python
hash = md5(text)

if hash in cache:
    return cache[hash]
```

---

# 12. Hybrid Search Before the LLM

Instead of embedding every query:

```text
User

↓

LLM

↓

Vector Search
```

Use

```text
BM25

↓

Vector Search

↓

Reranker

↓

LLM
```

Many queries are answered with fewer retrieved documents.

---

# 13. Early Exit

Sometimes an LLM isn't needed.

Example

```text
User

↓

Health Check

↓

Simple SQL

↓

Return Result
```

Instead of

```text
SQL

↓

LLM
```

Python

```python
if query == "status":
    return database.status()
```

---

# 14. Asynchronous Tool Calls

Instead of

```text
Search

↓

SQL

↓

API

↓

LLM
```

Run in parallel

```python
results = await asyncio.gather(
    search(),
    sql(),
    api()
)
```

This mainly reduces latency, but can also reduce repeated retries and timeouts that increase cost.

---

# 15. Evaluate Before Upgrading Models

Don't assume a larger model is required.

Example

```
Small Model Accuracy

↓

94%
```

Large Model

```
96%
```

The extra 2% may not justify a 10× cost increase for all requests.

---

# Example Production Architecture

```text
                    User
                      │
                      ▼
              API Gateway
                      │
                      ▼
             Cache (Redis)
                      │
           Cache Hit? ───────► Return Response
                      │
                 Cache Miss
                      │
                      ▼
            Intent Classifier
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
 Simple Query     RAG Query      Complex Agent
      │               │                │
 Small Model     Medium Model     Large Model
      │               │                │
      └───────────────┼────────────────┘
                      ▼
              Store in Cache
                      │
                      ▼
                 Return Result
```

---

# Example Cost Optimization

Suppose:

```
100,000 requests/day
```

Without optimization

```
Prompt = 4000 tokens

Completion = 1000 tokens

Cost = $1000/day
```

After optimization:

| Optimization               | Savings |
| -------------------------- | ------- |
| Cache                      | 20%     |
| Smaller model routing      | 30%     |
| Better retrieval           | 20%     |
| Conversation summarization | 15%     |
| Output token limits        | 10%     |

These percentages are **not additive** because they overlap, but together they can easily reduce the total bill by **50–80%** in many real-world workloads.

---

# What Senior AI Engineers Measure

Track these metrics continuously:

* **Cost per request**
* **Cost per active user**
* **Input tokens/request**
* **Output tokens/request**
* **Cache hit rate**
* **Model routing distribution** (small vs. medium vs. large)
* **Average retrieved tokens**
* **Average conversation context size**
* **Latency**
* **Task success rate** (ensure cost reductions don't significantly hurt quality)

---

# A Practical Cost-Reduction Strategy

If you need to cut costs quickly without sacrificing much quality, prioritize these steps:

1. **Cache identical and semantically similar requests** using Redis or another cache.
2. **Route requests to the smallest capable model** based on task complexity.
3. **Improve RAG retrieval** so only the most relevant context is sent.
4. **Summarize long conversations** instead of replaying the entire history.
5. **Set realistic output token limits** and instruct the model to be concise.
6. **Use deterministic code or APIs instead of the LLM** for calculations, database lookups, and other structured operations.
7. **Monitor token usage and cost per endpoint** so regressions are detected immediately.

These techniques are widely used in production AI systems and, when combined, can often reduce LLM spending by **50% or more** while maintaining a similar level of user experience.
