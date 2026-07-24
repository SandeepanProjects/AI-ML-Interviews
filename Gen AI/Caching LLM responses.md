Caching LLM responses is one of the easiest ways to reduce **both cost and latency**. In production systems, companies like OpenAI customers, Anthropic customers, Microsoft, Uber, and Stripe often cache responses because many users ask the same or very similar questions.

A good caching strategy can reduce:

* **LLM cost by 30–80%**
* **Latency from seconds to milliseconds**

---

# Where Caching Fits

```text
                    User
                      │
                      ▼
               API Gateway
                      │
                      ▼
                Cache Layer
                      │
          ┌───────────┴───────────┐
          │                       │
      Cache Hit              Cache Miss
          │                       │
          ▼                       ▼
   Return Response          Call LLM
          │                       │
          └──────────────┬────────┘
                         ▼
                   Store in Cache
                         │
                         ▼
                    Return Result
```

---

# Level 1: Exact Prompt Cache

This is the simplest type of cache.

User 1:

```
What is Python?
```

LLM generates:

```
Python is a programming language...
```

Store:

```text
Key:
"What is Python?"

↓

Value:
"Python is a programming language..."
```

When another user asks the **exact same question**, return the cached answer.

---

## Python Example

```python
import hashlib

cache = {}

def llm_with_cache(prompt):
    key = hashlib.sha256(prompt.encode()).hexdigest()

    if key in cache:
        print("Cache Hit")
        return cache[key]

    print("Calling LLM")

    answer = llm(prompt)

    cache[key] = answer

    return answer
```

---

# Problem with Exact Caching

These prompts are different strings:

```
What is Python?

Tell me about Python.

Explain Python.

Can you explain Python?
```

An exact cache misses all of them, even though they mean almost the same thing.

---

# Level 2: Semantic Cache (Recommended)

Instead of comparing text, compare **embeddings**.

```
"What is Python?"

↓

Embedding

↓

Vector Database
```

Another user asks:

```
Explain Python
```

Embedding:

```
Very Close

↓

Similarity = 0.97

↓

Return Cached Answer
```

---

## Semantic Cache Architecture

```text
                 User Prompt
                      │
                      ▼
               Generate Embedding
                      │
                      ▼
             Vector Database Cache
                      │
          Similar Response Exists?
              │                │
             Yes              No
              │                ▼
              ▼           Call LLM
       Return Cached           │
          Response             ▼
                        Store Embedding
```

---

## Example Using FAISS

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
responses = []

index = faiss.IndexFlatL2(384)

def cached_llm(prompt):

    vector = model.encode([prompt])

    if index.ntotal > 0:

        distance, ids = index.search(
            vector,
            1
        )

        if distance[0][0] < 0.2:
            return responses[ids[0][0]]

    answer = llm(prompt)

    index.add(vector)

    responses.append(answer)

    texts.append(prompt)

    return answer
```

---

# Level 3: Redis Cache

In production, use Redis instead of an in-memory dictionary.

```text
Application

↓

Redis

↓

LLM
```

Example:

```python
import redis
import hashlib

r = redis.Redis()

def ask(prompt):

    key = hashlib.sha256(
        prompt.encode()
    ).hexdigest()

    cached = r.get(key)

    if cached:
        return cached.decode()

    answer = llm(prompt)

    r.setex(
        key,
        3600,
        answer
    )

    return answer
```

`setex` sets a TTL (time-to-live), here **1 hour**.

---

# Level 4: Cache Retrieval Results (RAG)

Instead of caching only the LLM output, cache document retrieval.

Without cache:

```text
Question

↓

Vector Search

↓

Reranker

↓

LLM
```

With cache:

```text
Question

↓

Redis

↓

Retrieved Documents
```

Example:

```python
retrieved_docs = redis.get(query_hash)
```

This avoids repeated vector searches.

---

# Level 5: Cache Embeddings

Embedding models also cost time and money.

Bad:

```text
Document

↓

Embedding

↓

Embedding Again

↓

Embedding Again
```

Good:

```text
Document

↓

Hash

↓

Already Embedded?

↓

Reuse Vector
```

Example:

```python
document_hash = hashlib.md5(
    document.encode()
).hexdigest()
```

If the hash already exists, reuse the stored embedding.

---

# Level 6: Multi-Level Cache

Production systems often use several cache layers.

```text
                User
                  │
                  ▼
           In-Memory Cache
                  │
          Hit? ─────────► Return
                  │
                 Miss
                  ▼
               Redis
                  │
          Hit? ─────────► Return
                  │
                 Miss
                  ▼
          Semantic Cache
                  │
          Hit? ─────────► Return
                  │
                 Miss
                  ▼
                 LLM
```

This minimizes expensive model calls.

---

# Cache Invalidation

A cache should not live forever.

Common strategies:

### Time-based

```
Expire after 1 hour
```

### Version-based

```text
Knowledge Base Version

↓

v1

↓

v2

↓

Invalidate Cache
```

### Event-based

```
Document Updated

↓

Delete Related Cache
```

---

# What Should NOT Be Cached?

Avoid caching:

* Personalized responses that include user-specific data.
* Responses containing sensitive or confidential information.
* Highly dynamic data (e.g., stock prices, weather, live sports scores) unless the TTL is very short.
* Requests where freshness is critical.

---

# Cache Keys

Instead of using only the prompt:

```
"What is Python?"
```

Use a richer key:

```text
model
+
system prompt version
+
retrieval version
+
user locale
+
normalized prompt
```

Example:

```python
key = hashlib.sha256(
    f"{model}:{system_prompt_version}:{prompt}".encode()
).hexdigest()
```

This prevents serving responses generated under different configurations.

---

# Cache Hit Rate

Measure:

```
Cache Hits
──────────────
Total Requests
```

Example:

```
80 Hits

20 Misses

↓

80%
```

Monitor:

* Cache hit rate
* Average cache lookup time
* Memory usage
* Eviction rate

---

# Production Example

```text
                     User
                       │
                       ▼
                 API Gateway
                       │
                       ▼
             Prompt Normalizer
                       │
                       ▼
              In-Memory Cache
                       │
                 Cache Miss
                       ▼
                   Redis
                       │
                 Cache Miss
                       ▼
               Semantic Cache
                       │
                 Cache Miss
                       ▼
               Retrieve Docs
                       │
                       ▼
                     LLM
                       │
                       ▼
              Store at All Layers
```

---

# Example with Redis + FAISS

```python
class CacheManager:

    def get(self, prompt):
        # 1. Check Redis
        # 2. Check Semantic Cache
        # 3. Return cached response

    def put(self, prompt, answer):
        # Store in Redis
        # Store embedding in FAISS
```

The application simply calls:

```python
answer = cache_manager.get(prompt)

if answer is None:
    answer = llm(prompt)
    cache_manager.put(prompt, answer)
```

---

# What Senior AI Teams Do

A typical production caching strategy looks like this:

| Layer | Technology                                           | Purpose                                      |
| ----- | ---------------------------------------------------- | -------------------------------------------- |
| L1    | In-memory (LRU cache)                                | Microsecond lookups for hot data             |
| L2    | Redis                                                | Shared cache across application instances    |
| L3    | Semantic cache (FAISS, Qdrant, Milvus, Redis Vector) | Reuse answers for similar prompts            |
| L4    | Retrieval cache                                      | Cache vector search and reranking results    |
| L5    | Embedding cache                                      | Avoid recomputing embeddings                 |
| L6    | CDN/Edge cache                                       | Cache static assets and public API responses |

## Best Practices

1. **Normalize prompts** (trim whitespace, standardize formatting) before hashing.
2. **Include model and prompt version** in cache keys.
3. **Use TTLs** appropriate to the data freshness requirements.
4. **Separate caches** for retrieval, embeddings, and LLM responses.
5. **Measure cache hit rate** and tune TTLs accordingly.
6. **Never cache sensitive or user-specific responses** unless they are isolated per user and protected by proper access controls.
7. **Use semantic caching** for natural language applications where users often ask the same question in different words.

A well-designed caching layer often provides the largest return on investment for production LLM systems because it simultaneously reduces **latency**, **API costs**, and **load on downstream services**.
