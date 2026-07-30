# Embeddings Explained End-to-End (with Code)

**Embeddings** are one of the most important concepts in AI, LLMs, RAG, semantic search, recommendation systems, and vector databases.

If you understand embeddings deeply, you understand the foundation of modern Generative AI.

---

# What is an Embedding?

An **embedding** is a **dense numerical vector** that represents the **meaning (semantic information)** of text, images, audio, or other data.

Instead of representing text as words, an embedding represents it as **numbers** in a high-dimensional space.

Example:

```text
"I love machine learning"

↓

Embedding Model

↓

[0.124, -0.381, 0.923, 0.104, ...]
```

That vector captures the sentence's meaning.

---

# Why Do We Need Embeddings?

Computers don't understand words.

For example:

```text
Cat

Dog

Car

Apple
```

To a computer, these are just character sequences.

An embedding converts them into mathematical vectors where similar concepts are close together.

---

# Traditional Keyword Search

Suppose a user searches:

> "automobile"

Document:

```text
This guide explains how to repair a car.
```

Keyword search fails because:

```text
automobile != car
```

No exact keyword match.

---

# Embedding-Based Search

Embedding model:

```text
Automobile

↓

[0.25, 0.91, ...]

Car

↓

[0.26, 0.89, ...]
```

These vectors are close together because they have similar meanings.

The search succeeds even without exact keyword matches.

---

# Visual Intuition

Imagine a 2D space (real embeddings use hundreds or thousands of dimensions).

```text
                  Vehicle

                     Truck

          Car

               Automobile

Dog

Cat


Apple
```

Similar concepts cluster together.

Real embedding models produce vectors with dimensions such as:

* 768
* 1024
* 1536
* 3072

depending on the model.

---

# Similar Sentences

```text
Sentence A:
I love programming.

Sentence B:
Coding is my favorite hobby.
```

Embeddings:

```text
A

↓

[0.12, 0.41, ...]

B

↓

[0.13, 0.40, ...]
```

Very close vectors.

---

Different meaning:

```text
The weather is rainy.
```

Embedding:

```text
[-0.82, 0.91, ...]
```

Far away.

---

# How Embeddings Are Generated

```text
Sentence

↓

Tokenizer

↓

Tokens

↓

Transformer Encoder

↓

Hidden Representations

↓

Pooling

↓

Embedding Vector
```

Unlike a chat model, an embedding model produces **one vector**, not text.

---

# Example Using LangChain

```python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector = embedding_model.embed_query(
    "What is machine learning?"
)

print(len(vector))
print(vector[:5])
```

Example output:

```python
1536

[
    0.018,
    -0.103,
    0.672,
    ...
]
```

The exact values vary by model.

---

# Embedding Multiple Documents

```python
documents = [
    "Machine learning is a branch of AI.",
    "Cats are friendly animals.",
    "Python is a programming language."
]

vectors = embedding_model.embed_documents(documents)

print(len(vectors))
```

Output:

```text
3 vectors
```

Each document gets its own embedding.

---

# Why Vectors?

Suppose:

```text
Dog

↓

[0.1, 0.2]

Cat

↓

[0.12, 0.21]

Car

↓

[-0.8, 0.6]
```

Distance:

```text
Dog <----> Cat

Small distance

Dog <-----------------------> Car

Large distance
```

Semantic similarity becomes a mathematical distance problem.

---

# Cosine Similarity

The most common similarity metric.

```python
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

v1 = [1, 2, 3]
v2 = [1, 2, 4]

print(cosine_similarity(v1, v2))
```

Output:

```text
0.99
```

Very similar.

---

Different vectors:

```python
v1 = [1, 0]
v2 = [0, 1]

print(cosine_similarity(v1, v2))
```

Output:

```text
0
```

No similarity.

---

# Embeddings in RAG

```text
PDF

↓

Chunking

↓

Embedding Model

↓

Vectors

↓

Vector Database
```

User query:

```text
Question

↓

Embedding

↓

Nearest Vectors

↓

Relevant Chunks

↓

LLM
```

The LLM never searches the database directly; it receives the retrieved chunks.

---

# Complete Example with Qdrant

```python
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

docs = [
    Document(page_content="Python is a programming language."),
    Document(page_content="Machine learning is part of AI."),
    Document(page_content="Cats are mammals.")
]

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embedding,
    url="http://localhost:6333",
    collection_name="demo"
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

results = retriever.invoke(
    "What is artificial intelligence?"
)

for doc in results:
    print(doc.page_content)
```

Likely output:

```text
Machine learning is part of AI.
Python is a programming language.
```

The first result is semantically related.

---

# Embedding Pipeline

```text
Document

↓

Chunking

↓

Embedding Model

↓

Vector

↓

Vector Database

↓

Similarity Search

↓

Relevant Chunks

↓

LLM
```

---

# Where Are Embeddings Used?

### 1. Semantic Search

```text
Search

↓

Embedding

↓

Nearest Documents
```

---

### 2. RAG

```text
Question

↓

Embedding

↓

Retrieve Context

↓

LLM
```

---

### 3. Recommendations

```text
User History

↓

Embedding

↓

Similar Products
```

---

### 4. Duplicate Detection

```text
Document A

↓

Embedding

↓

Compare

↓

Document B
```

---

### 5. Clustering

Group similar documents automatically.

---

### 6. Classification

Use embeddings as features for downstream machine learning models.

---

# Choosing an Embedding Model

Common considerations:

| Factor               | Why It Matters                                              |
| -------------------- | ----------------------------------------------------------- |
| Embedding quality    | Better semantic retrieval                                   |
| Vector dimension     | Higher dimensions may improve accuracy but increase storage |
| Latency              | Affects indexing and query speed                            |
| Cost                 | API pricing or infrastructure cost                          |
| Multilingual support | Important for global applications                           |
| Context length       | Determines how much text can be embedded at once            |

---

# Common Mistakes

### Embedding Entire PDFs

❌

```text
200 pages

↓

One embedding
```

Retrieval quality suffers.

---

### Correct

```text
PDF

↓

Chunk

↓

Embedding

↓

Store
```

Each chunk becomes independently searchable.

---

### Using Different Embedding Models

❌ Index with Model A:

```text
text-embedding-3-small
```

Query with Model B:

```text
another embedding model
```

Embedding spaces differ, so similarity scores become unreliable.

Use the same embedding model for indexing and querying.

---

### Ignoring Metadata

Store metadata alongside vectors.

```python
Document(
    page_content="Leave policy",
    metadata={
        "department": "HR",
        "source": "leave_policy.pdf",
        "page": 5
    }
)
```

Then filter during retrieval.

---

# Common Interview Questions

### Why do we need embeddings?

Embeddings convert unstructured data into vectors where semantic similarity can be measured mathematically, enabling semantic search, recommendations, clustering, and RAG.

---

### Why not use keywords?

Keyword search only matches exact words. Embeddings capture meaning, allowing queries such as "automobile" to retrieve documents about "cars."

---

### Why use cosine similarity?

Cosine similarity compares vector direction rather than magnitude, making it effective for measuring semantic similarity between embeddings.

---

### Why chunk before embedding?

Embedding entire documents produces coarse vectors and poor retrieval. Chunking creates finer-grained representations so the retriever can return only the relevant sections.

---

### Are embeddings generated by LLMs?

Embedding models are often based on transformer architectures, but they are optimized to produce vector representations rather than generate text. They are typically separate models from chat-oriented LLMs.

---

# Senior AI Engineer Interview Answer

> **Embeddings are dense vector representations that encode the semantic meaning of text, images, or other data into a high-dimensional numerical space. In RAG systems, documents are first chunked, embedded using an embedding model, and stored in a vector database. At query time, the user's question is embedded using the same model, and similarity search retrieves the closest document vectors, typically using cosine similarity or another distance metric. Those retrieved chunks are then provided to the LLM as context for grounded generation. Embeddings are fundamental to semantic search, recommendation systems, document clustering, duplicate detection, and modern retrieval-augmented AI applications.**
