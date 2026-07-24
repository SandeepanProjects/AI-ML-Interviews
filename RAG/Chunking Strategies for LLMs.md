Chunking is one of the **most important design decisions in a RAG (Retrieval-Augmented Generation) system**. In production AI systems, poor chunking usually causes worse retrieval than using a better embedding model.

A senior AI engineer doesn't ask:

> "Which embedding model should I use?"

They first ask:

> "How should I split my documents?"

---

# Why Chunking Exists

LLMs cannot process entire documents efficiently.

For example,

```
PDF (500 pages)

↓

Embedding Model

↓

Vector Database
```

We cannot embed the whole PDF as one vector.

Instead we split it.

```
Document

↓

Chunk 1
Chunk 2
Chunk 3
Chunk 4
...

↓

Embedding each chunk

↓

Store in Vector DB
```

Later, retrieval happens at chunk level.

---

# Ideal Chunk

A chunk should satisfy two properties.

It should be

* semantically meaningful
* small enough for precise retrieval

Bad chunk

```
The quick brown fox
jumped over...
```

Sentence broken halfway.

Good chunk

```
The quick brown fox jumped over the lazy dog.
It was chasing food across the field.
```

Complete thought.

---

# Real World Chunking Strategies

Companies rarely use only one strategy.

Common production strategies are

```
1. Fixed-size chunking
2. Sliding window
3. Recursive chunking
4. Sentence chunking
5. Paragraph chunking
6. Semantic chunking
7. Document-aware chunking
8. Hierarchical chunking
9. Agentic chunking
10. Hybrid chunking
```

Let's study each.

---

# 1 Fixed Size Chunking

Simplest.

Example

```
Document

1000 words

↓

Every 200 words

↓

Chunk1
Chunk2
Chunk3
Chunk4
Chunk5
```

Python

```python
def fixed_chunk(text, chunk_size=200):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(
            " ".join(words[i:i+chunk_size])
        )

    return chunks
```

Output

```
Chunk1
Words 0-199

Chunk2
Words 200-399

Chunk3
Words 400-599
```

Advantages

* simple
* fast
* scalable

Disadvantages

Cuts meaning.

Example

```
Machine learning is...

(chunk ends)

...used for prediction.
```

Meaning destroyed.

---

# 2 Sliding Window Chunking

Very common in production.

Instead of

```
0-200

201-400
```

We overlap.

```
0-200

150-350

300-500
```

Visualization

```
---------------Chunk1---------------

            ---------------Chunk2---------------

                         ---------------Chunk3-------------
```

Python

```python
def sliding_chunks(words,
                   size=200,
                   overlap=50):

    chunks=[]

    start=0

    while start < len(words):

        end=start+size

        chunks.append(
            " ".join(words[start:end])
        )

        start += size-overlap

    return chunks
```

Why overlap?

Without overlap

```
Question:

What happened after Chapter 2?
```

Suppose answer begins

```
last sentence of chunk1

first sentence of chunk2
```

Retriever misses it.

Overlap solves this.

Production overlap

```
10%

20%

25%
```

---

# 3 Recursive Chunking

Most popular in LangChain.

Idea

Split by

```
Paragraph

↓

Sentence

↓

Word

↓

Character
```

until chunk fits.

Pseudo

```
if paragraph too large

↓

split into sentences

↓

if sentence too large

↓

split into words

↓

if still large

↓

characters
```

Much better than fixed size.

---

# 4 Sentence Chunking

Each chunk contains complete sentences.

Example

```
Sentence1

Sentence2

Sentence3
```

↓

```
Chunk

Sentence1

Sentence2

Sentence3
```

Never breaks grammar.

Excellent for

* QA
* legal
* research

---

# 5 Paragraph Chunking

One paragraph = one chunk.

Example

```
Paragraph1

↓

Chunk1

Paragraph2

↓

Chunk2
```

Works well because authors usually organize one idea per paragraph.

Great for

* books
* blogs
* documentation

---

# 6 Semantic Chunking

One of the best methods.

Instead of fixed size,

we detect topic change.

Example

```
History of AI

History continues

History continues

↓

Topic changes

↓

Deep Learning

↓

Topic changes

↓

Transformers
```

Chunks become

```
Chunk1

History

-----------------

Chunk2

Deep Learning

-----------------

Chunk3

Transformers
```

Instead of

```
Every 300 words
```

we split by meaning.

Implementation idea

```
Sentence embeddings

↓

Similarity between adjacent sentences

↓

Large similarity drop

↓

New chunk
```

Python

```python
from sklearn.metrics.pairwise import cosine_similarity

# embeddings of sentences

similarity = cosine_similarity(
    embedding1.reshape(1,-1),
    embedding2.reshape(1,-1)
)
```

Low similarity

↓

Start new chunk.

This is widely used.

---

# 7 Document-Aware Chunking

Different documents need different rules.

Example

Legal Contract

```
Section 1

Section 2

Clause

Sub Clause
```

Chunk by

```
Section
```

Research paper

```
Abstract

Introduction

Methods

Results

Conclusion
```

Chunk by headings.

Markdown

```
#

##

###
```

Chunk by heading.

HTML

```
<h1>

<h2>

<h3>
```

Chunk by DOM.

Production systems always preserve structure.

---

# 8 Hierarchical Chunking

Store two levels.

```
Large Parent Chunk

↓

Small Child Chunks
```

Example

```
Chapter

↓

Sections

↓

Paragraphs
```

Retrieve

```
small chunk

↓

expand to parent
```

Example

```
Question

↓

Paragraph retrieved

↓

Return whole section
```

Improves context.

---

# 9 Agentic Chunking

Newest approach.

LLM decides.

Example

```
Read document

↓

Find concepts

↓

Create logical chunks
```

Prompt

```
Split this document into
self-contained knowledge units.
```

Very expensive.

Mostly used

* enterprise search
* law
* medicine

---

# 10 Hybrid Chunking

Most companies use hybrid.

Example

```
Split by Heading

↓

Paragraph

↓

Recursive

↓

Sliding Window
```

Example pipeline

```
Markdown

↓

Heading

↓

Paragraph

↓

Recursive

↓

Overlap

↓

Embedding
```

This gives best retrieval.

---

# Chunk Size Tradeoffs

This is probably the most common interview topic.

Imagine a document of 10,000 words.

## Small Chunks (100 tokens)

```
Chunk1

100 tokens

Chunk2

100 tokens
```

Advantages

✅ Highly precise retrieval

Retriever gets only relevant text.

Disadvantages

❌ Missing context

Example

```
Question

Explain OAuth authentication.

Chunk

"Refresh token"

Only contains

Refresh token

No explanation.
```

Model hallucinates.

---

## Large Chunks (1000 tokens)

```
Chunk

Entire chapter
```

Advantages

Lots of context.

Disadvantages

Retriever returns

```
1000 tokens

Only 50 useful.
```

Noise increases.

Embedding becomes average of many topics.

Similarity decreases.

---

# Visual Comparison

```
Small Chunk

[Topic A]

[Topic B]

[Topic C]

Precise
```

vs

```
Large Chunk

Topic A

Topic B

Topic C

Mixed together

Embedding represents everything.
```

---

# Retrieval Precision

Assume

```
Book

AI

ML

Python

Databases
```

Question

```
Explain indexing.
```

Small chunk

```
Database indexing only

Similarity

0.95
```

Large chunk

```
AI

ML

Python

Databases

Similarity

0.61
```

Embedding diluted.

---

# Storage Tradeoff

Suppose

```
100-page document
```

Chunk size

100 words

```
400 chunks
```

Chunk size

500 words

```
80 chunks
```

More chunks

↓

More vectors

↓

Higher storage

↓

Higher indexing cost

---

# Latency Tradeoff

More chunks

↓

Larger vector database

↓

Longer ANN search

↓

Higher latency

Modern vector databases like FAISS, HNSW, Qdrant, Milvus, and Pinecone are optimized for millions of vectors, but chunk size still affects index size, memory usage, and retrieval latency.

---

# Cost Tradeoff

Smaller chunks often require retrieving more results (`top_k`) to reconstruct enough context.

Example:

```
Chunk size = 100 tokens
top_k = 10

→ ~1,000 tokens sent to the LLM
```

Versus:

```
Chunk size = 500 tokens
top_k = 3

→ ~1,500 tokens sent to the LLM
```

Although fewer chunks are retrieved in the second case, each is larger. The optimal choice depends on the balance between retrieval precision and the prompt token budget.

---

# Real-World Recommendations

| Use Case                           | Chunking Strategy                     | Chunk Size (approx.) |         Overlap |
| ---------------------------------- | ------------------------------------- | -------------------: | --------------: |
| Documentation (APIs, product docs) | Recursive + heading-aware             |       300–600 tokens |          10–20% |
| Technical manuals                  | Recursive + paragraph                 |       400–800 tokens |          15–20% |
| Source code                        | Function/class-aware                  |       150–400 tokens | Minimal (0–10%) |
| Legal contracts                    | Clause/section-aware                  |     500–1,000 tokens |          10–15% |
| Research papers                    | Section-aware + semantic              |       300–700 tokens |          15–20% |
| Chat logs                          | Conversation turn-aware               |       200–500 tokens |    Usually none |
| FAQs                               | One Q&A per chunk                     |     Varies by answer |            None |
| Books                              | Chapter → paragraph → recursive       |       300–600 tokens |          15–25% |
| Enterprise knowledge bases         | Hybrid (heading + semantic + overlap) |       400–800 tokens |          10–20% |

---

# What Senior AI Engineers Do in Production

A production ingestion pipeline typically looks like this:

```text
PDF / DOCX / HTML / Markdown
            │
            ▼
Document Parsing
            │
            ▼
Structure Detection
(headings, tables, lists, code blocks)
            │
            ▼
Recursive / Semantic Chunking
            │
            ▼
Sliding Window Overlap
            │
            ▼
Metadata Attachment
(source, page, section, title, timestamps)
            │
            ▼
Embedding Generation
            │
            ▼
Vector Database (Qdrant / Milvus / Pinecone / FAISS)
```

Rather than choosing a single chunk size for all documents, production systems often **adapt the chunking strategy to the document type** and **evaluate retrieval quality** using metrics such as Recall@k, Precision@k, Mean Reciprocal Rank (MRR), and answer faithfulness. The "best" chunk size is therefore not universal—it's the one that yields the highest end-to-end retrieval and answer quality for your specific corpus and application.
