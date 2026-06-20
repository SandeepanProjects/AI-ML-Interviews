This is one of the **most important RAG interview topics**.

Many candidates say:

> "Chunking means splitting documents."

That is only **10% of the answer**.

A Senior AI Engineer should explain:

* Why chunking exists
* Why chunk size matters
* Why overlap is needed
* Different chunking strategies
* Tradeoffs
* Which chunking strategy to use
* Production considerations
* Implementation from scratch

---

# Why Do We Need Chunking?

Suppose we have a PDF

```text
Employee Handbook

300 Pages

250,000 words
```

Can we send this directly to an LLM?

```text
PDF
   │
   ▼
LLM
```

No.

Reasons:

* Context window limitation
* Slow inference
* Expensive
* Poor retrieval quality

Instead we split the document.

```text
Employee Handbook

↓

Chunk 1

↓

Chunk 2

↓

Chunk 3

↓

Chunk 4
```

Each chunk becomes an embedding.

---

# Complete RAG Flow

```text
PDF
 │
 ▼
Extract Text
 │
 ▼
Chunking
 │
 ▼
Embedding
 │
 ▼
Vector Database
 │
 ▼
Retriever
 │
 ▼
LLM
```

Notice

Embedding happens **after chunking**.

---

# Why Not Embed Entire PDF?

Suppose document

```text
Employee Handbook

200 pages
```

Embedding

↓

```text
One Vector
```

Now user asks

```text
How many annual leaves?
```

Retriever compares

```text
Question

↓

Entire Handbook
```

Very poor retrieval.

One vector represents

everything.

Instead

```text
Handbook

↓

Leave Policy

↓

Insurance

↓

Benefits

↓

Travel Policy
```

Each section gets its own embedding.

Now retrieval becomes accurate.

---

# Example

Document

```text
Page 1

Employees receive
20 annual leaves.

Page 2

Medical insurance
covers family.

Page 3

Travel reimbursement
is allowed.
```

If stored as one chunk

Embedding

↓

```text
Leave

Insurance

Travel

All mixed together
```

Similarity becomes noisy.

Chunking separates topics.

---

# Types of Chunking

There are four commonly discussed approaches.

```text
1. Fixed Chunking

2. Recursive Chunking

3. Sliding Window Chunking

4. Semantic Chunking
```

Let's study each.

---

# 1. Fixed Chunking

This is the simplest method.

Rule

```text
Every N characters

or

Every N tokens
```

Suppose

Chunk Size

```text
500 characters
```

Document

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZ...
```

Split

```text
Chunk 1

500 chars

------------

Chunk 2

500 chars

------------

Chunk 3

500 chars
```

Very simple.

---

## Visualization

```text
Document

──────────────────────────────

↓

500

↓

500

↓

500

↓

500
```

---

## Code

```python
text = """
Employees receive
20 annual leaves.

Medical leave
is separate.

Travel policy
...
"""

chunk_size = 50

chunks = []

for i in range(0, len(text), chunk_size):

    chunks.append(text[i:i+chunk_size])

print(chunks)
```

---

## Problems

Sentence may break.

Example

```text
Chunk 1

Employees receive
20 annual

------------------

Chunk 2

leaves every year.
```

Meaning destroyed.

---

Advantages

* Very fast
* Easy

Disadvantages

* Breaks sentences
* Breaks context
* Poor retrieval

---

Production?

Rarely used alone.

---

# 2. Recursive Chunking

Most commonly used in production.

Instead of

cutting blindly

Recursive Chunking

tries

```text
Paragraph

↓

Sentence

↓

Line

↓

Word
```

It only splits at smaller units if necessary.

---

Suppose

Document

```text
Paragraph 1

Employees receive
20 annual leaves.

Paragraph 2

Insurance policy...

Paragraph 3

Travel policy...
```

Recursive Chunker

tries

```text
Split by Paragraph
```

If paragraph too big

↓

```text
Split by Sentence
```

Still too big

↓

```text
Split by Space
```

Still too big

↓

```text
Split Character
```

---

Visualization

```text
Document

↓

Paragraph

↓

Sentence

↓

Word

↓

Character
```

---

Why?

Keeps meaning intact.

---

Code

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)

chunks = splitter.split_text(text)
```

Internally

```text
Try

"\n\n"

↓

"\n"

↓

"."

↓

" "

↓

Character
```

Very intelligent.

---

Advantages

* Preserves sentences
* Better retrieval
* Most popular
* Production ready

---

Disadvantages

Slightly slower.

---

# 3. Sliding Window Chunking

One of the most important concepts.

Suppose

Chunk Size

```text
500
```

Without overlap

```text
Chunk1

Employees receive
20 annual

-----------------

Chunk2

leaves every year.
```

Sentence breaks.

Now

Use overlap.

Chunk Size

```text
500
```

Overlap

```text
100
```

Result

```text
Chunk1

Employees receive
20 annual leaves.

------------------

Chunk2

20 annual leaves.

Unused leaves...

------------------

Chunk3

Unused leaves
can carry...
```

Notice

Information repeats.

---

Visualization

```text
Chunk1

AAAAAAAAAA

BBBBBBBBBB

CCCCCCCCCC

-------------

Chunk2

CCCCCCCCCC

DDDDDDDDDD

EEEEEEEEEE
```

Overlap

```text
CCCCCCCCCC
```

appears twice.

---

Why?

Suppose

Sentence

```text
Employees receive
20 annual leaves.

Unused leaves...
```

Without overlap

Sentence broken.

Overlap preserves context.

---

Code

```python
splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)
```

Very common interview question

> Why overlap?

Answer

To preserve context across chunk boundaries and improve retrieval accuracy.

---

Advantages

* Better semantic continuity
* Better embeddings
* Better retrieval

Disadvantages

* More storage
* Duplicate embeddings
* Slightly slower indexing

---

# 4. Semantic Chunking

This is becoming the preferred method for enterprise RAG.

Instead of

splitting every

500 tokens

It asks

```text
Does the topic change?
```

Example

Document

```text
Leave Policy

Employees receive...

Employees may...

Carry forward...

Insurance

Medical insurance...

Dental insurance...

Travel Policy

Flights...

Hotels...
```

Semantic Chunker

creates

```text
Chunk 1

Leave Policy

------------

Chunk 2

Insurance

------------

Chunk 3

Travel Policy
```

Instead of fixed size.

---

How?

Usually

Sentence Embeddings.

Compute similarity.

```text
Sentence1

↓

Embedding

Sentence2

↓

Embedding

Similarity

↓

High?

Same chunk

Low?

New chunk
```

---

Visualization

```text
Sentence A

↓

Embedding

↓

0.98

↓

Sentence B

↓

Embedding

↓

0.95

↓

Sentence C

↓

Embedding

↓

0.18

↓

New Chunk
```

---

Pseudo Code

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

sentences = [
    "Employees receive annual leave.",
    "Unused leave can be carried forward.",
    "Medical insurance covers family."
]

embeddings = model.encode(sentences)
```

Compute

Cosine Similarity

If similarity

```text
> 0.8

Same Chunk
```

Else

```text
New Chunk
```

---

Advantages

* Best retrieval
* Topic-aware
* Better context
* Fewer irrelevant chunks

Disadvantages

* Slower
* More expensive
* Requires embeddings during indexing

---

# Comparison

Suppose document

```text
Leave Policy

Employees receive 20 annual leaves.

Insurance

Medical insurance covers family.

Travel

Flights reimbursed.
```

### Fixed

```text
Chunk1

Leave Policy
Employees receive

------------

Chunk2

20 annual
Insurance

------------

Chunk3

Medical insurance...
```

Broken.

---

### Recursive

```text
Chunk1

Leave Policy
Employees receive...

------------

Chunk2

Insurance...

------------

Chunk3

Travel...
```

Better.

---

### Sliding Window

```text
Chunk1

Leave Policy
Employees receive...

------------

Chunk2

Employees receive...
Insurance...
```

Context preserved.

---

### Semantic

```text
Chunk1

Entire Leave Policy

------------

Chunk2

Entire Insurance

------------

Chunk3

Entire Travel
```

Best.

---

# Which Chunking Should You Use?

| Method         | Speed | Retrieval Quality | Production Usage      |
| -------------- | ----- | ----------------- | --------------------- |
| Fixed          | ⭐⭐⭐⭐⭐ | ⭐                 | Rare                  |
| Recursive      | ⭐⭐⭐⭐  | ⭐⭐⭐⭐              | Very Common           |
| Sliding Window | ⭐⭐⭐   | ⭐⭐⭐⭐⭐             | Extremely Common      |
| Semantic       | ⭐⭐    | ⭐⭐⭐⭐⭐             | Modern Enterprise RAG |

---

# Production Recommendation

In real enterprise systems, a single strategy is rarely enough.

A common production pipeline is:

```text
PDF
   │
   ▼
Recursive Chunking
   │
   ▼
Sliding Window Overlap
   │
   ▼
Semantic Validation
   │
   ▼
Embeddings
   │
   ▼
Vector Database
```

Typical configuration:

* **Chunk size:** 300–800 tokens (depends on the embedding model and document type)
* **Overlap:** 50–150 tokens (10–20% overlap is a common starting point)
* **Metadata:** document ID, page number, section heading, source URL, timestamp
* **Chunk boundaries:** Prefer paragraphs and sentences over arbitrary token counts

---

# Senior AI Engineer Interview Answer (3–5 Minutes)

> "Chunking is the process of splitting large documents into smaller, semantically meaningful units before generating embeddings. Since embedding models produce a single vector per chunk, the quality of chunking directly impacts retrieval accuracy. Fixed chunking splits documents into equal-sized pieces but often breaks sentences and loses context. Recursive chunking improves this by attempting to split on natural boundaries such as paragraphs and sentences before falling back to smaller separators, making it the most common production approach. Sliding window chunking introduces overlap between adjacent chunks so that information crossing chunk boundaries is preserved, significantly improving retrieval quality. Semantic chunking goes a step further by grouping text based on meaning rather than size, typically using sentence embeddings and similarity thresholds to detect topic changes. In production RAG systems, I usually combine recursive chunking with overlapping windows, and for high-value knowledge bases, I use semantic chunking to maximize retrieval precision while preserving document context.
