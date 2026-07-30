# Chunking Strategies Explained End-to-End (with Production Code)

**Chunking** is one of the **most important parts of a RAG system**.

A poor chunking strategy can reduce retrieval quality dramatically, even if you use the best embedding model and LLM.

A common production saying is:

> **Garbage chunks → Garbage retrieval → Garbage answers**

---

# What is Chunking?

Chunking is the process of **splitting large documents into smaller, meaningful pieces** before generating embeddings.

Instead of embedding an entire document, you embed many smaller chunks.

```text
Large PDF (200 pages)
        │
        ▼
Split into chunks
        │
        ▼
Chunk 1
Chunk 2
Chunk 3
...
Chunk N
        │
        ▼
Generate embeddings
        │
        ▼
Store in Vector Database
```

---

# Why Do We Need Chunking?

Suppose we have a 100-page employee handbook.

```
Employee_Handbook.pdf
```

Question:

> What is the maternity leave policy?

### Option 1: Embed the entire document

```text
100-page PDF
      │
      ▼
One embedding
```

Problems:

* The embedding represents the entire document.
* Fine-grained information is lost.
* Retrieval quality is poor.
* Context sent to the LLM becomes too large.

---

### Option 2: Chunk first

```text
100-page PDF

↓

Chunk 1
Chunk 2
Chunk 3
...
Chunk 300

↓

Embeddings

↓

Vector Database
```

Now only the relevant chunk is retrieved.

---

# Complete RAG Pipeline

```text
PDF

↓

Document Loader

↓

Chunking

↓

Embedding

↓

Vector Database

↓

Similarity Search

↓

Reranking

↓

LLM
```

Chunking is the first major preprocessing step.

---

# Example

Document:

```text
Page 1:
Company Overview

Page 2:
Office Rules

Page 3:
Leave Policy

Page 4:
Health Insurance

Page 5:
Travel Policy
```

Question:

> How many maternity leave days?

Without chunking:

```
Entire PDF
```

With chunking:

```
Chunk 3

↓

Leave Policy
```

The retriever returns only the relevant content.

---

# Strategy 1: Fixed-Size Chunking

The simplest approach.

```
Document

↓

Every 500 characters

↓

Chunk 1
Chunk 2
Chunk 3
```

---

### LangChain Code

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

chunks = splitter.split_text(document)
```

Example output:

```
Chunk 1

Chunk 2

Chunk 3
```

---

## Advantages

* Very fast
* Easy
* Simple implementation

## Disadvantages

It may split in the middle of a sentence.

Example:

```
Chunk 1

Machine learning is a field
```

```
Chunk 2

of Artificial Intelligence...
```

The sentence is broken.

---

# Strategy 2: Recursive Chunking (Most Common)

This is the default strategy in many LangChain RAG applications.

Instead of splitting blindly, it tries separators in order:

1. Paragraph
2. Sentence
3. Line
4. Space
5. Character

This preserves semantic structure better.

---

### Code

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
```

---

### Internally

```text
Paragraph

↓

Too Large?

↓

Split into Sentences

↓

Still Too Large?

↓

Split into Words

↓

Still Too Large?

↓

Split into Characters
```

This produces cleaner chunks.

---

# Why Overlap?

Suppose:

```
Sentence:

Employees receive 180 days of maternity leave.
```

Without overlap:

```
Chunk 1

Employees receive
```

```
Chunk 2

180 days of maternity leave
```

The sentence is split.

With overlap:

```
Chunk 1

Employees receive 180 days
```

```
Chunk 2

180 days of maternity leave
```

Shared text preserves context.

---

### Code

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

Typical values:

* chunk_size: 300–1000 characters or tokens (depending on your splitter)
* overlap: 10–20% of chunk size

---

# Strategy 3: Sentence Chunking

Split by sentences.

```python
import nltk

sentences = nltk.sent_tokenize(document)
```

Output:

```
Sentence 1

Sentence 2

Sentence 3
```

Useful when preserving sentence boundaries is important.

---

# Strategy 4: Paragraph Chunking

```
Paragraph

↓

Chunk
```

Example:

```python
chunks = document.split("\n\n")
```

Simple but may produce chunks that are too large or too small.

---

# Strategy 5: Token-Based Chunking

LLMs operate on **tokens**, not characters.

Instead of:

```
500 characters
```

Use:

```
300 tokens
```

This better aligns with model context limits.

---

### LangChain Code

```python
from langchain_text_splitters import (
    TokenTextSplitter
)

splitter = TokenTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(document)
```

---

# Strategy 6: Semantic Chunking

Instead of fixed sizes, split when the topic changes.

Example:

```
Chapter 1

Machine Learning

↓

Chunk
```

```
Chapter 2

Neural Networks

↓

Chunk
```

Rather than splitting every 500 characters.

This often yields more coherent chunks for retrieval.

---

### Conceptual Pipeline

```text
Sentence Embeddings

↓

Similarity

↓

Topic Change?

↓

Create New Chunk
```

---

# Strategy 7: Document-Aware Chunking

Treat different document structures differently.

Example:

```
PDF

↓

Tables

Images

Paragraphs

Headers
```

Split each element according to its type.

This is especially useful for enterprise documents.

---

# Strategy 8: Markdown Chunking

Markdown has natural structure.

```
# HR

##

###

```

Split by headings.

Example:

```python
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter
)

headers = [
    ("#", "H1"),
    ("##", "H2"),
]

splitter = MarkdownHeaderTextSplitter(headers)
```

---

# Strategy 9: Code Chunking

Never split code arbitrarily.

Instead:

```
Function

↓

Chunk
```

Better:

```
Class

↓

Methods

↓

Chunks
```

Useful for code assistants.

---

# Strategy 10: Sliding Window Chunking

```
Chunk 1

1–500
```

```
Chunk 2

400–900
```

```
Chunk 3

800–1300
```

Each chunk overlaps with the next.

---

# Which Chunk Size Should You Use?

It depends on the content.

| Content         | Suggested Chunking               |
| --------------- | -------------------------------- |
| PDFs            | Recursive + overlap              |
| Policies        | Paragraph-aware                  |
| Documentation   | Recursive or Markdown            |
| Source Code     | Syntax-aware (functions/classes) |
| HTML            | DOM/section-aware                |
| Books           | Chapter + paragraph              |
| Legal contracts | Clause-aware                     |
| Tables          | Table-aware parsing              |

There is no universal chunk size; evaluate retrieval quality for your use case.

---

# Production Example

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

loader = PyPDFLoader("employee_handbook.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    url="http://localhost:6333",
    collection_name="employee_docs"
)
```

Now each chunk has its own embedding.

---

# Chunking Flow

```text
PDF

↓

Chunk 1
Chunk 2
Chunk 3
Chunk 4

↓

Embedding

↓

Vector Database
```

---

# Common Mistakes

## Embedding Entire PDFs

❌

```
300-page PDF

↓

One embedding
```

Retrieval quality suffers.

---

## Tiny Chunks

```
20 characters
```

Problems:

* Missing context
* Poor semantic representation
* Many unnecessary vectors

---

## Huge Chunks

```
5000 tokens
```

Problems:

* Mixed topics
* Lower retrieval precision
* Higher embedding cost

---

## No Overlap

```
Sentence split in half
```

Important context may be lost at chunk boundaries.

---

# Advanced Enterprise Pipeline

```text
PDF

↓

OCR (if needed)

↓

Layout Parser

↓

Header Detection

↓

Table Detection

↓

Recursive Chunking

↓

Metadata Enrichment

↓

Embedding

↓

Vector Database

↓

Hybrid Search

↓

Reranker

↓

LLM
```

This preserves document structure and improves retrieval quality.

---

# Common Interview Questions

### Why not embed the entire document?

A single embedding averages the semantics of the whole document, making it difficult to retrieve specific sections accurately.

---

### Why use overlap?

Overlap preserves context when information spans chunk boundaries, reducing the chance of splitting important ideas.

---

### What is the best chunk size?

There is no universal answer. A good starting point is a few hundred tokens with 10–20% overlap, then optimize using retrieval evaluation metrics such as context precision and recall.

---

### Why is recursive chunking popular?

It preserves natural document boundaries by attempting to split on paragraphs and sentences before falling back to smaller separators, producing more coherent chunks than fixed-size splitting.

---

# Chunking Strategy Comparison

| Strategy    | Pros                   | Cons                  | Best Use Case                  |
| ----------- | ---------------------- | --------------------- | ------------------------------ |
| Fixed-size  | Simple, fast           | Breaks sentences      | Small demos                    |
| Recursive   | Preserves structure    | Slightly slower       | General RAG (most common)      |
| Token-based | Matches LLM context    | Tokenization overhead | Production LLMs                |
| Sentence    | High readability       | Variable chunk sizes  | QA systems                     |
| Paragraph   | Keeps logical sections | Uneven sizes          | Policies, reports              |
| Markdown    | Uses headings          | Markdown only         | Technical documentation        |
| Code-aware  | Preserves syntax       | Language-specific     | Code assistants                |
| Semantic    | Topic-aware            | More compute          | High-quality enterprise search |

---

# Senior AI Engineer Interview Answer

> **Chunking is the process of splitting large documents into smaller, semantically meaningful units before generating embeddings. This improves retrieval quality because embeddings represent focused pieces of information instead of entire documents. In production, I typically use recursive or token-based chunking with overlap to preserve context across chunk boundaries. The optimal strategy depends on the document type—for example, syntax-aware chunking for source code, heading-based chunking for Markdown, and clause-aware chunking for legal documents. After chunking, each chunk is embedded, stored in a vector database, retrieved through semantic or hybrid search, optionally reranked, and then provided to the LLM as grounded context. This approach leads to significantly more accurate and efficient RAG systems.
