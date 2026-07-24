# Build a Production-Grade Document Chunker for RAG

A **chunker** is responsible for splitting large documents into smaller, meaningful pieces before generating embeddings.

Without good chunking:

* Retrieval quality drops
* LLM hallucinates more
* Context window is wasted
* Important information is missed

Many RAG failures are caused by **poor chunking**, not poor embeddings or LLMs.

---

# Why Do We Need Chunking?

Suppose we have a 200-page PDF.

```text
Python Guide (200 pages)

↓

One huge document

↓

Embedding
```

Problems:

* Embedding loses local context.
* Vector search retrieves the entire document.
* LLM context window is wasted.
* Retrieval becomes inaccurate.

Instead, split it into smaller chunks.

```text
Python Guide

↓

Chunk 1
Chunk 2
Chunk 3
...
Chunk N

↓

Embedding Each Chunk

↓

Vector Database
```

Now the retriever returns only the relevant sections.

---

# Production RAG Pipeline

```text
PDF
 │
 ▼
Loader
 │
 ▼
Cleaner
 │
 ▼
Chunker
 │
 ▼
Embedder
 │
 ▼
Vector DB
 │
 ▼
Retriever
 │
 ▼
LLM
```

---

# Chunk Object

Each chunk should carry metadata.

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict
```

Example

```python
Chunk(
    id="python_001",
    text="Python supports generators...",
    metadata={
        "page": 12,
        "section": "Generators",
        "source": "python.pdf"
    }
)
```

---

# Method 1 — Fixed-Size Chunking

The simplest strategy.

```python
class FixedChunker:

    def __init__(self, chunk_size=200):
        self.chunk_size = chunk_size

    def chunk(self, text):

        words = text.split()

        chunks = []

        for i in range(0, len(words), self.chunk_size):

            chunk = " ".join(
                words[i:i+self.chunk_size]
            )

            chunks.append(chunk)

        return chunks
```

Example

```python
text = " ".join(
    ["Python"] * 1000
)

chunker = FixedChunker(100)

chunks = chunker.chunk(text)

print(len(chunks))
```

Output

```text
10
```

---

# Problem with Fixed Chunking

Suppose:

```text
Chunk 1

Introduction...

Neural Networks...

Transformers

Chunk Ends
```

Chunk 2

```text
continue Transformer architecture...
```

The sentence is split in half.

Semantic meaning is lost.

---

# Method 2 — Overlapping Chunking

Solution:

```text
Chunk 1

Words 1-200

Chunk 2

Words 150-350

Chunk 3

Words 300-500
```

Some content appears in both chunks.

Implementation:

```python
class OverlapChunker:

    def __init__(
        self,
        chunk_size=200,
        overlap=50
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = start + self.chunk_size

            chunk = " ".join(
                words[start:end]
            )

            chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks
```

This preserves context across chunk boundaries.

---

# Method 3 — Recursive Chunking

Instead of splitting by word count, split on increasingly smaller separators.

Priority:

```text
Document

↓

Chapter

↓

Section

↓

Paragraph

↓

Sentence

↓

Words
```

Implementation

```python
class RecursiveChunker:

    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size

    def chunk(self, text):

        if len(text) <= self.chunk_size:
            return [text]

        for separator in [
            "\n\n",
            "\n",
            ". ",
            " "
        ]:

            parts = text.split(separator)

            if len(parts) > 1:

                chunks = []
                current = ""

                for part in parts:

                    candidate = (
                        current + separator + part
                        if current else part
                    )

                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = part

                if current:
                    chunks.append(current)

                return chunks

        return [text]
```

This keeps paragraphs and sentences intact whenever possible.

---

# Method 4 — Sentence Chunking

```python
import nltk

nltk.download("punkt")

from nltk.tokenize import sent_tokenize

class SentenceChunker:

    def __init__(self, max_sentences=5):
        self.max_sentences = max_sentences

    def chunk(self, text):

        sentences = sent_tokenize(text)

        chunks = []

        for i in range(
            0,
            len(sentences),
            self.max_sentences
        ):

            chunks.append(
                " ".join(
                    sentences[i:i+self.max_sentences]
                )
            )

        return chunks
```

Ideal for legal, academic, and news documents.

---

# Method 5 — Semantic Chunking

Instead of splitting by length, split where the topic changes.

Pipeline

```text
Sentence Embeddings

↓

Similarity

↓

Topic Change

↓

New Chunk
```

Pseudo-code

```python
embeddings = embed(sentences)

for i in range(len(sentences)-1):

    similarity = cosine(
        embeddings[i],
        embeddings[i+1]
    )

    if similarity < 0.75:
        start_new_chunk()
```

This produces chunks aligned with semantic boundaries.

---

# Production Chunker

```python
class Chunker:

    def __init__(
        self,
        chunk_size=500,
        overlap=100
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text,
        metadata
    ):

        words = text.split()

        chunks = []

        index = 0
        start = 0

        while start < len(words):

            end = start + self.chunk_size

            chunk_text = " ".join(
                words[start:end]
            )

            chunks.append(
                Chunk(
                    id=f"chunk_{index}",
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "chunk": index
                    }
                )
            )

            index += 1
            start += self.chunk_size - self.overlap

        return chunks
```

Example

```python
metadata = {
    "source": "python.pdf",
    "page": 12
}

chunks = Chunker().chunk(
    document_text,
    metadata
)
```

Each chunk retains its source information for citations and debugging.

---

# Integrate with the Retriever

```python
chunks = chunker.chunk(
    document_text,
    metadata
)

for chunk in chunks:

    embedding = model.encode(
        chunk.text,
        normalize_embeddings=True
    )

    vector_db.add(
        chunk,
        embedding
    )
```

Now the vector database stores chunks instead of entire documents.

---

# Chunk Size Trade-offs

| Chunk Size  | Advantages        | Disadvantages                                        |
| ----------- | ----------------- | ---------------------------------------------------- |
| 100 words   | Precise retrieval | Loses context                                        |
| 300 words   | Good balance      | Moderate embedding cost                              |
| 500 words   | Rich context      | Lower retrieval precision                            |
| 1000+ words | Fewer embeddings  | May exceed LLM context and reduce retrieval accuracy |

Typical production settings:

* Chunk size: **300–800 tokens**
* Overlap: **10–20% of chunk size**

Choose values based on your embedding model, document type, and LLM context window.

---

# Production Folder Structure

```text
chunking/
│
├── chunker.py
├── semantic_chunker.py
├── recursive_chunker.py
├── sentence_chunker.py
├── overlap_chunker.py
├── tokenizer.py
├── metadata.py
├── tests/
└── benchmark.py
```

---

# End-to-End Pipeline

```text
PDF
 │
 ▼
Document Loader
 │
 ▼
Text Cleaning
 │
 ▼
Recursive Chunker
 │
 ▼
Overlap Chunking
 │
 ▼
Embedding Model
 │
 ▼
Vector Database
 │
 ▼
Retriever
 │
 ▼
Cross-Encoder Reranker
 │
 ▼
Prompt Builder
 │
 ▼
LLM
```

---

# Senior AI Engineer Best Practices

A production chunking system should:

* Preserve document structure (headings, paragraphs, tables)
* Use **token-aware** chunk sizes instead of character counts
* Apply **10–20% overlap** to preserve context
* Store rich metadata (source, page, section, chunk ID)
* Support different strategies for PDFs, HTML, Markdown, and code
* Avoid splitting code blocks, tables, or lists in the middle
* Use semantic or recursive chunking for long-form content
* Benchmark retrieval quality (Recall@K, Context Precision, Faithfulness) when tuning chunk size and overlap

A well-designed chunker often has a greater impact on RAG accuracy than changing the embedding model, because retrieval quality starts with how documents are segmented.
