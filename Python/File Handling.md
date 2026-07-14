# File Handling for Large Files (Senior AI Engineer Interview)

One of the most common Senior Python interview questions is:

> **"How would you efficiently read a 10 GB CSV file?"**

A junior engineer might answer:

```python
with open("data.csv") as f:
    data = f.read()
```

A senior engineer immediately recognizes the problem.

A **10 GB file does not fit comfortably into memory** on many systems, and even if it does, reading it all at once is wasteful.

The key principle is:

> **Never load more data than you need.**

---

# Problem

Suppose:

```text
customers.csv

Size = 10 GB
```

This code:

```python
with open("customers.csv") as f:
    data = f.read()
```

Internally:

```
Disk
   │
   ▼
Read Entire 10 GB
   │
   ▼
RAM
   │
   ▼
Python String
```

Problems:

* Huge memory usage
* Slow startup
* Possible `MemoryError`
* Garbage collection overhead
* Poor scalability

---

# Solution 1 — Streaming

Streaming means:

> Read the file incrementally instead of all at once.

Python file objects are iterators.

```python
with open("customers.csv") as f:
    for line in f:
        process(line)
```

Memory usage:

```
Disk

↓

One line

↓

Process

↓

Discard

↓

Next line
```

Memory remains almost constant.

---

## Why is this efficient?

The operating system buffers file reads.

Python requests a block of bytes.

The OS loads something like:

```
8 KB

↓

16 KB

↓

64 KB
```

instead of reading the whole file.

---

# Example

Count rows.

```python
count = 0

with open("customers.csv") as f:

    for line in f:
        count += 1

print(count)
```

Even for a 100 GB file:

Memory usage stays almost constant.

---

# AI Example

Large training corpus.

```
Wikipedia

↓

100 GB

↓

One document

↓

Tokenizer

↓

Embedding

↓

Store

↓

Next document
```

Never load everything.

---

# Solution 2 — Chunking

Sometimes processing one line at a time is inefficient.

Instead:

Read chunks.

Example:

```python
CHUNK_SIZE = 1024 * 1024

with open("large.txt", "rb") as f:

    while True:

        chunk = f.read(CHUNK_SIZE)

        if not chunk:
            break

        process(chunk)
```

Reads:

```
1 MB

↓

1 MB

↓

1 MB

↓

...
```

Instead of:

```
10 GB
```

---

## Why Chunking?

Reading one byte at a time:

```
Disk

↓

Read

↓

Read

↓

Read
```

Millions of system calls.

Chunking:

```
Disk

↓

1 MB

↓

Process

↓

Next 1 MB
```

Much fewer I/O operations.

---

# Chunking CSV with pandas

Very common interview question.

```python
import pandas as pd

for chunk in pd.read_csv(
    "customers.csv",
    chunksize=100000
):

    process(chunk)
```

Flow:

```
CSV

↓

100k rows

↓

Process

↓

Next 100k rows
```

Memory stays stable.

---

# AI Example

Embedding generation.

Instead of:

```
10 million rows

↓

Embedding API
```

Use:

```
1000 rows

↓

Embedding

↓

Store

↓

Next 1000
```

Reduces memory usage and allows checkpointing.

---

# Solution 3 — Generator

A generator allows lazy reading.

```python
def read_lines(path):

    with open(path) as f:

        for line in f:
            yield line
```

Usage:

```python
for line in read_lines("big.txt"):
    process(line)
```

Internally:

```
yield

↓

One line

↓

Pause

↓

Resume
```

No huge memory allocation.

---

# Solution 4 — Memory Mapping (`mmap`)

Sometimes you need random access into a huge file.

Instead of:

```
Disk

↓

Read

↓

Copy into RAM
```

Use memory mapping.

```python
import mmap

with open("large.txt", "r+b") as f:

    mm = mmap.mmap(
        f.fileno(),
        0
    )

    print(mm[:100])

    mm.close()
```

---

## What is Memory Mapping?

Normal read:

```
Disk

↓

Python Buffer

↓

Python Object
```

Memory map:

```
Disk

↓

Virtual Memory

↓

Pages loaded only when accessed
```

The operating system loads pages **on demand**.

This is called **demand paging**.

---

## Benefits

* Faster random access
* Less copying
* Lower memory usage
* OS manages caching efficiently

---

## AI Example

Large embedding file.

```
vectors.bin

↓

Memory map

↓

Access vector #1

↓

Access vector #500000

↓

Only required pages loaded
```

Useful for:

* FAISS indexes
* Large binary datasets
* Model weights
* Feature stores

---

# Buffered Reading

Python already buffers file reads.

Example:

```python
with open(
    "file.txt",
    buffering=8192
) as f:

    ...
```

The default buffering is usually sufficient.

Avoid reading one byte at a time.

---

# Reading Binary Files

```python
with open("image.jpg", "rb") as f:

    while True:

        chunk = f.read(4096)

        if not chunk:
            break

        process(chunk)
```

Common for:

* Images
* PDFs
* Videos
* Audio

---

# Parallel File Processing

Suppose:

```
1000 PDFs
```

Sequential:

```
Read

↓

OCR

↓

Embedding
```

Use multiprocessing:

```python
from multiprocessing import Pool

with Pool(8) as pool:
    pool.map(process_pdf, files)
```

Each worker processes different files concurrently.

---

# Compress While Streaming

```python
import gzip

with gzip.open(
    "logs.gz",
    "rt"
) as f:

    for line in f:
        process(line)
```

No need to decompress the entire archive first.

---

# Streaming JSON

Instead of:

```python
import json

data = json.load(open("large.json"))
```

Prefer newline-delimited JSON (JSONL):

```python
import json

with open("data.jsonl") as f:

    for line in f:
        obj = json.loads(line)

        process(obj)
```

Each line is an independent JSON object.

---

# AI Pipeline Example

Suppose you have:

```
50 GB PDFs

↓

Extract Text

↓

Chunk

↓

Embedding

↓

Vector DB
```

Efficient pipeline:

```
PDF Reader (stream)

↓

Chunk Generator

↓

Embedding Batch

↓

Qdrant

↓

Next Chunk
```

Memory stays nearly constant regardless of total corpus size.

---

# Performance Comparison

| Method                           | Memory   | Speed                   | Best Use                |
| -------------------------------- | -------- | ----------------------- | ----------------------- |
| `read()`                         | High     | Fast for small files    | Small files             |
| Line-by-line iteration           | Very Low | Good                    | Large text/CSV          |
| Chunking                         | Low      | Excellent               | Large binary/text files |
| Generator                        | Very Low | Excellent               | Streaming pipelines     |
| `pandas.read_csv(chunksize=...)` | Moderate | Excellent               | Large CSV analytics     |
| `mmap`                           | Very Low | Excellent random access | Huge binary/text files  |

---

# Interview Questions

## How would you read a 10 GB CSV?

A strong answer:

> I would avoid loading the entire file into memory. If processing row-by-row, I'd stream it using the file iterator. For tabular analysis with pandas, I'd use `read_csv(..., chunksize=...)` to process manageable batches. This keeps memory usage predictable and allows checkpointing.

---

## What is streaming?

> Streaming processes data incrementally as it is read, rather than loading the entire dataset into memory first.

---

## What is chunking?

> Chunking reads fixed-size blocks or batches of records, reducing I/O overhead and controlling memory consumption.

---

## What is memory mapping?

> Memory mapping maps a file into virtual memory so the operating system loads only the required pages on demand. It is efficient for large files and random-access workloads.

---

## When would you use `mmap`?

Use it when you need:

* Fast random access
* Large binary files
* Huge indexes (e.g., FAISS)
* Model weights
* Very large datasets that don't fit comfortably in RAM

Avoid it for simple sequential CSV processing, where streaming is usually simpler and just as effective.

---

# Senior AI Engineer Summary

| Scenario                  | Recommended Approach                                |
| ------------------------- | --------------------------------------------------- |
| 10 GB CSV processing      | Stream rows or use `pandas.read_csv(chunksize=...)` |
| 100 GB text corpus        | Generator + streaming                               |
| Large PDFs                | Chunk + multiprocessing                             |
| Massive binary embeddings | `mmap`                                              |
| Large compressed logs     | Stream with `gzip.open()`                           |
| LLM ingestion pipeline    | Streaming → Chunking → Batch embeddings → Vector DB |

**Production rule:** Design your pipeline so that **memory usage depends on the chunk size, not the total dataset size**. This allows the same code to process 1 GB or 1 TB datasets with only minor configuration changes.
