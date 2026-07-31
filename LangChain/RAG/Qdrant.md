# Qdrant Explained Properly (Production-Level Guide with Code)

Qdrant is one of the most popular **vector databases** used in production RAG systems. It stores embeddings together with metadata and supports efficient similarity search, filtering, and hybrid retrieval.

A typical production RAG pipeline looks like:

```text
                   PDF / HTML / Docs
                           │
                           ▼
                    Document Loader
                           │
                           ▼
                  Text Chunking
                           │
                           ▼
                  Embedding Model
                           │
                           ▼
             Dense Vector + Metadata
                           │
                           ▼
                      Qdrant
                           │
             Similarity / Hybrid Search
                           │
                           ▼
                     Top K Chunks
                           │
                           ▼
                      Reranker
                           │
                           ▼
                          LLM
```

---

# Why Qdrant?

A relational database cannot efficiently answer:

> "Find documents semantically similar to this question."

For example:

Document:

```text
Employees receive 180 days of maternity leave.
```

User asks:

```text
How much pregnancy leave do employees get?
```

The words are different.

Keyword search may fail.

Embeddings capture semantic meaning.

Qdrant performs nearest-neighbor search on those embeddings.

---

# Core Concepts

Every record in Qdrant contains three parts:

```text
Point
│
├── ID
├── Vector
└── Payload (metadata)
```

Example:

```text
ID: 123

Vector:
[0.12, -0.44, 0.89, ...]

Payload:
{
   "department":"HR",
   "document":"Leave Policy",
   "page":12
}
```

The payload allows metadata filtering during search.

---

# Install

```bash
pip install qdrant-client
pip install langchain-qdrant
pip install langchain-openai
```

---

# Running Qdrant

Using Docker:

```bash
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

Production usually runs Qdrant as a clustered service on Kubernetes with persistent volumes.

---

# Connecting to Qdrant

```python
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333
)
```

Cloud:

```python
client = QdrantClient(
    url="https://YOUR_CLUSTER",
    api_key="YOUR_API_KEY"
)
```

---

# Creating a Collection

A collection is similar to a table.

```python
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams

client.create_collection(
    collection_name="company_docs",

    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)
```

`size` must match your embedding model.

Examples:

| Embedding Model        | Vector Size |
| ---------------------- | ----------: |
| text-embedding-3-small |        1536 |
| text-embedding-3-large |        3072 |
| BAAI/bge-base          |         768 |
| all-MiniLM-L6-v2       |         384 |

---

# Generate Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector = embeddings.embed_query(
    "How many maternity leave days?"
)
```

The vector contains 1536 floating-point numbers.

---

# Insert Documents

```python
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="company_docs",

    points=[

        PointStruct(

            id=1,

            vector=vector,

            payload={

                "text":
                    "Employees receive 180 days.",

                "department":
                    "HR",

                "document":
                    "Leave Policy"

            }

        )

    ]
)
```

---

# Searching

Generate a query embedding.

```python
query_vector = embeddings.embed_query(
    "Pregnancy leave"
)
```

Search.

```python
results = client.search(
    collection_name="company_docs",

    query_vector=query_vector,

    limit=5
)
```

Display results.

```python
for hit in results:

    print(hit.score)

    print(hit.payload["text"])
```

Output:

```text
0.93
Employees receive 180 days.

0.81
Parental leave policy.

0.67
Insurance benefits.
```

---

# Metadata Filtering

Enterprise systems almost always filter results.

Example:

Only search HR documents.

```python
from qdrant_client.models import Filter
from qdrant_client.models import FieldCondition
from qdrant_client.models import MatchValue

results = client.search(

    collection_name="company_docs",

    query_vector=query_vector,

    query_filter=Filter(

        must=[

            FieldCondition(

                key="department",

                match=MatchValue(
                    value="HR"
                )

            )

        ]

    ),

    limit=5

)
```

This prevents retrieving unrelated departments.

---

# Using LangChain

```python
from langchain_qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(

    client=client,

    collection_name="company_docs",

    embedding=embeddings

)
```

Insert documents.

```python
vector_store.add_texts(

    texts=[

        "Employees receive 180 days.",

        "VPN setup guide.",

        "Travel reimbursement."

    ],

    metadatas=[

        {"department":"HR"},

        {"department":"IT"},

        {"department":"Finance"}

    ]

)
```

Search.

```python
docs = vector_store.similarity_search(

    "Pregnancy leave",

    k=3

)

for doc in docs:

    print(doc.page_content)
```

---

# Retriever

```python
retriever = vector_store.as_retriever(

    search_kwargs={
        "k":5
    }

)
```

Use it inside a RAG chain.

```python
docs = retriever.invoke(
    "How many maternity leave days?"
)
```

---

# Production Ingestion Pipeline

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size=800,

    chunk_overlap=150

)

chunks = splitter.split_documents(documents)

vector_store.add_documents(chunks)
```

A typical ingestion workflow:

```text
PDF
 │
 ▼
Loader
 │
 ▼
Chunker
 │
 ▼
Embedding Model
 │
 ▼
Qdrant
```

---

# Hybrid Search

Dense vectors are excellent for semantic meaning.

Keyword search is excellent for exact terms.

Combine both:

```text
Query
   │
   ▼
Vector Search
   │
BM25
   │
   ▼
Merge Results
   │
   ▼
Reranker
```

Hybrid retrieval improves recall in enterprise search.

---

# Reranking

Retrieve 50 candidates.

```python
docs = retriever.invoke(query)
```

Rerank them using a cross-encoder before sending the top 5 to the LLM.

```text
Top 50
   │
   ▼
Cross Encoder
   │
   ▼
Top 5
```

This greatly improves Context Precision.

---

# Updating Documents

When policies change:

```python
client.upsert(

    collection_name="company_docs",

    points=[

        PointStruct(

            id=1,

            vector=new_vector,

            payload={

                "text":
                    "Employees receive 200 days.",

                "department":
                    "HR"

            }

        )

    ]
)
```

Using the same ID replaces the existing point.

---

# Deleting Documents

```python
from qdrant_client.models import PointIdsList

client.delete(

    collection_name="company_docs",

    points_selector=PointIdsList(

        points=[1]

    )
)
```

---

# Production Architecture

```text
                PDF / HTML / DOCX
                       │
                       ▼
                Document Loader
                       │
                       ▼
             Recursive Chunking
                       │
                       ▼
            OpenAI/BGE Embeddings
                       │
                       ▼
                   Qdrant
              (Vectors + Metadata)
                       │
              Similarity Search
                       │
               Metadata Filtering
                       │
                   Reranker
                       │
                       ▼
                     GPT-5
```

---

# Best Practices

### 1. Store Metadata

Always save useful metadata.

```python
metadata = {

    "department":"HR",

    "document":"Leave Policy",

    "page":15,

    "version":"2.1",

    "created_at":"2026-01-10"
}
```

This enables filtering and traceability.

---

### 2. Keep Chunks Small

Recommended starting point:

```text
Chunk Size:
600–1000 tokens

Overlap:
100–200 tokens
```

Avoid splitting sentences across chunks.

---

### 3. Batch Upserts

Instead of inserting one document at a time:

```python
client.upsert(
    collection_name="company_docs",
    points=batch_points
)
```

Batching significantly improves ingestion throughput.

---

### 4. Separate Collections

Instead of one huge collection:

```text
hr_documents

finance_documents

legal_documents
```

Or use a shared collection with strict metadata filtering for multi-tenant systems.

---

### 5. Use Metadata Filters

Example:

```python
department="HR"

country="India"

version="2026"
```

This improves both latency and precision.

---

### 6. Monitor

Track:

* Search latency
* QPS
* Recall
* Context Precision
* Context Recall
* Cache hit rate
* Collection size
* Index health

---

# Common Interview Questions

### Why Qdrant instead of PostgreSQL?

PostgreSQL is excellent for structured data. Qdrant is optimized for approximate nearest-neighbor search over high-dimensional vectors with efficient filtering.

---

### What is stored in Qdrant?

Each point contains:

* ID
* Embedding vector
* Payload (metadata)

---

### Why store metadata?

Metadata enables filtering, document attribution, access control, versioning, and easier debugging.

---

### Can Qdrant support millions of vectors?

Yes. It is designed for large-scale vector search and supports optimized indexes, persistence, snapshots, and distributed deployments. Exact scalability depends on hardware, vector size, indexing parameters, and deployment architecture.

---

# Production Folder Structure

```text
rag-platform/
│
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   └── ingest.py
│
├── retrieval/
│   ├── qdrant_client.py
│   ├── retriever.py
│   ├── reranker.py
│   └── hybrid_search.py
│
├── services/
│   ├── rag_service.py
│   └── search_service.py
│
├── api/
│   └── routes.py
│
└── monitoring/
    ├── metrics.py
    └── logging.py
```

This separation keeps ingestion, retrieval, serving, and monitoring independently testable and scalable.

---

# Senior AI Engineer Interview Answer

> **In production RAG systems, I use Qdrant as the vector database to store document embeddings together with rich metadata such as document ID, department, version, and page number. During ingestion, documents are chunked, embedded, and batch-upserted into Qdrant. At query time, I embed the user's question, perform similarity search with metadata filters, retrieve a larger candidate set, rerank the results with a cross-encoder, and pass only the highest-quality chunks to the LLM. I monitor retrieval latency and quality using metrics such as Context Precision and Context Recall, batch writes for efficient ingestion, and use metadata filtering for security, multi-tenancy, and improved retrieval accuracy. This architecture scales well for enterprise knowledge search while maintaining low latency and high retrieval quality.
