# RAG (Retrieval-Augmented Generation) End-to-End

**Retrieval-Augmented Generation (RAG)** is a technique that enables an LLM to answer questions using **external knowledge** instead of relying only on its training data.

Instead of asking the LLM to remember everything, RAG retrieves relevant documents first and then provides them as context for generation.

A production RAG system has **two distinct pipelines**:

1. **Offline Indexing Pipeline (Build the Knowledge Base)**
2. **Online Query Pipeline (Answer User Questions)**

Understanding this separation is critical in Senior AI Engineer interviews.

---

# High-Level Architecture

```text
                     OFFLINE PIPELINE
                (Runs once or periodically)

        Documents
            │
            ▼
      Document Loaders
            │
            ▼
      Document Cleaning
            │
            ▼
         Chunking
            │
            ▼
      Metadata Extraction
            │
            ▼
        Embeddings
            │
            ▼
      Vector Database
(Qdrant/Pinecone/Milvus)


────────────────────────────────────────────────────────


                     ONLINE PIPELINE

         User Question
               │
               ▼
        Query Rewrite
               │
               ▼
         Embeddings
               │
               ▼
      Similarity Search
               │
               ▼
      Retrieved Chunks
               │
               ▼
          Reranker
               │
               ▼
     Context Compression
               │
               ▼
       Prompt Template
               │
               ▼
            LLM
               │
               ▼
     Grounded Answer
```

---

# Example Documents

Suppose we have three PDFs.

```
Employee Handbook.pdf

Security Policy.pdf

Leave Policy.pdf
```

Employee asks:

> "How many maternity leave days are available?"

The answer should come from **Leave Policy.pdf**, not from the model's memory.

---

# PART 1 — Offline Indexing Pipeline

This pipeline builds the searchable knowledge base.

---

# Step 1: Load Documents

LangChain provides document loaders.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("leave_policy.pdf")

documents = loader.load()

print(documents[0].page_content)
```

Output

```
Maternity leave is 180 days...
```

Each document contains:

```python
Document(
    page_content="...",
    metadata={
        "source":"leave_policy.pdf",
        "page":3
    }
)
```

---

# Step 2: Clean Documents

Production systems clean documents before indexing.

Example:

```python
def clean(text):

    text = text.replace("\n"," ")

    text = text.strip()

    return text
```

You may also:

* remove headers
* remove footers
* remove duplicate pages
* OCR scanned PDFs
* normalize whitespace

---

# Step 3: Chunk Documents

LLMs cannot efficiently search large documents.

Instead:

```
100-page PDF

↓

Chunk 1

Chunk 2

Chunk 3

...

Chunk N
```

LangChain:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)

chunks = splitter.split_documents(documents)
```

Result

```
Chunk 1

Chunk 2

Chunk 3
```

---

# Why Chunking?

Without chunking:

```
Entire PDF

↓

Embedding

↓

Poor retrieval
```

With chunking:

```
Chunk

↓

Embedding

↓

Better retrieval
```

---

# Step 4: Create Embeddings

Embeddings convert text into vectors.

```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
```

Generate vectors.

```python
vectors = embedding.embed_documents(

    [chunk.page_content for chunk in chunks]

)
```

Output

```
[

0.12,

-0.34,

0.67,

...

]
```

Each chunk gets its own vector.

---

# Step 5: Store in Vector Database

Example with Qdrant.

```python
from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(

    chunks,

    embedding,

    url="http://localhost:6333",

    collection_name="company_docs"

)
```

Database

```
Chunk A

↓

Vector

↓

Metadata

↓

Stored
```

---

# PART 2 — Online Query Pipeline

User asks

```
How many maternity leave days are available?
```

---

# Step 1: Query Rewrite

Instead of searching:

```
maternity leave
```

Rewrite:

```
employee maternity leave policy days
```

Example:

```python
prompt = f"""

Rewrite this enterprise search query.

Question:

{question}

"""
```

---

# Step 2: Query Embedding

```python
query_vector = embedding.embed_query(

    question

)
```

---

# Step 3: Similarity Search

```python
retriever = vectorstore.as_retriever(

    search_kwargs={"k":5}

)
```

Retrieve.

```python
docs = retriever.invoke(question)
```

Output

```
Chunk 3

Chunk 5

Chunk 9
```

---

# How Similarity Search Works

Stored vectors:

```
Chunk A

↓

[0.1,0.2,...]

Chunk B

↓

[0.3,0.4,...]
```

User question

↓

Embedding

↓

Similarity

↓

Nearest vectors

Most vector databases use cosine similarity or related ANN algorithms (e.g., HNSW) to find the closest chunks efficiently.

---

# Step 4: Hybrid Search (Production)

Vector search

*

BM25 keyword search

```
User Query

↓

Vector Search

+

BM25

↓

Merge Results
```

Why?

Vector search misses

```
Invoice-3482
```

BM25 retrieves exact IDs.

---

# Step 5: Reranking

Initial retrieval

```
Doc A

Doc B

Doc C

Doc D
```

Cross encoder

↓

```
Doc C

Doc A

Doc D
```

Example

```python
reranked = reranker.compress_documents(

    docs,

    question

)
```

---

# Step 6: Context Compression

Instead of sending

```
25 pages
```

Compress to

```
3 paragraphs
```

Example

```python
compressed = compressor.compress_documents(

    reranked,

    question

)
```

Benefits

* cheaper
* faster
* fewer hallucinations

---

# Step 7: Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""

Answer ONLY using the supplied context.

Question:

{question}

Context:

{context}

If the answer is not in the context, say:

"I don't know."

""")
```

---

# Step 8: LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

chain = prompt | llm
```

Invoke

```python
response = chain.invoke(

    {

        "question":question,

        "context":compressed

    }

)
```

Output

```
Employees receive 180 days of maternity leave.
```

---

# Step 9: Return Citations

Production systems return:

```json
{
  "answer":"Employees receive 180 days.",

  "sources":[

      "LeavePolicy.pdf page 3"

  ]
}
```

---

# Complete LangChain LCEL Pipeline

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template(
    """
Answer ONLY using the provided context.

Question:
{question}

Context:
{context}
"""
)

llm = ChatOpenAI(model="gpt-4.1-mini")

def retrieve(inputs):
    docs = retriever.invoke(inputs["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {
        "question": inputs["question"],
        "context": context,
    }

rag_chain = (
    retrieve
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke(
    {"question": "How many maternity leave days are available?"}
)

print(answer)
```

---

# Production RAG Pipeline

```
User

↓

Authentication

↓

Query Rewrite

↓

Hybrid Retrieval

↓

Metadata Filter

↓

Reranker

↓

Context Compression

↓

LLM

↓

Guardrails

↓

Response
```

---

# Production Improvements

A production-grade RAG system usually includes:

| Feature                 | Why                                             |
| ----------------------- | ----------------------------------------------- |
| Query rewriting         | Improves retrieval quality                      |
| Hybrid search           | Better recall than vectors alone                |
| Metadata filtering      | Tenant, department, permissions                 |
| Cross-encoder reranking | Better ranking quality                          |
| Context compression     | Reduces tokens and latency                      |
| Redis cache             | Avoids repeated retrieval/LLM calls             |
| Conversation memory     | Supports follow-up questions                    |
| Retrieval grading       | Retries when context is weak                    |
| LangGraph               | Enables retries, loops, and conditional routing |
| Observability           | LangSmith, OpenTelemetry, Prometheus            |

---

# Common Interview Questions

### Why not send the whole document to the LLM?

Large documents exceed context windows, increase latency and cost, and often reduce answer quality. Chunking and retrieval provide only the most relevant information.

---

### Why do we need embeddings?

Embeddings map text into a vector space where semantically similar content is close together. This enables similarity search beyond exact keyword matching.

---

### Why use hybrid search?

Vector search finds semantic matches, while BM25 finds exact keywords such as error codes, invoice IDs, or API names. Combining both improves recall.

---

### Why rerank?

The vector database optimizes for fast approximate retrieval, not perfect ordering. A reranker evaluates the retrieved candidates more accurately before passing them to the LLM.

---

### How do you reduce hallucinations?

* Retrieve relevant context.
* Instruct the model to answer only from the provided context.
* Return "I don't know" when evidence is missing.
* Use reranking and context compression.
* Evaluate with metrics such as context precision, context recall, and faithfulness.

---

# Traditional Search vs RAG

| Traditional Search      | RAG                                               |
| ----------------------- | ------------------------------------------------- |
| Returns documents       | Returns synthesized answers grounded in documents |
| User reads results      | LLM summarizes relevant information               |
| Keyword matching        | Semantic retrieval + optional keyword search      |
| No reasoning            | Can reason over retrieved context                 |
| No citations by default | Can return source citations                       |

---

# Senior AI Engineer Interview Answer

> **I view RAG as two separate pipelines: an offline indexing pipeline and an online query pipeline. During indexing, documents are loaded, cleaned, chunked, enriched with metadata, embedded, and stored in a vector database. During query time, the user's question is optionally rewritten, embedded, and used for hybrid retrieval across vector and keyword indexes. The retrieved documents are reranked, compressed, and passed to the LLM through a prompt that instructs it to answer only from the supplied context. The response includes citations, while the platform uses caching, metadata filtering, retrieval evaluation, and observability to provide secure, scalable, and reliable enterprise search.
