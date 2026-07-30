# Graph RAG (Knowledge Graph + Vector RAG) using LangGraph & LangChain

A **Graph RAG** system extends traditional RAG by combining **vector search** with a **knowledge graph**. Instead of retrieving only semantically similar chunks, the system also traverses relationships between entities (people, companies, products, regulations, etc.).

This is increasingly common in enterprise AI because many questions require **relationships**, not just isolated documents.

---

# Why Traditional RAG Is Sometimes Insufficient

Suppose your documents contain:

```text
Document 1:
John works at Microsoft.

Document 2:
Microsoft acquired GitHub.

Document 3:
GitHub created Copilot.
```

Question:

> **"Which company employs the creator of Copilot?"**

A vector search may retrieve only **Document 3**, missing the chain of relationships:

```text
John
 │
works_at
 ▼
Microsoft
 │
acquired
 ▼
GitHub
 │
created
 ▼
Copilot
```

A graph-aware retriever can traverse these links to answer more accurately.

---

# High-Level Architecture

```text
                     User Question
                           │
                           ▼
                    Query Rewriter
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      Vector Retriever          Graph Retriever
             │                           │
             ▼                           ▼
      Similar Chunks           Related Entities
             └─────────────┬─────────────┘
                           ▼
                   Context Merger
                           ▼
                    Context Grader
                     │          │
                  Enough?      No
                     │          │
                     ▼          │
                LLM Generator ◄─┘
                     │
                     ▼
                  Final Answer
```

---

# Technology Stack

| Layer         | Technology                 |
| ------------- | -------------------------- |
| Orchestration | LangGraph                  |
| LLM Framework | LangChain                  |
| Embeddings    | OpenAI / Azure OpenAI      |
| Vector DB     | Qdrant / Pinecone / Milvus |
| Graph DB      | Neo4j                      |
| Database      | PostgreSQL                 |
| Cache         | Redis                      |
| Monitoring    | LangSmith + OpenTelemetry  |

---

# Project Structure

```text
graph_rag/

app/
│
├── graph/
│     workflow.py
│     state.py
│
├── retrieval/
│     vector.py
│     graph.py
│     merger.py
│
├── ingestion/
│     chunker.py
│     embeddings.py
│     entity_extractor.py
│
├── prompts/
├── monitoring/
├── api/
└── tools/
```

---

# Step 1: Shared Graph State

```python
from typing import TypedDict

class GraphRAGState(TypedDict):
    question: str

    rewritten_query: str

    vector_docs: list

    graph_docs: list

    merged_context: list

    answer: str

    sufficient_context: bool

    retries: int
```

Each LangGraph node reads and updates this shared state.

---

# Step 2: Ingest Documents

Load documents.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("company_docs.pdf")

documents = loader.load()
```

Split into chunks.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
```

---

# Step 3: Create Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

vectors = embedding.embed_documents(
    [c.page_content for c in chunks]
)
```

Store in Qdrant.

```python
from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    chunks,
    embedding,
    url="http://localhost:6333",
    collection_name="enterprise_docs"
)
```

---

# Step 4: Build the Knowledge Graph

Extract entities and relationships.

Suppose the document says:

```text
Satya Nadella is CEO of Microsoft.

Microsoft acquired GitHub.

GitHub developed Copilot.
```

Store it in Neo4j.

```cypher
CREATE
(:Person {name:'Satya Nadella'})
-[:CEO_OF]->
(:Company {name:'Microsoft'})

(:Company {name:'Microsoft'})
-[:ACQUIRED]->
(:Company {name:'GitHub'})

(:Company {name:'GitHub'})
-[:DEVELOPED]->
(:Product {name:'Copilot'})
```

Graph:

```text
Satya Nadella
      │
   CEO_OF
      ▼
 Microsoft
      │
  ACQUIRED
      ▼
 GitHub
      │
 DEVELOPED
      ▼
 Copilot
```

---

# Step 5: Query Rewriter

Improve ambiguous questions.

```python
def rewrite_query(state):

    rewritten = llm.invoke(
        f"""
Rewrite this search query.

Question:
{state['question']}
"""
    )

    return {
        "rewritten_query": rewritten.content
    }
```

---

# Step 6: Vector Retrieval

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k":5}
)

def vector_search(state):

    docs = retriever.invoke(
        state["rewritten_query"]
    )

    return {
        "vector_docs": docs
    }
```

---

# Step 7: Graph Retrieval

Example using Neo4j.

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

def graph_search(state):

    query = """
    MATCH (n)-[r]-(m)
    WHERE n.name CONTAINS $entity
    RETURN n,r,m
    """

    with driver.session() as session:

        result = session.run(
            query,
            entity=state["rewritten_query"]
        )

        rows = list(result)

    return {
        "graph_docs": rows
    }
```

This returns related entities and relationships rather than document chunks.

---

# Step 8: Merge Context

```python
def merge_context(state):

    merged = (
        state["vector_docs"]
        + state["graph_docs"]
    )

    return {
        "merged_context": merged
    }
```

---

# Step 9: Grade Retrieval Quality

```python
def grade_context(state):

    enough = len(state["merged_context"]) >= 5

    return {
        "sufficient_context": enough,
        "retries": state["retries"] + 1
    }
```

---

# Step 10: Conditional Routing

```python
MAX_RETRIES = 2

def routing(state):

    if state["sufficient_context"]:
        return "generate"

    if state["retries"] >= MAX_RETRIES:
        return "generate"

    return "vector_search"
```

The workflow retries retrieval when context is weak.

---

# Step 11: Generate Answer

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer only using the supplied context.

Question:
{question}

Context:
{context}
""")

chain = prompt | llm

def generate(state):

    response = chain.invoke(
        {
            "question": state["question"],
            "context": state["merged_context"]
        }
    )

    return {
        "answer": response.content
    }
```

---

# Step 12: Build the LangGraph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(GraphRAGState)

builder.add_node("rewrite", rewrite_query)
builder.add_node("vector_search", vector_search)
builder.add_node("graph_search", graph_search)
builder.add_node("merge", merge_context)
builder.add_node("grade", grade_context)
builder.add_node("generate", generate)

builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "vector_search")
builder.add_edge("rewrite", "graph_search")

builder.add_edge("vector_search", "merge")
builder.add_edge("graph_search", "merge")

builder.add_edge("merge", "grade")

builder.add_conditional_edges(
    "grade",
    routing,
    {
        "vector_search": "vector_search",
        "generate": "generate",
    },
)

builder.add_edge("generate", END)

graph = builder.compile()
```

---

# Execution Flow

```text
START
  │
Rewrite Query
  │
 ┌──────────────┐
 ▼              ▼
Vector      Graph Search
Search
 │              │
 └──────┬───────┘
        ▼
 Merge Context
        │
 Grade Context
        │
 Enough?
   │         │
  No        Yes
   │         ▼
Vector     Generate
Search        │
              ▼
             END
```

---

# Example

Question:

> "Who manages the team that built Copilot?"

Traditional RAG:

```text
Retrieved:
GitHub developed Copilot.
```

Graph RAG:

```text
GitHub
   ▲
Acquired by
   │
Microsoft
   ▲
CEO
   │
Satya Nadella
```

The graph traversal provides richer context for reasoning.

---

# Production Enhancements

A production Graph RAG system often includes:

* Hybrid retrieval (BM25 + vector + graph)
* Entity extraction during ingestion
* Graph expansion with configurable depth
* Cross-encoder reranking
* Metadata filters (tenant, department, document type)
* Redis caching
* PostgreSQL checkpointing
* Human review for sensitive outputs
* LangSmith and OpenTelemetry tracing

---

# Common Interview Questions

### Why combine a graph database with a vector database?

A vector database retrieves semantically similar text. A graph database models explicit relationships between entities. Together they improve recall and reasoning for relationship-heavy questions.

---

### When should you use Graph RAG?

Graph RAG is valuable when your domain contains interconnected entities, such as financial transactions, supply chains, healthcare records, legal precedents, or organizational hierarchies.

---

### How do you build the graph?

During ingestion, extract entities and relationships using NLP or an LLM, normalize them, and store them in a graph database such as Neo4j. Documents are still chunked and embedded into a vector database.

---

### How do you prevent graph traversal from becoming too expensive?

Limit traversal depth, filter by relationship types, cap the number of returned nodes, cache frequent traversals, and rerank the combined results before sending them to the LLM.

---

# Traditional RAG vs Graph RAG

| Feature              | Traditional RAG             | Graph RAG                       |
| -------------------- | --------------------------- | ------------------------------- |
| Retrieval            | Vector similarity           | Vector + relationship traversal |
| Best for             | FAQ, manuals, documentation | Connected knowledge domains     |
| Entity relationships | Weak                        | Strong                          |
| Multi-hop reasoning  | Limited                     | Excellent                       |
| Complexity           | Lower                       | Higher                          |
| Infrastructure       | Vector DB                   | Vector DB + Graph DB            |

---

# Senior AI Engineer Interview Answer

> **I design Graph RAG by combining semantic retrieval with knowledge graph traversal. Documents are ingested, chunked, embedded, and stored in a vector database, while entities and relationships are extracted into a graph database such as Neo4j. A LangGraph workflow rewrites the query, performs vector and graph retrieval in parallel, merges and grades the retrieved context, retries retrieval if necessary, and finally generates a grounded answer. The workflow maintains shared state, supports checkpointing and conditional routing, and integrates monitoring through LangSmith and OpenTelemetry. This architecture is particularly effective for enterprise domains where understanding relationships between entities is as important as retrieving relevant documents.**
