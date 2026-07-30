# Enterprise AI Copilot for Knowledge Search (Production Architecture)

An **Enterprise Knowledge Copilot** is one of the most common AI applications in large organizations. It enables employees to ask questions in natural language and retrieve information from internal knowledge sources such as:

* Confluence
* SharePoint
* Google Drive
* Jira
* GitHub
* Slack
* Microsoft Teams
* PDFs
* Word documents
* Wikis
* Internal databases

Unlike a simple RAG chatbot, an enterprise copilot must support:

* Multi-tenant isolation
* RBAC (Role-Based Access Control)
* Hybrid retrieval (BM25 + Vector)
* Reranking
* Conversation memory
* Tool calling
* Multi-step reasoning
* Source citations
* Human approval for sensitive actions
* Observability
* Production scalability

---

# Functional Requirements

Users should be able to ask:

* "What is our PTO policy?"
* "Summarize the Q2 architecture proposal."
* "Who owns the Payments service?"
* "Show Jira tickets related to authentication."
* "Find the API documentation for customer onboarding."
* "Compare the latest security policy with last year's version."

The system should retrieve information from multiple enterprise systems and generate a grounded answer with citations.

---

# High-Level Architecture

```text
                         Employee
                             │
                  Web / Teams / Slack UI
                             │
                             ▼
                   FastAPI API Gateway
                             │
                 OAuth2 / SSO Authentication
                             │
                  RBAC + Tenant Validation
                             │
                             ▼
                  LangGraph Orchestrator
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
 Query Rewrite        Memory Loader        Intent Classifier
       │
       ▼
  Retrieval Coordinator
       │
 ┌─────┼──────────┬────────────┬──────────────┐
 ▼     ▼          ▼            ▼              ▼
Vector BM25   Confluence   SharePoint     Jira/GitHub
Search Search    Tool         Tool            Tools
 └─────┴──────────┴────────────┴──────────────┘
                    │
                    ▼
             Context Merger
                    │
                    ▼
               Cross Encoder
                 Reranker
                    │
                    ▼
            Context Compression
                    │
                    ▼
               LLM Generator
                    │
                    ▼
         Structured Response + Citations
                    │
                    ▼
               Redis Cache
```

---

# Technology Stack

| Layer         | Technology                 |
| ------------- | -------------------------- |
| API           | FastAPI                    |
| Workflow      | LangGraph                  |
| LLM Framework | LangChain (LCEL)           |
| LLM           | OpenAI / Azure OpenAI      |
| Vector DB     | Qdrant / Pinecone          |
| Search        | Elasticsearch / OpenSearch |
| Cache         | Redis                      |
| Metadata      | PostgreSQL                 |
| Monitoring    | LangSmith + OpenTelemetry  |
| Deployment    | Kubernetes                 |

---

# Project Structure

```text
enterprise_copilot/

app/
│
├── api/
├── auth/
├── graph/
│     workflow.py
│     state.py
│
├── retrievers/
│     vector.py
│     bm25.py
│
├── connectors/
│     confluence.py
│     sharepoint.py
│     jira.py
│     github.py
│
├── reranker/
├── prompts/
├── monitoring/
├── memory/
└── evaluation/
```

---

# Step 1: Graph State

Every LangGraph node shares one state.

```python
from typing import TypedDict

class CopilotState(TypedDict):
    user_id: str
    tenant_id: str
    role: str

    question: str
    rewritten_query: str

    retrieved_docs: list
    reranked_docs: list

    answer: str

    retries: int
    sufficient_context: bool
```

---

# Step 2: Query Rewriting

Improve ambiguous questions.

```python
def rewrite(state):

    prompt = f"""
Rewrite the question for enterprise search.

Question:
{state['question']}
"""

    rewritten = llm.invoke(prompt)

    return {

        "rewritten_query":
        rewritten.content
    }
```

Example:

```
Original:
PTO

↓

Rewritten:
Paid Time Off policy for employees
```

---

# Step 3: Hybrid Retrieval

### Vector Search

```python
vector_docs = vector_retriever.invoke(
    state["rewritten_query"]
)
```

---

### BM25 Search

```python
keyword_docs = bm25.invoke(
    state["rewritten_query"]
)
```

---

### Confluence Tool

```python
docs = confluence_tool.invoke(
    state["rewritten_query"]
)
```

---

### Jira Tool

```python
tickets = jira_tool.invoke(
    state["rewritten_query"]
)
```

---

### GitHub Search

```python
repos = github_tool.invoke(
    state["rewritten_query"]
)
```

---

# Step 4: Merge Results

```python
def merge(state):

    merged = (

        state["vector_docs"]

        + state["keyword_docs"]

        + state["jira_docs"]

        + state["github_docs"]

    )

    return {

        "retrieved_docs": merged

    }
```

---

# Step 5: Metadata Filtering

Every document contains metadata.

Example:

```python
{
    "tenant":"EY",

    "department":"HR",

    "visibility":"employee"
}
```

Filter retrieval:

```python
retriever.invoke(

    query,

    filter={

        "tenant":state["tenant_id"]

    }

)
```

Tenant A never sees Tenant B documents.

---

# Step 6: RBAC

Suppose a Finance document.

```python
{
    "department":"Finance"
}
```

Before returning it:

```python
def authorize(user, doc):

    return doc.department in user.roles
```

Unauthorized documents are removed before reaching the LLM.

---

# Step 7: Reranking

Initial retrieval:

```
Doc A

Doc B

Doc C

Doc D
```

Cross encoder:

```
↓

Doc C

Doc A

Doc D
```

Code:

```python
reranked = reranker.compress_documents(

    state["retrieved_docs"],

    state["question"]

)
```

---

# Step 8: Context Compression

Instead of sending 30 pages.

```
30 pages

↓

4 paragraphs
```

```python
compressed = compressor.compress_documents(

    reranked,

    state["question"]

)
```

Reduces cost.

---

# Step 9: Answer Generation

```python
prompt = ChatPromptTemplate.from_template("""

Answer only using context.

Question:

{question}

Context:

{context}

""")

chain = (

    prompt

    | llm

    | StrOutputParser()

)
```

Invoke:

```python
answer = chain.invoke(

    {

        "question":state["question"],

        "context":compressed

    }

)
```

---

# Step 10: Retrieval Grader

```python
def grade(state):

    enough = len(

        state["retrieved_docs"]

    ) >= 5

    return {

        "sufficient_context":enough,

        "retries":

        state["retries"]+1

    }
```

---

# Step 11: Conditional Routing

```python
MAX_RETRIES = 2

def router(state):

    if state["sufficient_context"]:

        return "generate"

    if state["retries"] >= MAX_RETRIES:

        return "generate"

    return "retrieve"
```

Graph loops until context quality is acceptable or the retry limit is reached.

---

# Step 12: Conversation Memory

Store previous turns.

```python
class CopilotState(TypedDict):

    history:list

    answer:str
```

Example:

```
User:
Who owns Payments?

↓

Agent:
Platform Team.

↓

User:
Show their latest design document.
```

The follow-up uses the previous answer.

Persist memory using a PostgreSQL-backed checkpointer.

---

# Step 13: Build the LangGraph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(CopilotState)

builder.add_node("rewrite", rewrite)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("generate", generate)

builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "grade")

builder.add_conditional_edges(
    "grade",
    router,
    {
        "retrieve": "retrieve",
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
Retrieve
  │
Hybrid Search
  │
Merge Results
  │
Rerank
  │
Grade Context
  │
Enough?
 ┌───────┴────────┐
 │                │
No               Yes
 │                │
Retrieve      Generate
                  │
                 END
```

---

# Observability

Monitor:

* Query latency
* Retrieval latency
* Reranker latency
* LLM latency
* Token usage
* Cost
* Cache hit rate
* Retrieval score
* Hallucination rate

Use:

* LangSmith
* OpenTelemetry
* Prometheus
* Grafana

---

# Production Deployment

```text
Employees
     │
Load Balancer
     │
FastAPI Pods
     │
LangGraph Workers
     │
Redis
PostgreSQL
Qdrant
Elasticsearch
Confluence
SharePoint
Jira
GitHub
     │
Monitoring Stack
```

Workers remain stateless; workflow state, cache, and indexes are externalized.

---

# Advanced Enhancements

A production enterprise copilot typically adds:

* **Connector framework** for systems such as SharePoint, Confluence, Slack, Teams, GitHub, and Google Drive.
* **Incremental indexing** using webhooks or scheduled sync jobs.
* **Document versioning** so answers can reference the latest approved content.
* **Structured output** with answer, confidence score, and citations.
* **Streaming responses** while retrieval continues in the background.
* **PII detection and redaction** before prompts and logs.
* **Multi-model routing**, using smaller models for query rewriting and larger models for answer generation.
* **Evaluation pipelines** measuring context precision, context recall, faithfulness, and answer relevance.

---

# Common Interview Questions

### Why use LangGraph instead of a single LCEL chain?

Enterprise search often requires retries, conditional routing, checkpointing, and multiple retrieval sources. LangGraph provides explicit workflow control and shared state for these scenarios.

---

### Why combine BM25 and vector search?

Vector search captures semantic similarity, while BM25 excels at exact keywords such as policy IDs, ticket numbers, and API names. Hybrid retrieval improves recall across different query types.

---

### How do you enforce document security?

Authentication identifies the user, RBAC and tenant filters are applied during retrieval, and only authorized documents are passed to the LLM. The model is never responsible for access control.

---

### How do you reduce hallucinations?

Ground responses in retrieved context, use hybrid retrieval with reranking, retry when context quality is low, instruct the model to answer only from supplied evidence, and evaluate outputs using metrics such as faithfulness.

---

# Senior AI Engineer Interview Answer

> **I design an enterprise knowledge copilot as a LangGraph-orchestrated RAG system. The workflow authenticates the user, rewrites the query, performs hybrid retrieval across vector indexes and enterprise connectors such as Confluence, SharePoint, Jira, and GitHub, merges and reranks the results, grades retrieval quality, and retries if necessary before generating a grounded answer with citations. All retrieval is filtered by tenant and RBAC permissions, conversation state is checkpointed for continuity, Redis caches frequent queries, and LangSmith, OpenTelemetry, Prometheus, and Grafana provide end-to-end observability. This architecture is scalable, secure, resilient, and suitable for enterprise production deployments.
