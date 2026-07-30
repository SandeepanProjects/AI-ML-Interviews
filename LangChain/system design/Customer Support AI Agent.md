A **Customer Support AI Agent** is one of the most common **Senior AI Engineer** system design interview questions because it combines nearly every enterprise AI concept:

* LangGraph workflows
* Tool calling
* RAG
* Multi-agent coordination
* Human-in-the-loop
* Memory
* Checkpointing
* Authentication
* RBAC
* Observability
* Retry and fallback
* Production deployment

Below is a production-oriented design.

---

# Problem Statement

Users should be able to ask questions like:

* "Where is my order?"
* "I want a refund."
* "My payment failed."
* "Cancel my subscription."
* "Reset my password."
* "Talk to a human."

The agent should:

* Answer FAQs from company documents
* Retrieve customer-specific data
* Execute business actions
* Escalate complex issues
* Maintain conversation history
* Never expose unauthorized customer information

---

# High-Level Architecture

```text
                  Customer
                      │
          Web / Mobile / WhatsApp
                      │
                      ▼
                FastAPI Gateway
                      │
             JWT Authentication
                      │
                  RBAC Check
                      │
                      ▼
              LangGraph Workflow
                      │
      ┌───────────────┼───────────────────┐
      ▼               ▼                   ▼
 Intent Router   Memory Loader     Customer Profile
      │
      ▼
      ┌───────────────┬───────────────┬──────────────┐
      ▼               ▼               ▼
 FAQ Agent     Order Agent    Billing Agent
      │               │               │
      └───────────────┼───────────────┘
                      ▼
               Response Generator
                      ▼
             Human Escalation?
              │              │
             Yes            No
              │              │
              ▼              ▼
      Support Dashboard    Return Reply
```

---

# Technology Stack

| Layer          | Technology                |
| -------------- | ------------------------- |
| API            | FastAPI                   |
| Workflow       | LangGraph                 |
| LLM            | LangChain                 |
| Memory         | PostgreSQL Checkpointer   |
| Cache          | Redis                     |
| Vector Search  | Qdrant                    |
| Authentication | JWT                       |
| Monitoring     | LangSmith + OpenTelemetry |
| Metrics        | Prometheus + Grafana      |

---

# Folder Structure

```text
customer_support/

app/
    graph.py
    state.py

    agents/
        router.py
        faq.py
        order.py
        billing.py
        escalation.py

    tools/
        order_tool.py
        refund_tool.py
        ticket_tool.py

    prompts/
    auth/
    memory/
    monitoring/
```

---

# Step 1: Define Shared State

Every node shares one state object.

```python
from typing import TypedDict, Optional

class SupportState(TypedDict):
    user_id: str
    question: str
    intent: str

    retrieved_docs: list

    order_details: Optional[dict]

    answer: str

    requires_human: bool
```

Instead of passing parameters between nodes, every node reads and updates this state.

---

# Step 2: Intent Router

Determine what the customer wants.

```python
def router(state):

    q = state["question"].lower()

    if "order" in q:
        return {"intent": "order"}

    if "refund" in q:
        return {"intent": "billing"}

    return {"intent": "faq"}
```

---

# Step 3: Build the Graph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(SupportState)

builder.add_node("router", router)
builder.add_node("faq", faq_agent)
builder.add_node("order", order_agent)
builder.add_node("billing", billing_agent)
builder.add_node("escalation", escalation_agent)
```

---

# Step 4: Conditional Routing

```python
def choose_agent(state):

    return state["intent"]
```

```python
builder.add_conditional_edges(
    "router",
    choose_agent,
    {
        "faq": "faq",
        "order": "order",
        "billing": "billing",
    },
)
```

Flow:

```text
START
   │
   ▼
Router
   │
 ┌─┼────────────┐
 ▼ ▼            ▼
FAQ Order   Billing
```

---

# Step 5: FAQ Agent (RAG)

```python
def faq_agent(state):

    docs = retriever.invoke(
        state["question"]
    )

    answer = rag_chain.invoke(
        {
            "question": state["question"],
            "context": docs
        }
    )

    return {

        "retrieved_docs": docs,

        "answer": answer
    }
```

Questions like:

> What is the return policy?

never require internal tools.

---

# Step 6: Order Agent

Retrieve customer-specific information.

```python
def get_order(user_id):

    return {

        "order_id": "A123",

        "status": "Shipped",

        "eta": "Tomorrow"
    }
```

Node:

```python
def order_agent(state):

    order = get_order(
        state["user_id"]
    )

    answer = (
        f"Your order "
        f"{order['order_id']} "
        f"is {order['status']}."
    )

    return {

        "order_details": order,

        "answer": answer
    }
```

---

# Step 7: Billing Agent

Protected tool.

```python
def refund_tool(user):

    require_permission(
        user,
        "request_refund"
    )

    return "Refund created."
```

Billing node:

```python
def billing_agent(state):

    result = refund_tool(
        state["user"]
    )

    return {

        "answer": result
    }
```

RBAC ensures unauthorized users cannot invoke sensitive operations.

---

# Step 8: Human Escalation

Escalate when confidence is low or the issue is sensitive.

```python
def escalation_router(state):

    if state["requires_human"]:
        return "escalation"

    return END
```

Escalation node:

```python
def escalation_agent(state):

    create_support_ticket(
        user_id=state["user_id"],
        summary=state["question"]
    )

    return {

        "answer":
        "Your issue has been forwarded to a support specialist."
    }
```

---

# Step 9: Memory

Store conversation history.

```python
class SupportState(TypedDict):

    history: list

    question: str

    answer: str
```

Before answering:

```text
Customer:
Where is my order?

↓

Agent:
Order shipped.

↓

Customer:
Can I change the address?
```

The second question depends on the previous conversation.

Persist state with a PostgreSQL checkpointer so conversations survive restarts.

---

# Step 10: Error Handling

Suppose the order service fails.

```python
def order_agent(state):

    try:

        order = get_order(
            state["user_id"]
        )

        return {

            "order_details": order

        }

    except Exception:

        return {

            "requires_human": True
        }
```

The workflow automatically routes to the escalation node.

---

# Step 11: Monitoring

Track each node:

```text
Router

120 ms

↓

Order

240 ms

↓

Generator

900 ms
```

Monitor:

* Node latency
* Tool failures
* Retrieval latency
* Token usage
* Cost
* Retry count

Use:

* LangSmith
* OpenTelemetry
* Prometheus
* Grafana

---

# Step 12: Security

Every request:

```text
User

↓

JWT

↓

RBAC

↓

Tool

↓

Audit Log
```

Never let the LLM decide authorization.

---

# Step 13: Checkpointing

```text
Router

↓

Checkpoint

↓

Order Lookup

↓

Checkpoint

↓

Response
```

If the service crashes after the order lookup, the workflow resumes from the last checkpoint instead of starting over.

---

# Complete Workflow

```text
                 Customer Message
                        │
                        ▼
                 Authentication
                        │
                        ▼
                  Load Memory
                        │
                        ▼
                 Intent Router
                        │
      ┌───────────┬──────────────┬──────────────┐
      ▼           ▼              ▼
 FAQ Agent   Order Agent   Billing Agent
      │           │              │
      └───────────┼──────────────┘
                  ▼
          Response Generator
                  │
          Needs Human?
           │           │
          No          Yes
           │           │
           ▼           ▼
      Return Reply   Create Ticket
                      │
                      ▼
                Human Support
```

---

# Scaling Strategy

For thousands of users:

* Run multiple FastAPI instances behind a load balancer.
* Keep LangGraph workers stateless.
* Store workflow state in PostgreSQL.
* Cache retrieval results in Redis.
* Use a distributed vector database such as Qdrant.
* Process long-running workflows asynchronously with a queue.
* Scale workers horizontally using Kubernetes.

---

# Production Enhancements

A production customer support system typically includes:

* Sentiment analysis to prioritize frustrated customers.
* Automatic language detection and translation.
* Retrieval grading with retry if context quality is poor.
* LLM fallback models during provider outages.
* Streaming responses for better UX.
* SLA-aware routing (VIP customers go to higher-priority queues).
* Analytics dashboards for ticket categories and resolution times.

---

# Common Interview Questions

### Why use LangGraph instead of a simple chain?

Customer support workflows involve conditional routing, retries, human escalation, checkpointing, and stateful conversations. LangGraph is designed for these non-linear workflows.

---

### Why separate FAQ, Order, and Billing agents?

Each agent owns one business capability, making the system easier to maintain, test, secure, and scale independently.

---

### How do you prevent unauthorized refunds?

The refund tool enforces RBAC internally. The LLM may request the tool, but the application validates the authenticated user's permissions before execution.

---

### How do you handle ambiguous questions?

The router can classify the request with an LLM or confidence score. If confidence is low, route to a clarification node or directly to human support instead of guessing.

---

# Senior AI Engineer Interview Answer

> **I design a customer support agent using LangGraph as a stateful workflow orchestrator. The graph authenticates the user, loads conversation memory, classifies intent, and routes requests to specialized agents such as FAQ (RAG), Order, or Billing. Each agent updates a shared graph state rather than calling other agents directly. Sensitive tools enforce RBAC and business rules independently of the LLM. Long-running conversations use PostgreSQL checkpointing, retrieval uses a vector database with reranking, and Redis caches repeated queries. The platform is instrumented with LangSmith, OpenTelemetry, Prometheus, and Grafana for tracing and monitoring, and supports retries, human escalation, and horizontal scaling with FastAPI and Kubernetes. This architecture is modular, secure, resilient, and production-ready.
