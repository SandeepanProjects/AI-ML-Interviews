Designing a **Banking Agentic AI Platform** is a classic **Staff AI Engineer / Principal AI Engineer / AI Architect** interview question.

A banking platform is significantly more demanding than a generic RAG chatbot because it must satisfy:

* Strict security and compliance
* Multi-tenancy
* Authentication and RBAC
* Human approval workflows
* Auditability
* Stateful execution
* Reliable tool execution
* Fraud detection
* Explainability
* High availability
* Low latency

The architecture below is representative of a production system.

---

# Requirements

The platform should support tasks such as:

* Account balance inquiries
* Transaction history
* Card blocking
* Fund transfers
* Loan eligibility
* Investment advice
* KYC document verification
* Fraud alerts
* Customer support
* Secure internal employee workflows

---

# High-Level Architecture

```text
                    Mobile/Web App
                           │
                    API Gateway (FastAPI)
                           │
              Authentication (OAuth2 / JWT)
                           │
              Authorization (RBAC + ABAC)
                           │
                    Banking Supervisor Agent
                           │
        ┌─────────────┬──────────────┬─────────────┐
        ▼             ▼              ▼
  Customer Agent  Payments Agent  Loans Agent
        │             │              │
        ├─────────────┼──────────────┤
        ▼             ▼              ▼
  RAG Retriever   Fraud Agent   Compliance Agent
        │
        ▼
  Knowledge Base (Policies, FAQs)
                           │
          PostgreSQL  Redis  Vector DB
                           │
                 Core Banking APIs
                           │
             Audit Logs + Monitoring
```

The **Supervisor Agent** decides which specialized agent should handle the request.

---

# Recommended Tech Stack

| Layer         | Technology                     |
| ------------- | ------------------------------ |
| API           | FastAPI                        |
| Workflow      | LangGraph                      |
| LLM Framework | LangChain                      |
| LLM           | OpenAI / Azure OpenAI / Claude |
| Memory        | PostgreSQL Checkpointer        |
| Cache         | Redis                          |
| Vector DB     | Qdrant                         |
| Database      | PostgreSQL                     |
| Monitoring    | LangSmith + OpenTelemetry      |
| Metrics       | Prometheus + Grafana           |
| Deployment    | Kubernetes                     |

---

# Folder Structure

```text
banking_ai/

app/
│
├── api/
├── auth/
├── graph/
├── agents/
│     supervisor.py
│     customer.py
│     payments.py
│     loans.py
│     fraud.py
│     compliance.py
│
├── tools/
│     account.py
│     payment.py
│     card.py
│     loan.py
│
├── retriever/
├── prompts/
├── monitoring/
├── memory/
└── evaluation/
```

---

# Shared Graph State

Every node reads and updates a shared state.

```python
from typing import TypedDict

class BankingState(TypedDict):
    user_id: str
    role: str

    question: str

    intent: str

    account: dict

    retrieved_docs: list

    approval_required: bool

    answer: str
```

Instead of passing parameters between nodes, each node returns updates to this shared state.

---

# Supervisor Agent

The supervisor classifies the request.

```python
def supervisor(state):

    q = state["question"].lower()

    if "transfer" in q:
        return {"intent": "payments"}

    if "loan" in q:
        return {"intent": "loans"}

    if "balance" in q:
        return {"intent": "customer"}

    return {"intent": "rag"}
```

---

# Build the LangGraph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(BankingState)

builder.add_node("supervisor", supervisor)
builder.add_node("customer", customer_agent)
builder.add_node("payments", payments_agent)
builder.add_node("loans", loans_agent)
builder.add_node("rag", rag_agent)
```

Conditional routing:

```python
def route(state):
    return state["intent"]

builder.add_conditional_edges(
    "supervisor",
    route,
    {
        "customer": "customer",
        "payments": "payments",
        "loans": "loans",
        "rag": "rag",
    },
)
```

---

# Customer Agent

Retrieves account details using a secure tool.

```python
from langchain_core.tools import tool

@tool
def get_balance(user_id: str):
    return {
        "account": "123456",
        "balance": 125000
    }
```

Agent:

```python
def customer_agent(state):

    result = get_balance.invoke(
        {"user_id": state["user_id"]}
    )

    return {
        "account": result,
        "answer": f"Your balance is ₹{result['balance']:,}"
    }
```

---

# Payments Agent

Transfers money.

```python
@tool
def transfer_money(
    from_account: str,
    to_account: str,
    amount: float
):
    ...
```

This tool **must not** execute immediately.

---

# Human Approval Workflow

Transfers above a threshold require approval.

```python
def payments_agent(state):

    amount = state["amount"]

    if amount > 50000:
        return {
            "approval_required": True
        }

    result = transfer_money.invoke(
        {
            "from_account": "...",
            "to_account": "...",
            "amount": amount
        }
    )

    return {
        "answer": result
    }
```

Conditional routing:

```python
def approval_router(state):

    if state["approval_required"]:
        return "human_review"

    return END
```

---

# RAG Agent

Used for policy questions.

Example:

> What is the home loan interest calculation?

Retriever:

```python
docs = retriever.invoke(
    state["question"]
)
```

Generation:

```python
answer = rag_chain.invoke(
    {
        "question": state["question"],
        "context": docs
    }
)
```

---

# Fraud Agent

Every payment request is scored.

```python
def fraud_agent(state):

    score = fraud_model.predict(
        state["transaction"]
    )

    if score > 0.8:

        return {

            "approval_required": True

        }

    return {}
```

High-risk transactions are paused automatically.

---

# RBAC

Protect every tool.

```python
PERMISSIONS = {
    "customer": {"view_balance"},
    "employee": {
        "view_balance",
        "approve_transfer"
    },
}
```

Decorator:

```python
from functools import wraps

def requires(permission):

    def decorator(func):

        @wraps(func)
        def wrapper(user, *args, **kwargs):

            if permission not in PERMISSIONS[user.role]:
                raise PermissionError()

            return func(user, *args, **kwargs)

        return wrapper

    return decorator
```

Usage:

```python
@requires("approve_transfer")
def approve_transfer(...):
    ...
```

The LLM can propose a tool call, but authorization is enforced by the application.

---

# Conversation Memory

State includes history.

```python
class BankingState(TypedDict):

    history: list

    question: str
```

Example:

```
User:
Show my last transaction.

↓

Agent:
₹2,500 at Grocery Store.

↓

User:
Dispute that transaction.
```

The second request relies on conversation state, which can be persisted with a PostgreSQL checkpointer.

---

# Observability

Instrument every node.

Track:

* Node latency
* Token usage
* Tool failures
* Retry count
* Cost
* Human approvals

Example metrics:

```
Supervisor: 120 ms
Retriever: 300 ms
Payments: 180 ms
LLM: 1.2 s
```

Use:

* LangSmith
* OpenTelemetry
* Prometheus
* Grafana

---

# Error Handling

If a core banking API is unavailable:

```python
def payments_agent(state):

    try:
        ...

    except Exception:

        return {

            "approval_required": True,

            "answer": "Payment requires manual review."

        }
```

The graph routes to a manual review queue instead of failing silently.

---

# Security

Every request flows through:

```
User
 │
 ▼
JWT Authentication
 │
 ▼
RBAC / ABAC
 │
 ▼
Supervisor Agent
 │
 ▼
Secure Tool Execution
 │
 ▼
Audit Logging
```

The LLM is **never** trusted to enforce permissions.

---

# Deployment

```
Clients
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
Core Banking APIs
   │
Monitoring Stack
```

All workers are stateless. Workflow state is externalized.

---

# Production Enhancements

A production banking platform often includes:

* Multi-factor authentication before high-value actions.
* Policy engines (e.g., Open Policy Agent) for centralized authorization.
* Encryption of sensitive data at rest and in transit.
* Personally identifiable information (PII) detection and masking in prompts, logs, and traces.
* Transaction idempotency to prevent duplicate transfers.
* Circuit breakers and retries for downstream banking APIs.
* Disaster recovery and multi-region deployments.
* Immutable audit logs for compliance.

---

# Common Interview Questions

### Why use LangGraph instead of a single agent?

Banking workflows are stateful and require branching, retries, approvals, and resumable execution. LangGraph provides explicit workflow control and checkpointing.

---

### Why separate Payments, Loans, and Customer agents?

Each domain has different business rules, permissions, prompts, and tools. Specialized agents are easier to test, secure, and evolve independently.

---

### How do you secure money transfers?

The workflow authenticates the user, authorizes the requested action with RBAC/ABAC, validates transaction parameters, performs fraud checks, optionally requires human approval, and only then invokes the payment tool. Every step is logged for auditing.

---

### How do you prevent hallucinations?

Policy questions use RAG with citations. Transactional actions rely on authoritative banking APIs rather than model-generated information. The model never invents balances or transaction results.

---

# Senior AI Engineer Interview Answer

> **I design a banking agentic platform using LangGraph as the orchestration layer with a supervisor agent routing requests to specialized agents such as Customer, Payments, Loans, Fraud, and RAG. Each agent updates a shared workflow state and invokes narrowly scoped tools protected by RBAC and business rules. High-risk actions, such as large transfers, pass through fraud scoring and human approval before execution. Conversation state is checkpointed in PostgreSQL, Redis provides caching, and a vector database supports retrieval for policies and documentation. The platform is instrumented with LangSmith, OpenTelemetry, Prometheus, and Grafana for end-to-end observability, while authentication, authorization, audit logging, and PII protection ensure compliance with enterprise banking requirements. This design emphasizes security, reliability, scalability, and traceability.
