Coordinating specialized agents is one of the **core problems in enterprise AI systems**.

In real-world applications, you rarely have a single agent that does everything. Instead, you have multiple specialized agents, each responsible for one domain.

Examples:

* Research Agent → Retrieves information
* SQL Agent → Queries databases
* Code Agent → Writes code
* Finance Agent → Performs financial analysis
* Email Agent → Sends emails
* Reviewer Agent → Validates output

The challenge is **how these agents work together without becoming tightly coupled**.

---

# Real Enterprise Example

Imagine an enterprise copilot.

User asks:

> Find last month's sales, compare them with this month, generate a report, and email it to the manager.

No single agent should perform everything.

Instead:

```text
                 User
                   │
                   ▼
            Supervisor Agent
                   │
      ┌────────────┼──────────────┐
      ▼            ▼              ▼
   SQL Agent   Analytics Agent  Email Agent
      │            │              │
      └────────────┼──────────────┘
                   ▼
              Shared State
                   ▼
             Final Response
```

Each agent has one responsibility.

---

# Why Not Let Agents Call Each Other?

A common beginner approach is:

```python
def sql_agent(query):
    data = run_sql(query)
    analytics_agent(data)

def analytics_agent(data):
    report = generate_report(data)
    email_agent(report)

def email_agent(report):
    send_email(report)
```

Problems:

* Agents are tightly coupled.
* Difficult to test.
* Difficult to replace one agent.
* Impossible to checkpoint individual steps.
* Hard to trace.

---

# Better Architecture

Use a **shared graph state**.

Every agent:

* Reads state
* Updates state
* Returns

No agent directly calls another.

---

# Step 1 — Define Shared State

```python
from typing import TypedDict

class AgentState(TypedDict):
    user_query: str

    sales_data: str

    analytics: str

    report: str

    email_status: str
```

Initially:

```python
{
    "user_query":
        "Compare monthly sales",

    "sales_data":"",

    "analytics":"",

    "report":"",

    "email_status":""
}
```

---

# Step 2 — SQL Agent

Only responsible for database access.

```python
def sql_agent(state):

    result = """
January : ₹12,00,000
February: ₹14,50,000
"""

    return {
        "sales_data": result
    }
```

State becomes

```python
{
    "sales_data":
    """
January : ₹12,00,000
February: ₹14,50,000
"""
}
```

---

# Step 3 — Analytics Agent

Reads SQL output.

```python
def analytics_agent(state):

    analysis = f"""
Sales increased.

Input:

{state["sales_data"]}
"""

    return {
        "analytics": analysis
    }
```

Notice:

No SQL call.

Only state.

---

# Step 4 — Writer Agent

```python
def writer_agent(state):

    report = f"""
Monthly Sales Report

Analysis

{state["analytics"]}
"""

    return {
        "report": report
    }
```

---

# Step 5 — Email Agent

```python
def email_agent(state):

    print("Sending...")

    print(state["report"])

    return {
        "email_status":
            "Sent"
    }
```

---

# Step 6 — Build LangGraph

```python
from langgraph.graph import (
    StateGraph,
    START,
    END
)

builder = StateGraph(AgentState)

builder.add_node("sql", sql_agent)
builder.add_node("analytics", analytics_agent)
builder.add_node("writer", writer_agent)
builder.add_node("email", email_agent)

builder.add_edge(START, "sql")
builder.add_edge("sql", "analytics")
builder.add_edge("analytics", "writer")
builder.add_edge("writer", "email")
builder.add_edge("email", END)

graph = builder.compile()
```

Execution

```python
result = graph.invoke({

    "user_query":
    "Compare monthly sales"

})

print(result)
```

Workflow

```text
START

↓

SQL Agent

↓

Analytics Agent

↓

Writer Agent

↓

Email Agent

↓

END
```

---

# Coordinator (Supervisor) Pattern

A supervisor can dynamically choose which specialist to run.

```python
def supervisor(state):

    query = state["user_query"].lower()

    if "sales" in query:
        return {
            "next_agent": "sql"
        }

    elif "email" in query:
        return {
            "next_agent": "email"
        }

    return {
        "next_agent": "research"
    }
```

Router

```python
def router(state):
    return state["next_agent"]
```

Graph

```python
builder.add_conditional_edges(
    "supervisor",
    router,
    {
        "sql": "sql",
        "research": "research",
        "email": "email"
    }
)
```

The supervisor coordinates specialists without knowing their internal implementation.

---

# Parallel Coordination

Some work is independent.

Example

```text
                 Planner

                    │

        ┌───────────┴───────────┐

        ▼                       ▼

Research Agent            SQL Agent

        │                       │

        ▼                       ▼

 research_notes          sales_data

        └───────────┬───────────┘

                    ▼

               Writer Agent
```

Research and SQL can run simultaneously because neither depends on the other.

Merge

```python
def writer(state):

    report = f"""
Research

{state["research_notes"]}

Sales

{state["sales_data"]}
"""

    return {
        "report": report
    }
```

---

# Planner–Executor Coordination

The planner creates tasks.

```text
Planner

↓

1 Search

2 SQL

3 Report

4 Email
```

Executor executes one task.

```python
def executor(state):

    task = state["plan"][state["step"]]

    if task == "Search":
        ...

    elif task == "SQL":
        ...

    elif task == "Email":
        ...
```

---

# Human-in-the-Loop Coordination

```text
Research

↓

Writer

↓

Reviewer

↓

Human Approval

↓

Email
```

Reviewer

```python
def reviewer(state):

    if "confidential" in state["report"]:
        return {
            "approved": False
        }

    return {
        "approved": True
    }
```

Router

```python
def approval_router(state):

    if state["approved"]:
        return "email"

    return "human_review"
```

---

# Failure Coordination

Suppose SQL fails.

```text
Supervisor

↓

SQL

↓

Failure

↓

Retry

↓

Fallback

↓

Writer
```

```python
def sql_agent(state):

    try:
        return {
            "sales_data": query_db()
        }

    except Exception:

        return {
            "error": "Database unavailable"
        }
```

Supervisor

```python
def supervisor(state):

    if state.get("error"):
        return {
            "next_agent":
            "fallback_sql"
        }

    return {
        "next_agent":
        "writer"
    }
```

---

# Coordination with Checkpointing

```text
Research

↓

Checkpoint

↓

SQL

↓

Checkpoint

↓

Writer

↓

Crash

↓

Resume Writer
```

Previously completed work is preserved.

---

# Production Architecture

```text
                           User
                             │
                             ▼
                    API (FastAPI)
                             │
                             ▼
                   LangGraph Supervisor
                             │
      ┌──────────────────────┼──────────────────────┐
      ▼                      ▼                      ▼
 Research Agent         SQL Agent             Tool Agent
      │                      │                      │
      ▼                      ▼                      ▼
 Vector DB             PostgreSQL              External APIs
      └──────────────────────┼──────────────────────┘
                             ▼
                      Shared Graph State
                             ▼
                       Writer Agent
                             ▼
                      Reviewer Agent
                             ▼
                    Human Approval (Optional)
                             ▼
                            END
```

---

# Communication Between Agents

Agents **do not call each other directly**.

Instead:

```text
Research Agent

writes

↓

research_notes

↓

Writer Agent

reads

↓

research_notes
```

Another example:

```text
SQL Agent

writes

↓

sales_data

↓

Analytics Agent

reads

↓

sales_data
```

The graph state is the communication channel.

---

# Best Practices

### 1. Give every agent one responsibility

Bad:

```text
Research + SQL + Email
```

Good:

```text
Research

SQL

Writer

Email
```

---

### 2. Use shared state

Avoid:

```python
research_agent()

writer_agent()
```

Prefer:

```python
return {
    "research_notes": notes
}
```

---

### 3. Use a coordinator

The supervisor should:

* Route work
* Handle retries
* Manage failures
* Merge outputs

Workers should not orchestrate other workers.

---

### 4. Parallelize independent work

If two agents don't depend on each other, execute them in parallel to reduce latency.

---

### 5. Add observability

Track:

* Which agent executed
* Execution time
* Token usage
* Cost
* Errors
* Retry count

---

# Interview Questions

### Why use specialized agents?

They improve modularity, reduce prompt complexity, simplify testing, and allow independent scaling and optimization.

---

### How do agents communicate?

Through the shared graph state. Each agent reads the fields it needs and writes the fields it owns.

---

### Who decides which agent runs next?

Typically a **Supervisor Agent** or a **Planner**, using conditional routing in LangGraph.

---

### Can agents run in parallel?

Yes. Independent tasks, such as document retrieval and SQL queries, can execute concurrently and later merge their outputs.

---

# Complete Enterprise Flow

```text
                   User Request
                         │
                         ▼
                 Supervisor Agent
                         │
              Classify the Request
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
 Research Agent      SQL Agent      API Agent
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                  Merge Results
                         ▼
                   Writer Agent
                         ▼
                  Reviewer Agent
                         │
                 Approved?
                  │          │
                 No         Yes
                  │          │
                  ▼          ▼
          Human Approval    Email Agent
                  │          │
                  └──────┬───┘
                         ▼
                        END
```

---

# Senior AI Engineer Interview Answer

> **I coordinate specialized agents using a shared graph state and an orchestration layer such as a Supervisor or Planner in LangGraph. Each agent has a single responsibility—for example, retrieval, SQL, report generation, or email delivery—and communicates by reading from and writing to the shared state rather than calling other agents directly. The coordinator decides execution order using conditional routing, supports parallel execution for independent tasks, merges outputs, handles retries and failures, and uses checkpointing so workflows can resume after interruptions. This architecture is modular, scalable, easy to test, and well suited for production AI systems.**
