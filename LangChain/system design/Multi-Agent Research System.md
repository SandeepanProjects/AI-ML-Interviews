A **Multi-Agent Research System** is one of the most common **Senior AI Engineer**, **Staff AI Engineer**, and **AI Architect** interview questions because it demonstrates:

* Multi-agent collaboration
* LangGraph workflows
* Tool calling
* RAG
* Planning
* Parallel execution
* Reflection
* Human-in-the-loop
* Memory
* Checkpointing
* Production monitoring

Real companies like OpenAI, Anthropic, Microsoft, Google, and Perplexity use similar architectures for deep research.

---

# Problem Statement

The user asks:

> "Research the impact of Generative AI on the banking industry and generate a report."

Instead of using one LLM call, the system should:

1. Understand the task
2. Break it into subtasks
3. Research multiple sources
4. Verify the information
5. Remove duplicates
6. Summarize findings
7. Write a report
8. Review the report
9. Return the final answer

---

# High-Level Architecture

```text
                    User
                      │
                      ▼
              FastAPI / API Gateway
                      │
                      ▼
              LangGraph Supervisor
                      │
      ┌───────────────┼───────────────────┐
      ▼               ▼                   ▼
 Planner Agent   Research Agent     Memory Agent
      │
      ▼
 Parallel Research Tasks
      │
 ┌──────────────┬──────────────┬──────────────┐
 ▼              ▼              ▼
Web Agent   RAG Agent     Database Agent
      │              │              │
      └──────────────┼──────────────┘
                     ▼
              Evidence Merger
                     ▼
             Reflection Agent
                     ▼
              Writer Agent
                     ▼
              Reviewer Agent
                     ▼
                 Final Report
```

Notice that the **Planner never performs research**. It only creates a plan.

---

# Responsibilities of Each Agent

| Agent          | Responsibility              |
| -------------- | --------------------------- |
| Planner        | Break work into subtasks    |
| Web Research   | Search web                  |
| RAG Research   | Search enterprise documents |
| Database Agent | Retrieve structured data    |
| Reflection     | Detect missing evidence     |
| Writer         | Produce report              |
| Reviewer       | Improve quality             |
| Supervisor     | Coordinate workflow         |

---

# Folder Structure

```text
research_system/

app/
│
├── graph/
│
├── agents/
│      planner.py
│      web.py
│      rag.py
│      database.py
│      reflection.py
│      writer.py
│      reviewer.py
│
├── tools/
├── prompts/
├── retriever/
├── memory/
├── monitoring/
└── api/
```

---

# Shared State

Every agent communicates through graph state.

```python
from typing import TypedDict

class ResearchState(TypedDict):

    question: str

    plan: list

    web_docs: list

    rag_docs: list

    db_results: list

    merged_context: list

    report: str

    review: str

    approved: bool
```

Each node updates part of the state.

---

# Planner Agent

The planner decomposes the task.

```python
def planner(state):

    return {

        "plan":[

            "Research AI adoption",

            "Research regulations",

            "Research risks",

            "Research market trends"

        ]
    }
```

Result:

```text
Plan

↓

AI Adoption

Regulations

Risks

Market Trends
```

---

# Parallel Research

Instead of sequential execution:

```text
Planner

↓

Web

↓

RAG

↓

Database
```

Run them simultaneously.

```text
            Planner
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
     Web       RAG      Database
```

LangGraph supports fan-out/fan-in workflows.

---

# Web Research Agent

```python
def web_agent(state):

    docs = web_search.invoke(

        state["question"]

    )

    return {

        "web_docs": docs

    }
```

---

# RAG Agent

```python
def rag_agent(state):

    docs = retriever.invoke(

        state["question"]

    )

    return {

        "rag_docs": docs

    }
```

---

# Database Agent

```python
def database_agent(state):

    rows = sql_tool.invoke(

        {

            "query":"SELECT ..."

        }

    )

    return {

        "db_results": rows

    }
```

---

# Merge Evidence

Combine all sources.

```python
def merge(state):

    merged = (

        state["web_docs"]

        + state["rag_docs"]

        + state["db_results"]

    )

    return {

        "merged_context": merged

    }
```

---

# Reflection Agent

Reflection checks whether the evidence is sufficient.

```python
def reflection(state):

    if len(state["merged_context"]) < 5:

        return {

            "approved": False

        }

    return {

        "approved": True

    }
```

Conditional routing:

```python
def router(state):

    if state["approved"]:

        return "writer"

    return "web_agent"
```

The graph loops until enough evidence is collected.

```text
Research

↓

Reflection

↓

Enough?

↓

No

↓

Research Again

↓

Yes

↓

Writer
```

---

# Writer Agent

```python
def writer(state):

    report = llm.invoke(

        state["merged_context"]

    )

    return {

        "report": report

    }
```

---

# Reviewer Agent

```python
def reviewer(state):

    improved = review_llm.invoke(

        state["report"]

    )

    return {

        "report": improved

    }
```

The reviewer focuses on clarity, consistency, and completeness rather than introducing new facts.

---

# Build the Graph

```python
from langgraph.graph import StateGraph

builder = StateGraph(ResearchState)

builder.add_node("planner", planner)
builder.add_node("web", web_agent)
builder.add_node("rag", rag_agent)
builder.add_node("database", database_agent)
builder.add_node("merge", merge)
builder.add_node("reflection", reflection)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)
```

Example flow:

```text
START
  │
Planner
  │
  ├──────┐
  ▼      ▼
 Web    RAG
  │      │
  └──┐ ┌─┘
     ▼ ▼
   Database
      │
    Merge
      │
 Reflection
      │
  Enough?
   │    │
  No   Yes
   │    ▼
   └──Writer
         │
      Reviewer
         │
        END
```

---

# Memory

Persist state between turns.

```python
class ResearchState(TypedDict):

    history: list

    report: str
```

Example:

```text
User

Research AI

↓

Report

↓

User

Expand section 3
```

The report is already stored.

---

# Checkpointing

Checkpoint after expensive nodes.

```text
Planner

↓

Checkpoint

↓

Research

↓

Checkpoint

↓

Writer
```

If the workflow crashes during writing, it resumes from the latest checkpoint instead of repeating research.

---

# Human Review

Before publishing:

```text
Writer

↓

Reviewer

↓

Human Approval

↓

Publish
```

For sensitive reports, require explicit approval before release.

---

# Monitoring

Track:

* Planner latency
* Research latency
* Retrieval quality
* Token usage
* Cost
* Loop count
* Retry count
* Final report generation time

Use:

* LangSmith
* OpenTelemetry
* Prometheus
* Grafana

---

# Scaling

For enterprise workloads:

```text
Users
  │
Load Balancer
  │
FastAPI
  │
LangGraph Workers
  │
Redis
PostgreSQL
Vector DB
Search APIs
```

Design principles:

* Stateless workers
* External checkpoint storage
* Parallel research
* Background queues for long-running tasks
* Horizontal scaling with Kubernetes

---

# Why This Architecture?

A single agent has to plan, search, analyze, and write in one prompt, which increases hallucination risk and reduces traceability.

By separating responsibilities:

* Planner focuses on decomposition.
* Research agents gather evidence.
* Reflection validates completeness.
* Writer synthesizes.
* Reviewer improves quality.

Each component can be tested, monitored, and scaled independently.

---

# Common Interview Questions

### Why use multiple agents instead of one?

Multiple specialized agents improve modularity, observability, and reliability. Each agent has a clear responsibility and can use different prompts, tools, or models.

---

### Why include a reflection agent?

Reflection evaluates whether enough evidence has been gathered before writing. This reduces hallucinations and avoids generating reports from weak context.

---

### Can research agents run in parallel?

Yes. Web search, enterprise RAG, and structured database queries are independent tasks and should execute concurrently to reduce latency.

---

### How do agents communicate?

Agents do not call each other directly. They communicate by reading from and writing to the shared LangGraph state. The graph orchestrates execution order.

---

### How do you recover from failures?

Persist checkpoints after expensive stages, retry transient failures, use fallback models or tools where appropriate, and resume execution from the last successful checkpoint.

---

# Senior AI Engineer Interview Answer

> **I design a multi-agent research system using LangGraph with a supervisor-managed workflow. A planner agent decomposes the user's request into research tasks, which are executed in parallel by specialized agents such as web search, enterprise RAG, and database retrieval. Their outputs are merged into a shared graph state, and a reflection agent evaluates whether the collected evidence is sufficient. If not, the workflow loops back for additional retrieval. Once the context is adequate, a writer agent generates the report and a reviewer agent improves its quality before returning the final result. Workflow state is checkpointed in PostgreSQL, retrieval is backed by a vector database, Redis is used for caching, and LangSmith, OpenTelemetry, Prometheus, and Grafana provide end-to-end observability. This architecture is modular, scalable, resilient, and well suited for production-grade research systems.
