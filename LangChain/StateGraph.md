`StateGraph` and `AgentState` are the **foundation of LangGraph**. If you understand these two concepts, you understand how LangGraph works internally.

A simple way to think about them is:

* **StateGraph = Workflow Engine**
* **AgentState = Shared Memory**

Just as a web framework has **routes** and **requests**, LangGraph has **graphs** and **state**.

---

# Why Was StateGraph Introduced?

Suppose you have a simple RAG pipeline.

```text
User

↓

Retrieve Documents

↓

Generate Answer

↓

Return
```

This works for simple applications.

Now suppose you need:

* Planning
* Conditional branching
* Human approval
* Retry logic
* Multiple agents
* Resume after crash

A simple chain is no longer sufficient.

LangGraph solves this by representing your application as a **graph**.

---

# What is a StateGraph?

A **StateGraph** is a directed graph where:

* Nodes perform work
* Edges determine the next node
* Every node reads and updates a shared state

Think of it as a workflow engine.

```text
          Planner
             │
      ┌──────┴──────┐
      ▼             ▼
 Retrieve       Calculator
      │             │
      └──────┬──────┘
             ▼
         Generator
             │
             ▼
            END
```

Unlike a normal chain, the execution path is not always fixed.

---

# Real-Life Analogy

Imagine ordering food online.

```text
Order Received

↓

Restaurant Accepts

↓

Food Prepared

↓

Driver Assigned

↓

Delivered
```

Every step updates the order.

```text
Status = "Preparing"

↓

Status = "Out for Delivery"

↓

Status = "Delivered"
```

The **workflow** is the graph.

The **order information** is the state.

LangGraph works exactly like this.

---

# What is AgentState?

`AgentState` is the shared data that flows through the graph.

Every node receives it.

Every node can update it.

Example:

```python
from typing import TypedDict

class AgentState(TypedDict):

    question: str

    documents: list

    answer: str
```

Initially

```python
state = {
    "question": "Explain LangGraph",
    "documents": [],
    "answer": ""
}
```

---

# Graph Execution

```text
Initial State

↓

Planner

↓

Updated State

↓

Retriever

↓

Updated State

↓

Generator

↓

Final State
```

Notice that the **same object** travels through the graph.

---

# Example

Suppose the user asks

> Explain RAG

Initial state

```python
state = {
    "question": "Explain RAG",
    "documents": [],
    "answer": ""
}
```

---

## Planner Node

```python
def planner(state):

    print(state)

    return {
        "plan": [
            "retrieve",
            "generate"
        ]
    }
```

Output

```python
{
    "plan": [
        "retrieve",
        "generate"
    ]
}
```

State becomes

```python
{
    "question": "Explain RAG",

    "plan": [
        "retrieve",
        "generate"
    ],

    "documents": [],

    "answer": ""
}
```

---

## Retriever Node

```python
def retrieve(state):

    docs = [
        "RAG combines retrieval and generation.",
        "It improves factual accuracy."
    ]

    return {
        "documents": docs
    }
```

State now

```python
{
    "question": "Explain RAG",

    "plan": [...],

    "documents": [

        "RAG combines retrieval...",

        "Improves factual accuracy"
    ],

    "answer": ""
}
```

---

## Generator Node

```python
def generate(state):

    context = "\n".join(
        state["documents"]
    )

    answer = f"""
Question:
{state["question"]}

Context:
{context}
"""

    return {
        "answer": answer
    }
```

Final state

```python
{
    "question": "Explain RAG",

    "documents": [...],

    "answer": "..."
}
```

---

# Building a StateGraph

## Step 1: Define State

```python
from typing import TypedDict

class AgentState(TypedDict):

    question: str

    documents: list

    answer: str
```

---

## Step 2: Create Graph

```python
from langgraph.graph import StateGraph

builder = StateGraph(AgentState)
```

Here you tell LangGraph:

> Every node will receive an `AgentState`.

---

## Step 3: Add Nodes

```python
builder.add_node(
    "retrieve",
    retrieve
)

builder.add_node(
    "generate",
    generate
)
```

Each node must:

* Receive state
* Return updated fields

---

## Step 4: Connect Nodes

```python
builder.set_entry_point("retrieve")

builder.add_edge(
    "retrieve",
    "generate"
)

builder.set_finish_point(
    "generate"
)
```

Workflow

```text
Retrieve

↓

Generate

↓

END
```

---

## Step 5: Compile

```python
graph = builder.compile()
```

Now execute

```python
result = graph.invoke({

    "question": "Explain LangGraph",

    "documents": [],

    "answer": ""
})
```

---

# State Before and After

Input

```python
{
    "question":"Explain LangGraph",

    "documents":[],

    "answer":""
}
```

↓

Retriever

↓

```python
{
    "documents":[
        "...",
        "..."
    ]
}
```

↓

Generator

↓

```python
{
    "answer":"LangGraph..."
}
```

↓

Output

```python
{
    "question":"Explain LangGraph",

    "documents":[...],

    "answer":"LangGraph..."
}
```

---

# Conditional Branching

The graph can decide where to go next.

```text
           Planner

         /          \

        /            \

Search Needed?     No Search

      │                 │

      ▼                 ▼

 Retriever          Generator

      │                 │

      └───────┬─────────┘

              ▼

             END
```

Decision function

```python
def router(state):

    if "weather" in state["question"]:

        return "search"

    return "generate"
```

Graph

```python
builder.add_conditional_edges(
    "planner",
    router
)
```

Unlike a chain, the path is determined at runtime.

---

# AgentState in Multi-Agent Systems

```text
Planner

↓

Shared State

↓

Research Agent

↓

Writer Agent

↓

Reviewer Agent
```

All agents read and update the same state.

Example

```python
class AgentState(TypedDict):

    question: str

    research: str

    draft: str

    review: str

    final_answer: str
```

---

# Checkpointing

State enables recovery.

```text
Planner

↓

Checkpoint

↓

Retriever

↓

Checkpoint

↓

Generator
```

If the application crashes

```text
Restart

↓

Load State

↓

Resume
```

The graph continues instead of starting over.

---

# Why Not Use Global Variables?

Bad approach:

```python
documents = []

answer = ""
```

Problems:

* Not thread-safe
* Doesn't support concurrent users
* Hard to debug
* Difficult to persist
* Difficult to resume

State solves these issues by making data explicit and isolated per execution.

---

# Production Example

```python
from typing import TypedDict

class AgentState(TypedDict):

    user_id: str

    question: str

    retrieved_docs: list

    tool_results: list

    retries: int

    messages: list

    final_answer: str
```

Each node updates only the fields it owns.

Example

Planner

```python
return {

    "plan": [...],

    "retries": 0
}
```

Retriever

```python
return {

    "retrieved_docs": docs
}
```

Generator

```python
return {

    "final_answer": answer
}
```

The state gradually becomes richer as execution progresses.

---

# StateGraph vs Chain

| Feature        | Chain     | StateGraph        |
| -------------- | --------- | ----------------- |
| Execution      | Linear    | Graph             |
| Branching      | Limited   | Yes               |
| Shared state   | No        | Yes               |
| Checkpointing  | No        | Yes               |
| Human approval | Difficult | Built-in patterns |
| Multi-agent    | Limited   | Yes               |
| Pause/Resume   | No        | Yes               |

---

# StateGraph vs AgentExecutor

| StateGraph                | AgentExecutor               |
| ------------------------- | --------------------------- |
| Explicit workflow         | Implicit reasoning loop     |
| Persistent state          | Limited state               |
| Checkpointing             | No native checkpointing     |
| Multi-agent orchestration | Primarily single-agent      |
| Easier to debug           | Harder to inspect execution |
| Production orchestration  | Preferred                   |

---

# Senior AI Engineer Interview Answer

A concise interview answer is:

> **StateGraph is LangGraph's workflow engine. It represents an application as a graph of nodes connected by edges, where each node performs a task and updates shared state. AgentState is that shared state—a structured object, often defined with `TypedDict` or a Pydantic model, that flows through every node. Each node reads the current state, performs work, and returns updates that are merged into the state. This design enables conditional branching, checkpointing, pause/resume, human-in-the-loop workflows, retries, and multi-agent orchestration while keeping the workflow deterministic and easy to debug.**
