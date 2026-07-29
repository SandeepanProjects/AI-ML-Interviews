Checkpointing is one of the **most important production features in LangGraph**. It allows your graph to **save its state after each node**, so it can resume execution later instead of starting from the beginning.

This is essential for:

* Human-in-the-loop workflows
* Long-running agents
* Multi-agent systems
* Fault recovery
* Durable execution
* Conversation persistence

A common interview question is:

* What is checkpointing?
* Why do we need checkpoints?
* How does LangGraph resume execution?
* How is checkpointing different from conversation memory?

---

# What is Checkpointing?

A checkpoint is a **snapshot of the graph state** stored after one or more nodes execute.

Without checkpointing:

```text
START

↓

Retrieve

↓

Generate

↓

Tool Call

↓

Crash ❌

↓

Restart

↓

START AGAIN
```

Everything must run again.

With checkpointing:

```text
START

↓

Retrieve

✔ Save Checkpoint

↓

Generate

✔ Save Checkpoint

↓

Tool

↓

Crash ❌

↓

Restart

↓

Resume From Generate ✔
```

The workflow resumes from the last saved state.

---

# Why is Checkpointing Important?

Imagine a travel booking agent.

```text
User

↓

Find Flights

↓

Find Hotels

↓

Ask User

↓

Book Flight

↓

Book Hotel
```

The agent asks:

> Do you approve this itinerary?

The user replies **2 hours later**.

Without checkpointing:

```text
Everything runs again.
```

The agent searches flights and hotels again.

With checkpointing:

```text
Resume exactly where
the user left.
```

Huge improvement.

---

# Graph State

Suppose our state is:

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    documents: list[str]
    answer: str
```

Initially:

```python
{
    "question": "Explain LangGraph",
    "documents": [],
    "answer": ""
}
```

---

# Execution Without Checkpointing

```text
State

↓

Retrieve

↓

Generate

↓

Crash
```

After restart:

```text
State Reset

↓

Retrieve Again

↓

Generate Again
```

---

# Execution With Checkpointing

```text
State

↓

Retrieve

↓

Checkpoint Saved

↓

Generate

↓

Checkpoint Saved

↓

Crash
```

Restart:

```text
Load Checkpoint

↓

Continue
```

---

# Simple In-Memory Checkpointer

LangGraph provides an in-memory checkpointer that's useful for development and testing.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

memory = InMemorySaver()

builder = StateGraph(dict)

builder.add_node("hello", lambda state: {"message": "Hello"})
builder.add_edge(START, "hello")
builder.add_edge("hello", END)

graph = builder.compile(
    checkpointer=memory
)
```

The graph now saves state in memory after node execution.

> Note: This data is lost when the process stops, so it is **not suitable for production**.

---

# Executing With a Thread ID

A checkpoint belongs to a specific execution (thread).

```python
config = {
    "configurable": {
        "thread_id": "user-123"
    }
}

result = graph.invoke(
    {},
    config=config
)
```

The `thread_id` identifies which checkpoint to load when execution resumes.

---

# Why Thread IDs Matter

Suppose two users interact simultaneously.

```text
User A

↓

Thread A

↓

Checkpoint A
```

```text
User B

↓

Thread B

↓

Checkpoint B
```

Each conversation has an independent checkpoint.

---

# Human-in-the-Loop Example

Workflow:

```text
START

↓

Plan Trip

↓

Ask Human

↓

WAIT

↓

Resume

↓

Book Hotel

↓

END
```

Node:

```python
def plan_trip(state):
    return {
        "plan": "Flight + Hotel"
    }
```

The graph pauses after `plan_trip`.

Later:

```text
User clicks

Approve
```

The graph loads the checkpoint and continues instead of recomputing the plan.

---

# Example Graph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    count: int


def step1(state):
    print("Step 1")
    return {
        "count": state["count"] + 1
    }


def step2(state):
    print("Step 2")
    return {
        "count": state["count"] + 1
    }


builder = StateGraph(State)

builder.add_node("step1", step1)
builder.add_node("step2", step2)

builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", END)

graph = builder.compile(
    checkpointer=InMemorySaver()
)

config = {
    "configurable": {
        "thread_id": "workflow-1"
    }
}

result = graph.invoke(
    {"count": 0},
    config=config
)

print(result)
```

Output:

```text
Step 1
Step 2

{'count': 2}
```

After each node, the state is checkpointed.

---

# Production Checkpointing

For production, store checkpoints in a durable database.

Typical architecture:

```text
              User

                │

                ▼

             FastAPI

                │

                ▼

            LangGraph

                │

      Save Checkpoint

                │

                ▼

          PostgreSQL
```

If the application restarts:

```text
FastAPI

↓

Load Checkpoint

↓

Resume Workflow
```

---

# Long-Running Agent

Suppose an insurance claim workflow takes several hours.

```text
Upload Documents

↓

OCR

↓

Fraud Detection

↓

Human Review

↓

Approval

↓

Payment
```

Without checkpoints:

```text
Crash

↓

Everything Restarts
```

With checkpoints:

```text
Crash

↓

Resume

↓

Human Review
```

---

# Checkpoint vs Memory

These two concepts are often confused.

| Checkpointing                              | Conversation Memory                        |
| ------------------------------------------ | ------------------------------------------ |
| Saves workflow execution state             | Stores information from conversations      |
| Used to resume execution                   | Used to answer future questions            |
| Internal graph mechanism                   | User-facing context                        |
| Temporary or durable                       | Often long-lived                           |
| Includes node state and execution progress | Includes chat history, summaries, or facts |

Example:

Conversation:

```text
User:

My name is Alice.
```

Memory stores:

```text
User name = Alice
```

Checkpoint stores:

```text
Current Node = Generate

Messages = [...]

Tool Result = ...

Retry Count = 2
```

---

# Checkpoint + Human Approval

```text
Retrieve

↓

Generate

↓

Checkpoint Saved

↓

WAIT

↓

Manager Approves

↓

Resume

↓

Send Email
```

No previous work is repeated.

---

# Failure Recovery

```text
Node A

↓

Checkpoint

↓

Node B

↓

Checkpoint

↓

Node C

↓

Crash
```

Restart:

```text
Load Checkpoint

↓

Node C
```

Instead of:

```text
Node A

↓

Node B

↓

Node C
```

---

# Multi-Agent Checkpointing

```text
Planner

↓

Checkpoint

↓

Research Agent

↓

Checkpoint

↓

Writer Agent

↓

Checkpoint

↓

Reviewer Agent
```

Every agent's progress is recoverable.

---

# Production Architecture

```text
                   User

                     │

                     ▼

                  FastAPI

                     │

                     ▼

                 LangGraph

          ┌──────────┼──────────┐

          ▼          ▼          ▼

      Planner     Tools     Retriever

          │          │          │

          └──────────┼──────────┘

                     ▼

              Checkpointer

                     ▼

               PostgreSQL

                     ▼

          Resume After Crash
```

---

# Best Practices

### Use durable storage

Use **PostgreSQL** (or another persistent backend) in production instead of an in-memory checkpointer.

---

### Use unique thread IDs

Each workflow execution or conversation should have a unique `thread_id`.

---

### Keep state compact

Only store the information needed to resume execution.

---

### Checkpoint before waiting

Save state before:

* Human approval
* Long-running API calls
* External workflows
* Potentially unreliable operations

---

### Combine with tracing

Use checkpointing together with LangSmith/OpenTelemetry to know both:

* **Where** execution stopped (trace)
* **How** to resume it (checkpoint)

---

# Common Interview Questions

### Why not just rerun the graph?

Some workflows may take minutes or hours, call expensive APIs, or require human approval. Re-running wastes time, money, and may produce different results.

---

### Does checkpointing replace memory?

No.

Checkpointing stores the **execution state of the graph**, while memory stores **information the agent uses across conversations or interactions**.

---

### What is stored in a checkpoint?

Typically:

* Current graph state
* Current node or execution position
* Messages
* Tool outputs
* Retry counters
* Intermediate results
* Metadata needed to resume execution

---

# Senior AI Engineer Interview Answer

> **Checkpointing in LangGraph is the mechanism for persisting workflow execution state so a graph can resume from the last completed step instead of restarting after interruptions. After each node (or configured execution point), the current state is saved using a checkpointer. In production, I use a durable backend such as PostgreSQL and identify each workflow with a unique `thread_id`. Checkpointing is essential for human-in-the-loop approvals, long-running agents, crash recovery, and durable execution. It complements conversation memory by preserving workflow progress rather than storing user knowledge.**


Resuming from a checkpoint is one of the **most powerful features of LangGraph**. It enables **durable execution**, meaning an agent can stop (because of a crash, human approval, or process restart) and later continue from where it left off.

A common interview question is:

* **How does LangGraph resume execution?**
* **How is state restored?**
* **How do thread IDs work?**
* **How do you pause and resume a workflow?**

---

# How Resume Works

Suppose we have this graph:

```text
START

↓

Retrieve Documents

↓

Generate Answer

↓

Human Approval

↓

Send Email

↓

END
```

Imagine execution stops here:

```text
START

↓

Retrieve Documents ✔

↓

Generate Answer ✔

↓

Human Approval

↓

WAIT
```

The graph saves a checkpoint.

Later:

```text
User Clicks Approve

↓

Load Checkpoint

↓

Continue

↓

Send Email

↓

END
```

Notice:

* Retrieve is **NOT** executed again.
* Generate is **NOT** executed again.
* Execution resumes at the next step.

---

# Step 1: Create a Graph

```python
from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    question: str
    answer: str
```

---

# Step 2: Define Nodes

```python
def retrieve(state):

    print("Retrieving...")

    return {
        "answer": "LangGraph uses checkpoints."
    }


def send(state):

    print("Sending answer...")

    print(state["answer"])

    return {}
```

---

# Step 3: Compile with a Checkpointer

```python
memory = InMemorySaver()

builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("send", send)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "send")
builder.add_edge("send", END)

graph = builder.compile(
    checkpointer=memory
)
```

Without a checkpointer:

```text
Graph

↓

Runs Once

↓

State Lost
```

With a checkpointer:

```text
Graph

↓

Checkpoint Saved

↓

Resume Later
```

---

# Step 4: Execute Using a Thread ID

Every workflow needs a unique identifier.

```python
config = {

    "configurable": {

        "thread_id": "user-100"

    }

}
```

Execute:

```python
result = graph.invoke(

    {

        "question": "Explain LangGraph",

        "answer": ""

    },

    config=config

)
```

The checkpoint is stored under:

```text
Thread

user-100
```

---

# Why Thread IDs Matter

Imagine two users.

```text
User A

↓

Thread A

↓

Checkpoint A
```

```text
User B

↓

Thread B

↓

Checkpoint B
```

Each workflow resumes independently.

---

# Viewing the Saved State

The checkpointer stores something similar to:

```python
{

    "question": "Explain LangGraph",

    "answer": "LangGraph uses checkpoints."

}
```

along with execution metadata, including the point in the graph where execution stopped.

---

# Human Approval Example

Suppose we pause after generating an answer.

```text
START

↓

Retrieve

↓

Generate

↓

WAIT

↓

Resume

↓

Send Email
```

State:

```python
class AgentState(TypedDict):

    answer: str

    approved: bool
```

Generate node:

```python
def generate(state):

    return {

        "answer": "Final Answer"

    }
```

Approval node:

```python
def wait_for_human(state):

    print("Waiting...")

    return {}
```

The graph stops here.

Checkpoint:

```python
{

    "answer": "Final Answer",

    "approved": False

}
```

---

# Resume Later

Two hours later:

```python
config = {

    "configurable": {

        "thread_id": "user-100"

    }

}
```

Resume:

```python
graph.invoke(

    {

        "approved": True

    },

    config=config

)
```

LangGraph loads the previous checkpoint, merges the new input (`approved=True`), and continues execution from the saved position instead of restarting from `START`.

Conceptually, the resumed state becomes:

```python
{

    "answer": "Final Answer",

    "approved": True

}
```

The workflow proceeds to the next node.

---

# Real Production Example

Imagine a loan approval workflow.

```text
START

↓

Collect Documents

↓

Fraud Detection

↓

Manager Approval

↓

WAIT

↓

Resume

↓

Approve Loan

↓

END
```

Checkpoint:

```python
{

    "customer": "Alice",

    "risk_score": 0.18,

    "documents": [...],

    "approved": False

}
```

After the manager approves:

```python
graph.invoke(

    {

        "approved": True

    },

    config=config

)
```

The workflow resumes directly from the approval stage.

---

# Crash Recovery

Without checkpoints:

```text
Node 1

↓

Node 2

↓

Crash

↓

Restart

↓

Node 1 Again
```

With checkpoints:

```text
Node 1 ✔

↓

Checkpoint

↓

Node 2 ✔

↓

Checkpoint

↓

Crash

↓

Restart

↓

Resume After Node 2
```

---

# Using a Database Checkpointer

In production, use a persistent backend instead of `InMemorySaver`.

Typical architecture:

```text
            FastAPI

               │

               ▼

           LangGraph

               │

         Checkpointer

               │

               ▼

          PostgreSQL
```

After a restart:

```text
Application

↓

Load Checkpoint

↓

Resume Workflow
```

---

# Complete Example

```python
from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    counter: int


def node1(state):

    print("Node 1")

    return {
        "counter": state["counter"] + 1
    }


def node2(state):

    print("Node 2")

    return {
        "counter": state["counter"] + 1
    }


memory = InMemorySaver()

builder = StateGraph(State)

builder.add_node("n1", node1)
builder.add_node("n2", node2)

builder.add_edge(START, "n1")
builder.add_edge("n1", "n2")
builder.add_edge("n2", END)

graph = builder.compile(
    checkpointer=memory
)

config = {
    "configurable": {
        "thread_id": "workflow-1"
    }
}

result = graph.invoke(
    {"counter": 0},
    config=config
)

print(result)
```

Output:

```text
Node 1

Node 2

{'counter': 2}
```

If execution pauses after `node1`, LangGraph records both:

* The current state (`counter = 1`)
* The next node to execute (`node2`)

When resumed with the same `thread_id`, execution continues from `node2`.

---

# What Happens Internally?

```text
Initial State

↓

Node 1

↓

Save Checkpoint

State:
counter=1

Next Node:
node2

↓

Crash

↓

Restart

↓

Load Checkpoint

↓

State Restored

↓

Run node2

↓

END
```

LangGraph persists **both the graph state and the execution position**.

---

# Best Practices

### Use persistent storage

* Development: `InMemorySaver`
* Production: PostgreSQL-backed or another durable checkpointer

---

### Use stable thread IDs

Examples:

```text
chat-user-123

order-987

workflow-456
```

Avoid generating a new thread ID for every request if you intend to resume the same workflow.

---

### Save before waiting

Checkpoint before:

* Human approval
* External API callbacks
* Long-running jobs
* Expensive computations

---

### Keep state serializable

The state should contain JSON-serializable or otherwise supported data types. Avoid storing database connections, open file handles, or other non-serializable objects.

---

# Interview Questions

### What is required to resume a workflow?

A checkpointer configured for the graph and the same `thread_id` used for the original execution.

---

### What is stored?

A checkpoint typically contains:

* Current graph state
* Execution position (which node to run next)
* Messages
* Tool outputs
* Metadata required to continue execution

---

### Does it rerun previous nodes?

No. LangGraph restores the saved state and continues from the last checkpoint instead of replaying completed nodes.

---

# Senior AI Engineer Interview Answer

> **LangGraph resumes execution by loading a previously saved checkpoint associated with a specific `thread_id`. The checkpoint contains both the graph state and the execution position, allowing the workflow to continue from the next node rather than restarting. During development, I use `InMemorySaver`, while in production I use a durable backend such as PostgreSQL. This enables human-in-the-loop workflows, crash recovery, and long-running agents to resume reliably without repeating completed work.**
