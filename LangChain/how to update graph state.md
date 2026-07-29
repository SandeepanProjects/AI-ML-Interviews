Updating graph state is **the core concept of LangGraph**. Every node **receives the current state**, performs some work, and **returns only the fields it wants to update**. LangGraph then merges those updates into the shared graph state.

This is one of the most common interview questions:

* How does state flow between nodes?
* How do nodes update state?
* What happens if multiple nodes update the same field?
* How do reducers work?
* How do you implement loops?

---

# What is Graph State?

The graph state is a shared object that is passed between nodes.

```text
             Initial State

{
    messages: [],
    user: "Alice",
    current_step: 0,
    documents: [],
    answer: ""
}

            │
            ▼

          Node A

            │
     Update messages

            ▼

          Node B

            │
      Update documents

            ▼

          Node C

            │
      Update answer
```

Every node sees the latest state.

---

# Step 1: Define the State

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: list[BaseMessage]
    documents: list[str]
    answer: str
    current_step: int
```

Initial state:

```python
state = {
    "messages": [],
    "documents": [],
    "answer": "",
    "current_step": 0
}
```

---

# How State Updates Work

A node **does not mutate** the state directly.

Instead, it **returns a dictionary containing only the updated fields**.

```python
def node(state: AgentState):

    return {
        "answer": "Hello"
    }
```

LangGraph internally merges it.

Before:

```python
{
    "messages": [],
    "answer": "",
    "current_step": 0
}
```

Node returns:

```python
{
    "answer": "Hello"
}
```

Final state:

```python
{
    "messages": [],
    "answer": "Hello",
    "current_step": 0
}
```

Only `answer` changed.

---

# Example 1: Update a Counter

State:

```python
from typing import TypedDict

class CounterState(TypedDict):
    counter: int
```

Node:

```python
def increment(state: CounterState):

    return {
        "counter": state["counter"] + 1
    }
```

Initial:

```python
{
    "counter": 5
}
```

After:

```python
{
    "counter": 6
}
```

---

# Example 2: Update Messages

```python
from langchain_core.messages import AIMessage

class ChatState(TypedDict):
    messages: list
```

Node:

```python
def chatbot(state: ChatState):

    response = AIMessage(
        content="Hello!"
    )

    return {
        "messages": state["messages"] + [response]
    }
```

Before:

```python
[
    HumanMessage("Hi")
]
```

After:

```python
[
    HumanMessage("Hi"),
    AIMessage("Hello!")
]
```

---

# Example 3: Update Multiple Fields

```python
class AgentState(TypedDict):

    documents: list[str]

    answer: str

    step: int
```

Node:

```python
def generate(state):

    return {

        "answer": "Paris",

        "step": state["step"] + 1
    }
```

Result:

```python
{

    "documents": [...],

    "answer": "Paris",

    "step": 2
}
```

---

# Real RAG Workflow

```text
Question

↓

Retrieve Node

↓

Generate Node

↓

END
```

State:

```python
class AgentState(TypedDict):

    question: str

    documents: list[str]

    answer: str
```

Retrieve node:

```python
def retrieve(state):

    docs = retriever.invoke(
        state["question"]
    )

    return {

        "documents": docs
    }
```

Generate node:

```python
def generate(state):

    context = "\n".join(
        state["documents"]
    )

    answer = llm.invoke(
        f"""
Context:
{context}

Question:
{state['question']}
"""
    )

    return {

        "answer": answer.content
    }
```

Notice how:

* The retrieve node only updates `documents`.
* The generate node uses `documents` and updates `answer`.

---

# Example 4: Tool Result

State:

```python
class AgentState(TypedDict):

    tool_result: str
```

Node:

```python
def calculator_node(state):

    result = str(eval("25*30"))

    return {

        "tool_result": result
    }
```

Output:

```python
{

    "tool_result": "750"
}
```

---

# Example 5: Track Progress

```python
class WorkflowState(TypedDict):

    current_step: int

    status: str
```

Node:

```python
def planner(state):

    return {

        "current_step":
            state["current_step"] + 1,

        "status":
            "retrieving"
    }
```

Execution:

```text
Step 0

↓

Planner

↓

Step 1

↓

Retriever

↓

Step 2

↓

Generator
```

---

# Updating State in a Loop

Suppose an agent retries until enough documents are found.

State:

```python
class AgentState(TypedDict):

    retry_count: int

    documents: list[str]
```

Retriever:

```python
def retrieve(state):

    docs = retriever.invoke(
        "question"
    )

    return {

        "documents": docs,

        "retry_count":

            state["retry_count"] + 1
    }
```

Each iteration updates:

```text
Retry

0

↓

1

↓

2

↓

3
```

---

# Conditional Updates

```python
def grade(state):

    if len(state["documents"]) >= 3:

        return {

            "status": "good"
        }

    return {

        "status": "retry"
    }
```

State changes depending on the condition.

---

# Updating State with Reducers

Without reducers:

```python
return {

    "messages":

        state["messages"]

        + [AIMessage("Hi")]
}
```

With a reducer (such as `add_messages`):

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]
```

Now the node simply returns:

```python
def chatbot(state):

    return {

        "messages": [
            AIMessage(
                content="Hello"
            )
        ]
    }
```

The reducer appends the new message instead of replacing the list.

---

# Full LangGraph Example

```python
from typing import TypedDict

from langgraph.graph import StateGraph


class State(TypedDict):
    count: int


def node1(state):
    return {
        "count": state["count"] + 1
    }


def node2(state):
    return {
        "count": state["count"] + 5
    }


builder = StateGraph(State)

builder.add_node("n1", node1)
builder.add_node("n2", node2)

builder.set_entry_point("n1")
builder.add_edge("n1", "n2")
builder.set_finish_point("n2")

graph = builder.compile()

result = graph.invoke(
    {
        "count": 0
    }
)

print(result)
```

Execution:

```text
Initial

count = 0

↓

Node1

count = 1

↓

Node2

count = 6
```

Final output:

```python
{
    "count": 6
}
```

---

# Updating State After Tool Execution

```python
class AgentState(TypedDict):

    messages: list

    tool_result: str
```

Tool node:

```python
def weather_tool(state):

    result = weather_api("Bangalore")

    return {

        "tool_result": result
    }
```

LLM node:

```python
def chatbot(state):

    prompt = f"""
Weather:

{state["tool_result"]}
"""

    response = llm.invoke(prompt)

    return {

        "messages": [

            AIMessage(
                content=response.content
            )
        ]
    }
```

---

# State Flow Through Nodes

```text
Initial State
{
    question
}

        │

        ▼

Retrieve Node

Returns

{
    documents
}

        │

        ▼

Generate Node

Returns

{
    answer
}

        │

        ▼

END
```

LangGraph merges each node's returned values into the shared state before invoking the next node.

---

# Common Mistakes

### ❌ Mutating the state directly

```python
def bad_node(state):
    state["count"] += 1
    return state
```

This works against LangGraph's update model and can lead to confusing behavior.

Prefer returning only the changes:

```python
def good_node(state):
    return {
        "count": state["count"] + 1
    }
```

---

### ❌ Returning the entire state unnecessarily

```python
return state
```

Instead:

```python
return {
    "answer": answer
}
```

Return only the fields you changed.

---

### ❌ Replacing accumulated messages

If your state uses a reducer like `add_messages`, don't reconstruct the full message list manually.

Instead of:

```python
return {
    "messages": state["messages"] + [new_message]
}
```

Use:

```python
return {
    "messages": [new_message]
}
```

The reducer will append it automatically.

---

# Best Practices

* Design state with clear, purpose-specific fields (messages, documents, tool results, metadata).
* Return only updated fields from each node.
* Use reducers (`add_messages` or custom reducers) for fields updated by multiple nodes.
* Keep nodes stateless; all shared information should flow through the graph state.
* Add counters (for retries or iterations) to support loops and prevent infinite execution.
* Persist state using a production checkpointer (for example, PostgreSQL-backed) if workflows must survive restarts.

---

# Senior AI Engineer Interview Answer

> **In LangGraph, each node receives the current immutable graph state and returns a partial state update rather than modifying the state in place. LangGraph merges these updates before executing the next node. For simple fields like strings or counters, the latest returned value replaces the old one. For accumulative fields such as chat messages, I define reducers like `add_messages` so multiple nodes can safely append new messages. This approach keeps nodes independent, makes workflows deterministic, and enables retries, conditional routing, and checkpointing without shared mutable state.**
