This is one of the **most important LangGraph concepts**. Many developers initially think the state is passed by reference and mutated in place. In reality, **each node receives the current state as input and returns only the fields it wants to update. LangGraph merges those returned fields into the graph state before executing the next node.**

Let's walk through it step by step.

---

# Overall Flow

Suppose we have this graph:

```text
          User
            │
            ▼
        Planner
            │
            ▼
       Retriever
            │
            ▼
       Generator
            │
            ▼
           END
```

Shared state:

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    plan: list
    documents: list
    answer: str
```

Initially:

```python
state = {
    "question": "Explain RAG",
    "plan": [],
    "documents": [],
    "answer": ""
}
```

This single state object represents the execution context.

---

# Step 1: Planner Node

Node function:

```python
def planner(state: AgentState):

    print("Planner received:")
    print(state)

    return {
        "plan": [
            "retrieve documents",
            "generate answer"
        ]
    }
```

Input:

```python
{
    "question": "Explain RAG",
    "plan": [],
    "documents": [],
    "answer": ""
}
```

Return value:

```python
{
    "plan": [
        "retrieve documents",
        "generate answer"
    ]
}
```

Notice the planner **does not return the entire state**.

It returns **only what changed**.

---

# LangGraph Merges the Update

Internally, LangGraph performs something conceptually similar to:

```python
graph_state.update(node_output)
```

So the new state becomes:

```python
{
    "question": "Explain RAG",

    "plan": [
        "retrieve documents",
        "generate answer"
    ],

    "documents": [],

    "answer": ""
}
```

The unchanged fields remain intact.

---

# Step 2: Retriever Node

```python
def retrieve(state: AgentState):

    print(state["plan"])

    docs = [
        "RAG retrieves relevant documents.",
        "Retrieved context improves accuracy."
    ]

    return {
        "documents": docs
    }
```

Input received:

```python
{
    "question": "Explain RAG",

    "plan": [
        "retrieve documents",
        "generate answer"
    ],

    "documents": [],

    "answer": ""
}
```

Return:

```python
{
    "documents": [
        "RAG retrieves relevant documents.",
        "Retrieved context improves accuracy."
    ]
}
```

Merged state:

```python
{
    "question": "Explain RAG",

    "plan": [
        "retrieve documents",
        "generate answer"
    ],

    "documents": [
        "RAG retrieves relevant documents.",
        "Retrieved context improves accuracy."
    ],

    "answer": ""
}
```

---

# Step 3: Generator Node

```python
def generate(state: AgentState):

    context = "\n".join(state["documents"])

    answer = f"""
Question:
{state['question']}

Context:
{context}
"""

    return {
        "answer": answer
    }
```

Input:

```python
{
    "question": "Explain RAG",

    "plan": [...],

    "documents": [...],

    "answer": ""
}
```

Return:

```python
{
    "answer": "Question...\nContext..."
}
```

Merged state:

```python
{
    "question": "Explain RAG",

    "plan": [...],

    "documents": [...],

    "answer": "Question...\nContext..."
}
```

Execution completes.

---

# Visualizing State Flow

```text
Initial State
────────────────────────────────────

question = Explain RAG

plan = []

documents = []

answer = ""


            │

            ▼

Planner

returns

{
   plan = [...]
}


            │

            ▼

LangGraph merges

question = Explain RAG

plan = [...]

documents = []

answer = ""


            │

            ▼

Retriever

returns

{
   documents=[...]
}


            │

            ▼

LangGraph merges

question = Explain RAG

plan = [...]

documents = [...]

answer=""


            │

            ▼

Generator

returns

{
    answer="..."
}


            │

            ▼

Final State
```

---

# Updating Multiple Fields

A node can update more than one field.

```python
def retrieve(state):

    docs = retriever.invoke(state["question"])

    return {

        "documents": docs,

        "retrieval_count": len(docs),

        "retrieval_time": 0.45
    }
```

All returned keys are merged into the state.

---

# Adding New Fields

Suppose your state is:

```python
class AgentState(TypedDict):

    question: str

    answer: str

    confidence: float
```

Generator:

```python
return {

    "answer": answer,

    "confidence": 0.92
}
```

Updated state:

```python
{
    "question":"...",

    "answer":"...",

    "confidence":0.92
}
```

---

# Reading State

Every node can access values from previous nodes.

```python
def reviewer(state):

    print(state["answer"])

    print(state["documents"])
```

Because earlier nodes populated those fields.

---

# Conditional Routing Uses State

```python
def router(state):

    if len(state["documents"]) == 0:

        return "web_search"

    return "generate"
```

Here the routing decision depends entirely on the current state.

---

# What About Lists?

Suppose:

```python
state = {

    "messages": [
        "Hello"
    ]
}
```

If your node returns:

```python
return {

    "messages": [
        "How are you?"
    ]
}
```

By default, that **replaces** the old list rather than appending to it.

To accumulate values across nodes, LangGraph provides reducer mechanisms.

Example (using `Annotated` with an append reducer):

```python
from typing import Annotated, TypedDict
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list[str], add]
```

Now:

Node A:

```python
return {
    "messages": ["Hello"]
}
```

Node B:

```python
return {
    "messages": ["How are you?"]
}
```

Final state:

```python
{
    "messages": [
        "Hello",
        "How are you?"
    ]
}
```

Without the reducer, the second return would overwrite the first list.

---

# Don't Mutate the State Directly

Avoid this pattern:

```python
def planner(state):

    state["plan"] = ["retrieve"]

    return state
```

While it may appear to work in simple cases, it's not the recommended pattern because it makes updates harder to reason about.

Instead, prefer:

```python
def planner(state):

    return {

        "plan": [
            "retrieve"
        ]
    }
```

This clearly communicates what changed and lets LangGraph manage state updates consistently.

---

# Complete Example

```python
from typing import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    question: str
    documents: list
    answer: str


def retrieve(state):
    return {
        "documents": [
            "LangGraph is a workflow framework."
        ]
    }


def generate(state):
    return {
        "answer": f"Answer based on: {state['documents'][0]}"
    }


builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.set_finish_point("generate")

graph = builder.compile()

result = graph.invoke(
    {
        "question": "Explain LangGraph",
        "documents": [],
        "answer": ""
    }
)

print(result)
```

Output:

```python
{
    "question": "Explain LangGraph",

    "documents": [
        "LangGraph is a workflow framework."
    ],

    "answer": "Answer based on: LangGraph is a workflow framework."
}
```

---

# Production Example

A more realistic state definition might look like:

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    user_id: str
    question: str
    messages: Annotated[list, add]
    retrieved_docs: list
    tool_results: dict
    retries: int
    answer: str
```

Different nodes update different parts of the state:

* **Planner** → `plan`
* **Retriever** → `retrieved_docs`
* **Tool node** → `tool_results`
* **Generator** → `answer`
* **Reviewer** → `feedback`
* **Retry node** → `retries`

Each node only returns the fields it changes, and LangGraph merges those updates into the shared state before moving to the next node.

---

# Senior AI Engineer Interview Answer

> **In LangGraph, the graph state is the shared execution context that flows through every node. Each node receives the current state as input, performs its work, and returns only the fields it wants to update. LangGraph automatically merges those updates into the existing state before executing the next node. This immutable-style update pattern makes workflows deterministic, simplifies debugging, supports checkpointing and retries, and enables multiple nodes to collaborate through a shared state without relying on global variables.**
