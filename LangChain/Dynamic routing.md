Dynamic routing is one of the **core features that makes LangGraph different from a traditional chain**.

Instead of following a **fixed sequence of nodes**, the graph **decides at runtime which node to execute next based on the current state**.

Think of it as an **`if-else` statement for your workflow**.

---

# Static Flow vs Dynamic Routing

## Static Flow (Traditional Chain)

The execution path is fixed.

```text
User
  │
  ▼
Retrieve
  │
  ▼
Generate
  │
  ▼
END
```

Code:

```python
retrieve()

generate()
```

No matter what the question is, both nodes always execute.

---

## Dynamic Routing (LangGraph)

The next node depends on the current state.

```text
              User
                │
                ▼
             Router
          ┌─────┴─────┐
          ▼           ▼
     Web Search    Calculator
          │           │
          └─────┬─────┘
                ▼
            Generate
                │
                ▼
               END
```

The router examines the question and chooses the appropriate path.

---

# Why Do We Need Dynamic Routing?

Suppose users ask different questions.

### User 1

```text
"What is LangGraph?"
```

Needs:

```text
Retrieve Documents

↓

Generate
```

---

### User 2

```text
"What is 542 × 39?"
```

Needs:

```text
Calculator

↓

Generate
```

---

### User 3

```text
"What is today's weather?"
```

Needs:

```text
Weather API

↓

Generate
```

Running every tool for every question would be:

* Slow
* Expensive
* Unnecessary

Instead, the workflow chooses only the required branch.

---

# Example State

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    answer: str
```

---

# Router Function

A router is simply a Python function.

```python
def router(state: AgentState):

    question = state["question"].lower()

    if "weather" in question:
        return "weather"

    if "calculate" in question:
        return "calculator"

    return "retriever"
```

Notice:

It **does not execute anything**.

It only returns the **name of the next node**.

---

# Building the Graph

```python
from langgraph.graph import StateGraph

builder = StateGraph(AgentState)

builder.add_node("weather", weather_node)
builder.add_node("calculator", calculator_node)
builder.add_node("retriever", retrieve_node)
builder.add_node("generate", generate_node)

builder.set_entry_point("router")
```

Now connect the router.

```python
builder.add_conditional_edges(
    "router",
    router,
    {
        "weather": "weather",
        "calculator": "calculator",
        "retriever": "retriever"
    }
)
```

This mapping means:

```text
Router returns "weather"

↓

Execute Weather Node
```

---

# Execution Example

Initial state:

```python
{
    "question": "What's today's weather?"
}
```

Router:

```python
def router(state):

    if "weather" in state["question"]:
        return "weather"

    return "retriever"
```

Returns:

```python
"weather"
```

Graph:

```text
Router

↓

Weather Node

↓

Generate

↓

END
```

The retriever is skipped entirely.

---

# Another Example

Input:

```python
{
    "question": "Explain RAG"
}
```

Router:

```python
return "retriever"
```

Execution:

```text
Router

↓

Retriever

↓

Generate

↓

END
```

---

# State Flow During Routing

Initial state:

```python
{
    "question": "Calculate 100 + 50",
    "answer": ""
}
```

Router reads:

```python
state["question"]
```

Returns:

```python
"calculator"
```

Calculator node:

```python
def calculator(state):

    return {

        "answer": "150"

    }
```

State becomes:

```python
{
    "question": "Calculate 100 + 50",

    "answer": "150"
}
```

---

# Routing Based on Previous Nodes

The router doesn't have to use the user's question.

It can use **any value already stored in the state**.

Example:

```python
class AgentState(TypedDict):

    retrieved_docs: list

    answer: str
```

Router:

```python
def router(state):

    if len(state["retrieved_docs"]) == 0:

        return "web_search"

    return "generate"
```

Flow:

```text
Retrieve

↓

Documents Found?

      │

  Yes │ No

      ▼

Generate    Web Search
```

This is a common RAG pattern.

---

# Multi-Agent Routing

Planner decides which specialist agent should work.

```text
              Planner

          /      |      \

         ▼       ▼       ▼

 Finance  Legal  HR Agent

          \      |      /

             Generator
```

Planner:

```python
def planner(state):

    q = state["question"]

    if "invoice" in q:
        return "finance"

    if "contract" in q:
        return "legal"

    return "hr"
```

Each question reaches the correct expert.

---

# Retry Routing

Dynamic routing is also used for retries.

```text
Generator

↓

Good Answer?

  │

Yes│No

  ▼

END

Retry
```

Router:

```python
def retry_router(state):

    if state["confidence"] < 0.8:

        return "retry"

    return "__end__"
```

---

# Human Approval Routing

```text
Planner

↓

Human Approval?

 │

Yes│No

 ▼

Continue

 END
```

Router:

```python
def approval_router(state):

    if state["approved"]:

        return "execute"

    return "__end__"
```

---

# Retrieval Grader Example

This is a common production workflow.

```text
Retrieve

↓

Grade Documents

↓

Enough Context?

   │

Yes│No

 ▼

Generate

Web Search

↓

Retrieve Again
```

Router:

```python
def grader_router(state):

    if state["score"] > 0.8:

        return "generate"

    return "web_search"
```

---

# How LangGraph Executes Dynamic Routing

Suppose the router returns:

```python
return "calculator"
```

LangGraph internally does something conceptually like:

```python
next_node = router(state)

execute(next_node)
```

If it returns:

```python
"weather"
```

LangGraph executes the weather node instead.

---

# Dynamic Routing vs Conditional Edges

They are closely related.

* **Dynamic routing** is the concept: choosing the next node at runtime.
* **Conditional edges** are the LangGraph mechanism that implements dynamic routing.

Example:

```python
builder.add_conditional_edges(
    "router",
    router,
    {
        "search": "search",
        "calculator": "calculator",
        "generate": "generate"
    }
)
```

The `router` function returns one of the keys (`"search"`, `"calculator"`, or `"generate"`), and LangGraph follows the corresponding edge.

---

# Production Example

Imagine an enterprise customer support agent.

```text
                     User
                       │
                       ▼
                 Intent Classifier
          ┌─────────┼─────────┐
          ▼         ▼         ▼
     Billing     Technical   Sales
          │         │         │
          └─────────┼─────────┘
                    ▼
              Response Generator
                    ▼
                   END
```

Router:

```python
def intent_router(state):

    intent = state["intent"]

    if intent == "billing":
        return "billing_agent"

    if intent == "technical":
        return "tech_agent"

    return "sales_agent"
```

Only the relevant specialist executes, reducing latency and cost.

---

# Interview Questions

### Why use dynamic routing instead of a chain?

Because a chain has a fixed execution path. Dynamic routing allows the workflow to make decisions at runtime based on the current state, improving efficiency and flexibility.

---

### Can the router modify the state?

Typically, the router only **reads** the state and returns the next node. If you need to update the state (for example, classify an intent and store it), it's cleaner to use a separate node that updates the state, followed by a router that reads that new field.

---

### Can routing create loops?

Yes. You can route back to an earlier node.

Example:

```text
Retrieve

↓

Grade

↓

Low Score?

 │

Yes│No

 ▼

Retrieve Again

END
```

This is how iterative retrieval and retry workflows are implemented. You should also include safeguards such as retry counters or maximum iteration limits to prevent infinite loops.

---

# Senior AI Engineer Interview Answer

> **Dynamic routing is the ability to choose the next node in a LangGraph workflow at runtime based on the current graph state. Instead of following a fixed sequence, a router function inspects state—such as the user's question, retrieved documents, confidence score, or intent—and returns the identifier of the next node. LangGraph implements this using conditional edges, enabling efficient workflows like tool selection, retry loops, retrieval grading, human approval checkpoints, and multi-agent orchestration while avoiding unnecessary computation.**
