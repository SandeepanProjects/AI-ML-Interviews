# Build a Mini LangGraph-like Agent (Senior AI Engineer Interview)

One of the most common **Senior AI Engineer**, **LLM Engineer**, and **Agentic AI** interview questions is:

> **"Implement a simplified LangGraph-like agent framework from scratch."**

The goal is to demonstrate that you understand **stateful execution**, **graph-based workflows**, **conditional routing**, **tool execution**, and **extensibility**—not just how to use a framework.

---

# What is LangGraph?

Unlike a simple sequential agent:

```text
User
   │
   ▼
LLM
   │
   ▼
Answer
```

LangGraph models execution as a **graph**:

```text
          +-----------+
          |   START   |
          +-----+-----+
                |
                v
          +-----------+
          |    LLM    |
          +-----+-----+
                |
      +---------+----------+
      |                    |
      v                    v
+-----------+        +-------------+
| Calculator|        |   Search    |
+-----------+        +-------------+
      |                    |
      +---------+----------+
                |
                v
          +-----------+
          |   END     |
          +-----------+
```

Each node:

* Reads state
* Updates state
* Decides next node

---

# Project Structure

```text
mini_langgraph/

    graph.py
    node.py
    state.py
    tools.py
    llm.py
    agent.py
    main.py
```

---

# Step 1 — Define Agent State

Every node shares a common state.

```python
from dataclasses import dataclass, field

@dataclass
class AgentState:
    user_input: str
    tool_result: str = ""
    final_answer: str = ""
    history: list = field(default_factory=list)
```

Unlike passing many variables, everything flows through a single state object.

---

# Step 2 — Base Node

```python
from abc import ABC, abstractmethod

class Node(ABC):

    @abstractmethod
    def run(self, state: AgentState):
        pass
```

Every graph node implements the same interface.

---

# Step 3 — LLM Node

```python
class LLMNode(Node):

    def run(self, state):

        text = state.user_input.lower()

        if any(op in text for op in ["+", "-", "*", "/"]):
            return "calculator"

        return "search"
```

Instead of generating an answer, this node decides which node executes next.

---

# Step 4 — Calculator Node

```python
class CalculatorNode(Node):

    def run(self, state):

        state.tool_result = str(eval(state.user_input))

        return "answer"
```

---

# Step 5 — Search Node

```python
class SearchNode(Node):

    def run(self, state):

        state.tool_result = (
            f"Search result for '{state.user_input}'"
        )

        return "answer"
```

---

# Step 6 — Answer Node

```python
class AnswerNode(Node):

    def run(self, state):

        state.final_answer = (
            f"Final Answer: {state.tool_result}"
        )

        return None
```

Returning `None` means execution stops.

---

# Step 7 — Graph Engine

```python
class Graph:

    def __init__(self):

        self.nodes = {}

    def add_node(self, name, node):

        self.nodes[name] = node

    def run(self, start, state):

        current = start

        while current:

            node = self.nodes[current]

            current = node.run(state)

        return state
```

This is the core execution engine.

---

# Step 8 — Build the Graph

```python
graph = Graph()

graph.add_node("llm", LLMNode())
graph.add_node("calculator", CalculatorNode())
graph.add_node("search", SearchNode())
graph.add_node("answer", AnswerNode())
```

---

# Execute

```python
state = AgentState(
    user_input="20*5"
)

result = graph.run("llm", state)

print(result.final_answer)
```

Output

```text
Final Answer: 100
```

---

# Different Input

```python
state = AgentState(
    user_input="What is Machine Learning?"
)

result = graph.run("llm", state)

print(result.final_answer)
```

Output

```text
Final Answer:
Search result for 'What is Machine Learning?'
```

---

# Execution Flow

For:

```text
2+8
```

Execution becomes:

```text
START
   │
   ▼
LLM
   │
   ▼
Calculator
   │
   ▼
Answer
   │
   ▼
END
```

For:

```text
Who is Alan Turing?
```

```text
START
   │
   ▼
LLM
   │
   ▼
Search
   │
   ▼
Answer
   │
   ▼
END
```

---

# Add Conditional Edges

Instead of embedding routing logic inside nodes, you can separate transitions.

```python
graph.add_edge("llm", "calculator", condition=is_math)

graph.add_edge("llm", "search", condition=is_search)

graph.add_edge("calculator", "answer")

graph.add_edge("search", "answer")
```

This makes workflows easier to modify.

---

# Add Memory

```python
state.history.append(
    {
        "role": "user",
        "content": state.user_input
    }
)
```

Each node can inspect prior history before making decisions.

---

# Add Retry Logic

```python
class SearchNode(Node):

    def run(self, state):

        for attempt in range(3):
            try:
                state.tool_result = search(state.user_input)
                return "answer"
            except Exception:
                pass

        raise RuntimeError("Search failed")
```

---

# Async Version

Many tools are network-bound, so make the graph asynchronous.

```python
class Node:

    async def run(self, state):
        ...
```

Engine:

```python
async def run(self, start, state):

    current = start

    while current:

        current = await self.nodes[current].run(state)

    return state
```

---

# Production Architecture

```text
                User
                  │
                  ▼
            Graph Engine
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
 Planner      Search Tool   Calculator
      │           │            │
      └───────────┼────────────┘
                  ▼
             Update State
                  │
                  ▼
            Next Graph Node
                  │
                  ▼
             Final Response
```

---

# Production Improvements

A production-ready LangGraph-style framework typically includes:

| Feature             | Purpose                                              |
| ------------------- | ---------------------------------------------------- |
| Persistent State    | Resume workflows after failures                      |
| Checkpointing       | Save execution progress between nodes                |
| Parallel Branches   | Execute independent nodes concurrently               |
| Conditional Routing | LLM or rule-based decisions                          |
| Human-in-the-Loop   | Pause for approval before risky actions              |
| Tool Registry       | Dynamically discover and invoke tools                |
| Streaming           | Stream intermediate progress and final tokens        |
| Memory              | Short-term conversation and long-term vector memory  |
| Observability       | Traces, metrics, logs, execution graph visualization |
| Error Recovery      | Retries, fallback models, alternate tool paths       |
| Multi-Agent Support | Delegate subgraphs to specialized agents             |

---

# Time Complexity

If:

* **N** = number of graph nodes executed
* **T** = average tool execution time

Then:

| Operation       | Complexity                       |
| --------------- | -------------------------------- |
| Graph traversal | **O(N)**                         |
| State updates   | **O(1)** per node                |
| Tool lookup     | **O(1)** (dictionary)            |
| Overall         | **O(N + Σ tool execution time)** |

---

# How to Explain This in an Interview

A strong explanation is:

> "I model the agent as a directed graph where each node performs a single responsibility: planning, tool execution, reasoning, or response generation. A shared state object flows through the graph, allowing nodes to read and update context without tight coupling. The graph engine repeatedly executes the current node, which returns the next node to visit. This design naturally supports conditional routing, retries, checkpoints, streaming, parallel branches, and human approval steps. It's the same architectural pattern used by frameworks like LangGraph, but implemented in a lightweight, extensible way."
