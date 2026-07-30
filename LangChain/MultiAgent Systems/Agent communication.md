Agent communication is one of the most important concepts in **LangGraph**, **LangChain**, **AutoGen**, and **CrewAI**.

A common interview question is:

* **How do multiple agents communicate?**
* **Do agents call each other directly?**
* **How is data passed between agents?**
* **How does LangGraph implement agent communication?**

The answer depends on the framework, but in **LangGraph**, agents **communicate through shared graph state**, not by directly invoking each other.

---

# How Agents Communicate

Suppose we have three agents:

* Research Agent
* Writer Agent
* Reviewer Agent

The workflow looks like this:

```text
              User

                │

                ▼

         Research Agent

                │
      Updates Shared State

                ▼

          Writer Agent

                │
      Reads Shared State

                ▼

         Reviewer Agent

                │
      Updates Shared State

                ▼

              END
```

Notice something important:

* Research Agent never calls Writer Agent.
* Writer Agent never calls Reviewer Agent.

Instead:

* One agent **writes** to state.
* The next agent **reads** from state.

This loose coupling makes systems easier to scale and test.

---

# Shared State

```python
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    research_notes: str
    draft: str
    feedback: str
    final_answer: str
```

Initially

```python
state = {
    "query": "Explain transformers",
    "research_notes": "",
    "draft": "",
    "feedback": "",
    "final_answer": ""
}
```

---

# Research Agent

Research agent retrieves information.

```python
def research_agent(state):

    notes = """
Transformer uses self-attention.
Introduced in 2017.
Encoder-decoder architecture.
"""

    return {
        "research_notes": notes
    }
```

State becomes

```python
{
    "query":"Explain transformers",

    "research_notes":
        "Transformer uses self-attention...",

    "draft":"",

    "feedback":""
}
```

The agent has communicated by updating the shared state.

---

# Writer Agent

The writer reads what the researcher produced.

```python
def writer_agent(state):

    draft = f"""
Article

{state['research_notes']}
"""

    return {
        "draft": draft
    }
```

Notice:

The writer never calls:

```python
research_agent()
```

Instead it reads

```python
state["research_notes"]
```

---

# Reviewer Agent

```python
def reviewer_agent(state):

    if len(state["draft"]) < 100:

        return {
            "feedback": "Too short"
        }

    return {
        "feedback": "Looks good",
        "final_answer": state["draft"]
    }
```

Again,

Communication happens through state.

---

# LangGraph Workflow

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

builder.add_node("research", research_agent)
builder.add_node("writer", writer_agent)
builder.add_node("reviewer", reviewer_agent)

builder.add_edge(START, "research")
builder.add_edge("research", "writer")
builder.add_edge("writer", "reviewer")
builder.add_edge("reviewer", END)

graph = builder.compile()
```

Execution

```python
result = graph.invoke({

    "query":"Explain transformers"

})
```

Flow

```text
Research

↓

State Updated

↓

Writer

↓

State Updated

↓

Reviewer

↓

END
```

---

# State Evolution

### Initial

```python
{
    "query":"Explain transformers"
}
```

---

### After Research

```python
{
    "query":"Explain transformers",

    "research_notes":
        "Self-attention..."
}
```

---

### After Writer

```python
{
    "query":"Explain transformers",

    "research_notes":"...",

    "draft":"Complete article..."
}
```

---

### After Reviewer

```python
{
    "query":"Explain transformers",

    "research_notes":"...",

    "draft":"...",

    "feedback":"Looks good",

    "final_answer":"..."
}
```

Every agent contributes part of the state.

---

# Communication Using Messages

Sometimes agents exchange structured messages rather than custom fields.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

Research Agent

```python
def researcher(state):

    return {
        "messages": [

            AIMessage(
                content="Found three papers."
            )

        ]
    }
```

Writer Agent

```python
def writer(state):

    history = state["messages"]

    latest = history[-1].content

    return {
        "messages":[

            AIMessage(
                content=f"Writing report using: {latest}"
            )

        ]
    }
```

The `add_messages` reducer automatically appends new messages to the existing history.

---

# Parallel Communication

Two agents can update different parts of the state simultaneously.

```text
               Planner

                  │

        ┌─────────┴──────────┐

        ▼                    ▼

Research Agent         SQL Agent

        │                    │

        ▼                    ▼

 research_notes        sql_result

        └─────────┬──────────┘

                  ▼

            Merge Agent
```

State

```python
class State(TypedDict):

    research_notes: str

    sql_result: str

    report: str
```

Merge Agent

```python
def merge(state):

    report = f"""
Research

{state["research_notes"]}

SQL

{state["sql_result"]}
"""

    return {
        "report": report
    }
```

---

# Communication Through Tools

Sometimes one agent exposes a capability as a tool that another agent can use.

```python
from langchain_core.tools import tool

@tool
def search_company(name: str) -> str:
    return f"Information about {name}"
```

Planner

```python
def planner(state):

    return {

        "next_action":
            "search_company"

    }
```

Executor

```python
result = search_company.invoke({

    "name":"OpenAI"

})
```

Here:

* Planner communicates **what** to do.
* Tool performs the work.

---

# Supervisor Communication

Supervisor

```text
User

↓

Supervisor

↓

Research Agent

↓

State Updated

↓

Writer Agent

↓

State Updated

↓

END
```

Supervisor reads

```python
state["research_notes"]
```

and decides:

```python
return "writer"
```

No direct function calls between workers.

---

# Communication with Checkpointing

```text
Research

↓

Checkpoint

↓

Writer

↓

Crash

↓

Restart

↓

Load State

↓

Reviewer
```

The recovered state is exactly how agents continue communicating after a restart.

---

# Communication Through Memory

Long-term memory can also be shared.

Research Agent

```python
memory.save(
    "Customer prefers PDF reports."
)
```

Writer Agent

```python
preference = memory.load(
    "Customer prefers PDF reports."
)
```

This is different from graph state:

* **Graph state** → current workflow
* **Memory** → information reused across workflows

---

# Enterprise Architecture

```text
                    User

                      │

                      ▼

               Supervisor Agent

                      │

      ┌───────────────┼────────────────┐

      ▼               ▼                ▼

 Research Agent   SQL Agent     API Agent

      │               │                │

      └───────────────┼────────────────┘

                      ▼

                Shared Graph State

                      ▼

                 Writer Agent

                      ▼

                Reviewer Agent

                      ▼

                     END
```

Every agent reads from and writes to the shared state.

---

# Communication Patterns

| Pattern          | Description                                     | Example                     |
| ---------------- | ----------------------------------------------- | --------------------------- |
| Shared State     | Agents exchange information through graph state | LangGraph                   |
| Messages         | Agents append conversation history              | `messages` + `add_messages` |
| Tools            | One agent requests work via tools               | Planner → Search Tool       |
| Supervisor       | Central coordinator routes work                 | Supervisor → Workers        |
| Planner–Executor | Planner creates tasks, executor performs them   | Plan → Execute              |
| Memory           | Agents share long-term knowledge                | User preferences            |

---

# Best Practices

1. **Never tightly couple agents**

Avoid:

```python
research_agent(state)
writer_agent(state)
```

inside another agent.

Prefer graph edges and shared state.

---

2. **Keep state structured**

Instead of

```python
state["data"]
```

use

```python
state["research_notes"]
state["sql_result"]
state["draft"]
```

---

3. **Use reducers for concurrent updates**

If multiple agents update the same field (for example, `messages`), define a reducer such as `add_messages` so updates are merged instead of overwritten.

---

4. **Separate workflow state from long-term memory**

* Workflow state: current execution
* Memory: persistent user or application knowledge

---

# Common Interview Questions

### Do agents call each other directly?

In LangGraph, generally **no**. Agents communicate by reading from and writing to the shared graph state, while graph edges determine execution order.

---

### How does one agent send information to another?

By updating the shared state. The next agent reads the updated fields it needs.

---

### What if two agents update the same state?

Use **reducers** (for example, `add_messages`) or design the state so each agent owns different fields. Without a reducer, conflicting updates can overwrite each other.

---

# Senior AI Engineer Interview Answer

> **In LangGraph, agents communicate through a shared graph state rather than directly invoking each other. Each agent reads the fields it needs from the state, performs its task, and returns a partial state update. LangGraph merges these updates and passes the resulting state to the next node in the workflow. For conversational data, I typically use a `messages` field with the `add_messages` reducer, while for business workflows I use structured fields such as `research_notes`, `sql_result`, and `draft`. This approach keeps agents loosely coupled, makes workflows easier to test, supports checkpointing and recovery, and scales well to complex multi-agent systems.
