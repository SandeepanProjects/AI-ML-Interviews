## What is a Cyclic Graph in LangGraph?

A **cyclic graph** is a graph where execution can **return back to a previously executed node**.

In a normal workflow (acyclic graph), execution moves only forward:

```text
id="2u9m9s"
Start
  |
  v
Node A
  |
  v
Node B
  |
  v
Node C
  |
  v
END
```

Once you leave a node, you never come back.

This is called a **DAG (Directed Acyclic Graph)**.

---

A cyclic graph allows loops:

```text
id="0i2f8b"
          ┌──────────────┐
          │              │
          ▼              │
      Retrieve ──► Grade │
          │              │
          ▼              │
      Generate           │
          │              │
          ▼              │
       Answer Good? ─────┘
```

Execution can go:

```
Retrieve
   ↓
Grade
   ↓
Not Good
   ↓
Retrieve again
```

---

# Why Are Cycles Needed in AI Agents?

Real AI systems are not always one-pass.

An agent often needs to:

* Try something
* Evaluate the result
* Improve it
* Try again

This naturally creates loops.

---

# Example 1: RAG Retrieval Improvement Loop

A basic RAG pipeline:

```text
id="3xwz5b"
Question

   ↓

Retriever

   ↓

LLM

   ↓

Answer
```

Problem:

What if retrieval returns bad documents?

Example:

User:

```
Explain Kubernetes GPU autoscaling
```

Retriever returns:

```
Docker basics
```

LLM may hallucinate.

---

A better design:

```text
id="4x7mpt"
                 ┌───────────────┐
                 │               │
                 ▼               │
             Retriever           │
                 │               │
                 ▼               │
           Retrieval Grader      │
                 │               │
        ┌────────┴────────┐      │
        │                 │      │
        ▼                 ▼      │
    Good Context       Bad Context
        │                 │      │
        ▼                 │      │
     Generate             │      │
                          │      │
                          └──────┘
```

Flow:

```
Retrieve
   |
Grade
   |
Poor?
   |
Yes
   |
Improve query
   |
Retrieve again
```

This is a cycle.

---

# Example 2: Agent Tool Retry Loop

An agent calls an API.

```text
id="p0s9ry"
Agent

 ↓

Tool Call

 ↓

Success?

 ├── Yes → Continue

 └── No → Retry Tool
```

Example:

```python
def tool_router(state):

    if state["tool_failed"]:
        return "retry_tool"

    return "continue"
```

Graph:

```text
id="z7c2fw"
Tool

 ↓

Error Handler

 ↓

Tool
```

The tool node connects back to itself.

---

# Example 3: Reflection Pattern

Reflection means:

1. Generate output
2. Critique it
3. Improve it

Graph:

```text
id="s7s8vv"
              ┌───────────┐
              │           │
              ▼           │
          Generator       │
              │           │
              ▼           │
          Reviewer        │
              │           │
       Good enough?       │
              │           │
       No ─────────────────┘

       Yes

        ↓

       END
```

Example:

Generator:

```
Write a marketing email.
```

Reviewer:

```
Too generic. Add customer benefits.
```

Generator runs again:

```
Improved email
```

---

# Example 4: Human-in-the-Loop Approval

Enterprise workflows often require approval.

```text
id="m0b7w9"
Agent

 ↓

Create Proposal

 ↓

Human Review

      |
      |
  Approved?
      |
 ┌────┴─────┐
 │          │
Yes        No
 │          │
 ▼          ▼
Deploy    Modify
           |
           |
           └───────┐
                   |
                   ▼
                Review
```

The human can reject and send the workflow backward.

---

# Example 5: Planning Agent

A planner may discover missing information.

```text
id="1q7z2a"
Planner

 ↓

Execute Plan

 ↓

Check Result

 ↓

Need More Steps?

        |
        |
       Yes

        ↓

      Planner
```

The agent keeps refining the plan.

---

# How to Create Cycles in LangGraph

Example:

```python
from langgraph.graph import StateGraph


builder = StateGraph(AgentState)


builder.add_node(
    "retrieve",
    retrieve
)

builder.add_node(
    "grade",
    grade
)

builder.add_node(
    "generate",
    generate
)


builder.add_edge(
    "retrieve",
    "grade"
)


builder.add_conditional_edges(
    "grade",
    grade_router,
    {
        "retry": "retrieve",
        "generate": "generate"
    }
)
```

Here:

```text
retrieve
    |
    v
 grade
    |
    |
    +-------> retrieve
```

is a cycle.

---

# Important: Cycles Need Stopping Conditions

A dangerous graph:

```text
id="3s6f7b"
Retrieve

  ↓

Grade

  ↓

Bad

  ↓

Retrieve

  ↓

Grade

  ↓

Bad

  ↓

Retrieve

  ...
```

This becomes an infinite loop.

---

## Add Maximum Iterations

State:

```python
class AgentState(TypedDict):

    question: str

    retries: int
```

Router:

```python
def router(state):

    if state["retries"] >= 3:
        return "generate"

    return "retrieve"
```

Now:

```
Attempt 1
Attempt 2
Attempt 3
Stop
```

---

# Production Cycle Example

A production RAG agent:

```text
id="x2bq8s"
                User Query

                    |
                    v

              Query Analyzer

                    |
                    v

               Retriever

                    |
                    v

            Retrieval Evaluator

             /             \

            /               \

       Good Context       Poor Context

            |                 |

            v                 |

        Generator             |

            |                 |

            v                 |

       Answer Checker         |

             |                |

        Correct?              |

          |                   |

       No --------------------┘

          |

         Yes

          |

        Response
```

---

# Cyclic Graph vs DAG

| Feature         | DAG          | Cyclic Graph    |
| --------------- | ------------ | --------------- |
| Direction       | Forward only | Can go backward |
| Loops           | No           | Yes             |
| Retries         | Hard         | Natural         |
| Reflection      | Difficult    | Easy            |
| Agent behavior  | Limited      | Flexible        |
| RAG improvement | Basic        | Advanced        |
| Self-correction | No           | Yes             |

---

# Why Cycles Are Powerful for Agents

Modern AI agents are not simple functions:

```
Input → Output
```

They behave more like humans:

```
Think
 ↓
Act
 ↓
Observe
 ↓
Evaluate
 ↓
Improve
 ↓
Act again
```

Cycles enable:

✅ Self-correction
✅ Tool retries
✅ Better retrieval
✅ Reflection
✅ Planning refinement
✅ Human feedback loops
✅ Autonomous agents

---

# Interview Answer (Senior AI Engineer)

> **A cyclic graph is a graph where execution can return to a previously executed node. In LangGraph, cycles enable iterative workflows where an agent can evaluate its own output and decide whether to continue, retry, or improve. They are useful for patterns such as retrieval refinement, reflection, tool retries, planning loops, and human-in-the-loop approval. However, cycles must have termination conditions such as maximum iterations, confidence thresholds, or explicit stop states to prevent infinite execution loops.**
