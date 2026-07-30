The **Planner–Executor Pattern** is one of the **most important agent design patterns** in LangGraph and enterprise AI systems.

It is used when a task is **too complex to solve in one LLM call**. Instead of asking one agent to do everything, the work is divided into two responsibilities:

1. **Planner** → decides **what needs to be done**
2. **Executor** → performs each step

This pattern is widely used in:

* Research assistants
* Coding agents
* Financial analysis
* Enterprise copilots
* Data analytics
* Autonomous workflows

It is one of the most common **Senior AI Engineer** interview questions.

---

# Why Do We Need a Planner?

Suppose a user asks:

> Find the latest AI regulations in Europe, summarize them, compare them with US regulations, and generate a report.

A single LLM call must:

* Search
* Read documents
* Compare regulations
* Summarize
* Write the report

```text
User
  │
  ▼
 One LLM
  │
  ▼
 Huge Prompt
```

Problems:

* Large context
* Poor reasoning
* Hard to recover if one step fails
* Difficult to monitor

---

## Planner–Executor Solution

```text
                 User
                   │
                   ▼
               Planner
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
   Search      Compare      Report
      │            │            │
      ▼            ▼            ▼
          Executor Executes
                   │
                   ▼
             Final Response
```

The planner **thinks**, the executor **acts**.

---

# Responsibilities

## Planner

The planner:

* Understands the goal
* Breaks work into steps
* Chooses tools
* Orders execution
* Updates the plan if needed

Example plan:

```text
1. Search EU regulations
2. Search US regulations
3. Compare
4. Generate report
```

---

## Executor

The executor:

* Executes one step
* Calls tools
* Stores results
* Returns updated state

It **doesn't create plans**.

---

# Real Enterprise Example

User:

> Analyze last month's sales and email the report.

Planner generates:

```text
Step 1 → Query sales database

Step 2 → Calculate revenue

Step 3 → Create report

Step 4 → Email report
```

Executor performs each step sequentially.

---

# State Definition

```python
from typing import TypedDict

class AgentState(TypedDict):
    user_query: str
    plan: list[str]
    current_step: int
    results: list[str]
    final_answer: str
```

Initially:

```python
{
    "user_query":
        "Analyze monthly sales",

    "plan": [],

    "current_step": 0,

    "results": [],

    "final_answer": ""
}
```

---

# Step 1: Planner Node

```python
def planner(state: AgentState):

    plan = [

        "Query database",

        "Calculate revenue",

        "Generate report"

    ]

    return {

        "plan": plan,

        "current_step": 0

    }
```

State becomes:

```python
{
    "plan": [

        "Query database",

        "Calculate revenue",

        "Generate report"

    ]
}
```

---

# Step 2: Executor Node

```python
def executor(state: AgentState):

    step = state["plan"][
        state["current_step"]
    ]

    print("Executing:", step)

    result = f"Finished: {step}"

    return {

        "results":
            state["results"] + [result],

        "current_step":
            state["current_step"] + 1

    }
```

---

# Execution Flow

Iteration 1

```text
Current Step

0

↓

Query Database

↓

Result Stored
```

State:

```python
{

"results":[

"Finished Query Database"

],

"current_step":1

}
```

---

Iteration 2

```text
Current Step

1

↓

Calculate Revenue

↓

Store Result
```

---

Iteration 3

```text
Current Step

2

↓

Generate Report

↓

Done
```

---

# Router

We loop until every step is complete.

```python
def should_continue(state):

    if state["current_step"] < len(state["plan"]):

        return "executor"

    return "finish"
```

---

# Finish Node

```python
def finish(state):

    report = "\n".join(

        state["results"]

    )

    return {

        "final_answer": report

    }
```

---

# Complete LangGraph

```python
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

builder = StateGraph(AgentState)

builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("finish", finish)

builder.add_edge(START, "planner")

builder.add_edge(
    "planner",
    "executor"
)

builder.add_conditional_edges(

    "executor",

    should_continue,

    {

        "executor": "executor",

        "finish": "finish"

    }

)

builder.add_edge("finish", END)

graph = builder.compile()
```

Execution:

```python
result = graph.invoke({

    "user_query":
    "Analyze sales",

    "plan": [],

    "results": [],

    "current_step": 0,

    "final_answer": ""

})
```

---

# Graph Flow

```text
             START

               │

               ▼

            Planner

               │

               ▼

           Executor

               │

       More Steps?

       │         │

      Yes       No

       │         │

       ▼         ▼

   Executor    Finish

       │

       ▼

      END
```

---

# Example: Research Agent

User:

> Compare LangGraph and AutoGen.

Planner:

```text
1 Search LangGraph

2 Search AutoGen

3 Compare

4 Write Summary
```

Executor:

```text
Execute Step 1

↓

Execute Step 2

↓

Execute Step 3

↓

Execute Step 4
```

---

# Example: Coding Agent

User:

> Build a REST API.

Planner:

```text
Create project

↓

Create FastAPI app

↓

Add endpoints

↓

Write tests

↓

Generate Dockerfile
```

Executor performs one step at a time.

---

# Dynamic Replanning

Sometimes a plan becomes invalid.

Example:

```text
Step 2

↓

Database Offline

↓

Planner Updates Plan

↓

Use Backup Database

↓

Executor Continues
```

Planner:

```python
def planner(state):

    if state.get("database_down"):

        return {

            "plan":[

                "Query backup DB",

                "Generate report"

            ]

        }

    return {

        "plan":[

            "Query primary DB",

            "Generate report"

        ]

    }
```

---

# Planner + Tool Calling

Planner:

```text
Need Weather

↓

Use Weather Tool
```

Executor:

```python
weather.invoke(
    city="Bangalore"
)
```

Planner doesn't execute tools.

Executor does.

---

# Planner + Human Approval

```text
Planner

↓

Plan Generated

↓

Human Reviews

↓

Approved

↓

Executor Starts
```

Useful for:

* Loan approval
* Medical diagnosis
* Financial reporting

---

# Planner + Checkpointing

```text
Planner

↓

Checkpoint

↓

Executor Step 1

↓

Checkpoint

↓

Executor Step 2

↓

Crash

↓

Resume Step 2
```

No completed work is repeated.

---

# Enterprise Architecture

```text
                      User

                        │

                        ▼

                    Planner

                        │

      ┌─────────────────┼─────────────────┐

      ▼                 ▼                 ▼

 SQL Query         Retrieval         API Call

      │                 │                 │

      └─────────────────┼─────────────────┘

                        ▼

                    Executor

                        ▼

                 Intermediate Results

                        ▼

                     Final Report
```

---

# Planner vs Supervisor

| Planner–Executor                    | Supervisor                                   |
| ----------------------------------- | -------------------------------------------- |
| Creates a task plan                 | Chooses which agent should work              |
| Breaks work into ordered steps      | Routes work to specialized agents            |
| Executor performs each planned step | Worker agents perform their own domain tasks |
| Best for sequential workflows       | Best for coordinating multiple specialists   |

Example:

Planner:

```text
Search

↓

Summarize

↓

Translate
```

Supervisor:

```text
Research Agent

↓

Finance Agent

↓

Email Agent
```

---

# Planner + Supervisor Together

Large enterprise systems often combine both patterns.

```text
                    User

                      │

                      ▼

                 Supervisor

          ┌─────────┼──────────┐

          ▼         ▼          ▼

   Research Team  Coding Team  Finance Team

          │

          ▼

       Planner

          │

          ▼

      Executor Loop

          │

          ▼

       Final Result
```

The supervisor decides **which specialist team** should handle the request, while each specialist team uses its own planner–executor workflow.

---

# Advantages

| Advantage              | Benefit                     |
| ---------------------- | --------------------------- |
| Smaller prompts        | Better reasoning            |
| Modular execution      | Easier debugging            |
| Retry individual steps | No need to rerun everything |
| Dynamic replanning     | Adapt to failures           |
| Better observability   | Track progress step by step |
| Checkpoint support     | Resume after interruptions  |

---

# Best Practices

* Keep planners focused on decomposition and strategy.
* Keep executors focused on one task at a time.
* Store the plan and current step in the graph state.
* Add retry logic and checkpointing for long-running plans.
* Allow replanning when assumptions change or a step fails.
* Log each planned step and execution result for observability.

---

# Common Interview Questions

### Why separate planning from execution?

Planning and execution require different reasoning. A planner focuses on strategy and task decomposition, while the executor focuses on reliable tool use and task completion. Separating them improves modularity, observability, and recovery.

---

### Can the planner change the plan?

Yes. If execution reveals new information (for example, a tool fails or required data is unavailable), the planner can generate a revised plan and the executor continues from the updated workflow.

---

### Can the executor skip steps?

Typically no. The executor follows the planner's instructions. If a step becomes invalid, control should return to the planner for replanning rather than having the executor invent a new workflow.

---

# Senior AI Engineer Interview Answer

> **The Planner–Executor pattern separates strategic reasoning from execution. The planner analyzes the user's goal, decomposes it into an ordered sequence of tasks, and stores the plan in the graph state. The executor then processes one step at a time, invoking tools, updating state, and looping until all steps are complete. If execution fails or new information appears, the planner can replan and the executor continues with the revised workflow. This architecture is easier to scale, debug, checkpoint, and monitor than relying on a single agent to reason and execute simultaneously.**
