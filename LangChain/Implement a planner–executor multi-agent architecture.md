This is one of the **most important Senior AI Engineer system design and coding interview questions**.

Companies like **OpenAI, Microsoft, Google, Amazon, Salesforce, Anthropic, Databricks, Snowflake, EY, Accenture, and Deloitte** frequently ask about planner–executor architectures because this pattern scales well for complex AI workflows.

Interviewers want to evaluate whether you understand:

* Multi-agent systems
* Task decomposition
* Planning vs execution
* State management
* Tool calling
* Parallel execution
* Recovery and retries
* Production orchestration

---

# What is a Planner–Executor Architecture?

Instead of asking one LLM to solve everything in one prompt, split the work into:

* **Planner Agent** → Breaks the task into smaller tasks.
* **Executor Agent(s)** → Complete each task.
* **Coordinator** → Collects results and produces the final answer.

Architecture

```text
                    User Request
                          │
                          ▼
                   Planner Agent
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      Task 1          Task 2          Task 3
          │               │               │
          ▼               ▼               ▼
    Executor A      Executor B      Executor C
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                    Result Aggregator
                          │
                          ▼
                    Final Response
```

---

# Example Problem

User asks:

> Research LangGraph, compare it with LangChain, and write a summary.

Instead of one prompt:

```text
LLM

↓

Huge Prompt

↓

Huge Response
```

Planner–Executor

```text
Planner

↓

Research LangGraph

Compare

Write Summary

↓

Executors

↓

Results

↓

Final Report
```

---

# Step 1: Define State

```python
from typing import TypedDict

class AgentState(TypedDict):
    user_request: str
    plan: list[str]
    completed_tasks: list[str]
    final_answer: str
```

Example

```python
{
    "user_request": "Compare LangChain and LangGraph",
    "plan": [],
    "completed_tasks": [],
    "final_answer": ""
}
```

---

# Step 2: Planner Agent

The planner converts a large goal into smaller tasks.

```python
from langchain_openai import ChatOpenAI

planner = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)
```

Prompt

```python
PLAN_PROMPT = """
You are an AI planner.

Break the task into
small executable steps.

Return a numbered list.

Task:

{task}
"""
```

Planner

```python
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_template(
    PLAN_PROMPT
)

planner_chain = planner_prompt | planner
```

Example

Input

```text
Create a report about LangGraph.
```

Output

```text
1. Explain LangGraph

2. Compare with LangChain

3. Give use cases

4. Write conclusion
```

---

# Step 3: Parse the Plan

```python
def create_plan(state):

    response = planner_chain.invoke(
        {
            "task": state["user_request"]
        }
    )

    tasks = response.content.split("\n")

    return {
        "plan": tasks
    }
```

State

```python
{
    "plan": [
        "Explain LangGraph",
        "Compare LangChain",
        "Write conclusion"
    ]
}
```

---

# Step 4: Executor Agent

Executor performs one task.

```python
executor = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)
```

Executor function

```python
def execute_task(task):

    prompt = f"""
Complete this task.

Task:

{task}
"""

    response = executor.invoke(prompt)

    return response.content
```

---

# Step 5: Execute Every Task

```python
def run_plan(state):

    outputs = []

    for task in state["plan"]:

        result = execute_task(task)

        outputs.append(result)

    return {
        "completed_tasks": outputs
    }
```

Example

```text
Task 1

↓

Explain LangGraph

↓

Executor

↓

Done

Task 2

↓

Compare

↓

Done
```

---

# Step 6: Aggregator

Finally combine all results.

```python
def summarize(state):

    prompt = f"""
Combine these results.

{state["completed_tasks"]}

Generate one report.
"""

    response = executor.invoke(prompt)

    return {
        "final_answer": response.content
    }
```

---

# Complete Flow

```text
               User Request
                     │
                     ▼
              Planner Agent
                     │
                     ▼
               Task List
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
 Executor 1     Executor 2     Executor 3
      │              │              │
      └──────────────┼──────────────┘
                     ▼
               Aggregator
                     ▼
              Final Report
```

---

# LangGraph Implementation

Planner Node

```python
def planner_node(state):

    plan = planner_chain.invoke(
        {
            "task": state["user_request"]
        }
    )

    return {
        "plan": plan.content.split("\n")
    }
```

Executor Node

```python
def executor_node(state):

    results = []

    for task in state["plan"]:

        results.append(
            execute_task(task)
        )

    return {
        "completed_tasks": results
    }
```

Aggregator Node

```python
def aggregator_node(state):

    answer = summarize(state)

    return answer
```

Graph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("aggregator", aggregator_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "aggregator")
workflow.add_edge("aggregator", END)

graph = workflow.compile()
```

Run

```python
result = graph.invoke(
    {
        "user_request":
        "Explain LangGraph"
    }
)

print(result["final_answer"])
```

---

# Parallel Executors

The planner can assign independent tasks to multiple executors simultaneously.

```text
                 Planner
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
Researcher      Calculator      Coder
      │             │             │
      └─────────────┼─────────────┘
                    ▼
              Result Merger
```

Python example

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_execute(tasks):

    with ThreadPoolExecutor() as pool:

        results = list(
            pool.map(
                execute_task,
                tasks
            )
        )

    return results
```

For I/O-bound LLM API calls, an `asyncio` implementation is often preferable.

---

# Enterprise Example

Financial Copilot

User

```text
Analyze Tesla stock.
```

Planner

```text
1 Collect stock price

2 Read news

3 Analyze sentiment

4 Build report
```

Executors

```text
Market Agent

News Agent

Sentiment Agent

Report Agent
```

Architecture

```text
                   User
                     │
                     ▼
               Planner Agent
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Market Agent  News Agent  Sentiment Agent
        │            │            │
        └────────────┼────────────┘
                     ▼
               Report Agent
                     ▼
               Final Answer
```

---

# Planner with Tool Calling

Planner

```text
Task 1

↓

Search Web
```

Executor

```text
Tool

↓

Search API
```

Result

```text
Data

↓

Planner
```

Planner

```text
Next Task
```

This allows the planner to adapt the plan based on intermediate results.

---

# Failure Recovery

Suppose one executor fails.

```text
Planner

↓

Executor

↓

Timeout

↓

Retry

↓

Fallback Model

↓

Success
```

Example

```python
def execute(task):

    try:
        return execute_task(task)

    except Exception:

        return fallback_model.invoke(task)
```

---

# Enterprise Architecture

```text
                      User
                        │
                        ▼
                  API Gateway
                        │
                        ▼
                 Planner Agent
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
 Research Agent   SQL Agent      Code Agent
       │                │                │
       └────────────────┼────────────────┘
                        ▼
                  Quality Checker
                        │
          Approved? ────┴────── Retry
                        │
                        ▼
                  Report Generator
                        ▼
                  Final Response
```

---

# Production Improvements

## 1. Dynamic Planning

Instead of creating all tasks upfront:

```text
Planner

↓

Task 1

↓

Executor

↓

Planner

↓

Task 2
```

The planner adapts based on previous results.

---

## 2. Confidence-Based Replanning

```text
Executor

↓

Confidence

↓

Low?

↓

Planner

↓

New Task
```

---

## 3. Human Approval

```text
Planner

↓

Large Money Transfer?

↓

Human

↓

Continue
```

---

## 4. Task Queue

Large enterprises often use a queue.

```text
Planner

↓

Kafka

↓

Executors

↓

Results
```

This improves scalability and resilience.

---

# Interview Follow-Up Questions

## 1. Why separate planner and executor?

The planner focuses on **reasoning and decomposition**, while executors focus on **doing one task well**. This separation improves modularity, reuse, and debugging.

---

## 2. Can executors use different models?

Yes.

Example:

* Research → Large reasoning model
* Code generation → Code-specialized model
* SQL generation → Smaller model
* Translation → Translation model

Choosing models based on the task can reduce cost and improve quality.

---

## 3. When should executors run in parallel?

Parallel execution is appropriate only when tasks are **independent**.

Example:

```text
Research competitors
Research customers
Research regulations
```

These can run simultaneously.

Not suitable:

```text
Write code
↓

Run tests
↓

Deploy
```

Each step depends on the previous one.

---

## 4. How do you recover from failures?

Common strategies:

* Retry with exponential backoff
* Use fallback models or tools
* Mark failed tasks and continue where possible
* Replan if a dependency cannot be completed
* Persist checkpoints so work can resume

---

## 5. How would you build this for production?

A production-grade planner–executor architecture typically includes:

* Structured planner output (for example, Pydantic models)
* Async or distributed task execution
* Retry and fallback policies
* Shared state and checkpointing
* Tool permissions and security
* Human approval for sensitive actions
* Observability (latency, cost, success rate)
* Caching of repeated task results
* Dynamic replanning when new information arrives

---

# Complete Production Workflow

```text
                    User Request
                          │
                          ▼
                  Planner Agent
                          │
                          ▼
                 Structured Task Plan
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   Research Agent    SQL Agent      Code Agent
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                 Result Aggregator
                          │
                          ▼
                  Quality Evaluator
                   /              \
             Accepted          Replan
                 │                │
                 ▼                ▼
            Final Response   Planner Agent
```

This planner–executor pattern is one of the most widely used architectures for complex AI applications because it separates **planning**, **execution**, and **validation**, making workflows more scalable, maintainable, and resilient than relying on a single agent to handle every step.
