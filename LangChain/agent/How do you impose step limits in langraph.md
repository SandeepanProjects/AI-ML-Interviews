# How to impose step limits in LangGraph — senior/staff AI engineer approach

In production LangGraph applications, step limits prevent an agent workflow from looping indefinitely, repeatedly calling tools, consuming excessive tokens, and running up costs.

For example, imagine your financial compliance review agent:

1. The planner decides which checks are needed.
2. A retrieval agent fetches policy evidence.
3. A compliance agent evaluates the claims.
4. An evaluator checks the result.
5. If the result is incomplete, the workflow loops back for another attempt.

If the evaluator keeps rejecting the result, the graph may continue looping. A production system needs explicit limits and a safe exit path.

The key is to distinguish LangGraph's built-in recursion limit from application-level step budgets. They solve related but different problems, and a robust system generally uses both.

## 1. The three limits you should understand

1\. Graph recursion limit

Framework-level protection

Limits how many graph supersteps LangGraph can execute before raising a recursion-limit error. This is the primary protection against uncontrolled graph cycles.

2\. Application-level step budget

Business and workflow control

Tracks how many planner decisions, tool calls, retries, or evaluator rework attempts are allowed. This gives you explicit, testable business rules.

3\. Operational budgets

Cost and reliability protection

Enforce deadlines, tool-call budgets, token budgets, model retry limits, and concurrency limits. A recursion limit alone does not enforce these.

One important distinction: a superstep is not the same as one agent action. In a graph, several independent nodes can execute in parallel during one superstep. So a recursion limit of 20 does not necessarily mean 20 LLM calls, 20 tools, or 20 planner decisions.

## 2. Method 1 — Set `recursion_limit` at runtime

This is the simplest built-in protection.

```
from langgraph.errors import GraphRecursionErrorresult = graph.invoke(    {"question": "Review this financial claim"},    config={        "recursion_limit": 20    },)
```

LangGraph raises `GraphRecursionError` when the graph exceeds its configured superstep limit. The limit is supplied directly in `config`, not inside `config["configurable"]`.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference

+1



### What happens when the limit is reached?

```
from langgraph.errors import GraphRecursionErrortry:    result = graph.invoke(        initial_state,        config={"recursion_limit": 20},    )except GraphRecursionError:    # Record the failure, alert/trace it, and return    # a controlled error or recover from a checkpoint.    result = {        "status": "step_limit_exceeded",        "message": "The workflow exceeded its execution limit.",    }
```

This is useful as a last line of defense, but it is not enough by itself.

Why? Suppose the workflow uses 19 supersteps and still has an incomplete compliance review. An exception at the limit does not automatically give your application a completed business result or a user-friendly fallback. If partial results are required, design that behavior explicitly.

Also, don't simply raise the limit from 20 to 500 to make an error disappear. First inspect the graph's routing and determine whether the loop is intentional.

## 3. Method 2 — Use `RemainingSteps` for graceful termination

This is a better approach when you want the graph to detect that it is approaching the limit and choose a safe exit before the framework raises an exception.

LangGraph provides `RemainingSteps`, a managed state value that exposes the remaining supersteps for the current graph execution.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



The design is:

Evaluate result

Is the result approved?

Yes

Finish successfully

No

Check budgets

Rework allowed

Revise → evaluate again

Limit approaching

Safe finalization

### Complete example: a bounded compliance-review loop

This example uses a small graph to demonstrate the same control pattern you could apply to your financial compliance platform. The evaluator is a placeholder for your actual validation logic.

```
from typing import Literalfrom typing_extensions import TypedDictfrom langgraph.graph import StateGraph, START, ENDfrom langgraph.managed import RemainingStepsfrom langgraph.errors import GraphRecursionErrorMAX_REWORKS = 3REMAINING_STEP_THRESHOLD = 2class ReviewState(TypedDict):    draft: str    approved: bool    rework_attempts: int    status: str    remaining_steps: RemainingStepsdef evaluate(state: ReviewState) -> dict:    """    Replace this with real checks:    policy evidence, claim support, citations,    schema validation, and compliance rules.    """    draft = state["draft"]    approved = "SUPPORTED" in draft    return {"approved": approved}def revise(state: ReviewState) -> dict:    """Illustrative revision; replace with your LLM or rules engine."""    return {
```

Why this is better than only setting `recursion_limit`:

- `RemainingSteps` allows the graph to exit before the hard limit is exhausted.
- `MAX_REWORKS` enforces a business-specific limit independently of graph size.
- The fallback does not pretend an incomplete review was approved.
- The exception handler is still present for unexpected routing bugs.

The example is intentionally small. In a real compliance system, you would persist the review status and audit record, and preserve any useful partial evidence through your application’s persistence layer.

## 4. Method 3 — Enforce explicit application-level budgets

A senior engineer does not treat a graph superstep limit as the complete agent budget.

For example, you may want to enforce these rules for each financial review:

| Budget                    | Example limit    | What it controls            |
| ------------------------- | ---------------- | --------------------------- |
| Graph supersteps          | 30               | Overall graph progression   |
| Evaluator rework attempts | 3                | Repeated revision loops     |
| Tool calls                | 8                | External tool usage         |
| Planner actions           | 5                | Agent task generation       |
| LLM retries per request   | 2                | Transient provider failures |
| Wall-clock deadline       | 60 seconds       | Total request duration      |
| Token or monetary budget  | Product-specific | Model usage and cost        |

These are illustrative values, not universal production defaults.

### A. Limit tool calls explicitly

A straightforward pattern is to track tool calls in graph state and increment the counter whenever a tool is actually invoked.

```
from typing_extensions import TypedDictMAX_TOOL_CALLS = 8class AgentState(TypedDict):    tool_calls_used: int    result: str    status: strdef can_call_tool(state: AgentState) -> bool:    return state["tool_calls_used"] < MAX_TOOL_CALLSdef execute_policy_search(state: AgentState) -> dict:    if not can_call_tool(state):        return {            "status": "tool_budget_exhausted",            "result": "Further tool execution was blocked.",        }    # Increment before executing the external operation.    # This conservatively charges failed attempts too.    updated_count = state["tool_calls_used"] + 1    # Replace with your actual policy search service.    result = search_policy_documents("relevant policy query")    return {        "tool_calls_used": updated_count,        "result": result,        "status": "in_progress",    }
```

Here, `search_policy_documents` represents an application service you must implement. It is not a built-in LangGraph function.

Production caveat: a check in a graph node is not a universal tool-call limiter. If the agent can call multiple tools directly inside one node, the node must enforce the budget around each individual call, or use a centralized tool-execution wrapper. If parallel branches share a budget, a simple state counter may be insufficient because concurrent updates can race. Use a durable, atomic budget reservation mechanism when a strict global limit is required.

### B. Limit planner actions and evaluator retries

The planner and evaluator should have separate budgets because they represent different kinds of work.

```
MAX_PLANNER_ACTIONS = 5MAX_EVALUATOR_RETRIES = 3def plan_next_action(state: dict) -> dict:    if state["planner_actions"] >= MAX_PLANNER_ACTIONS:        return {"status": "planner_budget_exhausted"}    # Call planner only after the budget check.    action = generate_next_action(state)    return {        "next_action": action,        "planner_actions": state["planner_actions"] + 1,    }def route_after_planning(state: dict) -> str:    if state["status"] == "planner_budget_exhausted":        return "safe_exit"    return "execute_action"
```

The routing function must be connected to the appropriate nodes using `add_conditional_edges`. Also, the counter needs to increment at the actual action boundary: otherwise, repeated planning or routing could evade the limit.

In a multi-agent system, enforce these limits at the component that owns the operation. A supervisor can set the budget, but specialist agents and tools should not be able to bypass it.

## 5. Important distinction: hard limit versus graceful exit

There are two different policies you can implement.

Hard limit

Stop execution when the configured limit is exceeded. Useful when continuing would violate a strict safety or cost boundary.

Graceful exit

Detect that the budget is nearly exhausted and route to a finalization node. Useful when you want to return partial results, explain uncertainty, or request human review.

For financial compliance, my recommendation is to use both:

1. Use `recursion_limit` as the graph-level safety net.
2. Use `RemainingSteps` to detect approaching exhaustion.
3. Use explicit counters for planner actions, tool calls, and evaluator rework.
4. Route to a terminal node when a budget is exhausted.
5. Preserve an `incomplete` or `needs_human_review` status rather than returning an unverified result as approved.
6. Enforce request deadlines and provider-level timeouts outside the graph as well.

## 6. What about fan-out/fan-in and subgraphs?

This matters for the multi-agent architecture you are building.

- Fan-out/fan-in: a graph may launch several specialist nodes in parallel. One superstep can involve multiple node executions, so a recursion limit does not cap the number of parallel tool calls.
- Subgraphs: test the limits of the complete parent/child execution path. Do not assume that a parent graph's application-level counter automatically enforces a shared tool or cost budget across all children.
- Retries: a node retry policy and an evaluator rework loop are different. Bound both.
- Checkpoint resume: a resumed workflow should preserve the original execution's remaining budget or deadline. Do not reset the budget just because the request is resumed.
- Distributed workers: store strict shared budgets in a durable service or database if multiple workers may execute the same workflow.

LangGraph documents the recursion limit as a superstep limit and recommends explicit termination conditions for cyclic graphs.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 7. What I would implement in a production repository

For a FastAPI + LangGraph financial compliance platform, I would put the controls in separate modules:

```
app/
├── workflows/
│   ├── compliance_graph.py
│   ├── routing.py
│   └── state.py
├── policies/
│   └── execution_budgets.py
├── agents/
│   ├── planner.py
│   ├── supervisor.py
│   └── evaluator.py
├── tools/
│   └── budgeted_executor.py
├── observability/
│   └── workflow_metrics.py
└── api/
    └── review_routes.py
```

The separation is intentional:

- `execution_budgets.py` defines approved limits by workflow type or tenant plan.
- `routing.py` contains explicit exit and retry decisions.
- `budgeted_executor.py` enforces tool-call budgets.
- `compliance_graph.py` sets the runtime recursion limit and assembles the graph.
- `workflow_metrics.py` records step counts, budget exhaustion, retries, latency, and cost.

Do not let an untrusted API caller choose an arbitrarily large `recursion_limit`. Derive limits from trusted server-side configuration and apply stricter policies to sensitive workflows.

## 8. Interview-ready answer

> “I impose step limits in LangGraph at multiple layers. First, I configure `recursion_limit` at invocation time to prevent runaway graph execution. Second, I use `RemainingSteps` to detect when the graph is approaching that limit and route to a graceful terminal node. Third, I enforce application-level budgets for planner actions, tool calls, and evaluator retries because graph supersteps do not correspond one-to-one with agent actions. Finally, I apply deadlines, model timeouts, and token or cost budgets, with observability and safe recovery. In a compliance workflow, exhausting a budget must never silently convert an incomplete review into an approved result.”

That is the key staff-level distinction: the framework limit protects graph execution; the application budget protects your business, cost, and safety requirements.