Workflow recovery is a **critical production capability** for AI agents and LangGraph systems.

In production, workflows fail because of:

* API failures
* LLM timeouts
* Database errors
* Network failures
* Rate limits
* Process crashes
* Human approval delays

A robust agent should **not restart from zero**. It should:

1. Save progress (checkpoint)
2. Detect failure
3. Restore previous state
4. Retry or continue from the failed step
5. Use fallback strategies

---

# Workflow Recovery Architecture

```text
                 User Request

                     │

                     ▼

                LangGraph

                     │

        ┌────────────┼────────────┐

        ▼            ▼            ▼

    Planner       Tools       Generator

        │            │            │

        ▼            ▼            ▼

   Checkpoint    Checkpoint   Checkpoint

                     │

                     ▼

              PostgreSQL

                     │

          Failure Happens ❌

                     │

                     ▼

            Load Checkpoint

                     │

                     ▼

             Resume Workflow
```

---

# Example Scenario

Imagine a customer support agent:

```text
START

↓

Retrieve Customer Data

↓

Call Payment API

↓

Generate Response

↓

Send Email

↓

END
```

Payment API fails:

```text
START

↓

Retrieve Customer Data ✔

↓

Payment API ❌

↓

Crash
```

Without recovery:

```text
Restart from START
```

With recovery:

```text
Load checkpoint

↓

Retry Payment API
```

---

# Step 1: Define Workflow State

```python
from typing import TypedDict


class AgentState(TypedDict):

    customer_id: str

    payment_status: str

    retry_count: int

    error: str
```

Example state:

```python
{
    "customer_id": "123",
    "payment_status": "",
    "retry_count": 0,
    "error": ""
}
```

---

# Step 2: Create Nodes

## Customer Lookup Node

```python
def get_customer(state):

    print("Fetching customer")

    return {

        "customer_id":
            state["customer_id"]

    }
```

---

## Payment API Node

This is where failures happen.

```python
import random


def payment_node(state):

    print("Calling payment API")


    if random.random() < 0.5:

        raise Exception(
            "Payment API failed"
        )


    return {

        "payment_status":
            "SUCCESS"

    }
```

---

# Step 3: Add Retry Handling

Instead of crashing:

```python
def safe_payment_node(state):

    try:

        result = payment_api()

        return {

            "payment_status":
            "SUCCESS"

        }


    except Exception as e:


        return {

            "error": str(e),

            "retry_count":
                state["retry_count"] + 1

        }
```

Now failure becomes part of the state.

Example:

```python
{
    "payment_status":"",
    "retry_count":1,
    "error":"Payment API failed"
}
```

---

# Step 4: Add Conditional Recovery

```python
def recovery_router(state):


    if state["error"]:

        if state["retry_count"] < 3:

            return "retry"


        else:

            return "failure"


    return "success"
```

Flow:

```text
Payment Node

     │

     ▼

Error?

 ┌───────┐

 Yes     No

 │        │

 ▼        ▼

Retry   Continue
```

---

# Step 5: Build LangGraph

```python
from langgraph.graph import (
    StateGraph,
    START,
    END
)


builder = StateGraph(AgentState)


builder.add_node(
    "customer",
    get_customer
)


builder.add_node(
    "payment",
    safe_payment_node
)


builder.add_node(
    "retry",
    safe_payment_node
)


builder.add_node(
    "failure",
    lambda state:{
        "payment_status":
        "FAILED"
    }
)


builder.add_edge(
    START,
    "customer"
)


builder.add_edge(
    "customer",
    "payment"
)


builder.add_conditional_edges(

    "payment",

    recovery_router,

    {

        "retry":"retry",

        "success":END,

        "failure":"failure"

    }

)


builder.add_edge(
    "retry",
    END
)


graph = builder.compile()
```

---

# Step 6: Add Checkpointing

Recovery requires persistence.

```python
from langgraph.checkpoint.memory import InMemorySaver


memory = InMemorySaver()


graph = builder.compile(

    checkpointer=memory

)
```

Now every step is saved.

---

# Step 7: Execute With Thread ID

```python
config = {

    "configurable": {

        "thread_id":
        "payment-workflow-001"

    }

}


result = graph.invoke(

    {

        "customer_id":"123",

        "payment_status":"",

        "retry_count":0,

        "error":""

    },

    config=config

)
```

Checkpoint:

```text
thread_id:

payment-workflow-001


State:

{
 customer_id:123,
 retry_count:1,
 error:"Payment failed"
}
```

---

# Step 8: Resume After Failure

Suppose your service crashes.

Restart application.

Use the same thread ID:

```python
config = {

    "configurable": {

        "thread_id":
        "payment-workflow-001"

    }

}
```

Resume:

```python
result = graph.invoke(

    {},

    config=config

)
```

LangGraph:

```text
Load checkpoint

↓

Restore state

↓

Continue execution
```

---

# Production Retry Pattern

A better retry implementation uses exponential backoff.

```python
import time


def call_with_retry(
    func,
    max_attempts=3
):

    for attempt in range(
        max_attempts
    ):

        try:

            return func()


        except Exception:

            wait = 2 ** attempt

            time.sleep(wait)


    raise Exception(
        "All retries failed"
    )
```

Execution:

```text
Attempt 1

Failure

Wait 1 sec


Attempt 2

Failure

Wait 2 sec


Attempt 3

Success
```

---

# Recovering LLM Failures

Example:

```python
def llm_node(state):

    try:

        response = llm.invoke(
            state["prompt"]
        )


        return {

            "answer":
            response.content

        }


    except Exception as e:


        return {

            "error":
            "LLM unavailable"

        }
```

---

# Fallback Model Recovery

Production systems often use multiple models.

```python
def generate_answer(state):

    try:

        return openai.invoke(
            state["question"]
        )


    except Exception:


        return anthropic.invoke(
            state["question"]
        )
```

Flow:

```text
GPT-5

  |

Failure

  |

Claude

  |

Success
```

---

# Human Recovery

For critical workflows:

```text
Loan Approval

↓

Risk Analysis

↓

Checkpoint

↓

Human Review

↓

Approve / Reject

↓

Resume
```

State:

```python
{
 "risk_score":0.2,
 "approval":null
}
```

After human input:

```python
graph.invoke(

{
 "approval":"APPROVED"
},

config=config

)
```

---

# Production Database Checkpointing

Instead of:

```python
InMemorySaver()
```

Use:

```text
PostgreSQL

Table:

checkpoints

-----------------

thread_id

state

next_node

timestamp

```

Example:

```text
payment-workflow-001

{
 retry_count:2,
 error:"timeout"
}

next_node:

payment_retry
```

---

# Recovery Strategies

| Failure          | Recovery                |
| ---------------- | ----------------------- |
| LLM timeout      | Retry + fallback model  |
| Tool failure     | Retry with backoff      |
| API rate limit   | Wait + retry            |
| Database failure | Transaction rollback    |
| Process crash    | Resume checkpoint       |
| Human approval   | Pause and resume        |
| Invalid output   | Re-prompt / repair      |
| Infinite loop    | Maximum iteration limit |

---

# Preventing Bad Recovery

Always track:

```python
class AgentState(TypedDict):

    retry_count:int

    max_retries:int

    error:str

    status:str
```

Example:

```python
if state["retry_count"] >= 5:

    stop_workflow()
```

Otherwise:

```text
Retry

↓

Retry

↓

Retry

↓

Infinite loop ❌
```

---

# Enterprise Recovery Flow

```text
                 Request

                    │

                    ▼

              LangGraph Agent

                    │

       ┌────────────┼────────────┐

       ▼            ▼            ▼

    LLM Call     Tool Call    Retrieval

       │            │            │

       ▼            ▼            ▼

    Error       Error        Error

       │            │            │

       ▼            ▼            ▼

 Retry Policy  Retry Policy  Re-ranking

       │

       ▼

 Checkpoint Restore

       │

       ▼

 Resume Execution

       │

       ▼

 Final Response
```

---

# Senior AI Engineer Interview Answer

> **I recover failed LangGraph workflows using a combination of checkpointing, retries, and state-based recovery. Each workflow execution has a unique thread ID, and after every important node execution the state is persisted. If a failure occurs, I restore the latest checkpoint and resume from the failed step rather than restarting the workflow. For transient failures I implement exponential backoff retries, for model failures I use fallback models, and for critical workflows I add human-in-the-loop recovery points. I also track retry counts and workflow status in state to prevent infinite recovery loops.**
