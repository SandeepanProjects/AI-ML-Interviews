# Implementing Human-in-the-Loop (HITL) in LangGraph

At a senior/staff AI engineer level, I would implement HITL as a durable workflow pause, human decision, and controlled resume mechanism—not simply as an API that asks a user to approve something.

For your Financial Advisor Copilot, consider this example:

1. A user asks for a portfolio rebalance recommendation.
2. LangGraph retrieves portfolio data and runs risk, performance, and compliance agents.
3. The workflow generates a proposed recommendation.
4. Before executing a high-impact action, the graph pauses for human approval.
5. A reviewer approves, rejects, or requests changes.
6. The graph resumes from its saved checkpoint and follows the appropriate branch.

HITL workflow

User request

LangGraph agents execute

Research · Risk · Performance · Compliance

Generate recommendation

Human approval — interrupt()

Workflow pauses and saves its state

Approve

Reject

Revise

Resume workflow

Continue from checkpoint and take the approved branch

The central LangGraph APIs are:

- `interrupt()` — pause execution and expose a value to the caller.
- A checkpointer — persist graph state.
- `Command(resume=...)` — provide the human's decision and resume.
- `thread_id` — identify the specific workflow execution.

## 1. Understand the execution lifecycle

The important detail is that LangGraph does not keep a Python function or HTTP request blocked while it waits for approval.

Instead:

1. The graph reaches `interrupt(payload)`.
2. LangGraph checkpoints its state and pauses the execution.
3. The application receives the interrupt payload.
4. The backend stores a pending approval request.
5. A reviewer approves or rejects it, possibly hours later.
6. The backend resumes the graph using the same `thread_id`.
7. The graph continues with the human's response.

This is supported by LangGraph's checkpointing and interrupt mechanism.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference+2



Important: When a node resumes, LangGraph re-executes that node from its beginning. It does not continue at the exact Python line after `interrupt()`. Therefore, keep irreversible side effects out of the code before the interrupt.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference+1



## 2. Recommended production architecture

For your Financial Advisor Copilot, I recommend separating the graph, API, approval database, and execution service.

FastAPI

Start workflow · Get approval status · Submit decision

LangGraph workflow

Supervisor · Research · Risk · Compliance · Report

Approval gate — interrupt()

Pause and checkpoint the graph

PostgreSQL

LangGraph checkpoints, approval records, audit trail

Worker / queue

Resume workflow asynchronously

Execution service

Execute approved action only

The key design principle is:

- LangGraph manages workflow state and execution.
- PostgreSQL stores durable checkpoints and application approval records.
- FastAPI handles authentication, authorization, and reviewer requests.
- A worker resumes long-running workflows.
- The execution service performs the actual business operation.

Do not use `InMemorySaver` as the production persistence layer. It is useful for local development and tests, but production needs durable shared persistence.

## 3. Project structure

```
financial_advisor_copilot/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── approval_routes.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   ├── services/
│   │   ├── approval_service.py
│   │   └── execution_service.py
│   │
│   ├── infrastructure/
│   │   ├── checkpointer.py
│   │   └── database.py
│   │
│   └── schemas/
│       └── approval.py
│
├── tests/
│   ├── test_approval.py
│   └── test_workflow_recovery.py
│
├── requirements.txt
└── .env
```

Install the core dependencies:

```
pip install \
  langgraph \
  langgraph-checkpoint-postgres \
  fastapi \
  uvicorn \
  asyncpg \
  psycopg[binary,pool] \
  pydantic
```

## 4. Define the graph state

`app/graph/state.py`

```
from typing import Literal, TypedDictclass AdvisorState(TypedDict, total=False):    request_id: str    tenant_id: str    user_id: str    question: str    recommendation: str    risk_score: float    compliance_status: str    approval_status: Literal[        "PENDING",        "APPROVED",        "REJECTED",        "REVISION_REQUIRED",    ]    reviewer_id: str    reviewer_comments: str    revised_recommendation: str    execution_result: dict    error: str
```

The state contains the information needed to resume the workflow. Avoid putting secrets, credentials, or unnecessary personal information into checkpointed state.

## 5. Implement the approval node

This is the most important part of the implementation.

`app/graph/nodes.py`

```
from typing import Literalfrom langgraph.types import interrupt, Commandfrom app.graph.state import AdvisorStatedef prepare_recommendation(    state: AdvisorState,) -> dict:    # In a real system, this result comes from    # research, risk, performance and compliance agents.    recommendation = (        "Rebalance 10% of the portfolio from "        "Asset A to Asset B."    )    return {        "recommendation": recommendation,        "approval_status": "PENDING",    }def approval_gate(    state: AdvisorState,) -> Command[    Literal[        "execute_action",        "reject_action",        "revise_recommendation",    ]]:    # IMPORTANT:
```

Notice that the approval node uses `Command` to update state and choose the next node. This is a useful pattern when a node itself determines the next destination.

A production system should validate the decision using a Pydantic model, including required fields for each action. The API should also reject invalid actions before resuming the graph.

## 6. Build the LangGraph workflow

`app/graph/workflow.py`

```
from langgraph.graph import StateGraph, START, ENDfrom app.graph.state import AdvisorStatefrom app.graph.nodes import (    prepare_recommendation,    approval_gate,    execute_action,    reject_action,    revise_recommendation,)def build_workflow(checkpointer):    builder = StateGraph(AdvisorState)    builder.add_node(        "prepare_recommendation",        prepare_recommendation,    )    builder.add_node(        "approval_gate",        approval_gate,    )    builder.add_node(        "execute_action",        execute_action,    )    builder.add_node(        "reject_action",        reject_action,    )
```

The graph now supports three paths:

- Approve → execute.
- Reject → finish without executing.
- Revise → return to approval for another review.

For a complex workflow, you can add a maximum revision count and an escalation path to prevent endless review loops.

## 7. Configure durable PostgreSQL checkpointing

`app/infrastructure/checkpointer.py`

```
from langgraph.checkpoint.postgres.aio import (    AsyncPostgresSaver,)from app.graph.workflow import build_workflowclass GraphRuntime:    def __init__(self):        self.checkpointer_context = None        self.checkpointer = None        self.graph = None    async def start(self, database_url: str):        self.checkpointer_context = (            AsyncPostgresSaver.from_conn_string(                database_url            )        )        self.checkpointer = (            await self.checkpointer_context.__aenter__()        )        await self.checkpointer.setup()        self.graph = build_workflow(            self.checkpointer        )    async def stop(self):        if self.checkpointer_context:            await self.checkpointer_context.__aexit__(
```

For example:

```
DATABASE_URL=postgresql://user:password@localhost:5432/advisor
```

The checkpointer must remain available for the lifetime of the application. In a multi-worker deployment, use a shared PostgreSQL database rather than a process-local checkpointer.

The checkpointer setup and graph compilation are infrastructure initialization tasks, not operations to perform on every API request. See the official [LangGraph interrupt documentation](https://docs.langchain.com/oss/python/langgraph/interrupts?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference+1



## 8. Start the workflow and detect the interrupt

When a user submits a request, the API starts the graph with a unique workflow ID.

Example:

```
from uuid import uuid4from fastapi import APIRouterfrom app.schemas.request import AdvisorRequestrouter = APIRouter()@router.post("/advisor")async def create_advisor_request(    request: AdvisorRequest,):    request_id = str(uuid4())    config = {        "configurable": {            "thread_id": request_id        }    }    result = await runtime.graph.ainvoke(        {            "request_id": request_id,            "tenant_id": request.tenant_id,            "user_id": request.user_id,            "question": request.question,            "risk_score": 0.65,            "compliance_status": "PASSED",        },        config=config,    )    if result.get("__interrupt__"):        # Persist an application-level approval record        # and notify the assigned reviewer.
```

The response might look like:

```
{
  "request_id": "workflow-123",
  "status": "PENDING_APPROVAL",
  "interrupts": [
    {
      "type": "portfolio_rebalance_approval",
      "request_id": "workflow-123",
      "recommendation": "Rebalance 10% of the portfolio from Asset A to Asset B.",
      "risk_score": 0.65,
      "compliance_status": "PASSED",
      "allowed_actions": [
        "approve",
        "reject",
        "revise"
      ]
    }
  ]
}
```

The API returns immediately. It does not wait for the reviewer.

## 9. Resume the workflow after human approval

This is where `Command(resume=...)` comes in.

For a simple local demonstration, you can resume directly:

```
from langgraph.types import Commandasync def resume_workflow(    request_id: str,    decision: dict,):    config = {        "configurable": {            "thread_id": request_id        }    }    result = await runtime.graph.ainvoke(        Command(resume=decision),        config=config,    )    if result.get("__interrupt__"):        return {            "status": "PENDING_APPROVAL",            "interrupts": [                item.value                for item in result["__interrupt__"]            ],        }    return {        "status": result.get("approval_status"),        "execution_result": result.get(            "execution_result"        ),    }
```

Example approval decision:

```
decision = {    "action": "approve",    "reviewer_id": "reviewer-101",    "comments": "Risk and compliance checks reviewed.",}result = await resume_workflow(    request_id="workflow-123",    decision=decision,)
```

Approved path

The graph resumes and routes to `execute_action`.

Example rejection:

```
decision = {    "action": "reject",    "reviewer_id": "reviewer-101",    "comments": "The proposed allocation exceeds the risk limit.",}
```

The graph routes to `reject_action` and does not execute the proposed action.

Critical production detail: Do not expose the simple `resume_workflow()` function directly to an unauthenticated endpoint. The `reviewer_id` must come from the authenticated server-side identity, not from a client-supplied JSON field.

## 10. Implement the approval API correctly

A production approval endpoint needs more than just resuming LangGraph.

It must:

1. Authenticate the reviewer.
2. Check that the reviewer is allowed to approve this request.
3. Check tenant and resource ownership.
4. Verify the request is still pending.
5. Validate the action and any revised recommendation.
6. Record the decision and audit details.
7. Resume the correct LangGraph thread.

Example API shape:

```
from typing import Literalfrom fastapi import APIRouter, Depends, HTTPExceptionfrom pydantic import BaseModel, Fieldrouter = APIRouter()class ApprovalRequest(BaseModel):    action: Literal["approve", "reject", "revise"]    comments: str = ""    revised_recommendation: str | None = Field(        default=None,        max_length=5000,    )@router.post("/approvals/{request_id}/decision")async def submit_approval(    request_id: str,    body: ApprovalRequest,    reviewer=Depends(get_current_reviewer),):    if (        body.action == "revise"        and not body.revised_recommendation    ):        raise HTTPException(            status_code=422,            detail="Revised recommendation is required",        )    # This service must atomically verify:    # - approval exists and is pending    # - reviewer has permission
```

The `record_decision_and_enqueue()` method is an application service contract here; its database implementation is not included in this illustrative API snippet.

### Why enqueue the resume instead of resuming inside the HTTP request?

Imagine this sequence:

- Reviewer clicks Approve.
- Your API saves the approval decision.
- Your server crashes before calling `graph.ainvoke()`.

The database says approved, but the graph remains paused.

For production, use a transactional outbox:

```
BEGIN TRANSACTION

1. Update approval status to APPROVED.
2. Insert audit record.
3. Insert workflow_resume event into outbox.

COMMIT
```

A background worker reads the outbox and resumes the graph. If the worker crashes, it retries the event.

Use an idempotency key such as:

```
approval:{request_id}:{decision_version}
```

Also use a database conditional update so that only one reviewer decision can win:

```
UPDATE approvals
SET
    status = 'APPROVED',
    reviewer_id = :reviewer_id,
    decided_at = NOW()
WHERE
    request_id = :request_id
    AND tenant_id = :tenant_id
    AND status = 'PENDING';
```

Check the affected row count. If it is zero, the request was already decided, is not pending, or does not belong to that tenant. Do not resume it again.

This design gives you durable decision handling even when the API or worker crashes.

## 11. How do you handle failures and recovery?

This is particularly important in senior/staff-level interviews.

| Failure                                       | Recommended solution                               |
| --------------------------------------------- | -------------------------------------------------- |
| API crashes after approval                    | Transactional outbox retries resume                |
| Worker crashes during resume                  | Retry using the same thread ID and idempotency key |
| Reviewer submits twice                        | Conditional database update                        |
| Reviewer submits invalid decision             | Validate before resume                             |
| Reviewer lacks permission                     | Reject before recording decision                   |
| Workflow resumes but action service times out | Retry with business idempotency key                |
| Reviewer never responds                       | Expiry policy and escalation                       |
| Recommendation changes after approval         | Invalidate old approval and require new approval   |
| Another tenant tries to approve               | Tenant-scoped authorization                        |

### Important: approval is not the same as execution

For financial or other high-impact actions, make sure the approval is tied to the exact recommendation version.

For example:

```
{
  "request_id": "workflow-123",
  "recommendation_version": 4,
  "recommendation_hash": "sha256:...",
  "action": "approve",
  "reviewer_id": "reviewer-101"
}
```

If the recommendation changes after the reviewer approves it, the old approval must not authorize the new recommendation.

The execution service should verify the approved version and hash before performing the operation.

### Prevent duplicate execution

Suppose the graph resumes, calls an external service, and the worker crashes before the graph records success.

On retry, the external service could receive the same operation again.

Use an idempotency key at the business-operation boundary:

```
async def execute_rebalance(    request_id: str,    recommendation: str,):    idempotency_key = (        f"portfolio-rebalance:{request_id}"    )    return await portfolio_service.rebalance(        recommendation=recommendation,        idempotency_key=idempotency_key,    )
```

The receiving service must actually enforce that key; merely generating it is not enough.

## 12. Important LangGraph HITL mistakes to avoid

These are common mistakes that can create serious production issues.

Mistake 1: Using `interrupt_before` for business approval

```
graph = builder.compile(    interrupt_before=["execute_action"])
```

This is useful for debugging and stepping through nodes, but for application-level human approval, prefer `interrupt()` because it allows you to send a structured review payload and receive a decision.

Mistake 2: Putting side effects before `interrupt()`

Incorrect:

```
async def approval_node(state):    await payment_service.execute_payment()    decision = interrupt("Approve payment?")
```

The node can run again from the beginning after resumption. You must never put the irreversible action before the approval gate.

Mistake 3: Catching the interrupt exception

Avoid:

```
try:    decision = interrupt("Approve?")except Exception:    return {"approval_status": "FAILED"}
```

`interrupt()` uses a special control-flow exception. Catching it broadly can swallow the interrupt and break the pause mechanism.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://langchain-ai.github.io\&sz=32)

Docs by LangChain+1



Mistake 4: Using a new thread ID when resuming

Incorrect:

```
config = {    "configurable": {        "thread_id": str(uuid4())    }}
```

Correct:

```
config = {    "configurable": {        "thread_id": original_request_id    }}
```

The same thread ID identifies the checkpoint to resume.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference+1



Mistake 5: Trusting the human decision without authorization

A client must not be able to approve another user's or another tenant's workflow simply by knowing its request ID.

Authorization belongs in your API and application service, not in the LLM or the interrupt payload.

## 13. How should you test HITL?

I would cover at least these scenarios:

| Test                       | Expected result                              |
| -------------------------- | -------------------------------------------- |
| Workflow reaches approval  | Graph pauses                                 |
| Reviewer approves          | Execution node runs                          |
| Reviewer rejects           | Execution node does not run                  |
| Reviewer requests revision | Recommendation is revised and reviewed again |
| Invalid decision           | Resume is rejected                           |
| Duplicate approval         | Only one decision is accepted                |
| Wrong tenant               | Access denied                                |
| Worker crashes             | Workflow can resume                          |
| Recommendation changes     | Previous approval becomes invalid            |
| External action retries    | No duplicate business operation              |

A simple integration test for the core pause/resume behavior:

```
import pytestfrom langgraph.checkpoint.memory import InMemorySaverfrom langgraph.types import Command@pytest.mark.asyncioasync def test_approval_and_resume():    checkpointer = InMemorySaver()    graph = build_workflow(checkpointer)    config = {        "configurable": {            "thread_id": "test-workflow-123"        }    }    # Start workflow.    result = await graph.ainvoke(        {            "request_id": "test-workflow-123",            "tenant_id": "tenant-1",            "user_id": "user-1",            "question": "Rebalance my portfolio",            "risk_score": 0.5,            "compliance_status": "PASSED",        },        config=config,    )    # Verify graph paused.    assert result.get("__interrupt__")    # Resume with a test approval.    result = await graph.ainvoke(
```

This test uses an in-memory checkpointer intentionally. Add separate integration tests using PostgreSQL to verify persistence across application restarts and worker processes.

## 14. How I would implement this in your multi-agent architecture

For your Financial Advisor Copilot, I would not ask a human to approve every agent's output.

Instead, I would introduce approval gates based on business risk.

| Workflow stage                  | Human involvement                          |
| ------------------------------- | ------------------------------------------ |
| Research agent                  | Usually no approval                        |
| Risk agent                      | Usually no approval                        |
| Performance agent               | Usually no approval                        |
| Compliance agent                | Escalate high-risk or ambiguous cases      |
| Report generation               | Optional review                            |
| Portfolio rebalance execution   | Mandatory approval                         |
| High-impact financial operation | Strong authorization and approval controls |

The supervisor can decide whether the workflow needs human review based on deterministic policy rules, not simply an LLM's opinion.

For example:

```
def requires_human_approval(state):    if state["risk_score"] >= 0.70:        return True    if state["compliance_status"] != "PASSED":        return True    if state.get("action_type") == "EXECUTE_TRADE":        return True    return False
```

In a real financial system, the risk thresholds and mandatory approval conditions should be configured and enforced by a versioned policy service.

## 15. Senior AI Engineer interview answer

If the interviewer asks, "How would you implement Human-in-the-Loop in LangGraph?", I would answer:

> I implement HITL using LangGraph's interrupt and checkpointing mechanisms.
>
> First, I identify critical workflow stages where human intervention is required, such as financial transactions, compliance exceptions, or high-risk recommendations.
>
> When the graph reaches an approval node, I call `interrupt()` with a structured payload containing the proposed action, supporting evidence, risk assessment, and available decisions.
>
> LangGraph checkpoints the current workflow state and pauses execution. My FastAPI application exposes the pending approval to an authorized reviewer.
>
> Once the reviewer submits a decision, the backend authenticates and authorizes the reviewer, records the decision in PostgreSQL, and resumes the graph using `Command(resume=...)` with the original `thread_id`.
>
> Based on the response, the graph either executes the approved action, rejects it, or sends it for revision.
>
> For production, I also implement durable PostgreSQL checkpointing, transactional outbox processing, idempotent resume operations, audit logging, tenant-level authorization, timeouts, and recovery mechanisms.
>
> Most importantly, I ensure that an approval is bound to the exact version of the proposed action and that the execution service independently verifies authorization before performing any irreversible operation.

## 16. What you should be able to explain in an interview

HITL knowledge checklist

0 of 12 covered

0%

How interrupt() works internally

Why a checkpointer is mandatory

How Command(resume=...) works

Why the same thread_id is required

Why the interrupted node executes again from its beginning

How to implement approve / reject / revise

How to integrate HITL with FastAPI

How to persist checkpoints in PostgreSQL

How to handle duplicate approvals and worker crashes

How to implement RBAC and audit trails

How to make external actions idempotent

How to test workflow recovery

My recommendation: Learn the basic `interrupt()` and `Command(resume=...)` implementation first, then focus on durable checkpointing, transactional approval handling, and idempotent execution. Those production concerns are what distinguish a basic LangGraph implementation from a staff-level design.

Official references: [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts?utm_source=chatgpt.com) · [LangGraph human-in-the-loop concepts](https://docs.langchain.com/oss/python/langgraph/concepts/human_in_the_loop/?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference+2
