# How to Make an AI Workflow Resumable — Senior/Staff AI Engineer Guide

In production AI systems, a workflow is resumable when it can stop unexpectedly, preserve its execution state durably, and continue from the last safe checkpoint without repeating completed work or corrupting results.

This is especially important in your enterprise financial compliance and multi-agent RAG systems, where a workflow may:

- Retrieve policies from Qdrant.
- Run multiple specialist agents.
- Call an expensive LLM several times.
- Wait for a human compliance officer to approve a finding.
- Execute a background job that takes several minutes.
- Survive worker crashes, deployments, network failures, or queue retries.

A simple AI script running from top to bottom is not enough. If the process crashes after five LLM calls, restarting the script from the beginning may repeat those calls, duplicate side effects, and lose the human approval state.

Let's build this using LangGraph, PostgreSQL-backed checkpointing, FastAPI, and durable workflow state.

## 1. Understand the architecture first

FastAPI

Start workflow · Check status · Resume

Durable job / workflow record

PostgreSQL: tenant, status, version, approval state

LangGraph workflow

Typed state · Checkpoints · Interrupts · Recovery

PostgreSQL

Persist graph checkpoints

Workers

Execute and resume jobs

External services

LLM · Qdrant · Policy repository · Human approval

There are two different things to persist:

1. Workflow state: what the workflow has completed, its outputs, and what it needs next.
2. Operational state: whether the job is running, awaiting approval, failed, or completed; which tenant owns it; and whether a worker currently holds the execution lease.

LangGraph checkpointing solves the first part. A production job system must also handle the second part.

## 2. Why a normal workflow is not resumable

Consider this code:

```
async def review_document(document: str):    chunks = await ingest_document(document)    evidence = await retrieve_policy_evidence(chunks)    findings = await analyze_with_llm(evidence)    approval = await request_human_approval(findings)    return await finalize_review(approval)
```

If the process crashes during `finalize_review()`, the application has no durable record of which steps succeeded unless you explicitly saved that information elsewhere.

A common but insufficient fix is to save a Python dictionary or use a global variable. That state disappears on process termination and cannot be shared reliably across workers.

The correct approach is to create durable checkpoints at meaningful workflow boundaries, with explicit recovery and side-effect handling.

## 3. Build a resumable LangGraph workflow

We will implement a realistic compliance workflow with these stages:

1. Retrieve policy evidence.
2. Generate a compliance draft.
3. Pause and wait for human approval.
4. Finalize the review after approval.
5. Resume from the saved checkpoint after a process restart.

### Step 3.1 — Install dependencies

```
pip install \
  langgraph \
  langgraph-checkpoint-postgres \
  "psycopg[binary,pool]" \
  fastapi \
  uvicorn \
  pydantic
```

Configure PostgreSQL using an environment variable:

```
export DATABASE_URL="postgresql://app:password@localhost:5432/ai_workflows"
```

Use a secrets manager for production credentials, TLS where required, and a database role with appropriately limited permissions.

LangGraph's checkpointer persists graph state by `thread_id`. The in-memory checkpointer is suitable for local experiments but loses its state when the process exits, so use durable storage for production.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



### Step 3.2 — Define a typed workflow state

Create `app/workflows/compliance/state.py`.

```
from typing import Literal, TypedDictclass ComplianceState(TypedDict, total=False):    workflow_id: str    tenant_id: str    document_id: str    policy_evidence: list[dict]    draft: dict    approval_status: Literal[        "pending",        "approved",        "rejected",    ]    approval_reason: str | None    reviewer_id: str | None    final_result: dict
```

The state contains the minimum information required to continue the workflow.

Senior engineering decision: avoid putting unnecessary information into checkpoints. Don't persist database connection objects, live clients, open file handles, or secrets. Store document references and version identifiers rather than large documents whenever practical.

For financial compliance, `tenant_id`, `document_id`, and policy versions must be tied to an authorized workflow record. A field in graph state is not itself proof of authorization.

### Step 3.3 — Define the graph nodes

Create `app/workflows/compliance/graph.py`.

The external services below are represented by small async functions so you can replace them with your actual Qdrant retriever, LLM service, and PostgreSQL repository.

```
from typing import Literalfrom langgraph.graph import StateGraph, START, ENDfrom langgraph.types import interrupt, Commandfrom app.workflows.compliance.state import ComplianceState# Replace with your real retrieval service.async def retrieve_policy_evidence(state: ComplianceState) -> dict:    evidence = [        {            "policy_id": "POL-101",            "text": "Unsupported guaranteed-return claims are prohibited.",            "policy_version": "v3",        }    ]    return {"policy_evidence": evidence}# Replace with your real LLM call and output validation.async def generate_draft(state: ComplianceState) -> dict:    evidence = state["policy_evidence"]    draft = {        "risk_level": "HIGH",        "violations": [            {                "policy_id": item["policy_id"],                "reason": "The claim requires supporting evidence.",            }            for item in evidence        ],        "policy_versions": [            item["policy_version"] for item in evidence        ],    }    return {        "draft": draft,        "approval_status": "pending",    }async def human_approval(    state: ComplianceState,) -> dict:    # Execution pauses here until an authorized reviewer responds.    decision = interrupt({        "type": "compliance_approval",        "workflow_id": state["workflow_id"],        "document_id": state["document_id"],        "draft": state["draft"],        "message": "Approve or reject this compliance review.",    })    # The resume payload must also be validated at the API boundary.    approved = decision["approved"]    return {        "approval_status": "approved" if approved else "rejected",        "approval_reason": decision.get("reason"),        "reviewer_id": decision["reviewer_id"],    }def route_after_approval(    state: ComplianceState,) -> Literal["finalize", "end"]:    if state["approval_status"] == "approved":        return "finalize"    return "end"async def finalize_review(state: ComplianceState) -> dict:    # Replace with an idempotent repository operation.    result = {        "workflow_id": state["workflow_id"],        "status": "approved",        "draft": state["draft"],        "reviewer_id": state["reviewer_id"],    }    return {"final_result": result}def build_graph(checkpointer):    builder = StateGraph(ComplianceState)    builder.add_node("retrieve", retrieve_policy_evidence)    builder.add_node("draft", generate_draft)    builder.add_node("approval", human_approval)    builder.add_node("finalize", finalize_review)    builder.add_edge(START, "retrieve")    builder.add_edge("retrieve", "draft")    builder.add_edge("draft", "approval")    builder.add_conditional_edges(        "approval",        route_after_approval,        {            "finalize": "finalize",            "end": END,        },    )    builder.add_edge("finalize", END)    return builder.compile(checkpointer=checkpointer)
```

The approval node uses LangGraph's `interrupt()` mechanism. When interrupted, the graph saves state through its checkpointer. Resuming with the same `thread_id` and a `Command(resume=...)` continues the suspended workflow. The node containing the interrupt starts executing again, so operations before the interrupt must be safe to repeat.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference

+1



One important detail: the `finalize_review()` above is illustrative. In production, its database write must be idempotent, and it must verify that the reviewer decision has been durably accepted.

## 4. Configure durable PostgreSQL checkpointing

Create the graph once during application startup and reuse the checkpointer. Do not create a fresh in-memory saver for every API request.

Create `app/main.py`:

```
import osfrom contextlib import asynccontextmanagerfrom fastapi import FastAPIfrom langgraph.checkpoint.postgres.aio import AsyncPostgresSaverfrom app.workflows.compliance.graph import build_graph@asynccontextmanagerasync def lifespan(app: FastAPI):    database_url = os.environ["DATABASE_URL"]    async with AsyncPostgresSaver.from_conn_string(        database_url    ) as checkpointer:        # Initialize checkpoint tables/migrations during deployment        # or controlled application initialization.        await checkpointer.setup()        app.state.compliance_graph = build_graph(checkpointer)        yieldapp = FastAPI(lifespan=lifespan)
```

For a larger deployment, run database migrations as a controlled deployment step rather than having every application replica attempt schema setup simultaneously. Use connection pooling, connection limits, checkpoint retention, and database backups appropriate to your recovery objectives.

The PostgreSQL checkpointer is responsible for persisting graph checkpoints. It is not a replacement for your application's workflow, approval, or audit tables.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference

+1



## 5. Start and resume through FastAPI

A resumable workflow needs a stable identifier that survives HTTP requests and worker restarts.

Use the same workflow ID as the LangGraph `thread_id`, provided you authorize access to it on every request. A UUID helps avoid accidental collisions, but an unguessable identifier is not authorization.

The following API example demonstrates the two main operations. It assumes that `get_current_principal()` authenticates the caller and `workflow_repository` provides durable workflow ownership and status records.

```
from uuid import uuid4from typing import Literalfrom fastapi import Depends, HTTPException, Requestfrom pydantic import BaseModel, Fieldfrom langgraph.types import Commandclass StartReviewRequest(BaseModel):    document_id: strclass ApprovalRequest(BaseModel):    decision: Literal["approve", "reject"]    reason: str | None = Field(default=None, max_length=2000)@app.post("/workflows")async def start_review(    body: StartReviewRequest,    request: Request,    principal=Depends(get_current_principal),):    workflow_id = str(uuid4())    # Verify document access before starting the workflow.    await document_repository.require_access(        tenant_id=principal.tenant_id,        document_id=body.document_id,    )    # Persist workflow ownership and initial status before execution.    await workflow_repository.create(        workflow_id=workflow_id,        tenant_id=principal.tenant_id,        document_id=body.document_id,
```

The repository and authentication functions above are integration points, not built-in FastAPI or LangGraph APIs. For production, enqueue the workflow to a worker instead of keeping the HTTP request open while the graph executes. The worker should use the same persistent graph and workflow ID.

Now implement the resume endpoint:

```
@app.post("/workflows/{workflow_id}/approval")async def resume_review(    workflow_id: str,    body: ApprovalRequest,    request: Request,    principal=Depends(get_current_principal),):    # Check tenant ownership and reviewer authorization.    workflow = await workflow_repository.get_owned_workflow(        workflow_id=workflow_id,        tenant_id=principal.tenant_id,    )    if workflow is None:        raise HTTPException(404, "Workflow not found")    if workflow.status != "awaiting_approval":        raise HTTPException(            409,            "Workflow is not awaiting approval",        )    await reviewer_policy.require_can_approve(        principal=principal,        workflow=workflow,    )    config = {        "configurable": {            "thread_id": workflow_id,        }    }    # The identity comes from authentication, never the request body.    resume_value = {        "approved": body.decision == "approve",
```

This illustrates the API shape, but a production implementation must make the workflow status transition, approval record, and concurrency control robust against duplicate requests and process crashes. In particular, do not mark the workflow completed merely because an approval was submitted: verify the graph's actual terminal state and persisted final result.

A strong implementation records the approval decision and an idempotency key in one transaction, then lets a durable worker resume the graph. This avoids an HTTP retry creating duplicate approval events or concurrent graph executions.

## 6. Understand what happens when the workflow resumes

Let's walk through the execution lifecycle.

Step 1

Workflow starts

The worker retrieves policy evidence and saves graph progress in PostgreSQL.

Step 2

Draft generated

The structured result is stored in workflow state and validated.

Step 3

Workflow pauses

The approval interrupt persists the suspended graph state. The worker can release its resources.

Step 4

Application restarts

The process loses its in-memory objects, but the PostgreSQL checkpoint remains available.

Step 5

Reviewer approves

The API authorizes the reviewer and submits a resume command using the original workflow ID.

Step 6

Execution continues

LangGraph restores the saved state, processes the approval, and runs the appropriate downstream nodes.

The crucial detail is that the application does not reconstruct the workflow by asking the LLM what happened earlier. It restores the recorded execution state.

## 7. Checkpointing alone does not guarantee exactly-once execution

This is one of the most important senior-engineer concepts.

Suppose a node performs these actions:

```
async def publish_review(state):    await external_compliance_api.publish(        state["final_result"]    )    return {"published": True}
```

What happens if the external API publishes the review, but the worker crashes before LangGraph records the node's completion?

When the workflow resumes, the external operation may run again.

This is the difference between durable workflow state and durable side-effect management.

### Use idempotency keys for side effects

```
async def publish_review(state, compliance_client):    workflow_id = state["workflow_id"]    await compliance_client.publish(        payload=state["final_result"],        idempotency_key=f"compliance-review:{workflow_id}",    )    return {"published": True}
```

This works only if the receiving service actually enforces the idempotency key and returns the original result for duplicate requests.

If the external system doesn't support idempotency, use an outbox or durable operation ledger with reconciliation, or a compensating action where appropriate. You cannot guarantee exactly-once effects merely by adding a checkpoint.

### Production rules for workflow nodes

| Node type      | Design requirement                                                     |
| -------------- | ---------------------------------------------------------------------- |
| Retrieval      | Safe to rerun, or cache results by immutable query/index version       |
| LLM generation | Accept that retries can produce different results and incur extra cost |
| Database write | Use transactions, unique constraints, and idempotency                  |
| External tool  | Use idempotency keys, operation status checks, or reconciliation       |
| Human approval | Persist reviewer identity, decision, timestamp, and audit record       |
| Finalization   | Make terminal state transitions atomic and repeatable                  |

LangGraph checkpointing provides fault-tolerant state recovery, but your application remains responsible for making external effects safe.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 8. How do you recover automatically from worker crashes?

For production, don't rely on someone manually calling the resume endpoint whenever a worker crashes. Introduce a durable job queue and a recovery process.

A typical architecture is FastAPI + Celery + Redis or RabbitMQ + PostgreSQL.

- PostgreSQL stores workflow ownership, status, approval records, and graph checkpoints.
- The queue distributes execution work to workers.
- Workers execute or resume the graph using the same workflow ID.
- A recovery scheduler detects jobs that were running but lost their worker.
- A retry policy distinguishes transient failures from permanent failures.

A durable workflow record might contain:

```
workflow_id
tenant_id
status
thread_id
attempt_count
lease_owner
lease_expires_at
last_error_code
created_at
updated_at
```

### Recovery logic

```
async def recover_stale_workflows(repository, queue):    stale_workflows = await repository.find_expired_leases()    for workflow in stale_workflows:        if workflow.status in {            "completed",            "rejected",            "cancelled",        }:            continue        if workflow.attempt_count >= 5:            await repository.mark_failed(                workflow.workflow_id,                reason="MAX_ATTEMPTS_EXCEEDED",            )            continue        claimed = await repository.claim_recovery_lease(            workflow_id=workflow.workflow_id,        )        if not claimed:            # Another recovery worker claimed this workflow.            continue        await queue.enqueue(            workflow_id=workflow.workflow_id,            idempotency_key=(                f"resume:{workflow.workflow_id}:"                f"{workflow.attempt_count}"            ),        )
```

This is illustrative recovery pseudocode; `find_expired_leases`, `claim_recovery_lease`, and `enqueue` must be implemented with transactional or otherwise atomic concurrency controls.

There is a subtle but important problem here: a database transaction and a queue publish generally cannot be made atomic just by calling them sequentially. If the process crashes after claiming the workflow but before publishing the job, work can get stuck.

A common solution is a transactional outbox:

1. In one database transaction, claim the workflow and insert an outbox event.
2. An outbox publisher delivers pending events to the queue.
3. The worker claims the workflow using a lease or fencing token.
4. Duplicate deliveries are harmless because execution is deduplicated.
5. Expired leases and unprocessed outbox events are recovered automatically.

Also, do not automatically resume workflows awaiting human approval just because their worker lease expired. Those workflows are intentionally paused. Resume them only when an authorized decision or an explicit timeout policy permits it.

## 9. What should you test before calling a workflow production-ready?

The most valuable tests are failure-injection tests.

### Resumability test plan

0/8

Kill the worker after retrieval and verify that execution recovers.

Restart the application while a workflow awaits approval.

Submit the same approval twice and verify no duplicate finalization.

Simulate an external API succeeding just before a worker crash.

Run two workers against the same workflow and verify concurrency protection.

Attempt to resume a workflow from another tenant and verify access is denied.

Change the graph or schema version and verify incompatible checkpoints are handled safely.

Verify retry limits, terminal failure states, and alerting.

You should also test checkpoint retention, database failover, backup restoration, and observability. A checkpoint that exists but cannot be deserialized by a newer deployment is not useful recovery.

## 10. Common mistakes to avoid

| Mistake                                                      | Better approach                                                   |
| ------------------------------------------------------------ | ----------------------------------------------------------------- |
| Using an in-memory checkpointer in production                | Use durable checkpoint storage                                    |
| Generating a new thread ID when resuming                     | Reuse the original workflow ID                                    |
| Saving only the final answer                                 | Persist progress at workflow boundaries                           |
| Assuming checkpointing prevents duplicate API calls          | Make side effects idempotent                                      |
| Trusting the approval payload                                | Authenticate the reviewer and validate authorization              |
| Allowing multiple workers to resume concurrently             | Use atomic claims, leases, and fencing                            |
| Treating all failures as retryable                           | Classify errors and bound retries                                 |
| Updating graph code without considering existing checkpoints | Version state and support migrations or controlled draining       |
| Keeping large document contents in every checkpoint          | Store immutable references and version identifiers where possible |
| Using checkpoint state as the audit log                      | Maintain a separate durable, append-only audit trail              |

## 11. Alternative approaches and when to use them

| Approach                            | Best use case                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| LangGraph + PostgreSQL checkpointer | Stateful LLM workflows, agent graphs, human approval, resumable reasoning                        |
| Celery + PostgreSQL job records     | Background execution and distributed task processing; workflow state must be designed separately |
| Temporal                            | Long-running distributed workflows, durable timers, retries, signals, and complex orchestration  |
| Custom state machine + database     | Small, predictable workflows where explicit transitions are sufficient                           |

For your multi-agent financial compliance platform, I would start with LangGraph's PostgreSQL checkpointing and a durable job queue. If the system grows into complex multi-day business processes with many external systems, timers, compensating actions, and strict operational guarantees, evaluate Temporal or a comparable workflow orchestration platform.

## 12. Staff-level interview answer

> I make AI workflows resumable by separating durable workflow state from execution infrastructure. I use a persistent checkpoint store, such as PostgreSQL-backed LangGraph checkpointing, and a stable workflow ID to restore execution after a crash or human interruption.
>
> I design the graph around meaningful boundaries, use typed state and explicit transitions, and handle human approval with durable interrupts. For distributed execution, I use a queue, atomic workflow claims, leases, bounded retries, and recovery of expired jobs.
>
> I don't assume checkpointing guarantees exactly-once execution. External side effects are protected using idempotency keys, transactions, an outbox, or reconciliation. I separately persist approval records and audit events, enforce tenant authorization on every resume operation, and test worker crashes, duplicate messages, concurrency, and deployment compatibility.
>
> Finally, I monitor workflow age, checkpoint failures, retry counts, stuck jobs, and recovery time so that resumability is a measurable production capability rather than just a feature that works in a local demo.

The core principle is: persist the state, make recovery explicit, and make every retried operation safe. That is what turns an AI workflow from a fragile script into a production-grade system.