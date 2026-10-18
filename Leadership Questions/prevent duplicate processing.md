# How to prevent duplicate processing when a user submits the same request twice

As a senior or staff AI engineer, I solve this using idempotency, durable request tracking, database constraints, and concurrency control.

In an enterprise AI application such as your FastAPI + Celery + PostgreSQL + Redis + LangGraph financial-advisor platform, duplicate requests can happen for several reasons:

- A user double-clicks the Submit button.
- A mobile app retries after a network timeout.
- A load balancer or client resends a request.
- The API successfully accepts a request, but the response never reaches the client.
- A Celery task is redelivered after a worker crash.
- Two API replicas receive the same request concurrently.

The critical distinction is that duplicate HTTP requests and duplicate business operations are not necessarily the same thing.

For example, a user might intentionally ask the same financial question twice because they want a fresh answer. You should not automatically deduplicate every identical question. Instead, you need to identify whether the requests represent the same logical operation.

## 1. Understand the failure scenario

Imagine the user submits:

`POST /advisor/analyze-portfolio`

The following sequence occurs:

Client submits request

Request ID: request-123

FastAPI accepts request

Workflow created · Celery task enqueued

Response is lost

The client cannot tell whether the operation succeeded

Client submits again

Without protection, another workflow may start

Without duplicate protection, you could end up with:

- Two LangGraph workflows analyzing the same portfolio.
- Duplicate LLM calls and unnecessary cost.
- Repeated document ingestion or vector writes.
- Multiple notifications.
- Concurrent updates to the same business record.
- In the worst case, duplicate financial transactions or other external side effects.

The correct solution is not merely disabling the Submit button. The server must enforce duplicate protection because clients can retry, crash, or bypass the UI entirely.

## 2. The four main mechanisms

1\. Idempotency key

A client sends a stable key for one logical operation. Repeated submissions with that key return or reference the original operation rather than creating a new one.

2\. Database uniqueness

PostgreSQL enforces a unique constraint on the relevant identity. This closes race conditions when multiple API replicas receive the same key concurrently.

3\. Idempotent workflow steps

Celery retries and LangGraph resumption must not repeat completed side effects unnecessarily. Use durable step records, unique operation IDs, and safe upserts.

4\. Business-level constraints

Enforce rules such as one active ingestion per document version, one result per analysis run, or one execution per approved transaction ID.

These mechanisms solve different problems. The idempotency key handles repeated submissions; the database prevents concurrent duplicate creation; workflow idempotency handles retries after execution begins; business constraints protect the underlying domain.

## 3. Design the idempotency key correctly

An idempotency key identifies one logical operation, not merely the content of a request.

For example:

```
POST /advisor/analyze-portfolio
Authorization: Bearer <token>
Idempotency-Key: 8d5f8e10-portfolio-analysis-2026-001
Content-Type: application/json
```

```
{
  "portfolio_id": "portfolio-456",
  "analysis_type": "risk_assessment"
}
```

If the client retries this request with the same key and the same payload, the server should return the original workflow's status or result.

If the client sends a different payload with the same key, the server should reject it, typically with `409 Conflict`.

### What should the key contain?

A client-generated UUID is often a good default. The server scopes it to the authenticated user or tenant and the operation.

The effective uniqueness rule might be:

```
(tenant_id, authenticated_user_id, endpoint, idempotency_key)
```

Choose the scope based on your business semantics. If several authorized users in a tenant must share the same logical operation, user-level scoping might be too narrow; a tenant-level business operation ID may be more appropriate.

Do not use the request body hash alone as the idempotency key. Two identical financial questions can be intentional separate requests. A payload hash is useful for detecting whether a reused key has been associated with different content.

Also, don't trust a client-supplied tenant ID without checking it against authenticated permissions.

## 4. Store idempotency records in PostgreSQL

For a production FastAPI application, PostgreSQL is a good place to enforce durable idempotency. Redis can accelerate lookups, but it should not be the sole source of truth when losing its contents could create duplicate business operations.

A basic schema:

```
CREATE TABLE idempotency_records (
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    endpoint TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,

    workflow_id UUID NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'ACCEPTED',

    response_status INTEGER,
    response_body JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,

    PRIMARY KEY (
        tenant_id,
        user_id,
        endpoint,
        idempotency_key
    )
);
```

This schema illustrates the core idea. In production, add foreign keys where appropriate, validate status transitions, and establish a retention policy consistent with your retry window and business requirements.

The `workflow_id` identifies the actual workflow. The idempotency key identifies the submission. They are related, but they serve different purposes.

### What happens when the same key arrives twice?

| Existing record         | Incoming request            | Correct behavior                                                     |
| ----------------------- | --------------------------- | -------------------------------------------------------------------- |
| No record               | New key                     | Create the operation                                                 |
| `ACCEPTED` or `RUNNING` | Same key, same payload      | Return the existing workflow ID and status                           |
| `COMPLETED`             | Same key, same payload      | Return the original result or its resource URL                       |
| Any existing state      | Same key, different payload | Reject with `409 Conflict`                                           |
| Previously failed       | Same key, same payload      | Return the recorded failure/status or apply an explicit retry policy |

That final row needs a deliberate design. If a workflow failed, a client retry with the same key should not silently create an unrelated new operation. Either resume the existing workflow under its defined retry policy or require a new operation key.

## 5. Implement the FastAPI request flow

The core algorithm is:

1. Authenticate the caller and authorize access to the requested resource.
2. Validate the request and calculate a canonical payload hash.
3. Atomically claim the idempotency key in PostgreSQL.
4. If the key already exists, compare hashes and return the existing operation when they match.
5. If it is a new key, create the durable workflow record.
6. Ensure the work is reliably dispatched to Celery.
7. Return the workflow ID.

The atomic database insert is what protects against concurrent requests. A separate `SELECT` followed by `INSERT` is not enough: two replicas could both observe that no record exists.

Here is the essential SQL pattern:

```
INSERT INTO idempotency_records (
    tenant_id,
    user_id,
    endpoint,
    idempotency_key,
    request_hash,
    workflow_id,
    status
)
VALUES (
    :tenant_id,
    :user_id,
    :endpoint,
    :key,
    :request_hash,
    :workflow_id,
    'ACCEPTED'
)
ON CONFLICT (
    tenant_id,
    user_id,
    endpoint,
    idempotency_key
)
DO NOTHING
RETURNING workflow_id, request_hash, status;
```

If the insert returns a row, the request claimed the key. If it returns no row, load the existing record and compare its hash.

The claim, workflow creation, and durable dispatch intent should be coordinated in one database transaction. If you commit the idempotency record and then enqueue Celery separately, the API could crash between those operations, leaving a request accepted but never processed.

That is why production systems commonly use a transactional outbox.

## 6. Prevent the database-to-Celery dual-write problem

Consider this implementation:

```
await save_workflow(workflow_id)celery_task.delay(str(workflow_id))
```

It has a failure window:

- PostgreSQL commits the workflow.
- FastAPI crashes before publishing the Celery task.
- The workflow exists, but no worker processes it.

Reversing the order has a different problem: the worker could start before the database transaction commits.

### Use the transactional outbox pattern

In one PostgreSQL transaction, save both the idempotency record and an outbox event.

```
CREATE TABLE workflow_outbox (
    event_id UUID PRIMARY KEY,
    workflow_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,

    UNIQUE (workflow_id, event_type)
);
```

The transaction commits:

```
BEGIN
  |
  +-- Insert idempotency record
  |
  +-- Insert workflow record
  |
  +-- Insert WORKFLOW_REQUESTED outbox event
  |
COMMIT
```

A separate dispatcher publishes pending events to Celery and records successful publication.

```
@shared_taskdef process_workflow(workflow_id: str):    # Reload the durable workflow record.    # Claim the workflow using a lease or conditional update.    # Skip or resume according to the durable state.    # Persist every important step result.    ...
```

The dispatcher may publish the same event more than once if it crashes after publishing but before marking the event as published. That is expected: the consumer must deduplicate using the stable `workflow_id` or event ID.

The outbox ensures that a committed workflow has a durable dispatch intent. It does not make Celery delivery exactly once.

## 7. Handle simultaneous duplicate requests

This is where senior engineers distinguish a working demo from a reliable distributed system.

Imagine two FastAPI replicas receive the same request at almost the same instant:

```
Request A ──> API replica 1 ──┐
                             ├──> PostgreSQL
Request B ──> API replica 2 ──┘
```

Both requests might attempt to create a workflow. The unique constraint allows only one to claim the idempotency key.

The losing request then loads the existing record:

- If the payload hash matches, return the existing workflow.
- If the payload hash differs, return `409 Conflict`.
- Do not enqueue a second logical workflow.

The database constraint, not an in-process Python lock or Redis lock, is the final authority for key uniqueness.

### What response should the client receive?

For an asynchronous workflow API, I would typically return `202 Accepted` for the original submission:

```
{
  "workflow_id": "4f6d6a34-4f5d-4a6c-aabc-00c91e4c29b2",
  "status": "accepted",
  "status_url": "/workflows/4f6d6a34-4f5d-4a6c-aabc-00c91e4c29b2"
}
```

A repeated request with the same key and payload can return the same workflow ID and current status. You can return the same HTTP status as the original response or use a documented replay convention, but the behavior must be consistent.

Do not return `completed` just because the idempotency record exists. The response must reflect the actual persisted workflow state.

## 8. Prevent duplicate processing inside Celery and LangGraph

HTTP idempotency prevents duplicate workflow creation. It does not prevent the same Celery task from executing twice or the same LangGraph node from running again after recovery.

You need a second layer: step-level idempotency.

For example, your workflow might contain:

```
Workflow: portfolio-analysis-123
    |
    +--> retrieve_documents
    |
    +--> generate_embeddings
    |
    +--> analyze_portfolio
    |
    +--> persist_report
    |
    +--> notify_user
```

Persist the status and output of each completed step. Use stable step identifiers such as:

```
workflow_id + step_name + step_version
```

If the workflow is redelivered, it should reuse valid completed outputs rather than repeating expensive operations unnecessarily.

For each step, define its recovery behavior:

| Operation                            | Duplicate-prevention strategy                                    |
| ------------------------------------ | ---------------------------------------------------------------- |
| Read documents                       | Usually safe to repeat; use consistent versioning where needed   |
| Generate embeddings                  | Cache or persist by content hash and embedding-model version     |
| Upsert Qdrant vectors                | Deterministic point IDs and idempotent upserts                   |
| Generate an LLM answer               | Reuse a persisted result if the same execution should be resumed |
| Save a report                        | Unique workflow/report identity and transactional writes         |
| Send notification                    | Transactional outbox and downstream idempotency key              |
| Execute an external financial action | Unique business operation ID, authorization, and reconciliation  |

A subtle point: LLM generation is not inherently deterministic. Retrying an LLM call can produce a different answer and incur additional cost. If you need the same logical result after a crash, persist the original successful output before advancing the workflow.

Also, if a worker crashes after the LLM responds but before the result is saved, you may not be able to know whether the request was processed successfully. This is an uncertain outcome; your design must explicitly choose whether to retry, reconcile, or request intervention.

## 9. Should you use Redis locks?

Redis locks can be useful for reducing duplicate work, but they should not replace database uniqueness constraints.

For example, a short-lived lock might prevent two workers from simultaneously regenerating an expensive report. But locks can expire, workers can pause, and network failures can leave the caller uncertain about lock ownership.

For critical operations, combine:

- Database unique constraints.
- Atomic conditional updates or transactions.
- Lease ownership and fencing tokens where appropriate.
- Idempotent side effects.
- Durable workflow state.

A Redis lock is an optimization or coordination mechanism; it is not sufficient proof that an operation can execute only once.

## 10. How do you choose the right deduplication strategy?

There are three different requirements that are often confused.

### A. Same HTTP operation

The user double-clicks or the client retries.

Use: Idempotency key + PostgreSQL uniqueness.

### B. Same workflow step

Celery redelivers a task or LangGraph resumes execution.

Use: Durable step state + idempotent operations + checkpoints.

### C. Same business action

Two requests attempt to create the same report or execute the same transaction.

Use: Domain-level uniqueness + authorization + transaction reconciliation.

For example, a user submitting the same portfolio analysis twice with the same idempotency key should usually receive the same workflow. Submitting a new key may intentionally request a fresh analysis.

For a financial transaction, however, two different HTTP keys could still refer to the same business operation. That is why domain-level deduplication is essential for high-impact actions.

## 11. Production testing checklist

Before deploying, test these cases:

- The same request and key are submitted twice sequentially.
- Two identical requests arrive concurrently at different API replicas.
- The same key is reused with a different payload.
- The API crashes after committing the workflow but before publishing to Celery.
- The outbox dispatcher publishes an event twice.
- A Celery worker crashes after a side effect but before recording completion.
- A LangGraph node is replayed after a checkpoint recovery.
- The same notification event is delivered more than once.
- The idempotency record expires, and an old client retries.
- A different authorized user submits the same payload.
- A duplicate request arrives while the original workflow is still running.

Define a retention period for idempotency records that covers the maximum client retry window and your business requirements. For financial operations, keep permanent or long-lived business-operation identifiers where required, even if temporary HTTP idempotency records eventually expire.

## 12. Interview-ready answer

> "I prevent duplicate processing at multiple layers. At the API layer, the client supplies an idempotency key for each logical operation. I scope that key to the authenticated tenant, user or business context, and endpoint, and calculate a canonical request hash so that reusing a key with a different payload is rejected.
>
> In PostgreSQL, I enforce uniqueness with a database constraint and claim the key atomically. If a duplicate request arrives, I return the existing workflow ID and persisted status instead of creating another workflow.
>
> To avoid losing work between database commits and Celery publication, I use a transactional outbox. Since outbox events and Celery tasks may be delivered more than once, workers must also be idempotent. I persist workflow-step results, use stable operation IDs, and resume from durable checkpoints.
>
> For external side effects, I use transactions, unique constraints, an outbox, downstream idempotency keys where supported, and reconciliation for uncertain outcomes. I also test concurrent requests, worker crashes, duplicate delivery, and idempotency-record expiration.
>
> The key point is that exactly-once execution is not something I assume from the message broker. I build effectively-once business outcomes through durable state, atomic uniqueness, idempotent operations, and explicit recovery policies."

The staff-level takeaway: idempotency is an end-to-end architectural property. A UUID in the API request is only the beginning; the database, task queue, workflow engine, and external side effects must all participate in the design.