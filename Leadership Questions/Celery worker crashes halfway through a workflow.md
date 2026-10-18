# What if a Celery worker crashes halfway through a workflow?

As a senior or staff AI engineer, I treat this as a distributed workflow failure, not merely a worker-process crash.

In an enterprise AI application, such as your FastAPI + Celery + PostgreSQL + Redis + LangGraph financial-advisor platform, a worker might crash while ingesting documents, generating embeddings, executing agents, evaluating a RAG answer, or waiting for an LLM response.

The critical question is not just whether Celery retries the task. It is:

What work has already completed, what state has been persisted, what side effects may have occurred, and how can we resume without losing data or executing the same action twice?

## 1. What happens when a worker crashes?

Imagine this workflow:

FastAPI submits a workflow

Task ID: task-123

Celery worker starts

1\. Retrieve documents

Completed

2\. Generate embeddings

Completed and persisted

3\. LLM analysis — worker crashes

No final result committed

What happens next depends on your Celery configuration, broker, task acknowledgement settings, and where the workflow state is stored.

There are four important possibilities:

1. The task is redelivered. The broker may deliver it to another worker if it was not acknowledged and the relevant recovery conditions are met.
2. The task is marked failed. If the failure is caught and reported, or the task has already been acknowledged, automatic redelivery may not happen.
3. The task is retried. Your application or Celery retry policy schedules another attempt.
4. The task is lost from the application's perspective. The task may have been acknowledged, while the application has no durable record of how far the workflow progressed.

A crucial distinction: Celery task state such as `STARTED`, `RETRY`, or `FAILURE` is not the same as durable business-workflow state.

Even if the task is redelivered, Celery does not automatically know which agent steps, database writes, or external API actions completed.

## 2. Why duplicate execution is the biggest risk

Suppose your workflow performs these operations:

1. Reads a financial document.
2. Generates embeddings.
3. Stores vectors in Qdrant.
4. Calls an LLM to analyze the document.
5. Writes the final analysis to PostgreSQL.
6. Sends a notification.

The worker crashes after step 5 but before step 6.

When the task is retried, it might execute steps 1–5 again. That can lead to duplicate LLM charges, duplicate vectors, overwritten results, or repeated notifications.

An even more dangerous scenario occurs when a worker calls an external system, the external operation succeeds, and the worker crashes before recording the success locally. On retry, the system cannot safely assume the external operation failed.

Senior engineering principle: design for at-least-once execution and make operations idempotent wherever possible. Do not assume exactly-once execution across Celery, a database, an LLM provider, and external tools.

## 3. Configure Celery for worker-loss recovery

For tasks that are safe to execute more than once, late acknowledgement can help.

```
# app/celery_app.pyfrom celery import Celerycelery_app = Celery(    "financial_ai",    broker="redis://localhost:6379/0",    backend="redis://localhost:6379/1",)celery_app.conf.update(    task_acks_late=True,    task_reject_on_worker_lost=True,    worker_prefetch_multiplier=1,    task_track_started=True,    task_soft_time_limit=240,    task_time_limit=270,)
```

What these settings do:

- `task_acks_late=True`: acknowledges a task after execution rather than before execution.
- `task_reject_on_worker_lost=True`: requests requeueing when a worker process is lost while executing a late-acknowledged task.
- `worker_prefetch_multiplier=1`: limits prefetching to reduce the number of tasks held by a worker, subject to the worker pool and broker.
- `task_track_started=True`: records a `STARTED` state in the configured result backend.
- Time limits: constrain runaway tasks; they do not guarantee that every operation will terminate cleanly.

These settings are not a complete recovery system.

In particular, late acknowledgement does not guarantee redelivery in every failure scenario. Broker behavior, connection loss, shutdown mode, and Celery configuration matter. Requeued tasks can also execute more than once.

If Redis is your broker, configure and test its visibility timeout against the maximum task duration and expected recovery behavior. A visibility timeout that is too short can cause duplicate concurrent execution; one that is too long can delay recovery.

For long-running workflows, I would use late acknowledgement only alongside durable checkpoints and idempotent task logic.

## 4. Persist workflow progress in PostgreSQL

This is the most important architectural decision.

Do not rely on a worker's memory or Celery's result backend as the only record of workflow progress. Store the business workflow's status and completed steps durably.

For example, use a table like this:

```
CREATE TABLE workflow_runs (
    workflow_id UUID PRIMARY KEY,
    status TEXT NOT NULL,
    current_step TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    result JSONB,
    last_error TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE workflow_steps (
    workflow_id UUID NOT NULL
        REFERENCES workflow_runs(workflow_id),
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,
    output JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (workflow_id, step_name)
);
```

The first table records the workflow lifecycle. The second records the outcome of each step.

For example:

| Step                     | Persisted status       | Resume behavior                      |
| ------------------------ | ---------------------- | ------------------------------------ |
| Document extraction      | `COMPLETED`            | Reuse validated output               |
| Embedding generation     | `COMPLETED`            | Reuse or reconcile stored vectors    |
| LLM analysis             | `RUNNING` before crash | Reconcile outcome; retry if safe     |
| Final result persistence | `PENDING`              | Execute after analysis succeeds      |
| Notification             | `PENDING`              | Dispatch through a durable mechanism |

The status `RUNNING` does not prove the step is still running after a crash. On recovery, use leases, ownership tokens, or time-based recovery rules to detect abandoned work.

Also, do not mark a step `COMPLETED` before its output is durably stored. Otherwise, the workflow can resume from a step whose result does not actually exist.

## 5. Implement resumable, idempotent tasks

A task should receive a stable `workflow_id`, not generate a new workflow identity every time it retries.

Here is a simplified example of a resumable Celery task:

```
# app/tasks/document_workflow.pyfrom celery import shared_taskfrom celery.exceptions import MaxRetriesExceededError@shared_task(    bind=True,    acks_late=True,    max_retries=3,)def process_document(self, workflow_id: str):    try:        workflow = repository.get_workflow(workflow_id)        if workflow.status == "COMPLETED":            return workflow.result        # Each operation checks durable state before doing work.        extracted = run_step_once(            workflow_id,            "extract",            lambda: extract_document(workflow_id),        )        chunks = run_step_once(            workflow_id,            "chunk",            lambda: chunk_document(extracted),        )        embeddings = run_step_once(            workflow_id,            "embed",            lambda: generate_embeddings(chunks),        )        run_step_once(            workflow_id,            "index",            lambda: upsert_vectors(                workflow_id,                embeddings,            ),        )        analysis = run_step_once(            workflow_id,            "analyze",            lambda: analyze_with_llm(chunks),        )        result = persist_final_result(            workflow_id,            analysis,        )        mark_workflow_completed(workflow_id, result)        return result    except TransientWorkflowError as exc:        try:            raise self.retry(                exc=exc,                countdown=min(2 ** self.request.retries, 30),            )        except MaxRetriesExceededError:            mark_workflow_failed(workflow_id, str(exc))            raise
```

The functions `repository`, `run_step_once`, `extract_document`, `upsert_vectors`, and the other helpers are illustrative application interfaces, not built-in Celery APIs.

The critical part is `run_step_once()`. It must be implemented with durable storage and concurrency control; a simple in-memory dictionary or a check-then-run database query is insufficient.

A robust implementation should:

1. Acquire a lease or atomically claim the step.
2. Check whether a valid completed output already exists.
3. Run the operation.
4. Persist the output and completion status safely.
5. Release or expire the lease if the worker dies.
6. Reconcile uncertain outcomes before repeating external operations.

For database-only work, use transactions and unique constraints where appropriate. For Qdrant, use deterministic point IDs and idempotent upserts. For LLM calls, avoid assuming a timed-out request was never processed.

Notice that the task-level `max_retries` is only one layer of the policy. You must also define how the workflow handles exhausted retries, abandoned `RUNNING` steps, and failures that occur after a side effect but before its status is saved.

## 6. Prevent duplicate notifications and external actions

Consider the notification step. If an email is sent successfully but the worker crashes before saving `notification_sent = true`, retrying the task can send the email twice.

A common solution is the transactional outbox pattern.

PostgreSQL transaction

Save workflow result + insert outbox event

Outbox dispatcher

Claims pending events and delivers them

Notification provider

Use an idempotency key when supported

The result and outbox event are committed in the same PostgreSQL transaction. If the process crashes after committing, the event remains available for another dispatcher to process.

The dispatcher itself can also crash after sending the notification but before marking the event as delivered. Therefore, use a stable event ID as an idempotency key with the downstream service where supported.

For financial transactions or other high-impact actions, use stronger safeguards: durable action records, unique business operation IDs, reconciliation against the external system, and explicit authorization. A workflow retry must never be treated as authorization to repeat a transaction.

## 7. What if the workflow uses LangGraph?

Celery recovery and LangGraph recovery solve different problems.

- Celery manages task execution, delivery, acknowledgement, and retry.
- LangGraph manages graph execution, state transitions, checkpoints, and resumption when configured with a durable checkpointer.
- PostgreSQL can persist your application's business workflow records and results.

For a long-running agent workflow, I would use this pattern:

```
FastAPI
   |
   v
Create durable workflow record
   |
   v
Celery task starts graph
   |
   v
LangGraph checkpoint
   |
   v
Research agent completes
   |
   v
Persist state and checkpoint
   |
   v
LLM analysis agent
   |
   X Worker crashes
   |
   v
Celery redelivery / recovery scheduler
   |
   v
Load durable workflow state
   |
   v
Resume from a valid checkpoint
   |
   v
Persist final result
```

A LangGraph checkpoint is not automatically a guarantee that every external side effect is safe to replay. Graph node boundaries, checkpointer configuration, and side-effect handling all matter.

For example, if a node invokes an external tool and crashes before the graph records that node's completion, the node may run again. Make side effects idempotent and persist their outcomes independently when necessary.

For human-in-the-loop workflows, also persist the approval state. A worker crash must not silently bypass an approval gate or turn a pending approval into an approved action.

## 8. How do you detect and recover abandoned workflows?

A production system needs a recovery mechanism beyond Celery's automatic retries.

I would implement a workflow recovery scheduler that periodically finds workflows stuck in `RUNNING` beyond an expected duration.

For each candidate, it should:

1. Check whether the worker still owns a valid lease.
2. Verify that another worker has not already claimed the workflow.
3. Inspect the current step and last durable checkpoint.
4. Determine whether the step can be retried safely or needs reconciliation.
5. Atomically schedule recovery.
6. Escalate workflows that exceed the retry budget instead of retrying forever.

Use database transactions or atomic conditional updates to prevent two recovery workers from claiming the same workflow.

A simplified recovery query might look like this:

```
SELECT workflow_id, current_step, updated_at
FROM workflow_runs
WHERE status = 'RUNNING'
  AND updated_at < NOW() - INTERVAL '10 minutes';
```

This query identifies possible abandoned workflows; it does not establish that they are safe to restart. A production implementation must account for leases, heartbeats, workflow duration, and concurrent ownership.

For operational visibility, monitor:

| Metric                              | Why it matters                       |
| ----------------------------------- | ------------------------------------ |
| Worker crashes and unexpected exits | Detect execution instability         |
| Task retries per workflow           | Identify recurring failures          |
| Queue depth and oldest task age     | Detect backlog growth                |
| Workflows stuck in `RUNNING`        | Find abandoned executions            |
| Duplicate-step attempts             | Detect weak idempotency controls     |
| Recovery success rate               | Measure resilience                   |
| Workflow completion time            | Detect degradation                   |
| Failed and dead-lettered workflows  | Identify work requiring intervention |

For Celery workers, I would also inspect worker logs, broker connection health, memory usage, task duration, and task-level exception traces. If workers are being killed by out-of-memory events, repeated retries alone will make the problem worse.

## 9. Production checklist

Before deploying this architecture, verify the following.

### Worker resilience review

0 of 10

Configure and test acknowledgement and redelivery behavior

Use stable workflow IDs and durable step state

Make database writes and vector upserts idempotent

Implement bounded retries with exponential backoff and jitter

Persist LangGraph checkpoints when workflow resumption is required

Use an outbox or equivalent for reliable external event delivery

Add leases and concurrency control for recovery workers

Set time limits, retry budgets, and terminal failure states

Test worker termination at every workflow step

Alert on stuck workflows, queue age, and repeated crashes

## 10. Interview-ready answer

> "If a Celery worker crashes halfway through a workflow, the task may be redelivered, retried, or remain failed depending on acknowledgement settings, broker behavior, and the failure mode. I never assume that automatic redelivery resumes from the exact point of failure.
>
> I design for at-least-once execution. I persist workflow and step state in PostgreSQL, use durable LangGraph checkpoints where appropriate, and make individual operations idempotent. For worker-loss recovery, I configure late acknowledgement and appropriate worker-loss redelivery behavior, then validate those settings under actual crash scenarios.
>
> I pay particular attention to partial side effects. A database write, vector upsert, notification, or external financial action may have succeeded even if the worker did not record its success. I use transactions, unique operation IDs, deterministic vector IDs, transactional outboxes, and external idempotency keys where supported.
>
> Finally, I implement bounded retries, abandoned-workflow detection, concurrency-safe recovery, dead-letter or terminal failure handling, and observability around queue depth, retry counts, and stuck workflows. This allows the system to recover without losing completed work or blindly repeating dangerous actions."

The main takeaway: Celery can recover task execution, but your application must recover business progress. Durable state, idempotent operations, and safe resumption are what turn a task queue into a reliable enterprise workflow system.