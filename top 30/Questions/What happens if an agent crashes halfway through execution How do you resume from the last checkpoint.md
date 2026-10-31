# How to recover a LangGraph agent after it crashes halfway through execution

This is one of the most important production topics for a Senior/Staff AI Engineer interview.

In a production multi-agent system, an agent can crash because of:

- LLM provider timeout or outage.
- Vector database or Redis failure.
- Worker/container restart.
- Invalid LLM output.
- Python exception inside an agent.
- Network failure while calling an external API.
- Process termination after completing an operation but before saving its result.

The correct solution is durable checkpointing + controlled retries + idempotent execution + workflow recovery.

For your Financial Advisor Copilot, imagine this workflow:

1. Supervisor creates a plan.
2. Research agent retrieves financial documents.
3. Risk agent calculates portfolio risk.
4. Performance agent calculates returns.
5. Compliance agent checks policies.
6. Evaluator validates all agent outputs.
7. Report generator creates the final recommendation.

Suppose the Research and Risk agents complete successfully, but the Performance agent crashes.

We should not rerun the entire workflow unnecessarily. We want to recover from the latest durable checkpoint, preserve completed work, and retry the failed operation.

## 1. Understand how LangGraph checkpointing works

LangGraph provides a checkpointer that saves graph state as execution progresses.

The important concepts are:

| Concept      | Purpose                                       |
| ------------ | --------------------------------------------- |
| State        | Shared data passed between graph nodes        |
| Node         | A unit of work, such as a research agent      |
| Checkpoint   | Persisted snapshot of graph state             |
| Thread ID    | Identifies one workflow execution             |
| Checkpointer | Saves and retrieves checkpoints               |
| Resume       | Continues execution from saved workflow state |

A checkpoint is not simply a Python variable saved somewhere. It represents persisted graph execution state, including the graph's state and information needed to continue execution.

Example: Multi-agent execution

1. Supervisor

   Completed

   Checkpoint 1: execution plan saved
2. Research agent

   Completed

   Checkpoint 2: research result saved
3. Risk agent

   Completed

   Checkpoint 3: risk result saved
4. Performance agent

   Crashed

   The node did not complete successfully
5. Compliance agent

   Not started
6. Evaluator

   Not started

Recovery

Load the latest checkpoint

Retry the failed node

Continue the remaining workflow

Important distinction: LangGraph normally checkpoints at graph execution boundaries (supersteps), not after every Python statement inside a node. If a node crashes before its result is committed, that node may run again from its beginning. Completed work from prior committed supersteps can be retained. Parallel nodes in the same superstep require additional care because their results are committed together at the superstep boundary.

## 2. The most important recovery rule

There are two different failure situations, and you must handle them differently.

Scenario A: A node fails temporarily

For example, the Performance agent calls an LLM API and receives a timeout.

Use a LangGraph retry policy:

```
Performance agent
      |
      v
LLM API timeout
      |
      v
Retry after 1 second
      |
      v
Retry after 2 seconds
      |
      v
Retry after 4 seconds
      |
      v
Success or retries exhausted
```

Scenario B: The worker or container crashes

For example:

- Research completed.
- Risk completed.
- Performance started.
- The Kubernetes pod was terminated.

The application process is gone. The new worker must load the same graph checkpoint using the original `thread_id` and resume execution.

The key difference:

| Situation                             | Recovery mechanism                                 |
| ------------------------------------- | -------------------------------------------------- |
| Temporary node failure                | `RetryPolicy`                                      |
| Retry attempts exhausted              | Surface the error and inspect the checkpoint       |
| Process/container crash               | Resume from persisted checkpoint                   |
| Invalid agent output                  | Validate, retry or route to an error-handling node |
| External operation may have completed | Idempotency key and reconciliation                 |

LangGraph also supports pending writes, which help preserve successful node results when another node in the same parallel superstep fails.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.langchain.com\&sz=32)

Docs by LangChain



## 3. Production architecture

For the Financial Advisor Copilot, I would use this architecture:

FastAPI

Accept request and create workflow thread

LangGraph workflow

Supervisor → Research → Risk → Performance → Compliance

PostgreSQL checkpointer

Persist state and execution checkpoints

Redis / queue

Schedule recovery jobs

Worker

Execute and resume graph

Evaluator → Report generator

Continue after recovery

The database is the durable source for workflow execution state. Redis or a queue can coordinate jobs, but it should not replace the checkpointer.

## 4. Create the project structure

```
financial_advisor_copilot/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   ├── infrastructure/
│   │   └── checkpointer.py
│   │
│   ├── services/
│   │   ├── research_service.py
│   │   ├── risk_service.py
│   │   ├── performance_service.py
│   │   └── compliance_service.py
│   │
│   └── workers/
│       └── recovery_worker.py
│
├── requirements.txt
└── .env
```

Install:

```
pip install \
  langgraph \
  langgraph-checkpoint-postgres \
  fastapi \
  uvicorn \
  "psycopg[binary,pool]" \
  asyncpg \
  pydantic
```

## 5. Define the shared graph state

File: `app/graph/state.py`

```
from typing import TypedDict, Anyclass AdvisorState(TypedDict, total=False):    # Request information    request_id: str    tenant_id: str    user_id: str    question: str    portfolio_id: str    # Supervisor    execution_plan: list[str]    # Specialist outputs    research_result: dict[str, Any]    risk_result: dict[str, Any]    performance_result: dict[str, Any]    compliance_result: dict[str, Any]    # Workflow metadata    completed_agents: list[str]    failed_agent: str | None    retry_count: int    # Final output    evaluation_result: dict[str, Any]    final_report: str    # Error tracking    error: str | None    status: str
```

Why do we store these results in state?

Imagine Research completes successfully and returns 20 relevant financial document chunks.

If Performance subsequently fails, we do not want to retrieve and process those same documents again unnecessarily.

The research result is part of the persisted workflow state and can be consumed by subsequent nodes.

In a large production system, consider storing large document sets and intermediate artifacts in object storage, and keep only references and essential metadata in graph state.

## 6. Implement the specialist agents

File: `app/graph/nodes.py`

For this example, I am using service interfaces to keep agent orchestration separate from business logic.

```
from typing import Anyfrom app.graph.state import AdvisorStateclass TransientAgentError(Exception):    """Temporary failure that is safe to retry."""class PermanentAgentError(Exception):    """Failure that requires intervention."""async def supervisor_agent(    state: AdvisorState,) -> dict:    plan = [        "research",        "risk",        "performance",        "compliance",    ]    return {        "execution_plan": plan,        "completed_agents": [],        "retry_count": 0,        "status": "RUNNING",    }async def research_agent(    state: AdvisorState,) -> dict:
```

The `research_service`, `risk_service`, `performance_service`, and `compliance_service` are application dependencies that you would inject into these nodes. They represent your actual RAG, financial calculation, and compliance services.

Why separate the agents into individual nodes?

Suppose you put all four specialists into one node:

```
async def all_agents(state):    research = await run_research()    risk = await run_risk()    performance = await run_performance()    compliance = await run_compliance()    return {...}
```

If compliance fails, the entire node can execute again, repeating all four operations.

By splitting them into separate graph nodes, you improve recovery granularity, observability, testing, and retry control.

## 7. Configure automatic retries using RetryPolicy

LangGraph has a built-in retry policy. We should use it for failures that are likely to be temporary.

File: `app/graph/workflow.py`

```
from langgraph.graph import StateGraph, START, ENDfrom langgraph.types import RetryPolicyfrom app.graph.state import AdvisorStatefrom app.graph.nodes import (    TransientAgentError,    supervisor_agent,    research_agent,    risk_agent,    performance_agent,    compliance_agent,    evaluator_agent,    report_agent,)def build_workflow(checkpointer):    builder = StateGraph(AdvisorState)    # Supervisor    builder.add_node(        "supervisor",        supervisor_agent,    )    # Research agent    builder.add_node(        "research",        research_agent,        retry_policy=RetryPolicy(            max_attempts=3,            initial_interval=1.0,            backoff_factor=2.0,            max_interval=8.0,            retry_on=TransientAgentError,
```

Here is how the retry configuration works:

```
RetryPolicy(    max_attempts=3,    initial_interval=1.0,    backoff_factor=2.0,    max_interval=8.0,    retry_on=TransientAgentError,)
```

| Attempt | Wait before attempt |
| ------- | ------------------- |
| First   | Immediate           |
| Second  | 1 second            |
| Third   | 2 seconds           |

If all three attempts fail, LangGraph surfaces the exception. The workflow can then be recovered after the underlying problem is resolved.

The policy applies only to the specified transient exception. A permanent error should not be blindly retried.

Official reference: [LangGraph fault-tolerance and retry concepts](https://docs.langchain.com/oss/python/langgraph/fault-tolerance?utm_source=chatgpt.com).

## 8. Configure PostgreSQL checkpointing

This is the most important infrastructure component.

File: `app/infrastructure/checkpointer.py`

```
from langgraph.checkpoint.postgres.aio import (    AsyncPostgresSaver,)from app.graph.workflow import build_workflowclass GraphRuntime:    def __init__(self):        self._context = None        self.checkpointer = None        self.graph = None    async def start(self, database_url: str):        self._context = (            AsyncPostgresSaver.from_conn_string(                database_url            )        )        self.checkpointer = (            await self._context.__aenter__()        )        # Create checkpoint tables if necessary.        await self.checkpointer.setup()        # Compile graph with durable persistence.        self.graph = build_workflow(            self.checkpointer        )    async def stop(self):
```

Environment configuration:

```
DATABASE_URL=postgresql://postgres:password@localhost:5432/advisor
```

What happens internally?

```
LangGraph node executes
        |
        v
Node returns state update
        |
        v
LangGraph creates checkpoint
        |
        v
PostgreSQL persists checkpoint
        |
        v
Next graph step executes
```

The PostgreSQL checkpointer stores graph execution information, allowing the application to retrieve state after the original worker process disappears.

For production:

- Use a shared PostgreSQL database.
- Initialize checkpoint tables during application startup or a controlled migration/deployment step.
- Keep the checkpointer connection pool properly managed.
- Protect checkpoint data with appropriate access controls and retention policies.
- Do not use a new `thread_id` when recovering an existing execution.

## 9. Start the workflow

File: `app/main.py`

```
from uuid import uuid4from fastapi import FastAPI, HTTPExceptionfrom pydantic import BaseModelfrom app.infrastructure.checkpointer import GraphRuntimeapp = FastAPI()runtime = GraphRuntime()class AdvisorRequest(BaseModel):    tenant_id: str    user_id: str    portfolio_id: str    question: str@app.on_event("startup")async def startup():    await runtime.start(        "postgresql://postgres:password@localhost:5432/advisor"    )@app.on_event("shutdown")async def shutdown():    await runtime.stop()@app.post("/advisor")
```

For a production API, I would not execute a potentially long workflow directly in the HTTP request. The API should create a workflow record and enqueue a job. A background worker should execute the graph, while the API returns a request ID and a status endpoint.

Also, use FastAPI's lifespan mechanism for production resource management rather than the older startup/shutdown event decorators.

## 10. The most important part: Resume from the last checkpoint

Now let's simulate the actual production failure.

Suppose execution looks like this:

```
Supervisor          SUCCESS
Research            SUCCESS
Risk                 SUCCESS
Performance          FAILED
Compliance           NOT STARTED
Evaluator            NOT STARTED
Report               NOT STARTED
```

The Performance agent has exhausted its automatic retries because the external service is unavailable.

Your worker catches the exception and marks the workflow as `RECOVERY_REQUIRED` in your application database.

Once the service becomes healthy, a recovery worker resumes the same graph execution.

File: `app/workers/recovery_worker.py`

```
from app.infrastructure.checkpointer import runtimeasync def recover_workflow(request_id: str):    config = {        "configurable": {            "thread_id": request_id,        }    }    # 1. Load the most recent persisted graph state.    snapshot = await runtime.graph.aget_state(        config    )    if snapshot is None:        raise ValueError(            f"No checkpoint found for {request_id}"        )    # 2. Check whether there is unfinished work.    if not snapshot.next:        return {            "request_id": request_id,            "status": "NO_PENDING_WORK",            "state": snapshot.values,        }    # 3. Inspect where the graph will resume.    print("Current state:", snapshot.values)    print("Next nodes:", snapshot.next)    # 4. Resume the existing graph.
```

The crucial line is:

```
result = await runtime.graph.ainvoke(    None,    config=config,)
```

This tells LangGraph to continue from the checkpoint associated with the existing thread.

You are not sending the original request again. You are not creating a new graph thread. You are resuming the existing execution.

Why use `aget_state()` first?

It lets you inspect:

```
snapshot.valuessnapshot.nextsnapshot.configsnapshot.metadata
```

For example, you can see the persisted state and the next node or nodes scheduled for execution.

A checkpoint may also contain pending writes from successful nodes in a parallel superstep. LangGraph uses those writes to avoid rerunning successful siblings unnecessarily when a sibling fails.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.langchain.com\&sz=32)

Docs by LangChain



## 11. What happens during an actual recovery?

Let's walk through a concrete example.

First execution

1. Supervisor

   Checkpoint saved
2. Research

   Checkpoint saved

   Research result persisted
3. Risk

   Checkpoint saved

   Risk result persisted
4. Performance

   Failed after retries

Recovery worker starts

Load checkpoint using original thread ID

Resume pending Performance node

Performance succeeds

Compliance executes

Evaluator executes

Report generated

The expected result is that the Research and Risk results are reused, while Performance is retried and the remaining workflow continues.

One important nuance: if a node fails, its own uncommitted state update is not available as a completed node result. That node may execute from its beginning. This is why you should keep nodes reasonably small.

## 12. What if the crash happens inside the agent itself?

Consider this implementation:

```
async def performance_agent(state):    portfolio = await fetch_portfolio()    historical_returns = await fetch_returns()    benchmark = await fetch_benchmark()    result = await llm_analyze(        portfolio,        historical_returns,        benchmark,    )    return {        "performance_result": result    }
```

Suppose the LLM call fails.

The graph does not automatically save `portfolio` and `historical_returns` as separate graph checkpoints because they are local Python variables.

When the node retries or is resumed, it starts again from the beginning.

There are three solutions.

Solution A: Split the node into smaller nodes.

```
fetch_portfolio
       |
       v
fetch_historical_returns
       |
       v
fetch_benchmark
       |
       v
analyze_performance
```

Each successful node can produce a durable state update.

Solution B: Persist intermediate artifacts externally.

For example:

```
async def fetch_portfolio_node(state):    portfolio = await portfolio_service.fetch(        state["portfolio_id"]    )    artifact_id = await artifact_store.save(        key=f"{state['request_id']}/portfolio",        value=portfolio,    )    return {        "portfolio_artifact_id": artifact_id    }
```

Then the next node retrieves the artifact instead of fetching it again.

Solution C: Use an idempotent cache.

For expensive, deterministic operations, use a stable cache key:

```
cache_key = (    f"portfolio:{state['tenant_id']}:"    f"{state['portfolio_id']}:"    f"{portfolio_version}")
```

Only reuse the cached result when its input version and authorization scope match.

My preference is to split nodes at meaningful recovery boundaries and use external artifact storage for large intermediate results.

## 13. What if you have parallel agents?

Your financial advisor workflow may run Research, Risk, Performance, and Compliance concurrently.

For example:

```
                  Supervisor
                      |
          +-----------+-----------+
          |           |           |
       Research      Risk     Performance
          |           |           |
          +-----------+-----------+
                      |
                  Compliance
                      |
                   Evaluator
                      |
                    Report
```

LangGraph can execute nodes in parallel during the same superstep.

This introduces an important recovery consideration.

Suppose Research and Risk finish successfully, but Performance fails in the same superstep. LangGraph's checkpoint mechanism can preserve successful pending writes from the other nodes. On recovery, it can avoid rerunning those completed sibling nodes.

To accumulate parallel results correctly, use reducers in your state.

Example:

```
from typing import Annotated, TypedDictimport operatorclass ParallelAdvisorState(TypedDict, total=False):    request_id: str    tenant_id: str    portfolio_id: str    question: str    research_result: dict    risk_result: dict    performance_result: dict    # Each specialist appends its completion marker.    completed_agents: Annotated[        list[str],        operator.add,    ]
```

A specialist can return:

```
return {    "completed_agents": ["research"],    "research_result": result,}
```

The reducer combines updates rather than allowing one parallel node's list to overwrite another's.

Staff-level design point: Do not assume that every parallel agent has an independent durable checkpoint. The superstep and pending-write behavior matters. Use LangGraph's persistence model and test parallel failures explicitly.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.langchain.com\&sz=32)

Docs by LangChain



## 14. Prevent duplicate external operations

Checkpointing does not automatically make external side effects exactly-once.

This is one of the biggest mistakes engineers make.

Imagine your agent calls a financial execution API:

```
await trading_service.execute_trade(    portfolio_id=portfolio_id,    quantity=100,)
```

The following sequence is possible:

1. The trading API successfully executes the trade.
2. The worker crashes before LangGraph commits the node's result.
3. The graph resumes and runs the node again.
4. The trade may execute twice.

The solution is an idempotency key enforced by the receiving business service.

```
async def execute_trade_node(state):    request_id = state["request_id"]    idempotency_key = (        f"trade-execution:{request_id}:"        f"{state['recommendation_version']}"    )    result = await trading_service.execute_trade(        portfolio_id=state["portfolio_id"],        recommendation=state["approved_recommendation"],        idempotency_key=idempotency_key,    )    return {        "execution_result": result,        "status": "EXECUTED",    }
```

The trading service should persist the idempotency key and return the original result when it receives the same operation again.

For financial actions, I would also store the execution status independently in PostgreSQL and reconcile ambiguous outcomes before retrying.

## 15. Add workflow-level observability

When an incident happens, you need to answer:

- Which agent failed?
- What was its last successful checkpoint?
- How many retries happened?
- Which model and model version were used?
- How long did each node execute?
- Was the failure caused by the LLM, Qdrant, Redis, PostgreSQL, or application code?

I would track metrics such as:

| Metric                              | Purpose                        |
| ----------------------------------- | ------------------------------ |
| `workflow_duration_seconds`         | End-to-end execution time      |
| `agent_duration_seconds`            | Per-agent latency              |
| `agent_retry_total`                 | Number of retries              |
| `agent_failure_total`               | Agent failure frequency        |
| `workflow_recovery_total`           | Number of recovered workflows  |
| `checkpoint_write_duration_seconds` | Persistence latency            |
| `workflow_pending_total`            | Workflows waiting for recovery |

Every log should include:

```
{
  "request_id": "workflow-123",
  "tenant_id": "tenant-1",
  "agent": "performance",
  "attempt": 3,
  "error_type": "TransientAgentError",
  "status": "RECOVERY_REQUIRED"
}
```

Do not log API keys, raw secrets, or sensitive financial data.

## 16. Improve checkpoint durability for production

There is another subtle point: a successful node returning does not necessarily mean the checkpoint has been synchronously flushed to durable storage at that exact moment.

LangGraph supports checkpoint durability modes. When you need stronger guarantees that a checkpoint is persisted before the graph proceeds, use synchronous durability:

```
result = await runtime.graph.ainvoke(    initial_state,    config=config,    durability="sync",)
```

Use this when the cost of losing the most recent progress is greater than the additional checkpoint latency.

For ordinary workflows, choose the durability mode based on your recovery requirements and performance measurements. Do not assume that an external API operation and a LangGraph checkpoint are one atomic transaction.

See the [LangGraph persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence?utm_source=chatgpt.com).

## 17. Test crash recovery properly

I would write integration tests for at least these cases:

```
import pytestfrom langgraph.checkpoint.memory import InMemorySaver@pytest.mark.asyncioasync def test_resume_after_node_failure():    # Build the same graph, but inject a test    # performance service that fails on its first    # invocation and succeeds after recovery.    checkpointer = InMemorySaver()    graph = build_workflow(checkpointer)    config = {        "configurable": {            "thread_id": "recovery-test-001"        }    }    initial_state = {        "request_id": "recovery-test-001",        "tenant_id": "tenant-1",        "user_id": "user-1",        "portfolio_id": "portfolio-1",        "question": "Analyze my portfolio",        "status": "STARTED",    }    # First execution: configure the test service    # to keep failing until retry attempts are exhausted.    with pytest.raises(Exception):        await graph.ainvoke(            initial_state,
```

This is an integration-test outline: `performance_service.set_available()` is a test-double operation you would implement in your test fixture. Use PostgreSQL, rather than `InMemorySaver`, in a separate integration test that verifies recovery across a real process restart.

## 18. How I would deploy recovery in AWS/Kubernetes

For a production deployment, I would separate API execution from graph workers.

API Gateway / Load Balancer

FastAPI on ECS / EKS

Creates workflow record and queue event

SQS / Celery queue

Execution and recovery jobs

LangGraph workers

Run graph · retry · resume

RDS PostgreSQL

Checkpoints and workflow records

Redis

Cache and coordination

The recovery process should work as follows:

1. Worker receives a workflow job.
2. Worker runs LangGraph using its existing `thread_id`.
3. If the worker crashes, the queue makes the job eligible for redelivery after its visibility timeout or lease expires.
4. A new worker loads the graph checkpoint.
5. It checks the execution state and resumes using `ainvoke(None, config)`.
6. It records completion or schedules another recovery attempt.

For this to work reliably, I would add:

- A database workflow record with `PENDING`, `RUNNING`, `RECOVERY_REQUIRED`, `COMPLETED`, and `FAILED` states.
- Worker leases and heartbeat monitoring.
- A transactional outbox for publishing execution and recovery jobs.
- A maximum recovery-attempt policy.
- A dead-letter queue for workflows that exceed the retry limit.
- Idempotent business operations.
- A reconciliation job for workflows left in `RUNNING` after worker termination.

Do not allow two workers to recover the same workflow simultaneously. Use a database lease or atomic claim operation around the recovery job.

## 19. Common interview mistakes

| Incorrect approach                                | Correct approach                         |
| ------------------------------------------------- | ---------------------------------------- |
| Restart the whole workflow                        | Resume using the original thread         |
| Use a new thread ID                               | Reuse the original `thread_id`           |
| Store state only in Python variables              | Use a durable checkpointer               |
| Retry every error                                 | Retry only appropriate transient errors  |
| Put multiple expensive operations in one node     | Split at meaningful recovery boundaries  |
| Assume a successful API call is exactly-once      | Use idempotency keys                     |
| Use Redis as the only checkpoint store            | Use a durable graph checkpointer         |
| Assume every parallel node has its own checkpoint | Understand supersteps and pending writes |
| Ignore failed workflows indefinitely              | Use recovery scheduling and a DLQ        |

## 20. Senior/Staff AI Engineer interview answer

If an interviewer asks:

"What happens if an agent crashes halfway through execution? How do you resume from the last checkpoint?"

I would answer:

> I design the LangGraph workflow with durable checkpointing using PostgreSQL.
>
> Each specialist agent is represented as a separate graph node. This gives us clear recovery boundaries and allows successful node outputs to be persisted in graph state.
>
> I configure retry policies for transient failures such as LLM timeouts, rate limits, and temporary vector database outages.
>
> If retries are exhausted or the worker crashes, I inspect the latest checkpoint using `aget_state()` with the original thread ID.
>
> After the underlying issue is resolved, I resume the workflow using `ainvoke(None, config)`. LangGraph restores the persisted state and continues the unfinished execution rather than starting a new workflow.
>
> I also account for parallel execution, where successful pending writes from completed sibling nodes can be preserved.
>
> For external side effects, I use idempotency keys because checkpointing alone cannot guarantee exactly-once execution.
>
> Finally, I integrate recovery with a background worker, transactional outbox, worker leases, dead-letter queue, and OpenTelemetry monitoring.
>
> This gives us a workflow that can survive application crashes, worker restarts, and transient infrastructure failures without unnecessarily repeating expensive AI operations.

### The five things I would remember

1

PostgreSQL checkpointer — durable workflow state.

2

RetryPolicy — automatic transient failure recovery.

3

`aget_state()` — inspect the last persisted state.

4

`ainvoke(None, config)` — resume existing execution.

5

Idempotency — prevent duplicate external operations.

One important distinction to master for interviews: LangGraph checkpointing gives you durable workflow recovery, but it does not make every operation inside an agent transactional. You need both workflow-level recovery and application-level idempotency.

Official reading: [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence?utm_source=chatgpt.com) · [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance?utm_source=chatgpt.com) · [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.langchain.com\&sz=32)

Docs by LangChain+1
