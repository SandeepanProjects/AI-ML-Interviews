# How would you design an AI application capable of handling millions of requests?

This is a classic Senior/Staff AI Engineer system-design interview question.

The interviewer is not simply asking how to deploy an LLM behind FastAPI. They want to understand how you would design a distributed AI platform that can handle high traffic, unpredictable LLM latency, provider failures, expensive inference, concurrent users, and large-scale data retrieval without compromising reliability or security.

My approach would be to design the system around six principles:

1. Horizontally scalable stateless API services.
2. Asynchronous processing for long-running AI workflows.
3. Independent scaling of retrieval, inference, and orchestration.
4. Caching, batching, and model routing to control cost and latency.
5. Durable workflow execution with retries and recovery.
6. Strong observability, tenant isolation, and failure containment.

I'll use an enterprise RAG and multi-agent application as the example, similar to your Financial Advisor Copilot.

## 1. First, translate "millions of requests" into engineering requirements

Never start with Kubernetes or a particular LLM. First clarify what the traffic means.

For example, assume:

| Metric               | Example requirement                          |
| -------------------- | -------------------------------------------- |
| Registered users     | 10 million                                   |
| Daily active users   | 1 million                                    |
| Requests per day     | 10 million                                   |
| Average request rate | 116 requests/sec                             |
| Peak request rate    | 1,000 requests/sec                           |
| Target API latency   | p95 below 300 ms for admission               |
| AI response latency  | p95 below 5 seconds for interactive requests |
| Availability target  | 99.9%                                        |
| Data isolation       | Strict tenant-level isolation                |

These are illustrative design assumptions, not universal targets.

Important distinction: 10 million requests per day does not mean 10 million simultaneous LLM requests.

Average request rate:

\\[ \frac{10,000,000}{86,400} \approx 116 \text{ requests/sec} \\]

If peak traffic is 8–10 times average traffic, we should provision for approximately 1,000–1,200 incoming requests per second.

Also distinguish:

- API requests
- LLM calls
- Embedding calls
- Retrieval queries
- Tool executions
- Agent workflow executions

One user request may trigger multiple downstream operations. This amplification is one of the biggest challenges in scaling AI systems.

## 2. High-level production architecture

I would use a distributed architecture where every major component can scale independently.

[GitHub - FareedKhan-dev/scalable-rag-pipeline: A scalable RAG platform combining LangGraph agents, hybrid retrieval (Vector+Graph), and Ray orchestration on Kubernetes. · GitHub](https://images.openai.com/static-rsc-4/PJW01ne51Ziz9yKIXqRJHHSHBNQfBvgthqB5qcjvujT7Xmq8LqkOO59NdAcpqjAOzRmzXXZfB7PmnlURNJYfMdvlEqkSbiW8NNzItvPodKknswI9eGspexbjK9cwrQghpd0lOBv6lxELzN_kZR_hpV9ruJvtPtVLAI2w3HY9ZKg?purpose=inline)

[github.com](https://github.com/FareedKhan-dev/scalable-rag-pipeline?utm_source=chatgpt.com)

The architecture:

```
                   CLIENT APPLICATIONS
                           |
                           v
                    CDN / WAF
                           |
                           v
                   API GATEWAY
                Rate Limit / Auth
                           |
                           v
                 LOAD BALANCER
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
          FastAPI       FastAPI       FastAPI
          Instance      Instance      Instance
             |             |             |
             +-------------+-------------+
                           |
                           v
                   REQUEST ROUTER
                           |
                +----------+----------+
                |                     |
                v                     v
           Redis Cache          PostgreSQL
                |                     |
                +----------+----------+
                           |
                           v
                    JOB QUEUE
                 Kafka / SQS / RabbitMQ
                           |
                           v
                   AI WORKFLOW WORKERS
                           |
              +------------+------------+
              |            |            |
              v            v            v
          Retrieval     Agent         Model
          Service       Service       Router
              |            |            |
              v            v            v
           Qdrant       LangGraph     LLM Providers
              |            |            |
              +------------+------------+
                           |
                           v
                  RESPONSE / ARTIFACT
                           |
                           v
                     CLIENT / SSE
```

The critical architectural decision is that API servers should not be responsible for running every long-running AI workflow themselves.

They should authenticate, validate, enforce quotas, and submit work. Dedicated workers should execute the expensive operations.

## 3. Design each layer for horizontal scaling

### Layer 1: API gateway and stateless FastAPI services

The API layer receives requests and performs lightweight operations:

- Authentication and JWT validation.
- Tenant identification.
- Request schema validation.
- Rate limiting and quota checks.
- Idempotency validation.
- Request admission.
- Job submission.

I would keep API instances stateless so that I can add or remove instances without migrating user sessions.

For example:

```
                 Load Balancer
                       |
          +------------+------------+
          |            |            |
          v            v            v
       API Pod 1    API Pod 2    API Pod 3
          |            |            |
          +------------+------------+
                       |
                 Shared Services
```

If traffic increases, Kubernetes Horizontal Pod Autoscaler can add more API replicas based on CPU, memory, or custom metrics such as requests per second and request latency.

FastAPI example:

```
from uuid import uuid4from fastapi import FastAPI, Depends, HTTPExceptionfrom pydantic import BaseModel, Fieldapp = FastAPI()class AIRequest(BaseModel):    question: str = Field(min_length=1, max_length=5000)    portfolio_id: strclass JobResponse(BaseModel):    request_id: str    status: str@app.post("/v1/analysis", response_model=JobResponse)async def create_analysis(    payload: AIRequest,    user=Depends(authenticate_user),):    await authorize_portfolio(        user_id=user.id,        tenant_id=user.tenant_id,        portfolio_id=payload.portfolio_id,    )    request_id = str(uuid4())    await job_service.submit(        request_id=request_id,        tenant_id=user.tenant_id,        user_id=user.id,        payload=payload.model_dump(),
```

Here, `authenticate_user`, `authorize_portfolio`, and `job_service` are application dependencies that must be implemented and injected.

Notice that the endpoint does not call the LLM directly.

It submits a job and returns a request ID. The client can poll for status or subscribe to updates through Server-Sent Events (SSE).

For idempotency, I would also accept an `Idempotency-Key` header and persist its association with the request so that a client retry does not create duplicate jobs.

### Layer 2: Message queue and backpressure

This is one of the most important components in a large-scale AI application.

Imagine:

- Your system normally receives 200 requests per second.
- Suddenly, 2,000 requests per second arrive.
- Your LLM provider can only handle 500 concurrent requests.

If you send all requests directly to the LLM, you will experience:

- Provider rate-limit errors.
- Connection exhaustion.
- Increased latency.
- Memory pressure.
- Cascading failures.

I would introduce a durable queue between API admission and AI execution.

[Introduction to the Apache Kafka messaging system | Redpanda](https://images.openai.com/static-rsc-4/Q33J-It4zSnkThCkEa7C77pI4lZ1dLnNAFcDjeYSeqTnYQRW14zBjap6qTEjJ_TdGFW8vm99RGBQIaqMqDf01uBk_xjEu5PMlPoK1ZWD0fOYfN3uhfaT-j6_IB-SuZbUcjnRjSKEHpCW3BHbfhyyfEJ3B92lRAYyqv_pa1EPIXQ?purpose=inline)

[redpanda.com](https://www.redpanda.com/guides/kafka-use-cases-messaging-system?utm_source=chatgpt.com)

Example:

```
FastAPI Producers
       |
       v
   Job Queue
       |
       +---------------------+
       |                     |
       v                     v
  Worker Group A        Worker Group B
       |                     |
       v                     v
  RAG Workflows         Agent Workflows
       |                     |
       +----------+----------+
                  |
                  v
              LLM Router
```

Possible queue technologies:

| Technology    | When I would use it                                            |
| ------------- | -------------------------------------------------------------- |
| AWS SQS       | Managed background jobs, simple decoupling                     |
| Kafka         | High-throughput event streaming, replay and multiple consumers |
| RabbitMQ      | Flexible task routing and traditional job queues               |
| Redis Streams | Lightweight stream processing where Redis is already central   |

For your AWS-based architecture, SQS is a reasonable default for independent workflow jobs. Kafka becomes more attractive when you need event replay, multiple independent consumer groups, and high-volume event streaming.

#### Worker scaling

Workers should scale according to queue backlog and processing capacity, not only CPU.

For example:

```
Queue backlog = 10,000 jobs

Worker 1 -> 10 concurrent workflows
Worker 2 -> 10 concurrent workflows
Worker 3 -> 10 concurrent workflows
Worker 4 -> 10 concurrent workflows
```

If the backlog continues increasing, add workers, subject to the downstream LLM provider's concurrency and rate limits.

A useful metric is:

\\[ \text{Queue Wait Time} = \text{Job Start Time} - \text{Job Enqueue Time} \\]

Also monitor:

- Queue depth.
- Oldest message age.
- Job processing duration.
- Retry count.
- Dead-letter queue size.
- Worker saturation.

Important: A queue does not create more LLM capacity. It absorbs bursts and controls how quickly work reaches downstream services. If arrival rate remains above processing capacity, backlog and wait time will continue to grow. You need admission control, quotas, or additional inference capacity.

### Layer 3: AI workflow workers

I would separate the API service from the AI execution service.

For example:

```
app/
├── api/
│   ├── routes/
│   └── dependencies.py
│
├── application/
│   ├── job_service.py
│   └── workflow_service.py
│
├── workers/
│   ├── queue_consumer.py
│   └── workflow_worker.py
│
├── ai/
│   ├── orchestrator/
│   ├── agents/
│   ├── rag/
│   ├── embeddings/
│   └── model_router/
│
├── infrastructure/
│   ├── postgres/
│   ├── redis/
│   ├── qdrant/
│   └── messaging/
│
└── observability/
    ├── metrics.py
    └── tracing.py
```

The worker consumes a job, loads the relevant workflow state, executes the graph, persists results, and acknowledges the message only after durable completion.

A simplified worker:

```
async def process_job(message):    job = await job_repository.get(message.request_id)    if job.status == "COMPLETED":        return    await job_repository.mark_running(job.id)    try:        result = await workflow.ainvoke(            job.payload,            config={                "configurable": {                    "thread_id": job.id                }            },        )        await job_repository.mark_completed(            job_id=job.id,            result=result,        )    except TransientAIError:        # Let the queue's retry mechanism handle this.        raise    except Exception as exc:        await job_repository.mark_failed(            job_id=job.id,            error_code=classify_error(exc),        )        raise
```

This is a simplified execution example. A production implementation also needs:

- A durable LangGraph checkpointer.
- Worker lease or visibility-timeout management.
- Atomic job state transitions.
- Idempotent result persistence.
- Retry classification.
- Dead-letter handling.
- Recovery for jobs whose workers disappear.

One subtle issue: queue delivery is commonly at-least-once. A worker may execute a job and crash before acknowledging the message. Therefore, duplicate delivery must be expected.

I would use a unique job ID, a database uniqueness constraint, atomic claims or leases, and idempotent workflow operations.

## 4. Scale the LLM inference layer independently

This is where AI applications differ from conventional web applications.

A traditional API may execute a database query in milliseconds. An LLM request may take several seconds, consume substantial tokens, and be constrained by provider quotas.

I would introduce a centralized LLM Gateway / Model Router.

[LLM API Gateway Architecture Diagram for Multi-Provider Routing and Failover | flatkey.ai](https://images.openai.com/static-rsc-4/NK0Y7NzfByHSYsrlHlK9R_-YWfFYhNNiSKWu5KfW8BaubyooCis4xzAjsKMC09b_Qk80dPv83ofQvGQX3H3bTHTC7IM-WxXbRVGp4jxhuaz2ILQO01wSgXKGo8281amRjXi5HNwz0qcnZ66dIRue2iXAFobFd68pekM3o_q8gr8?purpose=inline)

[flatkey.ai](https://flatkey.ai/blog/llm-api-gateway-architecture?utm_source=chatgpt.com)

Example:

```
             AI Workflow Workers
                      |
                      v
                LLM Gateway
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
        Model A     Model B     Model C
        Provider    Provider    Self-hosted
          |           |           |
          +-----------+-----------+
                      |
                      v
                  Response
```

The model router should decide which model to use based on:

- Task complexity.
- Required output quality.
- Context window.
- Latency requirements.
- Provider availability.
- Cost per token.
- Rate limits.
- Data residency and security requirements.

For example:

| Task                       | Model strategy                    |
| -------------------------- | --------------------------------- |
| Intent classification      | Small, low-latency model          |
| Query rewriting            | Small model                       |
| Document summarization     | Cost-efficient model              |
| Complex financial analysis | Higher-capability model           |
| Structured extraction      | Model with reliable schema output |
| Embedding generation       | Dedicated embedding model         |

### Example: model routing

```
class ModelRouter:    def __init__(self, providers):        self.providers = providers    async def generate(self, task, prompt):        if task == "classification":            provider = self.providers["small"]        elif task == "financial_analysis":            provider = self.providers["advanced"]        else:            provider = self.providers["default"]        return await provider.generate(            prompt=prompt,        )
```

This is only the routing decision. A production router also needs provider-specific concurrency limits, token budgets, timeouts, circuit breakers, retries, usage tracking, and fallback policies.

Critical principle: Never allow every worker to independently retry an overloaded LLM provider. That can amplify the outage.

Use centralized concurrency controls and retry budgets. Retry transient failures with bounded exponential backoff and jitter. Do not retry permanent errors such as invalid requests or unauthorized access.

If a provider is unavailable, use a compatible fallback only if the fallback meets the task's quality, privacy, and output-contract requirements.

## 5. Scale the RAG retrieval layer

For an enterprise AI application, vector retrieval can become another major bottleneck.

Suppose you have:

- 10 million users.
- 100 million document chunks.
- Thousands of retrieval requests per second.
- Multiple tenants.
- Frequent document updates.

I would separate ingestion from online retrieval.

[Enterprise RAG: A Production Guide from Architecture to Multi-Tenant Security | by Pallavi Chandrashekar | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/_FC6cDDEyDiUjq5ihnqYv8MsqR2QaEvACNh-H03duVI0Wyz_vqfHQ_2Hgo6XaJBpOn5rLQlff8kEfTZMANVuiUYmtPksGHJyZUzZazHS3XeDycI51HR3eyhq3m1oNQ4wZeh6dluF1H1UlGKri03o2tRUtLkaB-9bUNkrbLwRNqI?purpose=inline)

[medium.com](https://medium.com/%40pallavi9964/enterprise-rag-a-production-guide-from-architecture-to-multi-tenant-security-f415c47ad36c?utm_source=chatgpt.com)

### A. Offline ingestion pipeline

```
Documents
    |
    v
Object Storage (S3)
    |
    v
Ingestion Queue
    |
    v
Document Parser
    |
    v
Chunking Service
    |
    v
Embedding Workers
    |
    v
Qdrant / Vector Database
```

This should run independently of user-facing query processing.

For high-volume ingestion:

- Batch embedding requests.
- Use asynchronous embedding workers.
- Deduplicate documents using content hashes.
- Track document versions.
- Make embedding writes idempotent.
- Use incremental updates instead of rebuilding everything.
- Delete or invalidate obsolete chunks safely.

### B. Online retrieval pipeline

```
User Query
    |
    v
Query Validation
    |
    v
Cache Lookup
    |
    v
Query Embedding
    |
    v
Vector Search + Keyword Search
    |
    v
Hybrid Result Fusion
    |
    v
Reranking
    |
    v
Context Construction
    |
    v
LLM
```

For large-scale retrieval, I would optimize:

1. Vector database indexing and filtering.
2. Query embedding latency.
3. Hybrid search efficiency.
4. Reranking candidate count.
5. Context token count.
6. Connection pooling.
7. Tenant and permission filters.

For Qdrant, I would evaluate:

- Sharding.
- Replication.
- Payload indexes for frequently filtered metadata.
- Collection design.
- Quantization.
- Query limits and timeouts.
- Snapshot and recovery strategy.

Do not create a separate vector collection for every tenant by default. The right design depends on tenant count, data volume, isolation requirements, and workload patterns. Shared collections with strict tenant filters, dedicated collections for larger tenants, or separate clusters for high-isolation tenants are all possible.

Security requirement: Tenant and document authorization filters must be derived from trusted server-side identity, never from LLM-generated parameters.

## 6. Use caching to reduce LLM calls and retrieval latency

At millions of requests, calling an LLM for every request is expensive.

I would introduce multiple caching layers.

[Semantic Caching: The Missing Layer in High-Performance LLM Systems](https://images.openai.com/static-rsc-4/rUwtJOxAfc4b7qX-G4spJ5lREpsaH9aTv15v0-YMcFGjHX2MCtk8xCdvBAOqhcQrP3j49YoNQEePKgG7-U-A9wTghBwp9lUjqSDiQ_IZGamggIRCwZE2-yF0BpxHaOc_2jxfDn8MJKZessPRfQnA1LUCum_P2uI4tablRRgnhs8?purpose=inline)

[linkedin.com](https://www.linkedin.com/pulse/semantic-caching-missing-layer-high-performance-llm-systems-sharma-jurzc?utm_source=chatgpt.com)

| Cache                | What to cache                                 | Technology                |
| -------------------- | --------------------------------------------- | ------------------------- |
| API response cache   | Identical safe responses                      | Redis                     |
| Semantic cache       | Equivalent questions and approved answers     | Redis + vector similarity |
| Retrieval cache      | Retrieved document IDs and versions           | Redis                     |
| Embedding cache      | Repeated query embeddings                     | Redis                     |
| Metadata cache       | Frequently accessed tenant/configuration data | Redis                     |
| Model response cache | Deterministic reusable outputs                | Redis                     |

### Example: semantic caching

Consider these questions:

```
Question 1:
What is the expense ratio of Fund A?

Question 2:
How much does Fund A charge as an expense ratio?
```

The wording is different, but the intent may be equivalent.

A semantic cache can identify similar queries and reuse a previously generated result.

However, I would not blindly return cached answers.

A safe cache key must consider:

```
tenant_id
+ user permission scope
+ normalized query
+ model version
+ prompt version
+ document/index version
+ relevant business context
```

For financial applications, stale answers can be dangerous. Portfolio balances and market prices should generally have short freshness windows or require live reads.

Also, semantic cache hits should only be allowed when the cached answer is valid for the requesting user's authorization scope.

### Example: Redis cache

```
import jsonfrom redis.asyncio import Redisclass ResponseCache:    def __init__(self, redis: Redis):        self.redis = redis    async def get(self, key: str):        value = await self.redis.get(key)        if value is None:            return None        return json.loads(value)    async def set(        self,        key: str,        value: dict,        ttl: int = 300,    ):        await self.redis.set(            key,            json.dumps(value),            ex=ttl,        )
```

This is a basic cache adapter. Production code should also enforce payload-size limits, serialization compatibility, cache versioning, and failure handling.

A cache outage should normally degrade performance, not make the entire AI application unavailable.

## 7. Scale PostgreSQL and Qdrant independently

One common mistake is to scale the FastAPI application while ignoring database bottlenecks.

At high traffic, you must manage database connections, read/write throughput, indexes, and query patterns.

### PostgreSQL architecture

```
             Application Services
                      |
                      v
               Connection Pool
                      |
             +--------+--------+
             |                 |
             v                 v
        Primary DB         Read Replicas
        (Writes)            (Reads)
```

I would use:

- Connection pooling with bounded pool sizes.
- Proper indexes.
- Read replicas for suitable read-heavy workloads.
- Partitioning for large append-heavy tables.
- PgBouncer where appropriate.
- Database query timeouts.
- Separate OLTP workloads from analytics.
- Automated backups and point-in-time recovery.

For example, the following tables should not necessarily share the same operational workload:

```
users
tenants
portfolios
documents
workflow_runs
workflow_checkpoints
tool_executions
audit_events
```

Large audit-event or workflow-event tables may need partitioning and separate archival policies.

Do not increase database connection counts without considering database capacity. If 100 worker pods each open 100 connections, you may create 10,000 connections and overwhelm PostgreSQL.

### Qdrant architecture

For vector search:

```
                Retrieval Service
                       |
                       v
                 Qdrant Cluster
                       |
          +------------+------------+
          |            |            |
          v            v            v
        Shard 1      Shard 2      Shard 3
          |            |            |
          +------------+------------+
                       |
                 Replicated Data
```

I would benchmark real query patterns before deciding on shard count, replication factor, indexing parameters, and collection layout.

The objective is not simply to store more vectors. It is to maintain predictable retrieval latency under concurrent reads and ingestion.

## 8. Prevent cascading failures

A million-request system must be designed to survive partial failures.

Imagine:

- OpenAI becomes unavailable.
- Qdrant starts responding slowly.
- Redis becomes unavailable.
- PostgreSQL connections are exhausted.
- A worker crashes halfway through a LangGraph workflow.

I would implement failure isolation at every service boundary.

| Failure                   | Production strategy                                         |
| ------------------------- | ----------------------------------------------------------- |
| LLM provider outage       | Circuit breaker, compatible fallback, bounded retry         |
| Qdrant unavailable        | Controlled degradation or approved alternate retrieval path |
| Redis unavailable         | Cache bypass where safe, bounded fallback                   |
| PostgreSQL unavailable    | Reject or defer operations requiring durable state          |
| Queue backlog increases   | Backpressure, admission control, worker scaling             |
| Worker crashes            | Queue redelivery and checkpoint recovery                    |
| Embedding provider outage | Pause or defer ingestion jobs                               |
| Invalid LLM output        | Schema validation, bounded repair or retry                  |
| Repeated agent loops      | Step limits, tool budgets and recursion limits              |

### Circuit breaker pattern

The circuit breaker prevents repeatedly calling a service that is already failing.

It has three states:

CLOSED

Normal operation

Requests are allowed through.

OPEN

Service failing

Requests fail fast; the unhealthy service is protected.

HALF-OPEN

Recovery testing

A limited number of probe requests are allowed.

Successful probes close the circuit; failed probes reopen it.

For distributed deployments, circuit-breaker state and concurrency controls need careful scoping. A per-process breaker is often sufficient for local protection, but it does not provide a global provider quota across all pods.

## 9. Control concurrency, rate limits, and token consumption

This is especially important for AI applications.

Suppose you have 1,000 worker processes, each allowed to make 20 concurrent LLM calls.

Your theoretical concurrency could reach:

\\[ 1,000 \times 20 = 20,000 \\]

That might be far above your provider's quota.

I would enforce several layers of limits:

| Limit                       | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| Per-user rate limit         | Prevent one user from dominating traffic |
| Per-tenant quota            | Control tenant resource consumption      |
| Global provider concurrency | Protect provider limits                  |
| Requests per minute         | Respect provider request quotas          |
| Tokens per minute           | Respect token throughput limits          |
| Per-workflow tool budget    | Prevent runaway agents                   |
| Maximum workflow duration   | Prevent stuck workflows                  |

For example, a tenant could have:

```
Maximum concurrent workflows: 20
Maximum requests/minute: 500
Maximum tokens/day: 5 million
Maximum tool calls/workflow: 15
Maximum workflow duration: 120 seconds
```

These are example limits, not fixed production defaults.

I would use Redis atomic operations or a dedicated quota service for distributed rate limiting. Token consumption should be reconciled against actual provider usage rather than relying only on estimated prompt size.

For expensive workflows, I would also implement tenant-level fair scheduling so that one high-volume customer cannot starve everyone else.

## 10. Make AI workflows resumable and idempotent

In a large distributed system, workers will crash. Messages will be redelivered. LLM providers will time out.

You cannot assume that every workflow executes exactly once.

For a multi-agent system, I would use LangGraph with a durable PostgreSQL checkpointer.

```
             Workflow Worker
                    |
                    v
              LangGraph Graph
                    |
           +--------+--------+
           |        |        |
           v        v        v
        Research   Risk   Compliance
           |        |        |
           +--------+--------+
                    |
                    v
                 Evaluator
                    |
                    v
               Final Report
                    |
                    v
              PostgreSQL
             Checkpoint Store
```

Each meaningful stage should have a recovery boundary.

For example:

```
async def execute_workflow(    graph,    request_id: str,    payload: dict,):    config = {        "configurable": {            "thread_id": request_id,        },    }    snapshot = await graph.aget_state(config)    if snapshot.values:        # Resume existing execution.        return await graph.ainvoke(            None,            config=config,        )    # Start a new execution.    return await graph.ainvoke(        payload,        config=config,    )
```

This example assumes that the graph is configured with a durable checkpointer and that the request ID is stable.

For a production worker, the database job record must also distinguish a new execution from an already completed execution. Do not start a new graph merely because a checkpoint lookup returns an empty state without first validating job ownership and status.

### Important: idempotency

Suppose a financial agent executes a trade, the trade succeeds, but the worker crashes before saving the result.

When the queue redelivers the job, the worker may execute the trade again.

I would use:

```
workflow_id + action_id = idempotency_key
```

And maintain a tool execution ledger:

```
tool_executions
------------------------------
id
tenant_id
workflow_id
action_id
idempotency_key
request_hash
status
result
created_at
updated_at
```

Use a unique constraint on the appropriate tenant-scoped idempotency key.

Before retrying an external financial operation, check whether it already succeeded. If the external system does not provide idempotency or status reconciliation, exactly-once execution cannot be guaranteed simply by using LangGraph checkpoints.

## 11. Observability: measure the entire AI request lifecycle

At scale, logs alone are not enough.

I would implement distributed tracing and metrics using OpenTelemetry.

[LLM Observability in Production: Tracing & Evaluation](https://images.openai.com/static-rsc-4/eumc74Zv12Wf_nc1RlcPU9dukVY6gW5NKmYvKzkc_Qw0B-uDDgCzo3MZlFRwyvuUpq6yvJUyE2GTvm7lc0qwl3Dxp-9yRGu64hkCdI_keuB4qryMm-fWFyst5z0oIUUfFxPMfsfGa4k9_fyvst2_RN4VtOqfAtLMFyuNrHEYvrM?purpose=inline)

[cloudrps.com](https://cloudrps.com/blog/llm-observability-production-tracing/?utm_source=chatgpt.com)

For one request, the trace could look like:

```
request_id: req-123

API Admission                20 ms
Queue Wait                  100 ms
Workflow Initialization     10 ms
Query Embedding             80 ms
Qdrant Retrieval            60 ms
Reranking                  120 ms
LLM Generation            1800 ms
Response Validation         20 ms
Persistence                 30 ms
----------------------------------
Total                    2240 ms
```

This makes it possible to identify whether the bottleneck is the API, queue, retrieval, model provider, or database.

### Metrics I would monitor

| Category    | Metrics                                                         |
| ----------- | --------------------------------------------------------------- |
| API         | RPS, p50/p95/p99 latency, HTTP errors                           |
| Queue       | Backlog, wait time, redelivery, DLQ                             |
| LLM         | Time to first token, total latency, tokens, errors              |
| RAG         | Embedding latency, retrieval latency, reranking latency         |
| Database    | Connections, query latency, lock waits                          |
| Agents      | Workflow duration, steps, retries, failure rate                 |
| Cost        | Cost per request, cost per tenant, cost per successful workflow |
| Reliability | Availability, recovery rate, circuit-breaker state              |

For tracing, I would propagate:

```
request_id
trace_id
workflow_id
tenant_id
agent_name
model_name
provider_name
tool_name
```

Avoid logging sensitive prompts, financial records, access tokens, or raw documents. Apply appropriate redaction and retention policies.

## 12. Deploy on Kubernetes and AWS

For an enterprise application, I would deploy independently scalable services on AWS using EKS or ECS, depending on operational complexity and workload requirements.

A possible deployment:

[Scaling EKS Architecture for 100k RPS with AWS Components | Akshay Shinde posted on the topic | LinkedIn](https://images.openai.com/static-rsc-4/xYBd-Gg351_eLJAO5czaTM6o1kmcqC6MkVKYJ-l82zJLtff2BzWcdFESLaihESP6X-ETfCk8G3P1r6sz5KuQzCl0ixY05i1jekJyEYQ39GHFIcZQmkTTDoIN6UZefp3wzHoLz_afCaE_GZcTNkoatxbvY3tQwx1iGudF6Ak9WYI?purpose=inline)

[linkedin.com](https://www.linkedin.com/posts/akshay-shinde99_aws-eks-cloudarchitecture-activity-7443524794081345537-kLSd?utm_source=chatgpt.com)

| Component           | AWS technology                                   |
| ------------------- | ------------------------------------------------ |
| API Gateway         | Amazon API Gateway                               |
| Load balancing      | Application Load Balancer                        |
| Compute             | EKS                                              |
| Job queue           | SQS or MSK                                       |
| Relational database | RDS PostgreSQL / Aurora PostgreSQL               |
| Cache               | ElastiCache Redis                                |
| Document storage    | S3                                               |
| Vector database     | Qdrant on EKS or managed vector service          |
| Secrets             | AWS Secrets Manager                              |
| Monitoring          | CloudWatch + OpenTelemetry                       |
| Tracing             | AWS X-Ray or an OpenTelemetry-compatible backend |
| Infrastructure      | Terraform                                        |
| Container registry  | ECR                                              |

### Kubernetes scaling

I would configure separate deployments for:

```
api-service
rag-retrieval-service
embedding-service
agent-worker-service
document-ingestion-service
llm-gateway-service
```

Each service has independent resource requests, limits, and scaling policies.

For example:

| Service       | Scaling signal                                |
| ------------- | --------------------------------------------- |
| API           | CPU, RPS, p95 latency                         |
| Retrieval     | Request concurrency, CPU, latency             |
| Embedding     | Queue backlog, GPU utilization if self-hosted |
| Agent workers | Queue depth, oldest job age                   |
| LLM gateway   | In-flight calls, provider quotas              |
| Ingestion     | Pending documents, processing lag             |

For worker scaling, I would consider KEDA because it supports event-driven scaling using queue metrics.

I would also configure:

- Multiple availability zones.
- Pod disruption budgets.
- Readiness and liveness probes.
- Graceful shutdown.
- Rolling deployments.
- Resource quotas.
- Autoscaling limits.
- Network policies.
- Secret rotation.
- Disaster recovery and backups.

Do not scale workers indefinitely. Worker autoscaling must respect provider quotas, database capacity, and downstream service limits.

## 13. Capacity planning: estimate how many workers you need

Let's work through a simplified example.

Assume:

- Incoming AI jobs: 1,000 per second at peak.
- Average workflow processing time: 4 seconds.
- Target worker utilization: 70%.

Using Little's Law:

\\[ L = \lambda \times W \\]

Where:

- \\(L\\) = average concurrent workflows.
- \\(\lambda\\) = arrival rate.
- \\(W\\) = average processing time.

Therefore:

\\[ L = 1000 \times 4 = 4000 \\]

You need approximately 4,000 concurrent workflow execution slots just to keep up with this arrival rate, before allowing for utilization headroom.

At 70% target utilization:

\\[ \text{Required concurrency} = \frac{4000}{0.7} \approx 5715 \\]

This does not mean 5,715 worker pods. Each pod can execute multiple workflows, subject to memory, connection pools, CPU, and downstream concurrency limits.

Also, if each workflow makes three LLM calls, your model layer might receive approximately:

\\[ 1000 \times 3 = 3000 \\]

LLM calls per second.

That is why model throughput and provider quotas must be planned separately from API throughput.

If the LLM provider cannot sustain this load, you must use some combination of:

- Admission control.
- Request prioritization.
- Queue-based scheduling.
- Smaller models.
- Batching where appropriate.
- Caching.
- Self-hosted inference.
- Reduced workflow complexity.

Staff engineer insight: The biggest bottleneck in an AI system is often not FastAPI or Kubernetes. It is the downstream inference capacity, token throughput, and cost of processing each request.

## 14. How I would implement this architecture in phases

I would not attempt to build the entire distributed architecture on day one.

| Phase | Implementation                                  |
| ----- | ----------------------------------------------- |
| 1     | FastAPI, authentication, PostgreSQL             |
| 2     | Redis caching and distributed rate limiting     |
| 3     | Durable queue and asynchronous workers          |
| 4     | RAG retrieval service and Qdrant                |
| 5     | LLM gateway and provider routing                |
| 6     | LangGraph orchestration and checkpoint recovery |
| 7     | Kubernetes autoscaling and workload isolation   |
| 8     | OpenTelemetry, cost tracking and SLO dashboards |
| 9     | Multi-region strategy and disaster recovery     |

This avoids premature microservices and unnecessary operational complexity.

## 15. Final architecture I would recommend for your Financial Advisor Copilot

Putting everything together:

## Enterprise AI platform

Client applications

Web, mobile and enterprise integrations

API Gateway + FastAPI

Authentication · RBAC · Rate limiting · Request admission

Redis

Cache and quotas

PostgreSQL

Jobs and state

Durable message queue

SQS / Kafka · Backpressure · Retry · DLQ

AI workflow execution layer

LangGraph

Supervisor and orchestration

RAG service

Retrieval and reranking

Agent services

Research, risk, compliance

LLM Gateway

Routing and concurrency

Qdrant

Vectors

LLM providers

Inference

PostgreSQL

Durable state

Cross-cutting platform capabilities

OpenTelemetry · Prometheus · Grafana · Audit logs · Security · Cost monitoring · CI/CD · Terraform

## 16. Interview-ready answer

If the interviewer asks this question, I would answer:

> I would start by defining the traffic profile, peak requests per second, latency SLOs, availability requirements, and the number of downstream LLM calls per request.
>
> Architecturally, I would use stateless API services behind an API gateway and load balancer. The API layer would handle authentication, tenant-level authorization, validation, rate limiting, and request admission.
>
> I would decouple long-running AI processing using a durable message queue such as SQS or Kafka. Dedicated workers would execute RAG and agent workflows, allowing API and AI execution capacity to scale independently.
>
> For the AI layer, I would introduce an LLM gateway that supports model routing, provider failover, concurrency limits, token budgets, and cost optimization.
>
> For RAG, I would separate offline ingestion from online retrieval, use a distributed vector database, optimize hybrid retrieval and reranking, and apply caching where data freshness and authorization allow it.
>
> I would use Redis for caching and distributed rate limiting, PostgreSQL for transactional state, and a durable LangGraph checkpointer for workflow recovery.
>
> Reliability would come from backpressure, circuit breakers, bounded retries, idempotency, dead-letter queues, and graceful degradation.
>
> Finally, I would deploy services independently on Kubernetes, use autoscaling based on workload-specific metrics, and implement end-to-end observability using OpenTelemetry.
>
> My main objective would be to scale each bottleneck independently while maintaining predictable latency, tenant isolation, availability, and cost per successful AI request.

## 17. Questions an interviewer will likely ask next

These are the follow-up questions I would prepare for at Staff/Senior AI Engineer level:

1. What happens when 10,000 requests arrive simultaneously?
2. How do you prevent the LLM provider from being overwhelmed?
3. How do you calculate the required number of workers?
4. How would you handle 100,000 concurrent users?
5. How do you prevent one tenant from consuming all available resources?
6. How would you scale Qdrant to hundreds of millions of vectors?
7. How would you reduce LLM cost by 50%?
8. What happens when Redis or PostgreSQL becomes unavailable?
9. How do you prevent duplicate execution when workers crash?
10. How would you deploy this across multiple AWS regions?

My recommendation: Start with question 1 and work through the failure and capacity scenarios. That is where interviewers distinguish someone who understands AI application development from someone who can architect and operate a large-scale AI platform.