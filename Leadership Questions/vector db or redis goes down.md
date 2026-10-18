# What if Vector DB or Redis goes down? — Senior AI Engineer approach

As a senior or staff AI engineer, I treat Vector DB and Redis failures differently because they have different responsibilities, failure modes, and recovery strategies.

In your FastAPI + PostgreSQL + Qdrant + Redis + LangGraph financial-advisor platform, a failure in either dependency can affect RAG quality, latency, workflow execution, and overall availability.

The first principle is:

Do not make the entire application unavailable just because one dependency is down. Degrade only the capabilities that depend on that service, while preserving security and correctness.

## 1. Understand the difference

### Vector database — Qdrant

Stores embeddings and searchable document chunks.

If Qdrant goes down, semantic retrieval may fail. The application might still have PostgreSQL records and source documents, but it may not be able to retrieve the right context for an LLM answer.

### Redis

Caches, rate limits, short-lived state, and sometimes queues or workflow coordination.

If Redis goes down, cache hits disappear, latency and database load may increase, and any feature relying on Redis-backed state may fail. The impact depends on what you store in Redis.

These failures are not equivalent. Qdrant is often critical to RAG correctness; Redis may be an optional performance optimization or a critical coordination dependency, depending on your design.

## 2. What happens if Qdrant goes down?

Consider this RAG request path:

FastAPI request

Query embedding

Qdrant unavailable

Search fails or times out

RAG answer generation

Cannot safely continue without adequate evidence

### User-visible impact

- Semantic search returns an error or times out.
- RAG cannot retrieve relevant policy documents or financial research.
- Agents that depend on retrieved evidence may fail or produce incomplete results.
- LLM costs can increase if retries or unnecessary fallback paths are triggered.
- If the application ignores retrieval failure and generates an answer anyway, it may hallucinate or provide unsupported financial guidance.

The most dangerous failure is a silent retrieval failure: the application treats a database outage as an empty search result and generates a confident answer without the required evidence.

### How I handle it

1. Apply short, bounded timeouts to Qdrant operations.
2. Retry only transient errors, with a strict retry budget.
3. Open a circuit breaker when persistent failures indicate the dependency is unhealthy.
4. Use an approved alternative retrieval source if one exists and meets the requirements.
5. Otherwise, return a controlled degraded response for evidence-dependent questions.
6. Monitor recovery and validate that document indexing is consistent before declaring full recovery.

A fallback could use PostgreSQL full-text search if you have indexed the relevant document content and implemented equivalent tenant and permission filters. It will not necessarily provide the same semantic retrieval quality as Qdrant.

Do not fall back to unrestricted database queries or unfiltered documents to make the request succeed.

## 3. How to implement Qdrant failure handling

The important design decision is to distinguish between retrieval unavailable and no relevant documents found.

Those are two different business outcomes.

```
from dataclasses import dataclassfrom typing import Protocol@dataclassclass RetrievedChunk:    text: str    document_id: str    score: floatclass RetrievalUnavailable(Exception):    passclass VectorSearch(Protocol):    async def search(        self,        query: str,        tenant_id: str,    ) -> list[RetrievedChunk]:        ...class QdrantRetriever:    def __init__(self, client):        self.client = client    async def search(        self,        query: str,        tenant_id: str,    ) -> list[RetrievedChunk]:        try:            # Implement embedding generation, bounded Qdrant            # search, and tenant/permission filters here.            return await self._search_qdrant(                query=query,                tenant_id=tenant_id,            )        except Exception as exc:            # Production code should catch only classified            # connection, timeout, and transient service errors.            if is_transient_vector_error(exc):                raise RetrievalUnavailable() from exc            raise
```

The helper methods above are placeholders for your actual Qdrant client and error-classification logic.

Then handle the failure at the application service layer:

```
async def answer_question(    question: str,    tenant_id: str,) -> dict:    try:        chunks = await retriever.search(            question,            tenant_id,        )    except RetrievalUnavailable:        return {            "status": "temporarily_unavailable",            "message": (                "Document search is temporarily unavailable. "                "Please retry shortly."            ),        }    if not chunks:        return {            "status": "no_evidence",            "message": (                "No sufficiently relevant documents were found."            ),        }    answer = await llm.generate(        build_grounded_prompt(question, chunks)    )    return {        "status": "completed",        "answer": answer,        "sources": [            chunk.document_id for chunk in chunks        ],    }
```

Here, `retriever`, `llm`, `build_grounded_prompt`, and `is_transient_vector_error` are application-specific dependencies that must be implemented.

### Senior-level design improvements

- Use dependency injection so Qdrant can be replaced or mocked without rewriting the business workflow.
- Enforce tenant and document permissions in every retrieval backend, including fallbacks.
- Use a bounded connection pool and explicit search deadline.
- Record retrieval status separately from the number of returned documents.
- Add a circuit breaker and a health check that tests the dependency without performing expensive searches.
- Test the failure path: Qdrant timeout, connection refusal, empty results, and recovery.

For a financial application, I would not generate a policy-based or evidence-dependent answer when required evidence cannot be retrieved. I would explicitly return a degraded status.

## 4. What happens if Redis goes down?

Redis requires a different approach because its failure impact depends on its role.

| Redis usage                  | What happens during outage?               | Recommended behavior                                           |
| ---------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| Response cache               | Cache misses increase                     | Bypass cache if safe                                           |
| Embedding or retrieval cache | More calls to the underlying services     | Bypass with load protection                                    |
| Rate limiting                | Limits may no longer be enforced          | Use a safe alternate limiter or reject protected traffic       |
| Session or token state       | Sessions may become unverifiable          | Preserve authentication guarantees; fail closed where required |
| Celery broker                | New jobs may not be accepted or delivered | Reject or durably queue work elsewhere                         |
| LangGraph checkpoint storage | Depends on the configured checkpointer    | Use durable checkpoint storage for important workflows         |
| Distributed locks            | Concurrent execution may become unsafe    | Stop affected operations until coordination is safe            |
| Temporary workflow state     | Agents may lose state or fail to resume   | Recover from durable storage, not assumptions                  |

Notice that Redis is not inherently disposable. If it only stores a response cache, it can often fail open. If it stores authorization state, critical locks, or the only copy of a workflow's progress, bypassing it may violate correctness or security.

### Scenario A: Redis is only a cache

This is the easiest failure mode.

```
Request
   |
   v
Try Redis
   |
   +-- Cache hit --> Return cached result
   |
   +-- Cache miss / Redis unavailable
                  |
                  v
             PostgreSQL
                  |
                  v
             Return result
```

The application remains available, but database traffic rises.

The dangerous scenario is a cache stampede: thousands of requests miss Redis simultaneously and overload PostgreSQL or the embedding service.

Mitigations include:

- Bounded database connection pools.
- Request coalescing or single-flight for identical cache keys.
- Per-key refresh locks where appropriate.
- Admission control and rate limits.
- TTL jitter to avoid synchronized expiration.
- A temporary limit on expensive requests during the outage.

Do not simply allow unlimited cache bypass.

## 5. Implement Redis cache failure handling in Python

Here is a practical pattern using `redis.asyncio`.

```
import jsonimport loggingfrom redis.asyncio import Redisfrom redis.exceptions import RedisErrorlogger = logging.getLogger(__name__)redis_client = Redis.from_url(    "redis://localhost:6379/0",    socket_connect_timeout=0.2,    socket_timeout=0.3,    decode_responses=True,)async def get_cached_or_load(    cache_key: str,    load_from_database,):    try:        cached = await redis_client.get(cache_key)        if cached is not None:            return json.loads(cached)    except (RedisError, OSError):        logger.warning("Redis cache unavailable")    # This path must have its own DB timeout,    # connection-pool limit, and load protection.    result = await load_from_database()    try:        await redis_client.set(            cache_key,            json.dumps(result),            ex=60,        )    except (RedisError, OSError):        logger.warning("Could not populate Redis cache")    return result
```

This example demonstrates cache fail-open behavior. The database loader and cache-key construction are application-specific.

In a production implementation, also consider:

- Cache keys must include the relevant tenant, user permissions, model version, and query parameters.
- Cache values should be validated when deserialized.
- Avoid caching sensitive personalized responses unless the access and invalidation rules are explicit.
- Set a bounded timeout on the database fallback.
- Prevent a Redis outage from turning every cache miss into an expensive downstream request.
- Distinguish an unavailable cache from a genuine cache miss in metrics.

A subtle point: the Redis operation may fail after the cache write actually succeeded. Cache writes should be safe to retry, but the application should not depend on the write having failed merely because its acknowledgment was lost.

### Scenario B: Redis is used for rate limiting or security

This is fundamentally different.

Suppose Redis enforces a limit of 100 requests per user per minute. If Redis fails and you silently allow all requests, the application loses its enforcement mechanism.

For security-sensitive controls, use a deliberate failure policy:

- Use another trusted limiter if one is available.
- Enforce conservative local limits as a temporary defense, recognizing that per-process limits are not globally accurate.
- Reject or defer protected operations if the required security guarantee cannot be maintained.
- Alert immediately and prevent the failure from becoming an unrestricted traffic or cost spike.

Never apply cache fail-open logic to authentication, authorization, or other critical security decisions.

## 6. What if both Qdrant and Redis go down?

Now consider your financial-advisor platform.

Application remains running

FastAPI · PostgreSQL · LLM provider

Qdrant down

Semantic retrieval unavailable

Redis down

Cache and Redis-backed features degraded

What the API should do

Continue safe independent operations

Read existing database records and serve authorized, non-LLM-dependent functionality.

Degrade evidence-dependent workflows

Do not invent financial answers or bypass required document permissions.

Preserve security and durability

Maintain authorization, protect the database from overload, and preserve workflow state.

The key architectural question is whether PostgreSQL is a reliable source of truth. If it holds the original documents, tenant relationships, and durable workflow records, it may support some degraded operations. If Redis is also the only store for an important piece of state, that capability cannot simply be assumed to survive.

## 7. Production architecture: isolate the failures

For an enterprise AI system, I would organize dependency handling as follows:

| Component                   | Resilience strategy                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------ |
| Qdrant                      | Bounded timeouts, transient retries, circuit breaker, permission-preserving fallback |
| Redis cache                 | Fail open, bounded fallback, cache-stampede protection                               |
| Redis security/coordination | Explicit fail-closed or trusted alternate mechanism                                  |
| PostgreSQL                  | Connection pooling, query timeouts, replication/failover, durable backups            |
| LangGraph                   | Durable checkpoints, step limits, controlled resume                                  |
| FastAPI                     | Request deadlines, admission control, health and readiness probes                    |
| Operations                  | Tracing, dependency metrics, alerts, tested recovery procedures                      |

### Liveness versus readiness

This distinction matters in Kubernetes.

- Liveness asks whether your process is functioning and should remain running.
- Readiness asks whether it can safely serve the relevant traffic.

If Qdrant is down, restarting every FastAPI pod will not fix Qdrant. Likewise, marking every pod unready might remove all capacity even for endpoints that do not need vector search.

For a multi-capability service, I would prefer capability-aware readiness or endpoint-specific degradation where feasible, rather than indiscriminately tying every endpoint to every dependency.

## 8. How do you recover when the services come back?

Recovery is a separate engineering problem. A dependency becoming reachable again does not necessarily mean the application is fully healthy.

### Qdrant recovery checklist

1. Confirm the collection and indexes are available.
2. Verify the service can perform a representative search.
3. Check whether document ingestion or indexing was interrupted.
4. Reconcile missing or outdated vectors against the authoritative document store.
5. Validate tenant filters, document permissions, and retrieval quality.
6. Close the circuit breaker gradually and monitor latency, errors, and relevance.

### Redis recovery checklist

1. Verify connectivity and Redis health.
2. Check eviction, memory usage, replication, and persistence settings as applicable.
3. Reconnect clients safely; avoid reconnect storms.
4. Expect a cold cache and protect PostgreSQL from a surge of misses.
5. Rebuild disposable cache entries gradually.
6. Reconcile any critical coordination or workflow state before resuming dependent operations.

If Redis is a cache, losing its contents may be acceptable. If it contains important state, recovery requirements must include durability, replication, backup, and reconciliation—not merely reconnecting the client.

## 9. Test these failures before production

I would include failure-injection tests in the release process.

| Test                                      | Expected result                                                |
| ----------------------------------------- | -------------------------------------------------------------- |
| Qdrant connection refused                 | Bounded failure; no fabricated RAG answer                      |
| Qdrant timeout                            | Controlled error or authorized fallback                        |
| Qdrant returns zero results               | Valid `no_evidence` outcome                                    |
| Redis cache unavailable                   | Safe fallback without database overload                        |
| Redis rate limiter unavailable            | Security policy remains enforced                               |
| Both dependencies unavailable             | Independent capabilities continue; dependent workflows degrade |
| Dependency recovers                       | Requests succeed without reconnect storms or duplicate work    |
| Dependency fails during an agent workflow | Durable state is preserved and safe resumption is possible     |

For distributed systems, test partial failures too: a network timeout may occur even when the remote service successfully processed an operation.

## 10. Senior AI engineer interview answer

> "I handle vector database and Redis outages according to their roles in the architecture.
>
> If Qdrant becomes unavailable, I apply bounded timeouts, controlled retries, and a circuit breaker. If an approved PostgreSQL full-text or alternate retrieval path exists, I can use it while preserving tenant permissions and evidence-quality requirements. Otherwise, I return a controlled degraded response rather than generating an unsupported answer. I also monitor ingestion consistency and validate retrieval quality after recovery.
>
> If Redis is used only for caching, I can fail open to the authoritative data source, but I protect that source from cache stampedes using bounded connection pools, admission control, and request coalescing. If Redis supports rate limiting, security, locks, queues, or workflow coordination, I apply an explicit failure policy rather than bypassing it.
>
> At the platform level, I use dependency-specific timeouts, circuit breakers, observability, capability-aware health checks, and durable workflow checkpoints. After recovery, I validate consistency, gradually restore traffic, and monitor for secondary failures.
>
> My goal is to isolate the failure, preserve security and correctness, keep independent capabilities available, and recover without causing a cascading outage."

The staff-level distinction is that you do not treat every dependency as equally critical. You classify each dependency by its role, define what can safely degrade, protect the remaining system from overload, and establish a tested recovery path.