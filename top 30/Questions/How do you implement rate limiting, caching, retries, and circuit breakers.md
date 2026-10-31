# How do you implement rate limiting, caching, retries, and circuit breakers?

This is a very important Senior AI Engineer / Staff AI Engineer system-design and coding interview question.

In a production AI application, these four mechanisms work together to protect your application from traffic spikes, excessive LLM costs, provider outages and cascading failures.

I would implement them in a centralized LLM Gateway using:

- FastAPI – API layer.
- Redis – Distributed rate limiting and caching.
- Async Python – Non-blocking execution.
- OpenAI / Anthropic – LLM providers.
- Circuit breaker – Provider failure protection.
- Exponential backoff – Temporary failure recovery.
- OpenTelemetry – Monitoring and tracing.

## 1. Understand the four mechanisms

Rate limiting

Controls how many requests a user or tenant can send within a time window.

Protects APIs, providers and budgets.

Caching

Stores previously generated results to avoid unnecessary LLM calls.

Reduces latency and token expenditure.

Retries

Re-executes requests when temporary failures occur.

Handles transient network and provider errors.

Circuit breaker

Temporarily stops requests to an unhealthy provider.

Prevents cascading failures.

## 2. Production architecture

I would not implement these mechanisms independently inside every agent or business service.

Instead, I would put them in the LLM Gateway.

Client requests

FastAPI API Gateway

Authentication · Tenant identification

LLM Gateway

1\. Rate limiter

2\. Cache lookup

3\. Retry manager

4\. Circuit breaker

5\. Provider selection

OpenAI

Anthropic

Local LLM

Redis

Rate-limit counters · Cache · Shared circuit state

The execution order is important:

1. Authenticate the user.
2. Check tenant and user rate limits.
3. Check cache.
4. If cache misses, check provider circuit breaker.
5. Execute the LLM request with retry and timeout.
6. Validate the response.
7. Cache eligible responses.
8. Return the response.

## 3. Rate limiting implementation

### What problem does rate limiting solve?

Imagine your AI application receives:

- User A: 100 requests/minute.
- User B: 500 requests/minute.
- User C: 10,000 requests/minute.

Without rate limiting, one user can consume the entire LLM budget and exhaust provider quotas.

I would implement multiple limits:

| Limit              | Example               |
| ------------------ | --------------------- |
| Per-user           | 20 requests/minute    |
| Per-tenant         | 500 requests/minute   |
| Global application | 5,000 requests/minute |
| Token budget       | 100,000 tokens/hour   |

These are illustrative limits; actual values should be based on provider quotas, workload and business requirements.

### Which algorithm should you use?

Common algorithms:

| Algorithm      | Characteristics                    |
| -------------- | ---------------------------------- |
| Fixed window   | Simple, but allows boundary bursts |
| Sliding window | More accurate request distribution |
| Token bucket   | Allows controlled bursts           |
| Leaky bucket   | Smooths outgoing traffic           |

For an AI application, I would generally use a token bucket for request and token budgets, or a Redis-backed sliding-window implementation where precise rolling limits are important.

### Production-oriented Redis sliding-window implementation

Install:

```
pip install redis fastapi uvicorn
```

`rate_limiter.py`

```
import redis.asyncio as redisimport timeimport uuidRATE_LIMIT_SCRIPT = """local key = KEYS[1]local now = tonumber(ARGV[1])local window = tonumber(ARGV[2])local limit = tonumber(ARGV[3])local member = ARGV[4]redis.call(    'ZREMRANGEBYSCORE',    key,    '-inf',    now - window)local count = redis.call('ZCARD', key)if count >= limit then    local oldest = redis.call(        'ZRANGE', key, 0, 0, 'WITHSCORES'    )    local retry_after = window    if oldest[2] then        retry_after = math.max(            1,            tonumber(oldest[2]) + window - now        )    end    return {0, count, retry_after}
```

How does this work?

1. Redis stores request timestamps in a sorted set.
2. Old requests outside the window are removed.
3. Redis counts the remaining requests.
4. If the limit is reached, the request is rejected.
5. Otherwise, the new request timestamp is inserted.

Redis Lua executes this operation atomically, which is important when multiple FastAPI workers are accessing the same counter.

### FastAPI integration

```
from fastapi import FastAPI, HTTPException, Dependsapp = FastAPI()redis_client = redis.from_url(    "redis://localhost:6379",    decode_responses=True)limiter = RedisRateLimiter(redis_client)async def check_rate_limit(user_id: str):    result = await limiter.allow(        user_id=user_id,        limit=20,        window=60    )    if not result["allowed"]:        raise HTTPException(            status_code=429,            detail="Rate limit exceeded",            headers={                "Retry-After": str(                    result["retry_after"]                )            }        )@app.post("/generate")async def generate(user_id: str):
```

In a real application, `user_id` must come from verified authentication, not from an arbitrary client-supplied parameter.

Staff-level consideration: Apply limits at user, tenant, provider and global levels. Also enforce token-based quotas because ten requests containing 100 tokens are very different from ten requests containing 100,000 tokens.

## 4. Caching implementation

### What problem does caching solve?

Consider a user asking:

"What is the difference between an index fund and an ETF?"

Your application calls the LLM and generates an answer.

Five seconds later, another user asks the same question.

Without caching, you make another LLM API call and pay again.

With caching, you can return the previously generated answer.

### Types of caching

| Type              | Usage                               |
| ----------------- | ----------------------------------- |
| Exact-match cache | Identical requests                  |
| Semantic cache    | Similar questions                   |
| Embedding cache   | Avoid repeated embedding generation |
| Retrieval cache   | Reuse document retrieval results    |
| Response cache    | Reuse final LLM responses           |

For production RAG, I would combine exact-match caching with carefully controlled semantic caching.

### Redis response caching

`cache.py`

```
import redis.asyncio as redisimport hashlibimport jsonclass LLMCache:    def __init__(self, redis_client):        self.redis = redis_client    def generate_key(        self,        tenant_id,        prompt,        model,        prompt_version    ):        payload = {            "tenant_id": tenant_id,            "prompt": prompt,            "model": model,            "prompt_version": prompt_version        }        serialized = json.dumps(            payload,            sort_keys=True        )        return hashlib.sha256(            serialized.encode()        ).hexdigest()    async def get(self, key):
```

### How to use it

```
async def generate_answer(    tenant_id,    prompt,    llm_gateway,    cache):    key = cache.generate_key(        tenant_id=tenant_id,        prompt=prompt,        model="gpt-4.1-mini",        prompt_version="v1"    )    # Step 1: Check cache    cached = await cache.get(key)    if cached is not None:        return {            "source": "cache",            "answer": cached        }    # Step 2: Cache miss, call LLM    answer = await llm_gateway.generate(prompt)    # Step 3: Store successful result    await cache.set(        key,        answer,        ttl=300    )
```

### Important production considerations

For your financial advisor copilot, I would not cache only on the basis of the user's question.

The cache key should also account for:

- Tenant ID.
- User permissions or authorization scope.
- Model and model configuration.
- System prompt version.
- Retrieved document IDs and versions.
- Relevant portfolio data version.
- Temperature and response format.

Otherwise, one tenant might receive another tenant's cached information.

Semantic caching needs additional safeguards. Similar questions do not necessarily have equivalent answers. For example:

"What was my portfolio value yesterday?"

and

"What is my portfolio value today?"

must not share a cached answer.

For dynamic financial information, use short TTLs or avoid caching entirely.

### Prevent cache stampedes

Imagine 1,000 users request the same uncached result simultaneously.

All 1,000 requests could call the LLM.

I would implement a single-flight mechanism or distributed lock so only one request generates the response while the others wait for the result.

For distributed deployments, use Redis locks with bounded waiting, unique ownership tokens and safe lock release. Never use an unconditional `DEL` to release a lock that may have expired and been acquired by another worker.

## 5. Retry implementation

### Why do we need retries?

LLM providers experience temporary failures.

Examples:

- Network connection failure.
- Request timeout.
- HTTP 429.
- HTTP 503.
- Temporary server errors.

Instead of immediately failing, retry the request when the error is recoverable.

### Exponential backoff

The formula is:

\\[ Delay = \min(MaxDelay, BaseDelay \times 2^{attempt}) \\]

Add random jitter:

\\[ ActualDelay = Delay + RandomJitter \\]

This prevents thousands of requests from retrying simultaneously.

### Production-oriented implementation

`retry.py`

```
import asyncioimport randomfrom app.llm.exceptions import (    LLMTimeoutError,    LLMRateLimitError,    LLMUnavailableError)RETRYABLE_ERRORS = (    LLMTimeoutError,    LLMRateLimitError,    LLMUnavailableError)async def retry_operation(    operation,    max_attempts=3,    base_delay=0.5,    max_delay=5.0,):    for attempt in range(max_attempts):        try:            return await operation()        except RETRYABLE_ERRORS:            if attempt == max_attempts - 1:                raise            delay = min(
```

Usage:

```
async def call_llm():    return await provider.generate(        prompt="Explain RAG architecture"    )response = await retry_operation(    call_llm,    max_attempts=3)
```

### When should you NOT retry?

| Error                   | Action                               |
| ----------------------- | ------------------------------------ |
| HTTP 400                | Do not retry                         |
| HTTP 401                | Fix credentials                      |
| HTTP 403                | Fix permissions                      |
| HTTP 429                | Retry according to provider guidance |
| HTTP 500                | Retry with backoff                   |
| HTTP 503                | Retry with backoff                   |
| Network timeout         | Retry within deadline                |
| Invalid JSON output     | Controlled validation/repair         |
| Context window exceeded | Reduce input or change model         |

Important: A retry must not blindly repeat an operation that has already produced an external side effect. For tool calls involving transactions, payments or database mutations, use idempotency keys and explicit execution records.

Also, use a total deadline across all retries. Never let retries continue indefinitely.

## 6. Circuit breaker implementation

### Why do we need a circuit breaker if we already have retries?

This is a common interview follow-up.

Consider:

- Your application receives 5,000 requests/minute.
- OpenAI is unavailable.
- Every request retries three times.
- Your application generates 15,000 failed provider calls.

Retries can amplify the outage.

A circuit breaker prevents this.

### Three states

| State     | Behavior                             |
| --------- | ------------------------------------ |
| CLOSED    | Allow provider requests              |
| OPEN      | Reject provider requests immediately |
| HALF_OPEN | Allow limited recovery probes        |

### Implementation

For a single-process demonstration, the following implementation shows the state machine:

`circuit_breaker.py`

```
import asyncioimport timeclass CircuitBreaker:    def __init__(        self,        failure_threshold=5,        recovery_timeout=30    ):        self.failure_threshold = failure_threshold        self.recovery_timeout = recovery_timeout        self.failures = 0        self.state = "CLOSED"        self.opened_at = None        self.lock = asyncio.Lock()        self.probe_active = False    async def call(self, operation):        async with self.lock:            if self.state == "OPEN":                elapsed = (                    time.monotonic() - self.opened_at                )                if elapsed < self.recovery_timeout:                    raise Exception(                        "Circuit breaker is OPEN"                    )
```

Usage:

```
breaker = CircuitBreaker(    failure_threshold=5,    recovery_timeout=30)async def call_provider():    return await provider.generate(        prompt="Explain vector databases"    )response = await breaker.call(call_provider)
```

### What happens in production?

Suppose five consecutive requests fail.

Request 1–5

CLOSED

Failures accumulate.

Failure threshold reached

OPEN

New requests are rejected or routed to another provider.

After 30 seconds

HALF-OPEN

One controlled recovery probe is allowed.

Probe succeeds

CLOSED

Normal traffic resumes.

This example is intentionally in-memory. Its lock protects only one Python process.

For a production deployment with multiple FastAPI workers or Kubernetes replicas, I would use a Redis-backed circuit breaker with atomic Lua scripts, a shared state per provider/model, and a distributed half-open probe lock. I would also count only provider failures, not application validation errors or user cancellations.

## 7. How do you combine all four mechanisms?

This is the most important part of the implementation.

You should not have four independent systems that do not communicate with each other.

They should work together in the LLM Gateway.

### Execution flow

```
async def execute_llm_request(    tenant_id,    user_id,    prompt,    cache,    limiter,    gateway):    # 1. Rate limiting    rate = await limiter.allow(        user_id=user_id,        limit=20,        window=60    )    if not rate["allowed"]:        raise Exception("Rate limit exceeded")    # 2. Cache lookup    key = cache.generate_key(        tenant_id=tenant_id,        prompt=prompt,        model="gpt-4.1-mini",        prompt_version="v1"    )    cached = await cache.get(key)    if cached is not None:        return cached    # 3. LLM call    # Gateway handles retries, circuit breakers    # and fallback provider selection.
```

This is the orchestration pattern. A production implementation should additionally enforce a total request deadline, tenant-level quotas, response validation and cache-stampede protection.

### Responsibility separation

| Component      | Responsibility                       |
| -------------- | ------------------------------------ |
| API Gateway    | Authentication and request limits    |
| RateLimiter    | User and tenant quotas               |
| Cache          | Avoid unnecessary model calls        |
| RetryManager   | Recover from transient errors        |
| CircuitBreaker | Protect unhealthy providers          |
| LLM Gateway    | Provider selection and orchestration |
| Observability  | Metrics, logs and tracing            |

## 8. How would you implement this for millions of requests?

For a large-scale AI application, I would make the following architectural decisions.

Scalable production architecture

Load Balancer

FastAPI 1

FastAPI 2

FastAPI 3

FastAPI N

Shared infrastructure

Redis Cluster

Cache and quotas

Redis State

Circuit coordination

LLM Gateway / Provider Router

OpenAI

Anthropic

Self-hosted

Prometheus · Grafana · OpenTelemetry

My design considerations:

1. Redis provides shared rate-limit and cache state.
2. Redis Cluster or a suitable managed Redis deployment provides horizontal scaling.
3. FastAPI workers remain stateless.
4. Provider-specific concurrency limits prevent overwhelming external APIs.
5. Background workers handle long-running, asynchronous AI jobs.
6. Circuit breakers isolate failures per provider.
7. Kubernetes autoscaling handles application workload.
8. Observability tracks latency, cost, throughput and failures.

One important distinction: millions of requests does not mean millions of simultaneous LLM calls. You need admission control, bounded concurrency, queues and capacity planning based on actual provider quotas.

## 9. Common mistakes interviewers look for

| Mistake                         | Correct approach              |
| ------------------------------- | ----------------------------- |
| Retry every error               | Retry only transient failures |
| Unlimited retries               | Bounded retries and deadlines |
| In-memory rate limiting         | Distributed rate limiting     |
| Cache everything                | Cache only eligible responses |
| One global circuit breaker      | Per-provider/model breakers   |
| Retry without jitter            | Exponential backoff + jitter  |
| Ignore token consumption        | Token-based quotas            |
| Cache without tenant ID         | Tenant-aware cache keys       |
| Use fallback without validation | Validate fallback responses   |
| Ignore duplicate requests       | Idempotency and deduplication |

## 10. Interview-ready answer

If an interviewer asks how you implement rate limiting, caching, retries and circuit breakers, I would answer:

"I implement these four mechanisms inside a centralized LLM Gateway.

For rate limiting, I use Redis-backed distributed rate limiting with user-level, tenant-level and provider-level quotas. I also consider token-based budgets because LLM requests have different token consumption.

For caching, I use Redis to cache eligible LLM responses, embeddings and retrieval results. I create cache keys using tenant ID, model, prompt version and relevant document versions. For semantic caching, I introduce similarity thresholds and additional authorization and freshness checks.

For retries, I use exponential backoff with jitter and retry only transient failures such as timeouts, rate limits and temporary provider errors. I enforce an overall request deadline to prevent excessive retries.

For circuit breakers, I implement CLOSED, OPEN and HALF-OPEN states. When a provider experiences repeated failures, I temporarily stop sending requests to it and route traffic to an approved fallback provider.

Finally, I integrate OpenTelemetry, Prometheus and Grafana to monitor latency, error rates, cache hit rates, token consumption, provider availability and fallback frequency.

My objective is to make the AI application scalable, cost-efficient, fault-tolerant and resilient to external provider failures."

## 11. What you should be able to explain in a Staff AI Engineer interview

I would prioritize these five advanced topics:

1. Distributed rate limiting: How do you implement Redis Lua-based token bucket rate limiting across 20 Kubernetes replicas?
2. Cache stampede: How do you prevent 1,000 simultaneous requests from generating the same LLM response?
3. Retry storms: How do you prevent retries from overwhelming an already unhealthy LLM provider?
4. Distributed circuit breakers: How do you maintain consistent breaker state across multiple FastAPI workers?
5. Streaming failure recovery: What happens if OpenAI fails after generating 500 tokens? Can you switch providers without corrupting the response?

The fifth question is particularly important: once tokens have been sent to a client, transparently switching providers can produce duplicated or inconsistent output. You need an explicit streaming recovery strategy rather than blindly restarting generation.

My recommendation: Implement these four components as reusable private Python packages in your AI platform:

```
ai-platform/
│
├── packages/
│   ├── rate-limiter/
│   ├── llm-cache/
│   ├── retry-manager/
│   └── circuit-breaker/
│
├── services/
│   ├── rag-service/
│   ├── agent-service/
│   └── financial-advisor-service/
│
└── infrastructure/
    ├── redis/
    ├── kubernetes/
    └── monitoring/
```

This is a strong Staff Engineer approach because these reliability mechanisms become reusable infrastructure rather than duplicated logic inside individual AI applications.