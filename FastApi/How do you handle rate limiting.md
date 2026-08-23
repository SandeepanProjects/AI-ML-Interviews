# How do you handle rate limiting in FastAPI?

Rate limiting controls **how many requests a client can make in a given time period**.

Example:

```text
User A → 100 requests/minute → allowed
User A → 101st request      → 429 Too Many Requests
```

This is important for:

* Preventing API abuse
* Protecting databases
* Protecting LLM APIs
* Preventing unexpected LLM costs
* Preventing brute-force attacks
* Protecting limited downstream resources
* Ensuring fair usage in multi-tenant systems

For a production distributed FastAPI application, I would typically use **Redis-based rate limiting**, because an in-memory counter does not work correctly across multiple FastAPI instances.

---

# 1. Why an in-memory rate limiter is not enough

Suppose we do this:

```python
request_counts = {}
```

This works on one server:

```text
User
  ↓
FastAPI Instance 1
  ↓
request_counts
```

But with multiple instances:

```text
                    Load Balancer
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         FastAPI Pod 1         FastAPI Pod 2
              │                     │
              ▼                     ▼
           counter=100           counter=100
```

The same user can bypass the limit by hitting different pods.

Instead:

```text
                    FastAPI Pods
                  /      |      \
                 ▼       ▼       ▼
                       Redis
                         │
                   Shared counter
```

Redis becomes the shared source for rate-limit state.

---

# 2. Common rate limiting algorithms

The most common algorithms are:

```text
1. Fixed Window
2. Sliding Window
3. Sliding Window Log
4. Token Bucket
5. Leaky Bucket
```

Let's understand each.

---

# 3. Fixed Window

Example:

```text
Limit = 100 requests/minute
```

Redis key:

```text
rate_limit:user:123:2026-08-23:10:30
```

Value:

```text
57
```

When the next minute starts:

```text
New key
Counter = 0
```

## Simple Redis implementation

```python
from redis.asyncio import Redis


async def is_allowed(
    redis: Redis,
    user_id: str,
    limit: int = 100,
    window: int = 60,
) -> bool:

    key = f"rate_limit:user:{user_id}"

    count = await redis.incr(key)

    if count == 1:
        await redis.expire(
            key,
            window,
        )

    return count <= limit
```

Usage:

```python
from fastapi import (
    Depends,
    HTTPException,
)


async def rate_limit_dependency(
    user=Depends(get_current_user),
    redis: Redis = Depends(get_redis),
):

    allowed = await is_allowed(
        redis=redis,
        user_id=str(user.id),
        limit=100,
        window=60,
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
        )
```

Protect an endpoint:

```python
@app.get(
    "/api/data",
    dependencies=[
        Depends(rate_limit_dependency)
    ],
)
async def get_data():
    return {
        "message": "success"
    }
```

---

# 4. Problem with naive `INCR` + `EXPIRE`

This code:

```python
count = await redis.incr(key)

if count == 1:
    await redis.expire(key, window)
```

has a potential problem.

Imagine:

```text
INCR succeeds
       ↓
Application crashes
       ↓
EXPIRE never executes
```

Now the key might remain forever.

A better implementation uses an **atomic Redis Lua script**.

---

# 5. Production-style Redis fixed window using Lua

```python
RATE_LIMIT_SCRIPT = """
local current = redis.call(
    'INCR',
    KEYS[1]
)

if current == 1 then
    redis.call(
        'EXPIRE',
        KEYS[1],
        ARGV[1]
    )
end

return current
"""
```

Python:

```python
from redis.asyncio import Redis


class FixedWindowRateLimiter:

    def __init__(
        self,
        redis: Redis,
    ):
        self.redis = redis

    async def check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ):

        count = await self.redis.eval(
            RATE_LIMIT_SCRIPT,
            1,
            key,
            window_seconds,
        )

        ttl = await self.redis.ttl(key)

        allowed = count <= limit

        return {
            "allowed": allowed,
            "limit": limit,
            "remaining": max(
                0,
                limit - count,
            ),
            "retry_after": max(
                0,
                ttl,
            ),
        }
```

---

# 6. Return proper rate-limit headers

Good APIs return headers like:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 47
Retry-After: 12
```

Let's implement this as middleware.

---

# 7. Redis rate-limiting middleware

```python
import time

from fastapi import Request
from fastapi.responses import JSONResponse


RATE_LIMIT_SCRIPT = """
local current = redis.call(
    'INCR',
    KEYS[1]
)

if current == 1 then
    redis.call(
        'EXPIRE',
        KEYS[1],
        ARGV[1]
    )
end

local ttl = redis.call(
    'TTL',
    KEYS[1]
)

return {current, ttl}
"""


@app.middleware("http")
async def rate_limit_middleware(
    request: Request,
    call_next,
):

    # Example: use IP address
    client_ip = request.client.host

    key = (
        f"rate_limit:"
        f"{client_ip}"
    )

    limit = 100
    window = 60

    redis = request.app.state.redis

    result = await redis.eval(
        RATE_LIMIT_SCRIPT,
        1,
        key,
        window,
    )

    count = result[0]
    ttl = result[1]

    if count > limit:

        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": (
                        "Too many requests"
                    ),
                }
            },
            headers={
                "Retry-After": str(ttl),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
            },
        )

    response = await call_next(request)

    response.headers[
        "X-RateLimit-Limit"
    ] = str(limit)

    response.headers[
        "X-RateLimit-Remaining"
    ] = str(
        max(0, limit - count)
    )

    return response
```

Now the flow is:

```text
Request
   │
   ▼
Rate Limit Middleware
   │
   ▼
Redis
   │
   ├── count <= limit
   │        │
   │        ▼
   │    FastAPI endpoint
   │
   └── count > limit
            │
            ▼
        HTTP 429
```

---

# 8. Middleware vs Dependency

This is an important interview question.

## Middleware

```text
Every request
      ↓
Rate limit
      ↓
Endpoint
```

Use middleware when the rate limit applies globally.

Example:

```text
100 requests/minute per IP
```

---

## Dependency

```text
Specific endpoint
      ↓
Rate limit
      ↓
Endpoint
```

Use a dependency when:

```text
/api/chat
→ 10 requests/minute

/api/search
→ 100 requests/minute

/api/health
→ unlimited
```

Example:

```python
@app.post(
    "/chat",
    dependencies=[
        Depends(llm_rate_limit)
    ],
)
async def chat():
    ...
```

---

# 9. Rate limiting by IP vs user vs API key

You need to choose the correct identity.

## By IP

```text
rate_limit:ip:192.168.1.10
```

Good for:

* Public APIs
* Login endpoints
* Anonymous users

Problem:

```text
100 employees
behind one NAT IP
```

might share the same limit.

---

## By authenticated user

```text
rate_limit:user:123
```

Good for:

* Authenticated applications
* SaaS applications

---

## By API key

```text
rate_limit:api_key:abc123
```

Good for:

* Public developer APIs
* B2B APIs

---

## By tenant

For enterprise SaaS:

```text
rate_limit:tenant:tenant_123
```

Example:

```text
Tenant A → 10,000 requests/minute
Tenant B → 1,000 requests/minute
```

For a multi-tenant AI application, I might use:

```text
tenant
    ↓
plan
    ↓
rate limit

FREE
→ 10 requests/minute

PRO
→ 100 requests/minute

ENTERPRISE
→ custom limit
```

---

# 10. Token Bucket algorithm

Fixed windows have a boundary problem.

Example:

```text
10:00:59 → 100 requests
10:01:00 → 100 requests
```

The user effectively sends:

```text
200 requests in 2 seconds
```

Token Bucket handles bursts better.

---

## How Token Bucket works

Imagine:

```text
Bucket capacity = 100 tokens
```

Each request consumes:

```text
1 token
```

Tokens refill:

```text
10 tokens/second
```

Example:

```text
Capacity: 100

████████████████████████
100 tokens

Request
   ↓
Consume 1
   ↓
99 tokens

Time passes
   ↓
Tokens refill
```

---

# 11. Token Bucket implementation with Redis Lua

A production-quality limiter should update state atomically.

```python
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]

local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')

local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

if tokens == nil then
    tokens = capacity
end

if last_refill == nil then
    last_refill = now
end

local elapsed = math.max(0, now - last_refill)

tokens = math.min(
    capacity,
    tokens + elapsed * refill_rate
)

local allowed = 0

if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
end

redis.call(
    'HMSET',
    key,
    'tokens',
    tokens,
    'last_refill',
    now
)

local ttl = math.ceil(
    (capacity / refill_rate) * 2
)

redis.call(
    'EXPIRE',
    key,
    ttl
)

return {
    allowed,
    math.floor(tokens)
}
"""
```

Python implementation:

```python
import time

from redis.asyncio import Redis


class TokenBucketRateLimiter:

    def __init__(
        self,
        redis: Redis,
    ):
        self.redis = redis

    async def check(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
        tokens: int = 1,
    ):

        now = time.time()

        result = await self.redis.eval(
            TOKEN_BUCKET_SCRIPT,
            1,
            key,
            capacity,
            refill_rate,
            now,
            tokens,
        )

        allowed = bool(result[0])
        remaining = int(result[1])

        return {
            "allowed": allowed,
            "remaining": remaining,
            "limit": capacity,
        }
```

Dependency:

```python
from fastapi import (
    Depends,
    HTTPException,
)


async def llm_rate_limit(
    request: Request,
    user=Depends(get_current_user),
):

    redis = request.app.state.redis

    limiter = TokenBucketRateLimiter(
        redis
    )

    result = await limiter.check(
        key=f"llm:user:{user.id}",
        capacity=20,
        refill_rate=20 / 60,
    )

    if not result["allowed"]:

        raise HTTPException(
            status_code=429,
            detail="LLM rate limit exceeded",
        )
```

Apply:

```python
@app.post(
    "/api/chat",
    dependencies=[
        Depends(llm_rate_limit)
    ],
)
async def chat():
    return {
        "message": "LLM response"
    }
```

This means:

```text
Maximum burst = 20 requests
Refill = 20 tokens/minute
```

---

# 12. Cost-based rate limiting for LLMs

For LLM applications, request count is sometimes not enough.

Consider:

```text
User A:
10 requests
100 tokens each

User B:
10 requests
100,000 tokens each
```

Both made:

```text
10 requests
```

But the cost is very different.

You can rate-limit by:

```text
Tokens
Cost
Concurrent LLM requests
Requests
```

Example:

```text
FREE PLAN
100,000 tokens/day

PRO PLAN
5,000,000 tokens/day
```

Redis key:

```text
tenant:{tenant_id}:token_usage:2026-08-23
```

Then increment by actual token usage:

```python
async def record_token_usage(
    redis: Redis,
    tenant_id: str,
    tokens: int,
):

    key = (
        f"tenant:{tenant_id}:"
        f"token_usage:2026-08-23"
    )

    total = await redis.incrby(
        key,
        tokens,
    )

    await redis.expire(
        key,
        86400,
    )

    return total
```

In production, derive the date dynamically in UTC and ensure the increment/TTL initialization is atomic.

---

# 13. Protecting login endpoints

Login endpoints are especially sensitive.

You can apply multiple limits:

```text
Per IP:
10 attempts/minute

Per email:
5 attempts/minute
```

Architecture:

```text
Login Request
     │
     ├── IP Rate Limit
     │
     ├── Email Rate Limit
     │
     └── Authentication
```

Example:

```python
async def login_rate_limit(
    email: str,
    request: Request,
):

    ip = request.client.host

    ip_key = f"login:ip:{ip}"
    user_key = f"login:email:{email}"

    # Check both limits
```

For endpoints behind a proxy, only trust forwarded client IP headers when the proxy infrastructure is explicitly configured as trusted.

---

# 14. Different limits for different endpoints

A real application might have:

```text
Endpoint                 Limit

/api/health              Unlimited
/api/users               100/min
/api/search              60/min
/api/chat                20/min
/api/upload              10/min
/api/login               5/min
```

You can represent this as configuration:

```python
RATE_LIMITS = {
    "/api/chat": {
        "capacity": 20,
        "refill_rate": 20 / 60,
    },
    "/api/search": {
        "capacity": 100,
        "refill_rate": 100 / 60,
    },
}
```

---

# 15. Middleware implementation with endpoint-specific limits

```python
RATE_LIMITS = {
    "/api/chat": 20,
    "/api/search": 100,
}


@app.middleware("http")
async def rate_limit_middleware(
    request: Request,
    call_next,
):

    path = request.url.path

    if path not in RATE_LIMITS:
        return await call_next(request)

    limit = RATE_LIMITS[path]

    identity = request.client.host

    key = (
        f"rate_limit:"
        f"{path}:"
        f"{identity}"
    )

    # Check Redis limiter
    allowed = await check_rate_limit(
        redis=request.app.state.redis,
        key=key,
        limit=limit,
        window=60,
    )

    if not allowed:

        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "RATE_LIMITED",
                    "message": "Too many requests",
                }
            },
        )

    return await call_next(request)
```

However, for authenticated identity, middleware may run before you have resolved the user dependency. In that case, a dependency or API gateway is often cleaner.

---

# 16. Rate limiting + concurrency limiting

These are different.

### Rate limiting

Controls:

```text
How many requests over time?
```

Example:

```text
100 requests/minute
```

### Concurrency limiting

Controls:

```text
How many requests at the same time?
```

Example:

```text
Maximum 10 concurrent LLM generations
```

Example:

```python
import asyncio


llm_semaphore = asyncio.Semaphore(10)


async def generate(prompt: str):

    async with llm_semaphore:

        return await llm_client.generate(
            prompt
        )
```

For multiple FastAPI pods:

```text
Pod 1 → 10
Pod 2 → 10
Pod 3 → 10
```

Total:

```text
30 concurrent requests
```

So use a distributed mechanism when you need a global cluster-wide concurrency limit.

---

# 17. Rate limiting architecture

For a production system:

```text
                    Client
                       │
                       ▼
                 API Gateway
                       │
                 Global Limit
                       │
                       ▼
                 Load Balancer
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
            Pod 1    Pod 2    Pod N
              │        │        │
              └────────┼────────┘
                       │
                       ▼
                     Redis
                       │
             Distributed Rate Limit
```

You may enforce different levels:

```text
1. CDN / WAF
2. API Gateway
3. Application
4. LLM/provider-level limits
```

Defense in depth is useful.

---

# 18. What happens when Redis is down?

This is a good interview question.

You need to decide:

## Fail open

```text
Redis unavailable
      ↓
Allow request
```

Good for:

```text
Normal low-risk endpoints
```

Risk:

```text
Abuse may increase
```

## Fail closed

```text
Redis unavailable
      ↓
Reject request
```

Good for:

```text
Expensive LLM endpoints
Payment-sensitive operations
Security-sensitive endpoints
```

Example:

```python
try:

    result = await limiter.check(...)

except Exception:

    logger.exception(
        "Rate limiter unavailable"
    )

    # Policy decision

    raise HTTPException(
        status_code=503,
        detail="Rate limiting unavailable",
    )
```

The correct policy depends on the endpoint and business requirements.

---

# 19. Testing rate limiting

Example using FastAPI dependency overrides is one approach, but the core test should verify:

```text
Request 1 → 200
Request 2 → 200
Request 3 → 200
Request 4 → 429
```

Pseudo-test:

```python
def test_rate_limit(client):

    for _ in range(3):

        response = client.get(
            "/api/chat"
        )

        assert response.status_code == 200

    response = client.get(
        "/api/chat"
    )

    assert response.status_code == 429
```

For production tests, use an isolated Redis instance/container and verify TTL, atomicity, concurrent requests, and behavior when Redis is unavailable.

---

# 20. Which algorithm should you use?

| Algorithm         | Best for                     |
| ----------------- | ---------------------------- |
| Fixed Window      | Simple APIs                  |
| Sliding Window    | More accurate request limits |
| Token Bucket      | APIs with controlled bursts  |
| Leaky Bucket      | Smooth output rate           |
| Concurrency Limit | Expensive operations         |
| Token/Cost Limit  | LLM applications             |

For a production AI application, I might use:

```text
API Gateway
    ↓
IP rate limit
    ↓
FastAPI
    ↓
Tenant rate limit
    ↓
Redis token bucket
    ↓
LLM concurrency limit
    ↓
Daily token/cost quota
    ↓
LLM Provider
```

---

# 21. Interview answer

If asked:

> **"How do you implement rate limiting in FastAPI?"**

A strong answer is:

> **"For a single instance, an in-memory limiter can work, but in production I use Redis because rate-limit state must be shared across multiple FastAPI instances. I choose the identity based on the API—IP for anonymous endpoints, user ID for authenticated APIs, API key for developer APIs, and tenant ID for multi-tenant systems."**
>
> **"For simple limits, I can use a fixed or sliding window, but for APIs that need to allow controlled bursts, I prefer a token bucket implemented atomically with Redis Lua. The limiter returns HTTP 429 with `Retry-After` and rate-limit headers. For LLM APIs, I often combine request limits with concurrent-generation limits and token or cost quotas because request count alone doesn't control spend."**
>
> **"I also decide fail-open versus fail-closed behavior if Redis is unavailable and monitor rate-limit rejections, Redis latency, and limiter failures."**

# Core production flow

```text
Request
   │
   ▼
Identify Client
(IP / User / API Key / Tenant)
   │
   ▼
Redis Atomic Rate Limit Check
   │
   ├── Allowed
   │      │
   │      ▼
   │   FastAPI Endpoint
   │
   └── Rejected
          │
          ▼
   HTTP 429 Too Many Requests
```

## Most important code concept

For a distributed FastAPI application:

```text
Multiple FastAPI Pods
        │
        ▼
Shared Redis
        │
        ▼
Atomic Counter / Token Bucket
```

That is the core idea behind scalable rate limiting.
