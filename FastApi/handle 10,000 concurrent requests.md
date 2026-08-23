# How do you handle 10,000 concurrent requests and use Redis caching?

This is a common **Senior Backend / AI Engineer interview question**.

The important answer is: **you don't handle 10,000 concurrent requests with one FastAPI process**. You design the whole system to scale horizontally.

A production architecture might look like this:

```text
                         10,000 Concurrent Requests
                                    │
                                    ▼
                         Load Balancer / API Gateway
                                    │
                   ┌────────────────┼────────────────┐
                   ▼                ▼                ▼
              FastAPI Pod 1    FastAPI Pod 2    FastAPI Pod N
                   │                │                │
                   └────────────────┼────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                  Redis Cluster             PostgreSQL
                    Cache/Rate              Primary +
                    Limits/Sessions          Replicas
                        │
                        ▼
                  LLM / External APIs
```

The goal is to prevent all 10,000 requests from reaching expensive resources.

---

# 1. First understand concurrency vs requests per second

These are different.

### 10,000 concurrent requests

Means:

```text
Request 1  ──────────────┐
Request 2  ──────────────┤
Request 3  ──────────────┤
...                      │
Request 10000 ───────────┘
```

Many requests are active simultaneously.

### 10,000 requests per second

Means:

```text
1 second
│
├── 10,000 requests arrive
│
```

This is usually much harder.

A good interview answer is:

> **"Before designing the system, I clarify whether 10,000 means concurrent in-flight connections or 10,000 RPS, because capacity planning is very different."**

---

# 2. Don't block the FastAPI event loop

This is critical.

Bad:

```python
@app.get("/users")
async def get_users():
    time.sleep(5)  # ❌ Blocks the event loop
```

Better:

```python
import asyncio


@app.get("/users")
async def get_users():
    await asyncio.sleep(5)
```

For I/O:

```text
FastAPI
   │
   ├── Request A waiting for DB
   ├── Request B waiting for Redis
   ├── Request C waiting for HTTP API
   │
   ▼
Event loop handles other requests
```

Use async clients for I/O:

```text
Async SQLAlchemy
Async Redis
Async HTTP client
Async LLM SDK
```

---

# 3. Basic production FastAPI structure

```text
app/
├── main.py
├── api/
│   └── users.py
├── services/
│   └── user_service.py
├── repositories/
│   └── user_repository.py
├── cache/
│   └── redis.py
├── middleware/
│   └── rate_limit.py
└── config/
    └── settings.py
```

---

# 4. Connect to Redis

Use the async Redis client.

```bash
pip install redis
```

Create:

```python
# app/cache/redis.py

from redis.asyncio import Redis


redis_client = Redis.from_url(
    "redis://localhost:6379",
    encoding="utf-8",
    decode_responses=True,
)
```

In production, don't necessarily create a client at import time. Use application lifespan.

---

# 5. Production Redis using FastAPI lifespan

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis


@asynccontextmanager
async def lifespan(app: FastAPI):

    redis = Redis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True,
        max_connections=100,
    )

    app.state.redis = redis

    try:
        yield

    finally:
        await redis.aclose()


app = FastAPI(
    lifespan=lifespan
)
```

Now Redis connection pooling is managed.

Access it:

```python
from fastapi import Request


def get_redis(
    request: Request,
) -> Redis:

    return request.app.state.redis
```

Dependency:

```python
from fastapi import Depends


@app.get("/test")
async def test(
    redis: Redis = Depends(get_redis),
):

    await redis.set(
        "hello",
        "world",
    )

    return await redis.get("hello")
```

---

# 6. Cache-aside pattern

The most common Redis caching pattern is:

```text
             Request
                │
                ▼
              Redis
                │
        ┌───────┴───────┐
        │               │
      Cache Hit       Cache Miss
        │               │
        ▼               ▼
    Return Data       Database
                        │
                        ▼
                    Save Redis
                        │
                        ▼
                    Return Data
```

This is called **Cache-Aside**.

---

# 7. Implement Redis caching

Suppose:

```text
GET /users/{user_id}
```

Repository:

```python
# repositories/user_repository.py

class UserRepository:

    def __init__(self, db):
        self.db = db

    async def get_by_id(
        self,
        user_id: int,
    ):

        result = await self.db.execute(
            select(User).where(
                User.id == user_id
            )
        )

        return result.scalar_one_or_none()
```

Service with Redis:

```python
import json

from redis.asyncio import Redis


class UserService:

    def __init__(
        self,
        repository: UserRepository,
        redis: Redis,
    ):
        self.repository = repository
        self.redis = redis

    async def get_user(
        self,
        user_id: int,
    ):

        cache_key = f"user:{user_id}"

        # 1. Try cache
        cached = await self.redis.get(
            cache_key
        )

        if cached:

            return json.loads(cached)

        # 2. Cache miss → database
        user = await self.repository.get_by_id(
            user_id
        )

        if not user:
            return None

        data = {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        }

        # 3. Store cache with TTL
        await self.redis.setex(
            cache_key,
            300,
            json.dumps(data),
        )

        return data
```

TTL:

```text
300 seconds = 5 minutes
```

---

# 8. Update and invalidate the cache

Suppose:

```text
PUT /users/123
```

The cache might contain old data.

```text
Redis:

user:123
{
  "name": "Old Name"
}
```

You update PostgreSQL:

```text
PostgreSQL:

name = "New Name"
```

You must handle the cache.

## Option A: Delete cache after database update

This is usually the simplest approach.

```python
async def update_user(
    self,
    user_id: int,
    data: dict,
):

    user = await self.repository.update(
        user_id,
        data,
    )

    await self.redis.delete(
        f"user:{user_id}"
    )

    return user
```

Next request:

```text
GET user
   │
   ▼
Cache miss
   │
   ▼
Database
   │
   ▼
Redis updated
```

This is commonly used with cache-aside.

---

# 9. Redis for 10,000 concurrent requests

Suppose 10,000 users request:

```text
GET /products/123
```

Without cache:

```text
10,000 requests
      │
      ▼
PostgreSQL
      │
      ▼
Database overloaded
```

With Redis:

```text
10,000 requests
      │
      ▼
FastAPI
      │
      ▼
Redis
      │
      ├── 9,900 Cache Hits
      │
      └── 100 Cache Misses
             │
             ▼
         PostgreSQL
```

The cache dramatically reduces database load.

---

# 10. Prevent cache stampede

This is an important senior-level topic.

Suppose the cache expires:

```text
product:123 expires
```

Then 10,000 requests arrive simultaneously:

```text
Request 1 ─┐
Request 2 ─┤
Request 3 ─┤
...        ├── Cache MISS
Request N ─┘
              │
              ▼
        All hit database ❌
```

This is called a **cache stampede** or **thundering herd**.

---

# 11. Solution: distributed Redis lock

Use Redis so only one request rebuilds the cache.

```text
Cache expires
      │
      ▼
Request A
      │
      ▼
Acquire Redis Lock
      │
      ▼
Database
      │
      ▼
Update Cache
      │
      ▼
Release Lock


Other requests
      │
      ▼
Wait briefly / retry cache
```

Example:

```python
import asyncio
import json

from redis.asyncio import Redis


class UserService:

    def __init__(
        self,
        repository,
        redis: Redis,
    ):
        self.repository = repository
        self.redis = redis

    async def get_user(
        self,
        user_id: int,
    ):

        cache_key = f"user:{user_id}"
        lock_key = f"{cache_key}:lock"

        # Check cache
        cached = await self.redis.get(
            cache_key
        )

        if cached:
            return json.loads(cached)

        # Try to acquire lock
        lock_acquired = await self.redis.set(
            lock_key,
            "1",
            nx=True,
            ex=10,
        )

        if lock_acquired:

            try:
                # Double check cache
                cached = await self.redis.get(
                    cache_key
                )

                if cached:
                    return json.loads(cached)

                # Query DB
                user = await self.repository.get_by_id(
                    user_id
                )

                if not user:
                    return None

                data = {
                    "id": user.id,
                    "name": user.name,
                }

                await self.redis.setex(
                    cache_key,
                    300,
                    json.dumps(data),
                )

                return data

            finally:
                await self.redis.delete(
                    lock_key
                )

        # Another request is rebuilding cache
        await asyncio.sleep(0.05)

        cached = await self.redis.get(
            cache_key
        )

        if cached:
            return json.loads(cached)

        # Fallback
        return await self.get_user(user_id)
```

### Production note

The recursive fallback above is useful for understanding the idea, but I would avoid unbounded recursion in production. Use a bounded retry loop and safe lock ownership semantics.

---

# 12. Better stampede protection with bounded retries

```python
import asyncio
import json


class UserService:

    async def get_user(
        self,
        user_id: int,
    ):

        cache_key = f"user:{user_id}"
        lock_key = f"{cache_key}:lock"

        cached = await self.redis.get(
            cache_key
        )

        if cached:
            return json.loads(cached)

        for _ in range(5):

            acquired = await self.redis.set(
                lock_key,
                "1",
                nx=True,
                ex=10,
            )

            if acquired:

                try:
                    # Double-check
                    cached = await self.redis.get(
                        cache_key
                    )

                    if cached:
                        return json.loads(cached)

                    user = await self.repository.get_by_id(
                        user_id
                    )

                    if not user:
                        return None

                    data = {
                        "id": user.id,
                        "name": user.name,
                    }

                    await self.redis.setex(
                        cache_key,
                        300,
                        json.dumps(data),
                    )

                    return data

                finally:
                    await self.redis.delete(
                        lock_key
                    )

            await asyncio.sleep(0.05)

            cached = await self.redis.get(
                cache_key
            )

            if cached:
                return json.loads(cached)

        # Controlled fallback
        return await self.repository.get_by_id(
            user_id
        )
```

---

# 13. Cache TTL with jitter

If thousands of cache keys expire at exactly:

```text
12:00:00
```

you can get a load spike.

Instead of:

```python
ttl = 300
```

use jitter:

```python
import random

ttl = 300 + random.randint(0, 60)
```

Then:

```python
await redis.setex(
    cache_key,
    ttl,
    json.dumps(data),
)
```

Expiration is distributed.

```text
Key 1 → 300 sec
Key 2 → 321 sec
Key 3 → 345 sec
Key 4 → 358 sec
```

This reduces synchronized cache expiration.

---

# 14. Rate limiting

10,000 requests may include abuse.

Redis can implement distributed rate limiting.

A simplified fixed-window example:

```python
async def check_rate_limit(
    redis,
    user_id: str,
):

    key = f"rate_limit:{user_id}"

    count = await redis.incr(
        key
    )

    if count == 1:
        await redis.expire(
            key,
            60,
        )

    if count > 100:
        return False

    return True
```

Usage:

```python
@app.get("/api/data")
async def get_data(
    user=Depends(get_current_user),
    redis=Depends(get_redis),
):

    allowed = await check_rate_limit(
        redis,
        user.id,
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
        )

    return {"data": "hello"}
```

In production, I would usually use an atomic Lua script or Redis's server-side primitives rather than composing multiple commands where race conditions matter.

---

# 15. Database connection pooling

If 10,000 requests arrive, don't create:

```text
10,000 database connections
```

Instead:

```text
10,000 requests
       │
       ▼
FastAPI Pods
       │
       ▼
Connection Pool
       │
       ├── Connection 1
       ├── Connection 2
       ├── Connection 3
       └── Connection N
       │
       ▼
PostgreSQL
```

Example:

```python
from sqlalchemy.ext.asyncio import (
    create_async_engine,
)


engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)
```

That means approximately:

```text
20 normal connections
+
10 temporary overflow connections
```

per application process, subject to actual SQLAlchemy configuration and deployment topology.

### Important capacity planning

If you run:

```text
10 pods × 4 workers × 30 connections
```

you could theoretically allow:

```text
1,200 database connections
```

That might overload PostgreSQL.

So pool size must be calculated across the entire deployment.

---

# 16. Concurrent external API calls

Don't do:

```python
result1 = await call_service_1()
result2 = await call_service_2()
result3 = await call_service_3()
```

That is sequential.

Instead:

```python
import asyncio


result1, result2, result3 = await asyncio.gather(
    call_service_1(),
    call_service_2(),
    call_service_3(),
)
```

Architecture:

```text
             Request
                │
                ▼
        ┌───────┼────────┐
        ▼       ▼        ▼
      API 1   API 2    API 3
        │       │        │
        └───────┼────────┘
                ▼
             Response
```

This reduces latency when calls are independent.

For high load, also use bounded concurrency.

---

# 17. Limit concurrency for expensive resources

You should not allow unlimited concurrent calls to an LLM or external API.

Example:

```python
import asyncio

llm_semaphore = asyncio.Semaphore(100)


async def call_llm(prompt: str):

    async with llm_semaphore:

        return await llm_client.generate(
            prompt
        )
```

This protects:

```text
LLM API
Database
External services
GPU inference servers
```

However, a process-local semaphore is not global across multiple pods. For cluster-wide limits, use a distributed limiter, queue, or provider-side limits.

---

# 18. Use horizontal scaling

One server:

```text
FastAPI Server
```

may not handle the required load.

Instead:

```text
                    Load Balancer
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
        Pod 1          Pod 2          Pod 3
           │             │             │
           └─────────────┼─────────────┘
                         │
                       Redis
```

Kubernetes can scale:

```text
3 Pods
  ↓
High traffic
  ↓
10 Pods
  ↓
20 Pods
```

Example HPA concept:

```text
CPU > 70%
     │
     ▼
Add pods
```

For async services, also monitor metrics beyond CPU:

* Request rate
* In-flight requests
* Event-loop lag
* P95 latency
* Connection pool saturation
* Redis latency
* Database latency

---

# 19. Redis architecture at scale

For production:

```text
                 FastAPI Pods
                      │
                      ▼
                Redis Cluster
                      │
             ┌────────┼────────┐
             ▼        ▼        ▼
          Node 1    Node 2    Node 3
```

Depending on the Redis setup, you may use:

* Primary/replica replication
* Sentinel for failover
* Redis Cluster for sharding
* Managed Redis services

For 10,000 concurrent users, the exact choice depends on workload, memory, key distribution, command mix, and availability requirements.

---

# 20. Cache what is expensive

Good candidates:

```text
User profile
Product catalog
Configuration
Feature flags
Permissions
Expensive DB queries
LLM responses (when appropriate)
RAG retrieval results (carefully)
```

Avoid blindly caching:

```text
Highly dynamic data
Sensitive data without tenant/user isolation
Large objects that exhaust memory
One-time unique responses
```

For multi-tenant systems, include tenant isolation:

```text
tenant:{tenant_id}:user:{user_id}
```

Never use only:

```text
user:{user_id}
```

if IDs are not globally unique or tenant boundaries matter.

---

# 21. LLM response caching

For an AI application:

```text
Question
   │
   ▼
Redis
   │
   ├── Cache hit → Return answer
   │
   └── Cache miss
           │
           ▼
         LLM
           │
           ▼
        Save Redis
```

Example:

```python
import hashlib
import json


def create_cache_key(
    user_query: str,
    model: str,
) -> str:

    raw = f"{model}:{user_query}"

    query_hash = hashlib.sha256(
        raw.encode()
    ).hexdigest()

    return f"llm:{query_hash}"
```

Service:

```python
async def get_answer(
    query: str,
):

    key = create_cache_key(
        query,
        model="my-model",
    )

    cached = await redis.get(key)

    if cached:
        return json.loads(cached)

    answer = await llm.generate(
        query
    )

    await redis.setex(
        key,
        3600,
        json.dumps(answer),
    )

    return answer
```

For real RAG, the cache key should usually include more context, such as:

```text
tenant
user/authorization scope
normalized query
model
prompt version
retrieval corpus/version
retrieval settings
```

Otherwise you risk returning stale or unauthorized answers.

---

# 22. Complete production request flow

For 10,000 concurrent requests:

```text
                         Client
                            │
                            ▼
                   API Gateway / LB
                            │
                     Rate Limiting
                            │
                            ▼
                       FastAPI Pods
                            │
                  ┌─────────┼─────────┐
                  │         │         │
                  ▼         ▼         ▼
               Redis      DB Pool    LLM/API
                  │         │         │
                  │      PostgreSQL   │
                  │                   │
                  └─────────┼─────────┘
                            ▼
                         Response
```

For a request:

```text
1. Request arrives
2. Authenticate
3. Rate limit
4. Check Redis
5. Cache hit → return immediately
6. Cache miss → call DB/service
7. Store result with TTL
8. Return response
```

---

# 23. What I would monitor

At this scale, monitoring is mandatory.

### FastAPI

```text
Requests/sec
Concurrent requests
P50 latency
P95 latency
P99 latency
Error rate
Event loop lag
```

### Redis

```text
Cache hit ratio
Memory usage
Evictions
Latency
Connection usage
Command rate
```

### Database

```text
Connection pool utilization
Slow queries
Query latency
CPU
Replication lag
```

### LLM

```text
Requests
Latency
Timeouts
Rate limits
Input tokens
Output tokens
Cost
```

---

# 24. Best interview answer

If asked:

> **"How would you handle 10,000 concurrent requests and use Redis caching?"**

A strong answer is:

> **"First, I would clarify whether 10,000 means concurrent in-flight requests or 10,000 RPS. I would not try to handle that with a single FastAPI process. I would use async, non-blocking I/O, connection pooling, and horizontally scale FastAPI behind a load balancer.**
>
> **For expensive reads, I would use Redis with the cache-aside pattern: check Redis first, return on a cache hit, and on a miss query PostgreSQL or the downstream service and populate Redis with a TTL. On writes, I would update the database and invalidate or refresh the affected cache keys.**
>
> **To handle high concurrency safely, I would protect against cache stampedes using locking, request coalescing, or stale-while-revalidate, add TTL jitter, and apply rate limits. I would use bounded concurrency for downstream resources, calculate database pool limits across all pods, and monitor P95/P99 latency, cache hit rate, event-loop lag, Redis latency, and database pool saturation.**
>
> **The system would scale horizontally with multiple FastAPI instances, while Redis reduces pressure on PostgreSQL and other expensive downstream services."**

## The core idea

```text
10,000 Requests
       │
       ▼
Load Balancer
       │
       ▼
Multiple FastAPI Instances
       │
       ▼
Redis Cache
       │
Cache Hit ──────────► Fast Response
       │
Cache Miss
       │
       ▼
Database / LLM / External API
       │
       ▼
Redis
       │
       ▼
Response
```

For an interview, the strongest keywords to mention are:

**Async I/O → Horizontal Scaling → Load Balancer → Redis Cache-Aside → Connection Pooling → Cache Stampede Protection → Rate Limiting → Bounded Concurrency → Observability.**
