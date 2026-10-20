# Production-grade rate limiter for AI systems — Staff Engineer implementation

For a production AI application built with FastAPI, Redis, PostgreSQL, and LLM providers, I would implement a distributed rate limiter backed by Redis, with an atomic Lua script, tenant-aware limits, per-user limits, endpoint-specific policies, and explicit handling for LLM costs and provider failures.

A rate limiter should do more than reject requests after a fixed number of calls. It should protect your API, control expensive LLM workloads, prevent one tenant from consuming another tenant's capacity, and remain predictable under concurrent traffic.

## 1. Architecture

Client / API consumer

JWT, API key, tenant ID

FastAPI middleware / dependency

Validate identity, classify endpoint, enforce body limits

Distributed rate limiter

Per tenant

Shared quota

Per user

Abuse control

LLM budget

Tokens and cost

Global limits

Capacity protection

Redis + atomic Lua execution

Allowed → RAG / agent workflow → LLM provider

Rejected → HTTP 429 + Retry-After

### Design decisions

| Concern                | Production choice                                  |
| ---------------------- | -------------------------------------------------- |
| Algorithm              | Token bucket                                       |
| Shared state           | Redis                                              |
| Atomicity              | Redis Lua script                                   |
| API framework          | FastAPI                                            |
| Identity               | Authenticated tenant and user IDs                  |
| Distributed deployment | All API replicas share Redis                       |
| LLM protection         | Request quota plus token/cost budget               |
| Observability          | Metrics, structured logs, tracing                  |
| Redis outage           | Explicit configurable fail-open/fail-closed policy |

Why token bucket? It permits small bursts while enforcing a long-term average request rate. A fixed-window limiter is simpler but can allow nearly twice the intended traffic around a window boundary. A sliding-window algorithm is another good choice when tighter rolling-window enforcement is required.

One important distinction: a request rate limiter does not, by itself, enforce an exact monthly monetary budget. That requires usage accounting, reservations, and reconciliation.

## 2. Project structure

I would keep the rate limiter in a reusable package rather than putting the Redis logic directly inside every FastAPI route.

```
ai-rate-limiter/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── identity.py
│   ├── limiter.py
│   ├── policies.py
│   └── routes.py
├── tests/
│   ├── test_limiter.py
│   └── test_api.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

The responsibilities are deliberately separated:

- `config.py`: validated runtime configuration.
- `identity.py`: identifies the authenticated tenant and user.
- `policies.py`: defines quotas per route and plan.
- `limiter.py`: implements the atomic Redis algorithm.
- `routes.py`: applies limits before expensive AI work.
- `main.py`: application lifecycle, Redis connections, health checks.

## 3. Install dependencies

`requirements.txt`

```

fastapi>=0.115,<1.0
uvicorn[standard]>=0.30,<1.0
redis>=5.2,<7.0
pydantic-settings>=2.6,<3.0
pytest>=8.0,<9.0
pytest-asyncio>=0.24,<2.0
httpx>=0.27,<1.0

```

Create a virtual environment and install:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Configuration

`app/config.py`

```
from functools import lru_cachefrom pydantic import Field, SecretStrfrom pydantic_settings import BaseSettings, SettingsConfigDictclass Settings(BaseSettings):    model_config = SettingsConfigDict(        env_file=".env",        env_file_encoding="utf-8",        extra="ignore",    )    app_name: str = "ai-rate-limiter"    environment: str = "local"    redis_url: str = "redis://localhost:6379/0"    redis_connect_timeout_seconds: float = Field(        default=1.0, gt=0    )    redis_socket_timeout_seconds: float = Field(        default=0.5, gt=0    )    # Choose deliberately based on endpoint risk.    # Fail closed is safer for expensive AI endpoints.    rate_limit_fail_closed: bool = True    # Use a stable, high-entropy secret in production.    identity_hmac_secret: SecretStr = SecretStr(        "replace-this-local-development-secret"    )    default_requests_per_minute: int = Field(        default=60, gt=0    )    default_burst: int = Field(default=10, gt=0)    # Illustrative limit, not an actual provider quota.    max_request_body_bytes: int = Field(        default=1_000_000, gt=0    )@lru_cachedef get_settings() -> Settings:    return Settings()
```

`.env.example`

```

ENVIRONMENT=local
REDIS_URL=redis://localhost:6379/0
REDIS_CONNECT_TIMEOUT_SECONDS=1.0
REDIS_SOCKET_TIMEOUT_SECONDS=0.5
RATE_LIMIT_FAIL_CLOSED=true
IDENTITY_HMAC_SECRET=replace-with-a-long-random-secret
DEFAULT_REQUESTS_PER_MINUTE=60
DEFAULT_BURST=10
MAX_REQUEST_BODY_BYTES=1000000

```

Do not commit real secrets to Git. In production, load them from a secrets manager.

## 5. Define the rate-limit policies

`app/policies.py`

```
from dataclasses import dataclass@dataclass(frozen=True, slots=True)class BucketPolicy:    name: str    capacity: int    refill_per_second: float@dataclass(frozen=True, slots=True)class RoutePolicy:    buckets: tuple[BucketPolicy, ...]# Capacity controls bursts; refill rate controls sustained traffic.PUBLIC_POLICY = RoutePolicy(    buckets=(        BucketPolicy(            name="tenant",            capacity=120,            refill_per_second=120 / 60,        ),        BucketPolicy(            name="user",            capacity=30,            refill_per_second=30 / 60,        ),    ))AI_POLICY = RoutePolicy(    buckets=(        BucketPolicy(            name="tenant_ai",
```

For example, the AI policy permits a burst of five requests per user and refills at five requests per minute. It is a token bucket, not a strict rule that every rolling 60-second interval must contain no more than five requests.

These are illustrative starting values. Production limits should be derived from measured concurrency, model throughput, provider quotas, tenant plans, and latency objectives.

## 6. Implement the distributed token bucket

This is the core implementation. The Lua script executes inside Redis, so simultaneous API replicas cannot race while reading and updating the same bucket.

It also uses Redis server time instead of relying on each application server's clock.

`app/limiter.py`

```
from __future__ import annotationsimport hashlibimport hmacimport mathfrom dataclasses import dataclassfrom typing import Sequencefrom redis.asyncio import Redisfrom app.policies import BucketPolicy, RoutePolicy# Each bucket is stored as a Redis hash:#   tokens       -> available tokens#   last_refill  -> last refill time, Redis server time## All keys in this script must be in the same Redis Cluster slot.# This implementation uses one hash tag per tenant and route scope.# It therefore works with Redis Cluster when all bucket keys in a# single call share the same {tag}.TOKEN_BUCKET_LUA = r"""local count = #KEYSlocal now_parts = redis.call("TIME")local now = tonumber(now_parts[1]) + tonumber(now_parts[2]) / 1000000local states = {}local retry_after = 0-- First pass: calculate every bucket without modifying Redis.-- If any bucket rejects the request, none of the buckets are charged.for i = 1, count do    local capacity = tonumber(ARGV[(i - 1) * 2 + 1])    local refill_rate = tonumber(ARGV[(i - 1) * 2 + 2])    local values = redis.call(        "HMGET", KEYS[i], "tokens", "last_refill"    )    local tokens = tonumber(values[1])    local last_refill = tonumber(values[2])    if tokens == nil then        tokens = capacity        last_refill = now
```

### Why this implementation matters

1. Atomic multi-bucket decision: if the user or tenant bucket rejects a request, no bucket is charged.
2. Distributed enforcement: API replicas share the same Redis state.
3. Clock consistency: refill calculations use Redis server time.
4. Privacy: identity values in Redis keys are HMAC-derived rather than raw user IDs.
5. Memory control: idle buckets expire.
6. Cluster compatibility: keys touched by one script share a hash slot.

There is one architectural trade-off: the hash tag is based on tenant and route scope, so requests from a tenant in the same route scope are assigned to one Redis Cluster slot. That preserves atomic tenant/user checks but can concentrate traffic. For very large tenants, partition the design deliberately or use a separate centralized quota service. Redis Cluster does not offer a general atomic transaction across arbitrary slots.

Also, this implementation charges one token per HTTP request. It does not yet reserve estimated LLM tokens or enforce a monetary budget; those need separate accounting.

## 7. Authenticate requests and derive identity

A critical security rule: never trust `X-Tenant-ID` or `X-User-ID` supplied by the caller. Those values must come from a validated JWT, API key, or another trusted authentication mechanism.

Here is a small API-key authentication example for local development.

`app/identity.py`

```
from dataclasses import dataclassimport hashlibimport hmacimport osfrom fastapi import Depends, HTTPException, Request, statusfrom fastapi.security import HTTPAuthorizationCredentialsfrom fastapi.security import HTTPBearerbearer = HTTPBearer(auto_error=False)@dataclass(frozen=True, slots=True)class Principal:    tenant_id: str    user_id: str    plan: strdef _load_keys() -> dict[str, Principal]:    # Demo format: API_KEYS="key1:tenant1:user1:pro,key2:tenant2:user2:basic"    # Use a secrets manager or identity provider in production.    raw = os.environ.get("API_KEYS", "")    result: dict[str, Principal] = {}    for entry in raw.split(","):        if not entry.strip():            continue        parts = entry.split(":")        if len(parts) != 4:            raise RuntimeError("Invalid API_KEYS entry")        key, tenant, user, plan = parts
```

This API-key example is intentionally minimal: it reloads credentials on each request and is not a complete enterprise identity system. For production, validate JWTs with a trusted issuer and audience, cache securely managed API-key records, support rotation and revocation, and obtain tenant membership from trusted server-side identity data.

The unused `Request` import can be omitted; the essential dependency is `get_principal`.

## 8. Enforce rate limits in FastAPI

`app/routes.py`

This dependency runs after authentication and before the protected endpoint executes. It returns HTTP 429 when any applicable bucket is exhausted.

```
from fastapi import APIRouter, Depends, HTTPException, Request, statusfrom fastapi.responses import JSONResponsefrom app.identity import Principal, get_principalfrom app.limiter import RateLimiterfrom app.policies import AI_POLICY, INGESTION_POLICY, PUBLIC_POLICYrouter = APIRouter()def get_limiter(request: Request) -> RateLimiter:    return request.app.state.rate_limiterdef make_limit_dependency(policy, scope: str):    async def enforce(        request: Request,        principal: Principal = Depends(get_principal),        limiter: RateLimiter = Depends(get_limiter),    ) -> Principal:        # Production systems should map plans to validated policies        # rather than trusting plan values from client input.        try:            decision = await limiter.check(                tenant_id=principal.tenant_id,                principal_id=principal.user_id,                route_scope=scope,                policy=policy,            )        except Exception as exc:            settings = request.app.state.settings            # Do not expose Redis connection details to the caller.            if settings.rate_limit_fail_closed:
```

The `JSONResponse` import is optional and can be removed. More importantly, production routes should use typed Pydantic request models instead of `dict`, and document ingestion must enforce authorization, file size, content-type, and queue capacity independently of request rate.

### 9. Application lifecycle and health checks

`app/main.py`

```
from contextlib import asynccontextmanagerfrom fastapi import FastAPI, HTTPException, Requestfrom redis.asyncio import Redisfrom app.config import get_settingsfrom app.limiter import RateLimiterfrom app.routes import router@asynccontextmanagerasync def lifespan(app: FastAPI):    settings = get_settings()    redis = Redis.from_url(        settings.redis_url,        decode_responses=False,        socket_connect_timeout=settings.redis_connect_timeout_seconds,        socket_timeout=settings.redis_socket_timeout_seconds,        health_check_interval=30,        max_connections=100,    )    try:        await redis.ping()        app.state.settings = settings        app.state.redis = redis        app.state.rate_limiter = RateLimiter(            redis=redis,            identity_secret=settings.identity_hmac_secret.get_secret_value(),        )        app.state.metrics = {"limiter_fail_open": 0}        yield
```

Run the API:

```
export API_KEYS="local-dev-key:tenant-1:user-1:pro"
uvicorn app.main:app --reload
```

For this local example, the bearer token is `local-dev-key`. Never use this development credential setup in a production deployment.

## 10. Run Redis locally with Docker

`docker-compose.yml`

```

services:
  redis:
    image: redis:7-alpine
    command:
      - redis-server
      - --appendonly
      - "yes"
      - --maxmemory
      - 256mb
      - --maxmemory-policy
      - noeviction
    ports:
      - "127.0.0.1:6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 2s
      retries: 10

volumes:
  redis_data:

```

Start it:

```
docker compose up -d redis
```

`noeviction` is intentional here: Redis must not silently evict rate-limit keys and thereby reset quotas. If Redis runs out of memory, writes can fail, so monitor memory and provision capacity appropriately. The application should then apply its configured failure policy.

`Dockerfile`

```

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

```

This Dockerfile is a starting point. For a hardened production image, pin dependencies, scan the image, configure resource limits, use TLS/authenticated Redis, and supply secrets through your deployment platform.

## 11. Add automated tests

A production rate limiter needs tests for both its algorithm and its HTTP behavior. The following tests use a real Redis instance because mocking Redis would not verify Lua atomicity.

`tests/test_limiter.py`

```
import osimport uuidimport pytestfrom redis.asyncio import Redisfrom app.limiter import RateLimiterfrom app.policies import BucketPolicy, RoutePolicy@pytest.fixtureasync def redis():    client = Redis.from_url(        os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1"),        decode_responses=False,    )    await client.flushdb()    yield client    await client.flushdb()    await client.aclose()@pytest.mark.asyncioasync def test_allows_requests_within_capacity(redis):    limiter = RateLimiter(        redis, identity_secret="t" * 40,        key_prefix=f"test:{uuid.uuid4().hex}",    )    policy = RoutePolicy(        buckets=(            BucketPolicy("user", capacity=2, refill_per_second=0.01),        )    )    first = await limiter.check(
```

Run the tests:

```
pip install pytest pytest-asyncio
pytest -v
```

These are foundational tests, not a complete certification of production behavior. Add concurrent-request tests, tenant-isolation tests, refill timing tests, Redis failure tests, and integration tests against the exact Redis topology used in deployment.

## 12. Important production correction: tenant quotas must be shared

There is one critical detail to fix in the initial implementation: a tenant bucket must not include the individual user's identity in its Redis key. Otherwise, each user receives a separate copy of the supposed tenant quota.

Make the following changes before using this implementation.

First, update `BucketPolicy` in `app/policies.py`:

```
@dataclass(frozen=True, slots=True)class BucketPolicy:    name: str    capacity: int    refill_per_second: float    scope: str = "user"  # "tenant" or "user"
```

Then declare tenant and user scopes explicitly:

```
PUBLIC_POLICY = RoutePolicy(    buckets=(        BucketPolicy(            name="tenant",            capacity=120,            refill_per_second=2.0,            scope="tenant",        ),        BucketPolicy(            name="user",            capacity=30,            refill_per_second=0.5,            scope="user",        ),    ))AI_POLICY = RoutePolicy(    buckets=(        BucketPolicy(            name="tenant_ai",            capacity=20,            refill_per_second=20 / 60,            scope="tenant",        ),        BucketPolicy(            name="user_ai",            capacity=5,            refill_per_second=5 / 60,            scope="user",        ),    ))
```

In `RateLimiter._key()`, replace the key construction with:

```
if bucket.scope not in {"tenant", "user"}:    raise ValueError("Invalid bucket scope")principal_hash = (    self._digest(principal_id)    if bucket.scope == "user"    else "shared")return (    f"{self._key_prefix}:{{{tag}}}:"    f"{tenant_hash}:{principal_hash}:"    f"{route_scope}:{bucket.name}")
```

This ensures all users within a tenant share the tenant bucket, while each user has an independent user bucket. The Lua script can still evaluate both atomically.

For a more complete test suite, add a test in which two different users share a tenant bucket of capacity one: the first request should pass and the second should be rejected.

## 13. What is still required before production deployment?

The code above provides a strong starting implementation, but I would not certify it for a high-volume enterprise AI service without the following work.

Security and identity

Validate JWT issuer, audience, expiry and tenant membership. Add API-key rotation, revocation, request-size limits, and per-tenant authorization.

Observability

Record allowed/rejected requests, Redis latency, Redis errors, fail-open events and rate-limit decisions. Avoid logging credentials, raw prompts or sensitive tenant data.

AI-specific budgets

Add estimated-token reservations before invoking an LLM, reconcile against actual usage afterward, and maintain tenant-level daily/monthly spending limits. Use durable accounting for monetary budgets.

Resilience and scaling

Test Redis failover and cluster behavior, connection-pool saturation, hot tenants, concurrent traffic, memory exhaustion, and provider throttling. Decide explicitly whether each endpoint should fail open or fail closed.

Correctness

Test quota sharing across users, tenant isolation, simultaneous requests, burst handling, refill accuracy, expiry, policy changes, and behavior after Redis restarts.

## 14. How to explain this in a Staff AI Engineer interview

> I designed a distributed, tenant-aware rate limiter using a token-bucket algorithm and Redis Lua scripts. Each request is evaluated against applicable tenant and user quotas atomically, preventing concurrent API replicas from independently granting the same remaining capacity. FastAPI dependencies enforce the limits before expensive RAG and agent workflows execute. I also define explicit failure policies for Redis outages and distinguish request quotas from token-based and monetary budget enforcement. The design includes observability, automated concurrency testing, and a deliberate Redis Cluster partitioning strategy.

The staff-level design principle: rate limiting is a capacity-control and cost-control subsystem, not merely an HTTP middleware feature. For an AI platform, combine request limits with token budgets, concurrency limits, bounded queues, provider-specific throttling, and durable usage accounting.
