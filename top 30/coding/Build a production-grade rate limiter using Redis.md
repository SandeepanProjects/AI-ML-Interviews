To build a production-grade rate limiter for FastAPI, the industry standard is to use a Token Bucket or Sliding Window Counter algorithm implemented via Redis. Using raw Python counters introduces race conditions; leveraging Redis Lua scripts ensures that operations are atomic, highly performant, and scale cleanly across multiple server instances.
Here is a comprehensive implementation using the sliding window counter algorithm via redis-om / redis-py (async client) wrapped inside a FastAPI dependency.

Project Dependencies

You will need to install the following packages:
bash
pip install fastapi uvicorn redis pydantic
Use code with caution.

1. The Core Redis Rate Limiter Middleware

This file (limiter.py) handles the connection pool to Redis and executes an atomic Lua script to track client requests within a rolling time window.
python
import time
from fastapi import HTTPException, Request, status
import redis.asyncio as aioredis

# Atomic Lua script for a Rolling Window rate limiter
# It removes expired timestamps, counts active ones, and pushes the new hit.
LUA_RATE_LIMITER = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

local clear_before = now - window

-- Remove elements outside the sliding window
redis.call('ZREMRANGEBYSCORE', key, 0, clear_before)

-- Count elements in current window
local current_requests = redis.call('ZCARD', key)

if current_requests < limit then
    -- Add current request timestamp
    redis.call('ZADD', key, now, now)
    -- Set TTL on the key to automatically clean up inactive keys
    redis.call('EXPIRE', key, window)
    return {1, current_requests + 1} -- Allowed, updated count
else
    return {0, current_requests} -- Denied, current count
end
"""

class RedisRateLimiter:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        # Initialize an asynchronous connection pool
        self.pool = aioredis.ConnectionPool.from_url(redis_url, decode_responses=True)
        self.redis = aioredis.Redis(connection_pool=self.pool)
        self.script = None

    async def init_script(self):
        """Registers the Lua script inside Redis cache to avoid resending it on every request."""
        if not self.script:
            self.script = self.redis.register_script(LUA_RATE_LIMITER)

    async def is_rate_limited(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        """
        Evaluates the rate limit using an atomic sliding window.
        Returns a tuple: (is_allowed: bool, current_count: int)
        """
        await self.init_script()
        now = time.time()
        
        # Execute script atomically on Redis server
        allowed, current_count = await self.script(keys=[key], args=[now, window, limit])
        return bool(allowed), current_count

    async def close(self):
        """Gracefully closes the connection pool."""
        await self.pool.disconnect()

# Singleton instance for application scope
rate_limiter = RedisRateLimiter()
Use code with caution.

2. The Dependency Factory

This class allows you to dynamically configure limits (e.g., 5 requests per 10 seconds) on individual endpoints using FastAPI's dependency injection system.
python
# dependencies.py
from fastapi import Request, HTTPException, status
from limiter import rate_limiter

class RateLimit:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds

    async def __call__(self, request: Request):
        # Identify the client by IP address (or use Auth tokens / API keys if applicable)
        client_ip = request.client.host if request.client else "unknown"
        
        # Unique namespace key for Redis sorted sets
        redis_key = f"rate_limit:{request.url.path}:{client_ip}"
        
        # Check against Redis
        allowed, current_count = await rate_limiter.is_rate_limited(
            key=redis_key, 
            limit=self.limit, 
            window=self.window
        )
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Maximum allowed: {self.limit} requests per {self.window} seconds."
                }
            )
        return True
Use code with caution.

3. Application Hookup

Integrate the rate limiter into a FastAPI application lifecycle (lifespan) to handle startup connection logic seamlessly.
python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from limiter import rate_limiter
from dependencies import RateLimit

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize scripts and pools
    await rate_limiter.init_script()
    yield
    # Shutdown: Clean connection pool safely
    await rate_limiter.close()

app = FastAPI(title="Production Rate Limited API", lifespan=lifespan)

# Standard endpoints with varying limits
@app.get("/api/public", dependencies=[Depends(RateLimit(limit=60, window_seconds=60))])
async def public_endpoint():
    return {"status": "success", "message": "Welcome to the public channel (60 req/min)."}

@app.post("/api/heavy-llm", dependencies=[Depends(RateLimit(limit=5, window_seconds=10))])
async def llm_generation():
    """Heavy GPU route aggressively limited to 5 requests every 10 seconds."""
    return {"status": "success", "message": "LLM text generated successfully."}
Use code with caution.

Production Advantages of this Setup

• Zero Race Conditions: Because Redis runs single-threaded and Lua execution is block-isolated, standard distributed race conditions (where multiple web workers read and write concurrently) are impossible.
• Sliding Window Protocol: Unlike a static fixed-window block (e.g., clearing exactly at 12:00), the sliding window removes timestamps continuously, protecting your system from burst attacks right at the turn of a minute block.
• Connection Multiplexing: aioredis.ConnectionPool scales out automatically across hundreds of incoming FastAPI asynchronous event loop concurrent steps.
To ready this code for your infrastructure, let me know:
• Do you want to swap out the IP address check for an API Key / JWT token identifier?
• Should we add HTTP response headers (like X-RateLimit-Limit, X-RateLimit-Remaining, and Retry-After) to notify the client's system?
