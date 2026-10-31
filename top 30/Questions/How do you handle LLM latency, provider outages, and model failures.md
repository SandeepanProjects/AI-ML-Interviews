# How do you handle LLM latency, provider outages, and model failures?

This is a very important Senior AI Engineer / Staff AI Engineer interview question because production AI systems cannot depend on a single LLM provider or assume that every model request will succeed.

My approach is to design the LLM integration layer for:

1. Low latency – Reduce the time taken to generate responses.
2. High availability – Continue serving requests when an LLM provider is unavailable.
3. Fault tolerance – Handle timeouts, rate limits, network failures and malformed responses.
4. Cost optimization – Avoid unnecessarily calling expensive models.
5. Graceful degradation – Return a useful response even when the primary model fails.
6. Observability – Identify whether the failure originates from the provider, model, prompt, retrieval or application.

Let's approach this as a production system using FastAPI, Python, OpenAI, Redis and asynchronous execution.

## 1. Production architecture

I would not call OpenAI directly from every business service.

Instead, I would introduce an LLM Gateway responsible for model selection, retries, timeouts, fallback, circuit breaking and observability.

Client / API Gateway

Authentication · Rate limiting · Request ID

FastAPI Application

Business service · Request validation

LLM Gateway

Timeout

Retry policy

Circuit breaker

Fallback routing

Model selection

Response validation

Primary

OpenAI

Fallback

Anthropic

Fallback

Self-hosted

Redis · Metrics · Logs · Traces

Shared circuit state · Caching · Monitoring

The key principle is:

Business services should depend on an LLM abstraction, not on a specific LLM provider.

## 2. How do you handle LLM latency?

First, I would identify where the latency is coming from.

In a RAG application, the total response time is approximately:

\\[ T\_{total}=T\_{retrieval}+T\_{reranking}+T\_{LLM}+T\_{postprocessing} \\]

For example:

| Component           | Latency  |
| ------------------- | -------- |
| Query embedding     | 100 ms   |
| Vector retrieval    | 150 ms   |
| Reranking           | 300 ms   |
| LLM generation      | 2,000 ms |
| Response validation | 100 ms   |
| Total               | 2,650 ms |

Here, LLM generation is the biggest contributor.

### My optimization strategy

A. Select the appropriate model

Do not use an expensive, large reasoning model for every request.

| Task                            | Model strategy            |
| ------------------------------- | ------------------------- |
| Classification                  | Small, fast model         |
| Query rewriting                 | Small model               |
| Simple RAG answer               | Fast model                |
| Complex financial analysis      | Strong reasoning model    |
| Final critical decision support | Strong model + validation |

B. Reduce input tokens

- Remove irrelevant retrieved chunks.
- Use reranking.
- Limit conversation history.
- Avoid unnecessarily long system prompts.
- Use appropriate chunk sizes.

C. Implement semantic caching

If the same or sufficiently similar question has already been answered, return a cached response.

Example:

```
import hashlibimport redis.asyncio as redisclass SemanticCache:    def __init__(self, client):        self.redis = client    async def get(self, key: str):        return await self.redis.get(key)    async def set(        self,        key: str,        value: str,        ttl: int = 300    ):        await self.redis.setex(key, ttl, value)
```

This example implements exact-key caching. For semantic caching, store embeddings and retrieve sufficiently similar queries, subject to tenant, authorization, document-version and freshness constraints.

D. Use streaming

Instead of waiting for the entire response, stream tokens to the client.

```
from fastapi.responses import StreamingResponseasync def stream_response(llm, prompt):    async for chunk in llm.astream(prompt):        if chunk:            yield chunk@app.post("/generate")async def generate(prompt: str):    return StreamingResponse(        stream_response(llm, prompt),        media_type="text/plain"    )
```

Streaming improves perceived latency, although it does not necessarily reduce total generation time.

E. Parallelize independent operations

For example, in a financial advisor copilot:

- Fetch portfolio information.
- Retrieve market research.
- Retrieve compliance policies.

These independent operations can run concurrently instead of sequentially.

Use `asyncio.gather()` with bounded concurrency and appropriate timeouts.

## 3. How do you handle provider outages?

Consider this production scenario:

Your application uses OpenAI as its primary provider.

Suddenly:

- OpenAI starts returning HTTP 503.
- Some requests time out.
- API requests begin accumulating.
- Your FastAPI workers are waiting for responses.
- Application latency increases.
- Eventually, your application starts failing.

If you simply retry every request, you can make the situation worse.

I would use three mechanisms.

### A. Retry with exponential backoff

Retry temporary failures such as:

- HTTP 429 – rate limit.
- HTTP 500 – internal server error.
- HTTP 502 – bad gateway.
- HTTP 503 – service unavailable.
- HTTP 504 – gateway timeout.
- Network connection failures.

Example:

```
import asyncioimport randomasync def retry_with_backoff(    operation,    max_attempts=3,    base_delay=0.5,    max_delay=5.0):    for attempt in range(max_attempts):        try:            return await operation()        except (TimeoutError, ConnectionError):            if attempt == max_attempts - 1:                raise            delay = min(                base_delay * (2 ** attempt),                max_delay            )            delay += random.uniform(0, 0.2)            await asyncio.sleep(delay)
```

The retry intervals are approximately:

| Attempt | Delay       |
| ------- | ----------- |
| 1       | 0.5 seconds |
| 2       | 1 second    |
| 3       | 2 seconds   |

This is a simplified example. In production, I would also classify provider-specific exceptions, respect `Retry-After`, enforce an overall request deadline, and avoid retrying non-retryable errors.

Important: Do not retry every error.

For example:

- Invalid API key: Do not retry.
- Invalid request: Do not retry.
- Context window exceeded: Modify the request or select a suitable model.
- Rate limit: Retry with backoff, subject to the deadline.
- Provider unavailable: Try an eligible fallback provider.

### B. Circuit breaker pattern

This is particularly important for production AI applications.

Suppose OpenAI is unavailable. Instead of sending 10,000 requests to OpenAI and waiting for them to fail, we temporarily stop calling it.

A circuit breaker has three states:

CLOSED

Normal operation

Requests are allowed. Failures are counted.

Failure threshold reached

OPEN

Provider temporarily blocked

Requests fail fast or are routed to a fallback provider.

Recovery timeout expires

HALF-OPEN

Recovery testing

Allow a limited number of test requests. Success closes the circuit; failure opens it again.

Here is an implementation:

```
import asyncioimport timeclass CircuitBreakerOpen(Exception):    passclass CircuitBreaker:    def __init__(        self,        failure_threshold=5,        recovery_timeout=30    ):        self.failure_threshold = failure_threshold        self.recovery_timeout = recovery_timeout        self.failures = 0        self.opened_at = None        self.state = "CLOSED"        self.lock = asyncio.Lock()        self.probe_in_progress = False    async def call(self, operation):        async with self.lock:            if self.state == "OPEN":                if (                    time.monotonic() - self.opened_at                    < self.recovery_timeout                ):                    raise CircuitBreakerOpen()
```

This demonstrates the state transitions. For production, I would improve it by:

- Counting only eligible provider failures, not application validation errors.
- Maintaining separate breakers per provider and model.
- Using Redis or another shared state store across multiple workers.
- Handling cancellation and process restarts.
- Adding a bounded half-open probe policy.

An important architectural point: an in-memory circuit breaker is not sufficient when you have multiple FastAPI instances.

## 4. How do you implement automatic fallback between LLM providers?

Suppose your application supports:

| Priority | Provider          | Purpose              |
| -------- | ----------------- | -------------------- |
| 1        | OpenAI            | Primary model        |
| 2        | Anthropic         | Secondary model      |
| 3        | Self-hosted model | Last-resort fallback |

The routing logic should look like this:

```
Incoming request
       |
       v
   OpenAI call
       |
       +---- Success ----> Return response
       |
       +---- Failure
                |
                v
          Retry eligible errors
                |
                v
          Still failing?
                |
                v
         Check circuit breaker
                |
                v
          Anthropic call
                |
          +-----+-----+
          |           |
        Success     Failure
          |           |
          v           v
       Return     Self-hosted
                   model
                      |
                      v
                  Response
```

However, fallback is not simply choosing another model.

You must consider:

1. Whether the fallback model supports the same task.
2. Whether it supports structured JSON output.
3. Whether it supports tool calling.
4. Whether its context window is sufficient.
5. Whether its response quality meets your requirements.
6. Whether sending the data to another provider is allowed by your security policy.

For example, a financial compliance application should not automatically send confidential customer information to a different external provider unless that provider is approved for that data.

## 5. Complete LLM Gateway implementation

I would use the Strategy pattern for providers and a Gateway/Facade abstraction for application services.

Suggested project structure:

```
app/
│
├── llm/
│   ├── base.py
│   ├── exceptions.py
│   ├── circuit_breaker.py
│   ├── retry.py
│   ├── providers.py
│   └── gateway.py
│
├── services/
│   └── answer_service.py
│
├── api/
│   └── routes.py
│
└── main.py
```

### File 1: `base.py`

Define a common interface. Every provider implements this contract.

```
from typing import Protocolclass LLMProvider(Protocol):    async def generate(        self,        prompt: str,        model: str,        max_tokens: int    ) -> str:        ...
```

### File 2: `exceptions.py`

```
class LLMError(Exception):    passclass LLMTimeoutError(LLMError):    passclass LLMRateLimitError(LLMError):    passclass LLMUnavailableError(LLMError):    passclass LLMConfigurationError(LLMError):    passclass LLMInvalidResponseError(LLMError):    pass
```

### File 3: `providers.py`

The providers encapsulate their respective SDKs.

```
import asynciofrom openai import AsyncOpenAIfrom anthropic import AsyncAnthropicfrom app.llm.exceptions import (    LLMTimeoutError,    LLMRateLimitError,    LLMUnavailableError,    LLMConfigurationError)class OpenAIProvider:    def __init__(self, api_key):        self.client = AsyncOpenAI(            api_key=api_key,            timeout=10.0,            max_retries=0        )    async def generate(        self,        prompt,        model,        max_tokens    ):        try:            response = await self.client.chat.completions.create(                model=model,                messages=[                    {"role": "user", "content": prompt}
```

These adapters illustrate the approach. Before deploying, verify the SDK exception hierarchy and response schema for the pinned SDK versions. Production adapters should also normalize connection errors, cancellation, malformed responses and provider-specific timeout exceptions.

### File 4: `retry.py`

Implement retries with a maximum number of attempts and an overall timeout.

```
import asyncioimport randomfrom app.llm.exceptions import (    LLMTimeoutError,    LLMRateLimitError,    LLMUnavailableError)RETRYABLE_ERRORS = (    LLMTimeoutError,    LLMRateLimitError,    LLMUnavailableError)async def execute_with_retry(    operation,    max_attempts=2,    base_delay=0.5):    for attempt in range(max_attempts):        try:            return await operation()        except RETRYABLE_ERRORS:            if attempt == max_attempts - 1:                raise            delay = base_delay * (2 ** attempt)            await asyncio.sleep(
```

### File 5: `gateway.py`

This is the most important component. It coordinates the providers, retries, circuit breakers and fallback routing.

```
import asyncioimport loggingfrom app.llm.retry import execute_with_retryfrom app.llm.circuit_breaker import CircuitBreakerfrom app.llm.exceptions import (    LLMError,    LLMConfigurationError,    CircuitBreakerOpen)logger = logging.getLogger(__name__)class LLMGateway:    def __init__(self, providers):        self.providers = providers        self.breakers = {            name: CircuitBreaker(                failure_threshold=5,                recovery_timeout=30            )            for name in providers        }        self.models = {            "openai": "gpt-4.1-mini",            "anthropic": "claude-sonnet-4-5"        }    async def generate(
```

The gateway expects a mapping of provider names to provider instances, ordered by fallback priority.

For example:

```
providers = {    "openai": openai_provider,    "anthropic": anthropic_provider}gateway = LLMGateway(providers)
```

A few production improvements are important here:

- Keep provider configuration separate from business logic.
- Use a distributed circuit breaker when running multiple application instances.
- Add an overall deadline covering all providers, not just individual calls.
- Add provider-specific rate limiters and concurrency limits.
- Emit metrics for every provider attempt.
- Do not log prompts, customer information, API keys or confidential responses.

The code above is a working architectural starting point, not a complete production deployment. In particular, the in-memory breaker and sequential fallback need distributed coordination and deadline-aware routing at scale.

## 6. How do you handle model failures?

Provider outages are not the only problem.

Sometimes the provider returns HTTP 200, but the model produces an incorrect or unusable response.

For example:

Expected output:

```
{
    "risk_level": "HIGH",
    "reason": "Portfolio concentration risk"
}
```

Actual output:

```
Your portfolio seems risky because...
```

The API request succeeded, but the application failed to receive the required output.

I would implement response validation.

### A. Validate the response schema

```
from pydantic import BaseModelfrom typing import Literalclass RiskAssessment(BaseModel):    risk_level: Literal[        "LOW",        "MEDIUM",        "HIGH"    ]    reason: str
```

Validate:

```
import jsondef validate_response(raw_response: str):    try:        data = json.loads(raw_response)        return RiskAssessment.model_validate(data)    except Exception as exc:        raise ValueError(            "Invalid LLM response schema"        ) from exc
```

### B. Implement a controlled repair strategy

If validation fails:

1. Capture the validation error.
2. Retry once with a corrective instruction.
3. Validate the response again.
4. If it still fails, use a compatible fallback model.
5. If all attempts fail, return a controlled error or require human review.

Do not endlessly ask the model to correct itself.

### C. Add semantic validation

Schema validation only checks the structure, not whether the answer is correct.

For a RAG application, I would also check:

- Whether the answer is supported by retrieved documents.
- Whether citations refer to actual retrieved sources.
- Whether the answer violates business rules.
- Whether the answer contains unsupported financial claims.
- Whether the answer satisfies the task-specific confidence threshold.

For high-impact financial decisions, a model-generated answer should not independently authorize a transaction or make a final compliance decision.

## 7. What if all LLM providers fail?

This is where graceful degradation becomes important.

Imagine your financial advisor copilot depends on an LLM to generate investment research.

If OpenAI and Anthropic are both unavailable, I would not simply return a blank response.

Instead:

| Application functionality    | Degraded behavior                 |
| ---------------------------- | --------------------------------- |
| Portfolio dashboard          | Continue using database           |
| Historical reports           | Serve cached results              |
| Financial calculations       | Use deterministic Python services |
| Compliance rules             | Run deterministic rule engine     |
| New AI research              | Return temporary unavailable      |
| Critical financial decisions | Require human review              |

For example:

```
async def get_portfolio_summary(    portfolio_service,    llm_gateway):    portfolio = await portfolio_service.get_portfolio()    try:        answer = await llm_gateway.generate(            prompt=f"Summarize this portfolio: {portfolio}"        )        return {            "status": "success",            "summary": answer        }    except Exception:        return {            "status": "degraded",            "summary": None,            "portfolio": portfolio,            "message": (                "AI summary is temporarily unavailable. "                "Portfolio data is still accessible."            )        }
```

This is an example of graceful degradation: core application functionality remains available even when AI functionality is unavailable.

## 8. How do you monitor LLM latency and failures?

I would integrate OpenTelemetry and Prometheus/Grafana.

I would monitor the following metrics:

| Metric                         | What it tells us                |
| ------------------------------ | ------------------------------- |
| `llm_request_duration_seconds` | LLM response latency            |
| `llm_time_to_first_token`      | Streaming responsiveness        |
| `llm_error_rate`               | Percentage of failed requests   |
| `llm_fallback_rate`            | How frequently fallback is used |
| `llm_timeout_count`            | Number of timed-out requests    |
| `llm_token_usage`              | Input and output tokens         |
| `llm_cost`                     | Model expenditure               |
| `llm_circuit_state`            | Provider availability           |
| `llm_schema_failure_rate`      | Invalid structured responses    |

I would build dashboards showing:

Example production monitoring dashboard

# 2.4s

P95 latency

# 0.8%

Error rate

# 3.2%

Fallback rate

# 99.9%

Service availability

Illustrative metrics, not live production measurements.

I would also create alerts:

- P95 latency exceeds the defined SLO.
- Error rate increases beyond the threshold.
- Provider circuit breaker opens.
- Fallback usage increases unexpectedly.
- Token consumption suddenly increases.
- Model response validation failures increase.

## 9. Additional production-level considerations

There are several things that distinguish a senior engineer's implementation from a basic retry-and-fallback implementation.

| Problem                      | Production solution                  |
| ---------------------------- | ------------------------------------ |
| Duplicate requests           | Idempotency keys                     |
| Retry storms                 | Exponential backoff + jitter         |
| Too many concurrent requests | Semaphore + rate limiter             |
| Multiple workers             | Redis-backed coordination            |
| Provider outage              | Circuit breaker + fallback           |
| Long-running requests        | Deadlines + cancellation             |
| Invalid model output         | Schema and semantic validation       |
| High costs                   | Model routing + caching              |
| Sensitive data leakage       | Tenant isolation + provider policies |
| Worker crash                 | Durable job queue + checkpointing    |
| Partial workflow execution   | LangGraph checkpoint and recovery    |

### Important correction to the gateway example

In the gateway code, `asyncio.wait_for()` can raise `asyncio.TimeoutError`. To make the retry policy work consistently, normalize that exception:

```
async def call_provider():    try:        return await asyncio.wait_for(            provider.generate(                prompt=prompt,                model=self.models[provider_name],                max_tokens=max_tokens            ),            timeout=12        )    except asyncio.TimeoutError as exc:        raise LLMTimeoutError(            "Provider request timed out"        ) from exc
```

This ensures the retry mechanism recognizes the timeout as retryable.

Also, retries should have a shared overall deadline. A 12-second timeout per attempt across multiple providers can otherwise create an unexpectedly long user request.

## 10. How would you handle this in your LangGraph multi-agent application?

Consider your Enterprise Multi-Agent Financial Advisor Copilot.

You have:

- Planner agent
- Research agent
- Portfolio analysis agent
- Compliance agent
- Supervisor agent
- Final response generator

What happens if the Research agent's LLM provider fails?

I would handle failures at two levels.

Level 1: LLM-level recovery

The LLM Gateway handles:

- Retry.
- Timeout.
- Circuit breaker.
- Provider fallback.
- Model switching.

Level 2: Workflow-level recovery

LangGraph handles:

- Node execution state.
- Checkpointing.
- Retryable node failures.
- Workflow resumption.
- Human intervention.

Example:

```
from langgraph.graph import StateGraph, ENDasync def research_node(state):    try:        response = await llm_gateway.generate(            prompt=state["research_query"]        )        return {            "research_result": response,            "research_status": "completed"        }    except Exception as exc:        return {            "research_status": "failed",            "error": str(exc)        }def route_after_research(state):    if state["research_status"] == "completed":        return "portfolio_analysis"    return "fallback_or_review"workflow = StateGraph(dict)workflow.add_node("research", research_node)workflow.add_node(
```

This is a simplified graph illustrating failure routing. For durable production recovery, compile the graph with a persistent checkpointer, use stable thread IDs, and ensure external side effects are idempotent. Transient provider failures should normally be handled inside the gateway before the workflow decides that a node has permanently failed.

## 11. Interview-ready answer (what I would actually say)

If an interviewer asks this question, you can answer:

"I handle LLM latency and failures using a centralized LLM Gateway rather than integrating individual providers directly into business services.

First, I optimize latency through model selection, prompt optimization, token reduction, semantic caching, streaming and parallel execution.

Second, I implement resilience using timeouts, exponential backoff with jitter, rate limiting and circuit breakers. If the primary provider becomes unavailable, the gateway routes requests to an approved fallback provider.

Third, I validate model responses using Pydantic schemas and semantic validation, because an HTTP 200 response does not guarantee a correct AI response.

For multi-agent workflows, I separate LLM-level recovery from workflow-level recovery. The LLM Gateway handles provider failures, while LangGraph checkpointing allows interrupted workflows to resume.

Finally, I use OpenTelemetry, Prometheus and Grafana to monitor P95/P99 latency, token consumption, error rates, fallback frequency, provider availability and cost.

My main objective is to ensure that the application remains available, predictable, secure and cost-efficient even when an LLM provider or model fails."

## 12. Important follow-up interview questions

These are the questions I would prepare next for a Senior/Staff AI Engineer interview:

1. How do you implement a distributed circuit breaker using Redis across multiple FastAPI workers?
2. How do you prevent retry storms when an LLM provider experiences an outage?
3. How do you implement automatic model routing based on latency, cost and response quality?
4. How do you handle streaming when the primary LLM provider fails halfway through generating a response?
5. How do you ensure that switching from OpenAI to Anthropic does not change the output schema or break downstream agents?
6. How do you implement LLM request deduplication and idempotency?
7. How do you design an LLM Gateway capable of handling 10,000 concurrent requests?

My recommendation: Focus particularly on questions 1, 3 and 4. They test whether you understand distributed systems, production reliability and AI-specific failure handling rather than just knowing how to call an LLM API.