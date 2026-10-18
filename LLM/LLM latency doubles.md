# How to handle a sudden doubling of LLM latency — Senior AI Engineer approach

As a senior or staff-level AI engineer, you should treat a sudden doubling of LLM latency as a production incident that requires measurement, isolation, mitigation, and root-cause analysis.

You should not immediately assume that the LLM provider is slow or simply increase the timeout. The latency increase could originate from your application, retrieval pipeline, model provider, network, concurrency limits, or a recent deployment.

For example, suppose your AI service normally responds in 2 seconds, but suddenly starts taking 4 seconds. Your objective is to determine exactly which component added the extra 2 seconds, protect the user experience, and prevent the incident from recurring.

## 1. First, understand the latency metrics

Normal latency

## 2 seconds

Incident latency

## 4 seconds

Illustrative values. A 100% increase in latency does not necessarily mean the model itself became 100% slower.

Monitor these metrics separately:

| Metric                     | What it tells you                                  |
| -------------------------- | -------------------------------------------------- |
| p50 latency                | Typical user experience                            |
| p95 latency                | Slow experience for a significant minority         |
| p99 latency                | Extreme tail latency                               |
| Time to first token (TTFT) | How quickly streaming starts                       |
| Inter-token latency        | How quickly subsequent tokens arrive               |
| Input/output tokens        | Whether prompt size or response length changed     |
| Error and timeout rates    | Whether the slowdown is causing failures           |
| Throughput and concurrency | Whether demand exceeds capacity                    |
| Cost per request           | Whether the system is using more tokens or retries |

Important: Average latency alone is not enough. If p50 remains at 2 seconds but p99 rises from 5 to 12 seconds, you may have a queueing, concurrency, or tail-latency problem rather than a system-wide slowdown.

## 2. Follow a structured incident-response process

Step 1 — Detect and scope

Compare current p50/p95/p99 against baseline. Check when the incident started, which model and endpoints are affected, and whether errors or traffic increased.

Step 2 — Decompose the request

Break total latency into application, retrieval, queueing, model time-to-first-token, generation, and post-processing.

Step 3 — Mitigate customer impact

Reduce unnecessary retries, enable bounded fallbacks, use cached results when safe, and apply admission control if the system is overloaded.

Step 4 — Find the root cause

Correlate traces, model/token metrics, provider health, queue depth, deployments, and retrieval performance.

Step 5 — Verify and prevent recurrence

Confirm that latency and error rates recover, run regression tests, update alerts, and document the incident.

## 3. Decompose latency before changing anything

Imagine your financial-advisor RAG application has this request flow:

```
Client
  |
  v
FastAPI
  |
  v
Authentication + request validation
  |
  v
Query embedding
  |
  v
Qdrant retrieval + reranking
  |
  v
Prompt construction
  |
  v
LLM provider
  |
  v
Output validation + response
```

Instrument each stage independently. Otherwise, you only know that the request is slow, not why.

### Example: latency breakdown

| Component                     | Baseline | During incident |
| ----------------------------- | -------- | --------------- |
| Authentication + API overhead | 100 ms   | 110 ms          |
| Embedding + retrieval         | 300 ms   | 350 ms          |
| LLM call                      | 1,400 ms | 3,250 ms        |
| Validation + serialization    | 200 ms   | 290 ms          |
| Total                         | 2,000 ms | 4,000 ms        |

Here, the LLM call accounts for most of the increase. But you still need to separate provider queueing, TTFT, generation speed, token count, and network delay.

The opposite can also happen: the model may be healthy, but your application is waiting for a database connection or a semaphore.

### Root-cause decision table

| What changed?                        | Likely cause                                           | What I investigate                                |
| ------------------------------------ | ------------------------------------------------------ | ------------------------------------------------- |
| TTFT increases                       | Provider queueing, large prompt, network or cold start | Provider metrics, prompt size, request timing     |
| TTFT is normal, generation is slower | More output tokens, lower generation throughput        | Output token count and tokens/second              |
| Both TTFT and generation slow        | Provider capacity, throttling, model/runtime change    | Provider health, model version, rate limits       |
| Retrieval latency rises              | Slow vector search, embedding service, reranker        | Qdrant latency, query plans, reranker timing      |
| Latency rises with traffic           | Saturation or queueing                                 | Concurrency, worker utilization, connection pools |
| Only one endpoint is affected        | Endpoint-specific code or prompt change                | Recent commits, configuration and traces          |
| Retries and errors rise together     | Timeouts, rate limiting, retry amplification           | Retry count, 429/5xx responses, backoff           |

These are diagnostic signals, not definitive conclusions. Validate them against traces and controlled tests.

## 4. Implement distributed tracing

At staff level, I would make each AI request traceable across the API, retrieval system, and model provider.

For example, each trace should include spans such as:

```
ai.request                         4,000 ms
├── auth.validate                    50 ms
├── embedding.generate              100 ms
├── retrieval.search                150 ms
├── reranker.score                  200 ms
├── llm.generate                  3,250 ms
│   ├── provider.wait / TTFT        900 ms
│   └── generation                2,350 ms
└── output.validate                 250 ms
```

These numbers are illustrative. Provider-internal queue time is not always exposed as a separate metric, so you may only have client-side timing and TTFT.

### Python implementation with OpenTelemetry

Install the instrumentation libraries:

```
pip install \
  fastapi uvicorn openai \
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-instrumentation-fastapi
```

Create `observability/tracing.py`:

```
from opentelemetry import tracefrom opentelemetry.sdk.resources import Resourcefrom opentelemetry.sdk.trace import TracerProviderfrom opentelemetry.sdk.trace.export import (    BatchSpanProcessor,    ConsoleSpanExporter,)resource = Resource.create({    "service.name": "financial-advisor-api",    "deployment.environment": "production",})provider = TracerProvider(resource=resource)provider.add_span_processor(    BatchSpanProcessor(ConsoleSpanExporter()))trace.set_tracer_provider(provider)tracer = trace.get_tracer("financial-advisor")
```

The console exporter is for local development. In production, export traces to your approved OpenTelemetry Collector and tracing backend.

Now instrument the model call:

```
import timefrom openai import AsyncOpenAIfrom opentelemetry import traceclient = AsyncOpenAI()tracer = trace.get_tracer("financial-advisor")async def generate_answer(prompt: str) -> str:    with tracer.start_as_current_span("llm.generate") as span:        start = time.perf_counter()        try:            response = await client.chat.completions.create(                model="YOUR_CONFIGURED_MODEL",                messages=[                    {"role": "user", "content": prompt}                ],                max_completion_tokens=500,            )            elapsed_ms = (                time.perf_counter() - start            ) * 1000            span.set_attribute(                "llm.latency_ms", elapsed_ms            )            span.set_attribute(                "llm.input_tokens",                response.usage.prompt_tokens,            )            span.set_attribute(                "llm.output_tokens",                response.usage.completion_tokens,            )            span.set_attribute(                "llm.model", response.model            )            return response.choices[0].message.content or ""        except Exception:            span.record_exception(                __import__("sys").exception()            )            span.set_status(                trace.Status(                    trace.StatusCode.ERROR                )            )            raise
```

Use the following to instrument the FastAPI application:

```
from fastapi import FastAPIfrom opentelemetry.instrumentation.fastapi import (    FastAPIInstrumentor,)app = FastAPI()FastAPIInstrumentor.instrument_app(app)
```

This is a starting point. Production code should also configure secure exporters, request-level metrics, exception classification, and redaction policies.

Security rule: never attach full prompts, document contents, access tokens, customer identifiers, or sensitive financial information to traces by default. Record safe metadata, token counts, latency, model version, and correlation IDs instead.

One more important detail: the non-streaming example measures the full model request duration. For streaming APIs, measure TTFT when the first token arrives and generation duration separately.

## 5. Mitigate the incident without making it worse

While the investigation is running, I would protect the service and reduce user impact.

### A. Control retries and timeouts

A common mistake is increasing timeouts and allowing every slow request to retry several times. This can make an overloaded provider even slower.

For example:

- 1,000 incoming requests
- Each request retries three times
- Potentially 4,000 model attempts instead of 1,000

Use bounded retries, exponential backoff, jitter, and an overall request deadline.

```
import asyncioimport randomimport timefrom openai import (    AsyncOpenAI,    APITimeoutError,    APIConnectionError,    RateLimitError,    InternalServerError,)client = AsyncOpenAI(    timeout=8.0,    max_retries=0,  # Application owns retry policy)RETRYABLE = (    APITimeoutError,    APIConnectionError,    RateLimitError,    InternalServerError,)async def call_llm(prompt: str) -> str:    deadline = time.monotonic() + 10.0    max_attempts = 3    for attempt in range(max_attempts):        remaining = deadline - time.monotonic()        if remaining <= 0:            raise TimeoutError("LLM deadline exceeded")        try:            async with asyncio.timeout(remaining):                response = await client.chat.completions.create(                    model="YOUR_CONFIGURED_MODEL",                    messages=[                        {"role": "user", "content": prompt}                    ],                    max_completion_tokens=400,                    timeout=min(6.0, remaining),                )            return response.choices[0].message.content or ""        except RETRYABLE:            if attempt == max_attempts - 1:                raise            remaining = deadline - time.monotonic()            delay = min(0.25 * (2 ** attempt), 1.0)            delay += random.uniform(0, 0.1)            if delay >= remaining:                raise TimeoutError(                    "Insufficient deadline for retry"                )            await asyncio.sleep(delay)    raise RuntimeError("Unexpected retry termination")
```

This is illustrative production-oriented code, not a complete retry framework.

Key engineering considerations:

- Retry only transient failures, not malformed requests or authorization failures.
- Respect provider `Retry-After` guidance when available.
- Keep retries within the overall deadline.
- Use one clearly defined retry owner to avoid nested retry multiplication.
- Ensure the request is safe to retry; non-idempotent tool actions require additional protections.
- Check the SDK's timeout and retry semantics for the installed version.

A timeout limits how long your application waits. It does not guarantee that remote processing stops immediately.

### B. Apply concurrency limits and admission control

If traffic exceeds available model capacity, requests may queue and latency may double.

Use a semaphore or a bounded work queue to control in-flight requests. A semaphore is a simple local protection; in a multi-replica deployment, it does not enforce a global limit.

```
import asynciollm_slots = asyncio.Semaphore(30)async def bounded_llm_call(prompt: str) -> str:    try:        async with asyncio.timeout(0.5):            await llm_slots.acquire()    except TimeoutError:        raise RuntimeError(            "LLM capacity exhausted; retry later"        )    try:        return await call_llm(prompt)    finally:        llm_slots.release()
```

In this example, the 500 ms limit applies to acquiring a slot, not to the LLM call itself.

At scale, I would additionally consider:

- Per-tenant concurrency limits and fair scheduling.
- Bounded queues and load shedding.
- Provider rate-limit-aware admission control.
- Autoscaling, while accounting for provider quotas.
- Circuit breakers to stop repeatedly calling an unhealthy dependency.

Do not just raise the concurrency limit. If the provider is saturated, that can increase queueing and throttling.

### C. Use fallbacks intelligently

If the primary model is unhealthy, a secondary model may preserve availability.

```
async def generate_with_fallback(prompt: str) -> str:    try:        return await call_primary_model(prompt)    except (TimeoutError, APITimeoutError):        return await call_secondary_model(prompt)
```

The two functions here are placeholders for your model adapters.

A production fallback should have its own deadline, concurrency limit, health check, and cost budget. Avoid triggering a fallback for every slow request: that may overload both providers.

For a financial-advisor application, a fallback model must meet the same minimum security and answer-quality requirements. If it cannot, return a clear temporary-unavailability response instead of an unreliable financial answer.

### D. Reduce work per request

If the model is slow because the request became more expensive, optimize the workload.

| Optimization                            | When it helps                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------- |
| Reduce prompt size                      | Excessive system prompts or retrieved context                                   |
| Reduce output token budget              | Long answers and unnecessary generation                                         |
| Use a smaller model                     | Tasks that do not require the strongest model                                   |
| Cache deterministic results             | Repeated, safe-to-cache queries                                                 |
| Parallelize independent retrieval calls | Independent operations are unnecessarily sequential                             |
| Remove unnecessary agents               | Multi-agent workflows perform redundant model calls                             |
| Stream output                           | Improves perceived responsiveness, though not necessarily total completion time |

For RAG, inspect retrieved chunk count, reranking cost, prompt token count, and whether the same context is being sent repeatedly.

Do not indiscriminately truncate evidence, bypass authorization filters, or cache personalized financial answers across users.

## 6. Connect the evidence to the correct fix

Here is how I would make the decision during the incident.

### Latency triage decision tree

LLM request latency doubled

Compare traces and model metrics

Model time increased

Check provider health, TTFT, output tokens, throttling, model version.

Mitigate: bounded retries, fallback, smaller model where safe.

Application time increased

Check DB pools, retrieval, embedding, CPU, event-loop blocking.

Mitigate: fix bottleneck, bound queues, remove unnecessary work.

Verify recovery and prevent recurrence

Compare p95/p99, error rates, throughput, cost, and answer quality.

## 7. What should you alert on?

I would use a combination of latency, reliability, and saturation alerts—not a single fixed latency threshold.

| Signal                      | Example alert condition                         |
| --------------------------- | ----------------------------------------------- |
| p95 end-to-end latency      | Above the endpoint's SLO for a sustained window |
| p99 latency                 | Significant tail-latency regression             |
| Error/timeout rate          | Error-budget burn or sustained increase         |
| Provider 429 responses      | Rate limiting above expected baseline           |
| Queue wait                  | Admission queue approaching its deadline        |
| Token throughput            | Generation throughput drops materially          |
| Fallback rate               | Secondary model usage increases unexpectedly    |
| Cost per successful request | Sudden increase after retries or token growth   |

Set the actual thresholds using historical traffic, business requirements, and the endpoint's service-level objective (SLO). Alert on sustained conditions to reduce noise, and use faster paging for severe user-facing failures.

For a multi-agent application, monitor the whole workflow and individual agent calls. A workflow might become slow because agents execute sequentially, a planner repeats its reasoning, or an evaluator triggers too many rework loops—even when each individual model call is healthy.

## 8. Staff-level prevention strategy

After resolving the incident, I would focus on preventing the same failure.

1. Baseline and capacity testing: Load-test realistic prompts, context lengths, output sizes, and concurrency levels.
2. Canary deployments: Compare latency, errors, and quality before rolling out model, prompt, or retrieval changes globally.
3. Model routing: Route simple tasks to cheaper, faster models and complex tasks to stronger models, with explicit quality requirements.
4. Resilience controls: Establish deadlines, bounded queues, circuit breakers, retry budgets, and safe fallbacks.
5. Observability: Correlate traces with model version, deployment, token counts, retrieval metrics, and provider status.
6. Regression testing: Maintain a representative evaluation set so latency optimizations do not silently damage answer quality.
7. Incident review: Record impact, timeline, root cause, mitigation, and assigned corrective actions.

## 9. How to answer this in a senior AI engineer interview

A strong answer would sound like this:

> "If LLM latency suddenly doubles, I treat it as a production incident. First, I establish the scope using p50, p95, p99, TTFT, throughput, error rates, and token usage. I then decompose end-to-end latency using distributed traces to isolate application overhead, retrieval, queueing, provider time, generation, and post-processing.
>
> While investigating, I protect the service with bounded retries, deadlines, concurrency control, and a circuit breaker. If appropriate, I route to a validated fallback model or serve a safe cached response. I avoid blindly increasing timeouts or concurrency because those actions can amplify overload.
>
> Next, I correlate the incident with traffic patterns, rate limits, prompt-size changes, model deployments, retrieval latency, and provider health. Once the bottleneck is identified, I apply the targeted fix and validate recovery against the latency SLO, error budget, throughput, cost, and answer-quality metrics.
>
> Finally, I add regression tests, capacity tests, and deployment safeguards to prevent recurrence. My priority is to restore service safely, identify the actual bottleneck, and make the system more resilient—not merely to make the timeout larger."

The key distinction: a mid-level engineer may focus on optimizing one slow model call. A senior or staff engineer investigates the entire request path, manages the incident, protects service-level objectives, considers cost and answer quality, and addresses the systemic cause.