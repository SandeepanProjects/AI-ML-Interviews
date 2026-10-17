# Staff AI Engineer Guide: Tracing LLM Calls, Monitoring Tokens, Latency, and Cost

In a production AI system, monitoring an LLM call is not as simple as measuring how long an HTTP request takes. A single user request may trigger a planner, multiple agents, several retrieval operations, tool calls, retries, and multiple LLM calls.

As a staff-level AI engineer, you need to answer four questions for every request:

1. Tracing: What happened, in what order, and which component caused a failure?
2. Token monitoring: How many input, output, cached and reasoning tokens were consumed?
3. Latency monitoring: Where did the time go, and which component is the bottleneck?
4. Cost monitoring: How much did the request cost, which model incurred the cost, and how can we reduce it?

You also need to connect those measurements to quality, reliability, privacy and business outcomes.

I'll use a production-style example based on a financial research copilot built with FastAPI, an LLM provider, RAG, and LangGraph-style agent orchestration. The same principles apply to other agentic AI systems.

## 1. Understand the observability architecture

Client request

Request ID, tenant ID, authenticated user context

FastAPI service

Request span, endpoint latency, status, queue time

Agent / RAG workflow

Planner → retrieval → tools → specialist agents → synthesis

LLM instrumentation

Provider call, model, tokens, retries, latency, status, estimated cost

Traces

Request → workflow → LLM spans

Metrics

Tokens, latency, errors, cost

Structured logs

Correlation IDs and operational events

Governance

Redaction, retention, access controls

These four observability outputs are complementary:

| Mechanism         | What it tells you                | Example                                   |
| ----------------- | -------------------------------- | ----------------------------------------- |
| Distributed trace | What happened across components  | Why did this request take 8 seconds?      |
| Metrics           | How the system behaves over time | Why did p95 latency increase?             |
| Structured logs   | Details about a specific event   | Which provider error occurred?            |
| Cost ledger       | Financial attribution            | Which tenant consumed the most LLM spend? |

A common implementation stack is:

- OpenTelemetry (OTel): trace context, spans and metrics instrumentation.
- OpenInference or provider/framework instrumentation: LLM-specific spans and metadata.
- Prometheus + Grafana: metrics and dashboards.
- Jaeger, Grafana Tempo or an OTel-compatible backend: trace storage and exploration.
- LangSmith or another LLM observability platform: agent and LangGraph execution inspection, depending on your stack.
- A database or warehouse: durable cost and usage reporting, reconciled with provider billing data.

You do not need every product. You need the capabilities, consistent identifiers and a clear data-retention policy.

## 2. LLM tracing: how to follow a request end to end

### 2.1 What is a trace, a span and a correlation ID?

Suppose a user asks:

> Analyze the risk of a particular investment and cite the relevant internal research.

Your workflow may execute these operations:

Trace: `trace-abc`

Entire user request

Span 1 — API request

120 ms

Span 2 — Agent workflow

6,200 ms

Span 3 — Vector retrieval

180 ms

Span 4 — Planner LLM

1,400 ms

Span 5 — Search tool

800 ms

Span 6 — Synthesis LLM

2,100 ms

Illustrative timings only. Parallel spans can overlap, so child durations should not simply be added together.

- Trace: the complete execution for a request.
- Span: one timed operation inside the trace, with a start, end, status, attributes and possibly child spans.
- Trace ID: identifies the whole trace.
- Span ID: identifies an individual operation.
- Correlation/request ID: an application-level identifier used to connect traces, logs, and business records. It can be the same as a trace identifier, but the concepts are not identical.

In an agentic system, the most important design principle is to preserve the parent-child relationships across the API, orchestration framework, retrieval layer, tools and provider calls.

### 2.2 What should an LLM span capture?

For every provider call, capture operational metadata such as:

```
{
  "operation": "llm.generate",
  "provider": "example-provider",
  "model": "example-model",
  "trace_id": "trace-abc",
  "request_id": "req-123",
  "agent_name": "research_agent",
  "input_tokens": 2400,
  "output_tokens": 420,
  "duration_ms": 1420,
  "retry_count": 0,
  "status": "ok"
}
```

This is a schema example, not a live call.

Useful attributes include:

- Provider, model and model version when available.
- Operation type: completion, chat, embedding, reranking or other inference.
- Input/output token counts, plus cached or reasoning token details when reported.
- Start time, duration, status and error category.
- Retry attempt and timeout information.
- Workflow, agent and tool names.
- Prompt-template version and retrieval configuration version.
- Request, trace and tenant identifiers, subject to privacy controls.
- Provider request ID for troubleshooting provider-side incidents.

Do not put API keys, raw authentication headers, full customer prompts, confidential documents or unrestricted model responses into telemetry by default. Prompt and response capture should be an explicit, access-controlled decision.

### 2.3 Implement tracing with OpenTelemetry

Install the basic dependencies:

```
pip install \
  fastapi uvicorn \
  openai \
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-otlp \
  opentelemetry-instrumentation-fastapi
```

Create `observability.py`:

```
from opentelemetry import tracefrom opentelemetry.sdk.resources import Resourcefrom opentelemetry.sdk.trace import TracerProviderfrom opentelemetry.sdk.trace.export import BatchSpanProcessorfrom opentelemetry.exporter.otlp.proto.http.trace_exporter import (    OTLPSpanExporter,)resource = Resource.create({    "service.name": "financial-ai-api",    "service.version": "1.0.0",    "deployment.environment.name": "production",})provider = TracerProvider(resource=resource)provider.add_span_processor(    BatchSpanProcessor(        OTLPSpanExporter(            # Configure the endpoint and credentials for your            # actual OTel collector/backend.        )    ))trace.set_tracer_provider(provider)tracer = trace.get_tracer("financial_ai")
```

The exporter must be configured for your OTel collector. Do not leave an unconfigured exporter in production; use environment-based configuration and verify that the backend receives spans.

Now instrument an LLM call:

```
from openai import AsyncOpenAIfrom opentelemetry import traceimport timeclient = AsyncOpenAI()tracer = trace.get_tracer("financial_ai")async def generate_answer(prompt: str) -> dict:    start = time.perf_counter()    with tracer.start_as_current_span("llm.generate") as span:        span.set_attribute("llm.provider", "openai")        span.set_attribute("llm.operation", "chat")        span.set_attribute("llm.model", "YOUR_MODEL_NAME")        try:            response = await client.chat.completions.create(                model="YOUR_MODEL_NAME",                messages=[                    {                        "role": "system",                        "content": "Answer accurately and concisely.",                    },                    {"role": "user", "content": prompt},                ],            )            usage = response.usage            elapsed_ms = (time.perf_counter() - start) * 1000            input_tokens = usage.prompt_tokens if usage else None            output_tokens = usage.completion_tokens if usage else None            span.set_attribute("llm.duration_ms", elapsed_ms)
```

Important production notes:

- Replace `YOUR_MODEL_NAME` with the model configured for your environment.
- The example uses the OpenAI Chat Completions API; other providers and APIs expose usage differently.
- Avoid attaching raw prompts or exception messages to spans without a reviewed redaction policy.
- Use provider/framework instrumentation where possible to capture retries, streaming, tool calls and token metadata consistently. A hand-written span only captures what your code explicitly records.
- Ensure the SDK and exporter are initialized once at application startup, not on every request.

### 2.4 Instrument the FastAPI request boundary

```
from fastapi import FastAPIfrom opentelemetry.instrumentation.fastapi import (    FastAPIInstrumentor,)app = FastAPI()FastAPIInstrumentor.instrument_app(app)
```

This creates HTTP-level tracing for the FastAPI application. Your service and agent code should add nested spans for work that automatic instrumentation cannot see.

For distributed execution, propagate W3C Trace Context across supported HTTP calls and queue messages. A Celery worker, for example, may execute after the original HTTP request has returned; propagate the trace context explicitly through the task headers if you want one connected trace.

Staff-level point: a trace is only useful if context survives asynchronous boundaries. Otherwise, you end up with unrelated traces for the API, worker, retrieval service and model provider.

## 3. Token usage monitoring

### 3.1 What are you actually counting?

A production LLM system may report several token categories.

| Token category      | Meaning                                                                  | Why it matters                           |
| ------------------- | ------------------------------------------------------------------------ | ---------------------------------------- |
| Input tokens        | Tokens in the submitted input                                            | Prompt size, context and input charges   |
| Output tokens       | Tokens generated by the model                                            | Generation cost and output length        |
| Cached input tokens | Input tokens served through a provider's caching mechanism               | May have a different price               |
| Reasoning tokens    | Provider-reported tokens used for internal reasoning on supported models | May affect output usage and cost         |
| Embedding tokens    | Tokens processed when generating embeddings                              | Often a separate pricing category        |
| Reranking usage     | Usage for a reranking API, if applicable                                 | May be priced per request, item or token |

Exact usage fields and billing rules depend on the provider, API, model and pricing plan. Do not assume that every provider exposes reasoning or cached token counts in the same way.

### 3.2 Why aggregate token usage is not enough

Suppose one user request invokes the model four times:

| Call                | Input tokens | Output tokens |
| ------------------- | ------------ | ------------- |
| Planner             | 1,200        | 150           |
| Research agent      | 2,500        | 400           |
| Tool-selection call | 800          | 80            |
| Final synthesis     | 4,000        | 600           |
| Total               | 8,500        | 1,230         |

The final response may contain only 600 generated tokens, but the full workflow consumed 9,730 input-plus-output tokens across its model calls.

This is why you should measure at two levels:

- Per LLM call: identify expensive prompts, agents and retries.
- Per end-user request: calculate the sum across the complete workflow.

Also track workflow dimensions such as tenant, feature, agent, model and success status. Keep high-cardinality identifiers such as request IDs in traces or logs, not as Prometheus metric labels.

### 3.3 Capture usage from the provider response

The previous example reads token counts from `response.usage`. In production, centralize this in an LLM wrapper or instrumentation callback so every call follows the same path.

A provider-neutral internal usage model might look like:

```
from dataclasses import dataclassfrom typing import Optional@dataclass(frozen=True)class TokenUsage:    input_tokens: Optional[int]    output_tokens: Optional[int]    cached_input_tokens: Optional[int] = None    reasoning_tokens: Optional[int] = None    @property    def total_input_output_tokens(self) -> Optional[int]:        if self.input_tokens is None or self.output_tokens is None:            return None        return self.input_tokens + self.output_tokens
```

`None` means the count is unavailable. Do not silently turn missing usage into zero: that can make dashboards and cost reports look healthier than they really are.

For a specific provider, build a small adapter that translates its response into this internal model. Preserve the raw provider usage structure in restricted diagnostic data if needed, while exposing normalized fields to your monitoring pipeline.

### 3.4 What if usage is missing?

There are three common methods:

1. Authoritative usage: read the usage object returned by the provider.
2. Token estimation: use a tokenizer compatible with the selected model or an approximate estimator before or after the call.
3. Billing reconciliation: compare accumulated usage with provider billing exports or usage APIs.

Use estimation for admission control and early budgeting, but do not treat it as authoritative billing. Tokenization differs across model families, and hidden formatting, tools, images, audio, cached input and provider-specific billing rules can affect the actual bill.

For agentic systems, record token usage for failed attempts too. A retry that returns an error can still consume tokens and incur charges.

## 4. Latency monitoring: measure more than total response time

### 4.1 The latency metrics that matter

A common mistake is to log only the duration of the final LLM call. In a multi-agent application, the end user experiences the entire workflow.

| Metric                     | Definition                             | Why it matters                        |
| -------------------------- | -------------------------------------- | ------------------------------------- |
| End-to-end latency         | Request arrival to completion          | User experience                       |
| LLM call latency           | One provider call start to completion  | Model performance and provider delays |
| Time to first token (TTFT) | Request start to first generated token | Streaming responsiveness              |
| Inter-token latency        | Time between streamed tokens           | Smoothness of generation              |
| Retrieval latency          | Time spent fetching context            | RAG bottlenecks                       |
| Tool latency               | Time spent in an external tool         | Search, database or API delays        |
| Queue wait time            | Time waiting before execution          | Worker capacity and backpressure      |
| Retry overhead             | Additional time due to retries         | Reliability and latency amplification |
| p50, p95, p99              | Latency percentiles                    | Typical versus tail performance       |

Mean latency alone is insufficient. If average latency is 1.5 seconds but p99 is 18 seconds, some users are experiencing a much slower system.

### 4.2 Example: identify the bottleneck

Imagine a request has the following sequential components:

Illustrative request latency breakdown

Sequential components of a 6.2-second request.

0 ms600 ms1.2K ms1.8K ms2.4K msRetrievalPlanner LLMSearch toolSynthesis LLMOther

The synthesis call is the largest single component, but `Other` is also substantial. A trace can reveal whether that time came from orchestration, queueing, serialization or another hidden operation.

This is a common staff-level workflow:

1. Identify the affected service or request class from metrics.
2. Use a trace to find the slow spans.
3. Compare slow traces with normal traces.
4. Identify the cause—for example, excessive prompt context, queue saturation, a slow tool or a retry.
5. Change one component and verify the improvement with a controlled comparison.

### 4.3 Measure the full duration correctly

For a non-streaming call, use a monotonic clock such as `time.perf_counter()` to measure elapsed time:

```
import timestart = time.perf_counter()# await call_llm(...)duration_ms = (time.perf_counter() - start) * 1000
```

For streaming, measure at least:

```
request_started_at
provider_request_started_at
first_token_received_at
last_token_received_at
```

Then derive:

```
provider_setup_ms = provider_request_started_at - request_started_at
ttft_ms = first_token_received_at - provider_request_started_at
generation_ms = last_token_received_at - first_token_received_at
```

The exact definitions must be consistent across dashboards. For example, one team might define TTFT from the user's request arrival, while another defines it from the provider request start. Both can be useful, but they are not interchangeable.

For parallel agent execution, do not sum every span's duration to estimate end-to-end latency. Parallel calls overlap. Use the trace timeline to understand the critical path.

### 4.4 Retries can hide latency problems

Consider this sequence:

```
LLM attempt 1: 2.0 seconds → timeout
Backoff:       0.5 seconds
LLM attempt 2: 1.8 seconds → success
```

The user sees at least 4.3 seconds before accounting for orchestration and other overhead. If you record only the successful attempt, your dashboard will understate the actual request duration and miss a reliability issue.

Record the duration, status and retry count for every attempt. Use bounded retries, timeouts, jittered backoff and retry rules that distinguish transient errors from permanent errors. For non-idempotent tool operations, do not blindly retry without an idempotency strategy.

## 5. Cost monitoring: calculate cost per call and per request

### 5.1 The basic calculation

For a model billed per million tokens:

\\[ \text{Cost} = \frac{I \times P_i + O \times P_o}{1{,}000{,}000} \\]

Where:

- \\(I\\) = billable input tokens
- \\(O\\) = billable output tokens
- \\(P_i\\) = input price per million tokens
- \\(P_o\\) = output price per million tokens

This basic equation must be adapted when the provider has separate rates for cached input, reasoning tokens, images, audio, batch processing or other billing dimensions.

### 5.2 A configurable cost calculator

Do not hard-code one model's prices throughout your application. Use a centrally maintained, versioned pricing configuration.

```
from dataclasses import dataclassfrom decimal import Decimal@dataclass(frozen=True)class TokenPricing:    input_per_million: Decimal    output_per_million: Decimal    cached_input_per_million: Decimal = Decimal("0")def calculate_llm_cost(    *,    input_tokens: int,    output_tokens: int,    pricing: TokenPricing,    cached_input_tokens: int = 0,) -> Decimal:    if min(input_tokens, output_tokens, cached_input_tokens) < 0:        raise ValueError("Token counts cannot be negative")    if cached_input_tokens > input_tokens:        raise ValueError("Cached input cannot exceed total input")    regular_input = input_tokens - cached_input_tokens    million = Decimal("1000000")    regular_input_cost = (        Decimal(regular_input)        * pricing.input_per_million        / million    )    cached_input_cost = (        Decimal(cached_input_tokens)        * pricing.cached_input_per_million        / million    )    output_cost = (        Decimal(output_tokens)        * pricing.output_per_million        / million    )    return regular_input_cost + cached_input_cost + output_cost# EXAMPLE RATES ONLY — not live provider pricing.example_pricing = TokenPricing(    input_per_million=Decimal("2.00"),    output_per_million=Decimal("8.00"),    cached_input_per_million=Decimal("0.50"),)cost = calculate_llm_cost(    input_tokens=8_500,    output_tokens=1_230,    cached_input_tokens=2_000,    pricing=example_pricing,)print(f"Estimated cost: ${cost:.6f}")
```

This function illustrates a simplified cached-input pricing model. The provider's actual billing contract may define cached tokens as a subset of input tokens, and other token categories may have separate rates or rules. Adapt the calculation to that provider's official usage and pricing schema. Do not count cached tokens twice.

### 5.3 Why cost needs a ledger

Metrics are excellent for trends and alerting, but a financial or enterprise application often needs durable usage records.

A usage event could contain:

```
{
  "request_id": "req-123",
  "trace_id": "trace-abc",
  "tenant_id": "tenant-42",
  "feature": "investment_research",
  "agent": "research_agent",
  "provider": "example-provider",
  "model": "example-model",
  "input_tokens": 2400,
  "output_tokens": 420,
  "cached_input_tokens": 300,
  "estimated_cost_usd": "0.004500",
  "pricing_version": "2026-10-01",
  "status": "success"
}
```

These values are illustrative. The event should be generated from the actual provider response and the pricing configuration applicable to that usage.

For an enterprise implementation, persist usage events to a durable store, such as PostgreSQL or a usage-event pipeline, with a unique event ID. Make ingestion idempotent to avoid duplicate charges in internal reporting when an event is retried.

Track at least:

- Cost per LLM call
- Cost per user request
- Cost per workflow, agent and feature
- Cost per tenant or team, with access controls
- Daily and monthly spend
- Failed-call and retry spend
- Estimated cost versus reconciled provider billing

Important: your calculated cost is an estimate unless it matches the provider's full billing rules. Reconcile it against provider billing data, and account for taxes, credits, negotiated rates, batch discounts or other commercial adjustments where relevant.

## 6. Implement production metrics with Prometheus

OpenTelemetry traces help investigate individual requests. Metrics help answer questions about the whole service over time.

Install:

```
pip install prometheus-client
```

Create `llm_metrics.py`:

```
from prometheus_client import Counter, HistogramLLM_CALLS = Counter(    "llm_calls_total",    "Number of LLM calls",    ["provider", "model", "operation", "status"],)LLM_INPUT_TOKENS = Counter(    "llm_input_tokens_total",    "Input tokens reported by the provider",    ["provider", "model"],)LLM_OUTPUT_TOKENS = Counter(    "llm_output_tokens_total",    "Output tokens reported by the provider",    ["provider", "model"],)LLM_COST_USD = Counter(    "llm_estimated_cost_usd_total",    "Estimated LLM cost in USD",    ["provider", "model"],)LLM_CALL_DURATION = Histogram(    "llm_call_duration_seconds",    "LLM call duration in seconds",    ["provider", "model", "operation"],    buckets=(        0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30    ),)REQUEST_DURATION = Histogram(
```

Record the observations around each provider call:

```
import timefrom llm_metrics import (    LLM_CALLS,    LLM_INPUT_TOKENS,    LLM_OUTPUT_TOKENS,    LLM_CALL_DURATION,    LLM_COST_USD,)async def observed_llm_call(    *,    provider: str,    model: str,    operation: str,    invoke,    pricing,):    start = time.perf_counter()    status = "error"    try:        response = await invoke()        # This extractor is for the OpenAI Chat Completions        # response shape. Other APIs require their own adapter.        usage = response.usage        if usage is None:            raise RuntimeError("Provider returned no usage metadata")        input_tokens = usage.prompt_tokens        output_tokens = usage.completion_tokens        # This simplified example assumes there are no cached        # input tokens. Production code should normalize the        # provider's detailed usage fields first.        cost = calculate_llm_cost(            input_tokens=input_tokens,            output_tokens=output_tokens,            pricing=pricing,        )        LLM_INPUT_TOKENS.labels(            provider, model        ).inc(input_tokens)        LLM_OUTPUT_TOKENS.labels(            provider, model        ).inc(output_tokens)        LLM_COST_USD.labels(            provider, model        ).inc(float(cost))        status = "success"        return response    finally:        duration = time.perf_counter() - start        LLM_CALL_DURATION.labels(            provider, model, operation        ).observe(duration)        LLM_CALLS.labels(            provider, model, operation, status        ).inc()
```

Usage:

```
response = await observed_llm_call(    provider="openai",    model="YOUR_MODEL_NAME",    operation="chat",    invoke=lambda: client.chat.completions.create(        model="YOUR_MODEL_NAME",        messages=[            {"role": "user", "content": "Summarize this report."}        ],    ),    pricing=example_pricing,)
```

This is an instrumentation example, not a complete production wrapper. In particular, a missing usage object should not be converted into a real zero-cost event, and a monitoring failure should not accidentally cause a successful LLM operation to be reported as a provider failure. Production code should distinguish model execution failures from telemetry failures and record missing usage explicitly.

For HTTP metrics, expose the Prometheus endpoint using the appropriate ASGI integration or a dedicated metrics endpoint, then configure Prometheus to scrape it. Keep the metrics endpoint protected according to your infrastructure's security model.

### Critical Prometheus design rule: avoid high-cardinality labels

Good metric labels:

- `provider`
- `model` (a bounded set)
- `operation`
- `status`
- `environment`

Dangerous metric labels:

- `request_id`
- `trace_id`
- Full prompt text
- User email
- Arbitrary tool arguments
- Unbounded tenant identifiers in a large multi-tenant system

High-cardinality labels can overwhelm the metrics backend. Put request-level identifiers in traces and structured logs instead. For tenant-level billing, use a ledger or analytics store rather than generating a new time series for every tenant by default.

## 7. Connect the measurements to an agentic workflow

Suppose your LangGraph workflow contains:

```
START
  ↓
Planner LLM
  ↓
Document retrieval
  ↓
Research agent LLM
  ↓
Search tool
  ↓
Risk analysis LLM
  ↓
Final synthesis LLM
  ↓
END
```

You need to observe both the whole workflow and each component.

For every LLM call, record:

```
trace_id
workflow_run_id
agent_name
provider
model
call_index
attempt_number
input_tokens
output_tokens
duration_ms
status
estimated_cost
```

For non-LLM components, record their own operational data:

```
retrieval_latency_ms
documents_retrieved
reranker_latency_ms
tool_latency_ms
tool_status
queue_wait_ms
```

### Instrument LangGraph

LangGraph supports streaming execution events and can be integrated with tracing systems. If you use a supported LLM and framework combination, framework-specific instrumentation may automatically capture nested operations.

For example, when using LangChain/LangGraph, a LangSmith tracing integration can show graph nodes, model calls and tool executions. Alternatively, OpenTelemetry can be used when you want traces in your broader platform observability system.

The exact configuration depends on your LangGraph, LangChain, provider and instrumentation versions. Do not assume a generic OTel span around `graph.ainvoke()` automatically captures each nested LLM call. You may need callbacks, provider instrumentation or explicit spans around node operations.

Conceptually:

```
async def run_workflow(graph, state, tracer):    with tracer.start_as_current_span(        "agent.workflow"    ) as span:        span.set_attribute("workflow.name", "financial_research")        try:            result = await graph.ainvoke(state)            span.set_attribute("workflow.status", "success")            return result        except Exception as exc:            span.record_exception(exc)            span.set_status(                trace.Status(trace.StatusCode.ERROR)            )            raise
```

Use actual framework callbacks or supported instrumentation for node-level events and token usage. The wrapper above traces only the overall workflow unless child operations create their own spans.

### Why this matters

Imagine total request cost increases by 40%. A workflow-level cost metric tells you the increase happened, but not why.

Per-agent and per-call telemetry may reveal:

- The planner started calling the model twice as often.
- Retrieval started returning 40 documents instead of 8.
- A failed tool triggered repeated synthesis calls.
- The model router started selecting a more expensive model.
- The final prompt grew because conversation history was not truncated.

That is the difference between merely collecting metrics and building observability that helps engineers make decisions.

## 8. Streaming LLMs: monitoring before the response finishes

Streaming changes what you can measure and when.

For a non-streaming response, you often receive token usage together with the final response. During streaming, you receive chunks over time. The final usage information may arrive only in the terminal event, if requested and supported by the provider.

You should separately monitor:

- Time to first token
- Total generation duration
- Inter-token delay
- Client disconnects
- Completed versus interrupted streams
- Final usage metadata availability
- Tokens generated before cancellation, where the provider exposes this information

Example using the OpenAI Chat Completions streaming interface:

```
import timefrom openai import AsyncOpenAIclient = AsyncOpenAI()async def stream_answer(prompt: str):    started = time.perf_counter()    first_token_at = None    final_usage = None    stream = await client.chat.completions.create(        model="YOUR_MODEL_NAME",        messages=[{"role": "user", "content": prompt}],        stream=True,        stream_options={"include_usage": True},    )    try:        async for chunk in stream:            now = time.perf_counter()            if first_token_at is None:                first_token_at = now                print(                    "TTFT ms:",                    round((first_token_at - started) * 1000, 2),                )            # Usage is commonly present on a final chunk when            # include_usage is supported. Not every chunk has it.            if chunk.usage is not None:                final_usage = chunk.usage            for choice in chunk.choices:                token_text = choice.delta.content
```

This example illustrates the measurements; a production implementation should send telemetry to a metrics/tracing backend instead of using `print`. It should also handle client cancellation, provider errors and stream cleanup.

A subtle but important point: TTFT is not the same as total latency. A model might start responding quickly and then generate very slowly. Conversely, a model might take a while before its first token but generate the rest rapidly. Both patterns affect user experience differently.

## 9. Budgets, alerts and automatic cost controls

Monitoring detects a problem. A production system also needs controls to prevent a problem from becoming a major incident.

### 9.1 Define budgets at multiple levels

A financial AI platform could define:

- Maximum estimated cost per request
- Maximum token budget per workflow
- Maximum number of LLM calls per workflow
- Maximum agent iterations
- Maximum wall-clock duration
- Maximum daily or monthly spend per feature or tenant

For example:

```
from dataclasses import dataclass@dataclass(frozen=True)class WorkflowBudget:    max_llm_calls: int = 12    max_total_tokens: int = 30_000    max_cost_usd: float = 0.20    max_duration_seconds: float = 30.0class BudgetExceeded(Exception):    passdef check_budget(    *,    calls: int,    total_tokens: int,    estimated_cost_usd: float,    elapsed_seconds: float,    budget: WorkflowBudget,) -> None:    if calls >= budget.max_llm_calls:        raise BudgetExceeded("LLM call limit reached")    if total_tokens >= budget.max_total_tokens:        raise BudgetExceeded("Token budget reached")    if estimated_cost_usd >= budget.max_cost_usd:        raise BudgetExceeded("Cost budget reached")    if elapsed_seconds >= budget.max_duration_seconds:        raise BudgetExceeded("Time budget reached")
```

These are illustrative limits. The caller must check the budget before dispatching another expensive operation, and the token and cost totals must be updated atomically or consistently as calls complete.

This function is a guard, not a complete budget-management service. For concurrent agents, multiple calls can pass a local check simultaneously and collectively exceed the limit. Production implementations need a centralized reservation or atomic budget mechanism, a safety margin for estimated usage, and reconciliation of actual usage after completion.

For agent workflows, combine these budgets with LangGraph step limits, maximum retry counts and explicit stop conditions. A token budget alone does not prevent infinite loops of cheap calls, and a step limit alone does not prevent one enormous prompt from being expensive.

### 9.2 Useful alerts

| Alert              | Example condition                               | What it can indicate                                     |
| ------------------ | ----------------------------------------------- | -------------------------------------------------------- |
| Latency regression | p95 latency above SLO for 10 minutes            | Provider slowdown, queueing or oversized context         |
| Error-rate spike   | LLM error ratio above a threshold               | Rate limiting, provider outage or bad requests           |
| Cost anomaly       | Spend significantly above baseline              | Traffic spike, retry loop or routing regression          |
| Token inflation    | Average input tokens rise sharply               | Prompt growth, duplicated history or excessive retrieval |
| Missing usage      | Large share of successful calls lack usage data | Instrumentation gap or unsupported response format       |
| Agent loop         | Calls per workflow exceed normal limits         | Bad routing, tool failure or termination bug             |
| Quality regression | Evaluation scores fall below release threshold  | Model, prompt, retrieval or data regression              |

Set alert thresholds based on actual traffic, service-level objectives and business budgets. A single global threshold can be misleading if interactive chat and offline batch jobs have very different latency expectations.

### 9.3 Cost optimization should be evidence-driven

Once you have tracing and metrics, you can make targeted improvements.

Reduce prompt size

Measure input tokens by agent and prompt-template version. Remove duplicated conversation history, irrelevant retrieved chunks and unnecessary tool schemas.

Route requests to suitable models

Use smaller models for classification or simple extraction and stronger models for complex reasoning when evaluation demonstrates acceptable quality.

Use caching where safe

Cache deterministic intermediate results or embeddings when the keys include relevant versions and tenant boundaries. Provider prompt caching may reduce cost for eligible repeated prefixes.

Reduce unnecessary agent calls

Use trace data to identify redundant planner calls, excessive retries and loops. Enforce explicit maximum steps and stop conditions.

Do not optimize solely for token count. A cheaper model that produces incorrect financial analysis can create a much larger business cost. Track quality and cost together, for example cost per successfully completed task or cost per answer meeting the evaluation threshold.

## 10. Production dashboards: what I would build

I would build separate dashboards for system health, LLM behavior, cost and quality.

### Dashboard A — Reliability and latency

Request success rate

# 99.5%

Illustrative target

Request p95

# 4.0 s

Illustrative target

Track request rate, p50/p95/p99, provider errors, timeouts, retries, queue wait and tool failures.

### Dashboard B — Token and cost usage

Tokens per request

# 9.7k

Illustrative average

Estimated cost/request

# $0.04

Illustrative value

Break down input/output tokens, cached usage, spend per model, agent, feature and tenant, plus daily/monthly budget consumption.

### Dashboard C — Quality and behavior

Track task completion, retrieval quality, citation validity, groundedness, refusal behavior, tool success and offline evaluation results.

All figures above are mock dashboard values, not actual measurements or recommended universal targets.

### Useful PromQL examples

Average observed LLM latency over five minutes:

```
sum(rate(llm_call_duration_seconds_sum[5m]))
/
sum(rate(llm_call_duration_seconds_count[5m]))
```

p95 LLM call latency, grouped by model:

```
histogram_quantile(
  0.95,
  sum by (le, model) (
    rate(llm_call_duration_seconds_bucket[5m])
  )
)
```

LLM error ratio:

```
sum(rate(llm_calls_total{status="error"}[5m]))
/
clamp_min(sum(rate(llm_calls_total[5m])), 1e-9)
```

Estimated spend rate in USD per hour:

```
3600 *
sum(rate(llm_estimated_cost_usd_total[5m]))
```

These queries assume the metric names shown earlier are exposed with compatible histogram and counter semantics. For an exact request-level cost distribution, use a cost ledger or analytics query; a cumulative counter is not sufficient to calculate a true per-request distribution.

## 11. Security, privacy and operational correctness

This is where staff-level implementation differs from a quick prototype.

### Data minimization

- Do not capture full prompts and completions by default.
- Redact secrets, access tokens, personal information and confidential document content.
- Restrict access to trace backends because prompt and response payloads may contain sensitive information.
- Define retention and deletion policies for telemetry and usage records.
- Use tenant-aware authorization and prevent cross-tenant exposure in traces and cost reports.

### Instrumentation must not become a new failure mode

A tracing backend can be unavailable while the LLM provider is healthy. Decide which telemetry failures should be non-blocking and which records must be durably preserved.

A typical design uses buffered, asynchronous export for traces and metrics, bounded queues, timeouts and a defined drop policy. Cost records may need a stronger durability guarantee than sampled debug traces, so a separate durable usage-event pipeline is often appropriate.

### Keep measurements consistent

- Use a monotonic clock for elapsed duration.
- Use UTC timestamps for event records.
- Define exactly when a request starts and finishes.
- Distinguish missing token usage from zero usage.
- Count failed and retried calls where billable usage may occur.
- Keep estimated and reconciled costs separate.
- Version pricing configurations and prompt templates.
- Test instrumentation during streaming, cancellation, retries, timeouts and parallel execution.

### Use sampling intelligently

Tracing every request with full payloads can be expensive and risky. Use head or tail sampling according to your backend's capabilities, retain error traces at a higher rate, and keep aggregate metrics for all traffic where feasible.

Do not rely on sampled traces for financial accounting. Use durable usage records and provider billing reconciliation for cost attribution.

## 12. How I would implement this in your FastAPI + LangGraph enterprise project

I would use a layered design:

```
app/
├── api/
│   └── routes.py
├── observability/
│   ├── tracing.py
│   ├── metrics.py
│   ├── logging.py
│   └── redaction.py
├── llm/
│   ├── client_factory.py
│   ├── instrumented_client.py
│   ├── usage_normalizer.py
│   └── pricing.py
├── workflow/
│   ├── graph.py
│   ├── nodes.py
│   └── budget.py
├── services/
│   └── usage_service.py
├── repositories/
│   └── usage_repository.py
└── monitoring/
    ├── alerts.py
    └── evaluation_metrics.py
```

Responsibilities:

- `client_factory.py`: creates provider clients with timeouts, authentication and configuration.
- `instrumented_client.py`: wraps each model invocation with spans, timing, usage extraction and error handling.
- `usage_normalizer.py`: converts provider-specific usage into a common schema.
- `pricing.py`: applies versioned provider/model pricing rules.
- `budget.py`: limits workflow steps, calls, time and estimated spend.
- `usage_service.py`: attributes usage to a request, workflow and tenant.
- `usage_repository.py`: persists durable usage events idempotently.
- `redaction.py`: applies telemetry data-minimization rules.
- `evaluation_metrics.py`: tracks quality separately from operational health.

The most important architectural rule is to instrument the shared LLM client boundary, rather than manually adding slightly different token and cost logic to every agent. That gives you consistent measurement even as you add new agents or tools.

## 13. Staff-level interview answer

> I implement LLM observability at multiple levels: request, workflow, agent, individual model call and provider. I use distributed tracing with OpenTelemetry or supported LLM framework instrumentation to correlate API requests, retrieval operations, tool calls and model invocations.
>
> For each model call, I capture provider, model, latency, status, retries, token usage and the relevant workflow identifiers. I normalize provider-specific usage metadata and calculate estimated cost using a versioned pricing configuration. I then persist usage events for request-level and tenant-level attribution and reconcile the estimates against provider billing.
>
> For performance, I track end-to-end latency, p50/p95/p99, time to first token for streaming, provider latency, queueing, retrieval and tool latency. I use metrics to detect regressions and traces to identify the critical path.
>
> I also enforce budgets for maximum calls, tokens, time and spend, while controlling retries and agent iterations. Finally, I protect telemetry through redaction, access controls and retention policies, and I monitor model quality alongside latency and cost. The objective is to make LLM behavior observable, costs attributable, incidents diagnosable and production operation predictable.

The key principle: traces explain an individual request, metrics reveal system-wide trends, usage records support cost attribution, and budgets prevent uncontrolled consumption. All four are necessary for production-grade LLM observability.