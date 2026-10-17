# How to trace an AI request — Staff AI Engineer / Staff Software Engineer level

In a production AI system, tracing means following one user request across every component it touches, from the API gateway to the LLM, retrieval pipeline, agents, tools, databases, external services, and final response.

For a simple API, tracing might show:

`HTTP request → database query → HTTP response`

For an enterprise AI application, a single request might look like this:

`FastAPI → authentication → LangGraph supervisor → planner → query embedding → pgvector retrieval → reranker → LLM → tool execution → evaluator → final response`

The difficult part is not just measuring how long the request took. You also need to understand which step was slow, which model was called, how many tokens were consumed, why a tool was invoked, where an error occurred, and whether the final answer was grounded in retrieved evidence.

I'll explain this using an enterprise financial-advisor copilot built with FastAPI, LangGraph, PostgreSQL/pgvector, Redis, and an LLM provider.

## 1. What does tracing an AI request actually mean?

Consider this user request:

> “Summarize my portfolio's risk exposure and explain why my technology allocation increased this month.”

Your AI system may execute these operations:

1\. API request

12 ms

Authenticate user, assign request ID, create trace.

2\. Agent orchestration

35 ms

Supervisor selects portfolio analysis and policy retrieval.

3\. Retrieval

180 ms

Embedding, vector search, filtering, reranking.

4\. LLM calls

1.4 s

Planning, tool selection, synthesis, token usage.

5\. Evaluation and response

250 ms

Validate output, apply policy checks, return response.

Illustrative end-to-end duration

# 1.88 s

Primary latency contributor

## LLM calls

These timings are illustrative, not measurements from a running system. Also, the timings can only be added directly if the stages are sequential and non-overlapping. With parallel agents, overlapping spans mean the total wall-clock latency is not the sum of all span durations.

Without tracing, you see that the API took 1.88 seconds. With tracing, you can identify where that time was spent and investigate the cause.

## 2. Understand the three observability signals

A staff-level design combines three complementary signals.

Distributed traces

Explain one request's execution path. They contain spans, parent-child relationships, timestamps, attributes, events, and errors.

Answer: “Why did this particular request take 8 seconds?”

Metrics

Aggregated measurements such as latency percentiles, error rates, token consumption, tool-call failures, and cache hit rate.

Answer: “Why has p95 latency increased across all requests?”

Logs

Timestamped application events, error details, policy decisions, and diagnostic context, correlated with the trace.

Answer: “Which validation rule rejected this tool call?”

These signals work together. A metric can identify an incident, a trace can locate the slow stage, and logs can explain the failure inside that stage.

For an AI system, I also track model-specific data such as model name/version, input/output token counts, provider latency, retries, timeouts, tool selection, retrieval quality, and evaluation results.

## 3. Choose a tracing architecture

For a production system, I would use OpenTelemetry as the common instrumentation layer, with an OpenTelemetry-compatible trace backend and separate AI-specific evaluation tooling where needed.

A practical architecture:

FastAPI + LangGraph + RAG + tools

Instrument requests, model calls, retrieval, database operations, and tool execution

OpenTelemetry SDK

Trace context · spans · attributes · events · exceptions

OTLP exporter / collector

Batching, sampling, redaction, routing, and export

Trace backend

Jaeger, Tempo, vendor APM

Metrics backend

Prometheus, Grafana

AI evaluation

Quality, grounding, cost

The exact backends are a deployment choice. The important thing is to standardize trace context and instrumentation instead of building a different tracing mechanism for every agent.

My design preference:

- OpenTelemetry for distributed tracing and cross-service correlation.
- Prometheus-compatible metrics for operational alerting.
- Structured JSON logs for application diagnostics.
- AI-specific traces or evaluation records for prompts, model responses, retrieval relevance, and answer quality—subject to strict data controls.

Don't assume that installing OpenTelemetry automatically traces every LLM provider, LangGraph node, Redis operation, and database call. Each library needs supported auto-instrumentation or explicit instrumentation.

## 4. Implement OpenTelemetry in FastAPI

Let's build a minimal foundation.

### Step 1: Install dependencies

```
pip install \
  fastapi uvicorn \
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-otlp \
  opentelemetry-instrumentation-fastapi
```

You will also need a running OTLP-compatible collector or backend to receive exported spans. Without one, you can still instrument locally, but you will not have a production trace UI.

### Step 2: Create a tracing module

`app/observability/tracing.py`

```
from opentelemetry import tracefrom opentelemetry.sdk.resources import Resourcefrom opentelemetry.sdk.trace import TracerProviderfrom opentelemetry.sdk.trace.export import BatchSpanProcessorfrom opentelemetry.exporter.otlp.proto.http.trace_exporter import (    OTLPSpanExporter,)from opentelemetry.trace import Status, StatusCoderesource = Resource.create({    "service.name": "financial-ai-api",    "service.version": "1.0.0",    "deployment.environment.name": "production",})provider = TracerProvider(resource=resource)exporter = OTLPSpanExporter(    # Prefer configuring this through environment variables    # and deployment configuration.)provider.add_span_processor(    BatchSpanProcessor(exporter))trace.set_tracer_provider(provider)tracer = trace.get_tracer("financial-ai")
```

This initializes a provider, associates spans with a service identity, and batches export.

For a real deployment, configure the OTLP endpoint, TLS, authentication, exporter timeouts, sampling, and resource attributes according to your collector/backend. Avoid creating multiple competing global providers in different modules. Initialize observability once during application startup.

### Step 3: Instrument FastAPI

`app/main.py`

```
from fastapi import FastAPIfrom opentelemetry.instrumentation.fastapi import (    FastAPIInstrumentor,)app = FastAPI(title="Financial AI API")FastAPIInstrumentor.instrument_app(app)@app.get("/health")async def health():    return {"status": "ok"}
```

This creates HTTP-level spans and propagates trace context where supported. Add middleware or dependencies for authentication and request metadata, but do not use a user ID, email address, or request payload as a high-cardinality metric label.

For production, also instrument outgoing HTTP requests, SQLAlchemy/database operations, Redis, and any message queue or Celery worker. Verify the instrumentation against the versions in your lockfile.

## 5. Add custom spans around AI operations

Automatic HTTP instrumentation tells you the API duration. It doesn't necessarily tell you whether the delay came from retrieval, the model, a reranker, or a tool.

Create spans around meaningful operations.

`app/services/answer_service.py`

```
from opentelemetry import tracetracer = trace.get_tracer("financial-ai")async def answer_question(    question: str,    tenant_id: str,):    with tracer.start_as_current_span("ai.answer") as root:        root.set_attribute("ai.workflow", "financial_rag")        root.set_attribute("ai.tenant_tier", "enterprise")        with tracer.start_as_current_span("retrieval.search") as span:            documents = await retrieve_documents(                question=question,                tenant_id=tenant_id,            )            span.set_attribute(                "retrieval.document_count",                len(documents),            )        with tracer.start_as_current_span("llm.generate") as span:            answer = await generate_answer(                question=question,                documents=documents,            )            span.set_attribute("ai.operation", "answer_generation")        with tracer.start_as_current_span("answer.validate") as span:            validated_answer = validate_answer(answer)        return validated_answer
```

The functions `retrieve_documents`, `generate_answer`, and `validate_answer` are application-specific placeholders. Replace them with your implementations.

Notice that we record the number of retrieved documents, not their full contents. The trace can tell us how many documents were returned without automatically exposing sensitive financial data.

### What a useful trace should contain

For a successful request, the hierarchy could look like:

```
HTTP POST /ask
└── ai.answer
    ├── auth.authorize
    ├── langgraph.invoke
    │   ├── supervisor.plan
    │   │   └── llm.call
    │   ├── retrieval.search
    │   │   ├── embedding.generate
    │   │   ├── pgvector.search
    │   │   └── reranker.score
    │   ├── portfolio.get_summary
    │   │   └── postgres.query
    │   ├── llm.generate
    │   └── answer.validate
    └── response.serialize
```

This is the level of detail that makes a trace operationally useful. Use spans for meaningful operations, not every trivial Python function; excessive span creation adds overhead and noise.

## 6. Trace LLM calls, token usage, latency, and cost

LLM calls need more detailed instrumentation than ordinary HTTP requests.

For each model call, I would capture:

- Provider and model identifier.
- Model version or deployment identifier, when available.
- Operation type: planning, classification, generation, reranking, or evaluation.
- Provider request ID.
- Input and output tokens.
- Cached input tokens, if the provider exposes them.
- Time to first token for streaming.
- Total generation latency.
- Retry count, timeout, rate-limit response, and error category.
- Estimated cost using a versioned pricing configuration.

### A. Instrument a model call

Here is a provider-agnostic wrapper for a non-streaming LLM client:

```
from time import perf_counterfrom opentelemetry import tracefrom opentelemetry.trace import Status, StatusCodetracer = trace.get_tracer("financial-ai.llm")async def traced_llm_call(    client,    *,    model: str,    messages: list[dict],):    start = perf_counter()    with tracer.start_as_current_span("llm.chat") as span:        span.set_attribute("gen_ai.operation.name", "chat")        span.set_attribute("gen_ai.request.model", model)        try:            response = await client.chat.completions.create(                model=model,                messages=messages,            )            usage = getattr(response, "usage", None)            if usage is not None:                span.set_attribute(                    "gen_ai.usage.input_tokens",                    usage.prompt_tokens,                )                span.set_attribute(                    "gen_ai.usage.output_tokens",
```

This example assumes an async client with an OpenAI-compatible chat-completions interface. Adapt the call and usage fields to the provider you actually use. Some providers expose usage only at the end of a streaming response, and some return different token categories.

For a production implementation, I would also record the provider's request ID, classify errors into timeout/rate-limit/connection/provider failures, and add bounded retries with exponential backoff and jitter.

Avoid recording raw messages or model responses by default. Even if a request is successful, the prompt may contain personally identifiable information, portfolio details, account identifiers, or confidential documents.

### B. Measure the right latency values

For non-streaming requests:

\\[ T\_{\text{LLM}} = T\_{\text{response received}} - T\_{\text{request sent}} \\]

For streaming requests, measure at least two separate values:

- Time to first token (TTFT): time from sending the request until the first generated token or content delta is received.
- Total generation time: time from sending the request until the stream finishes.

For an agentic request, also track:

- Total end-to-end latency.
- Time spent in retrieval and tool execution.
- Number of LLM calls per user request.
- Sequential versus parallel model-call duration.
- Queue wait time if the request is dispatched to Celery.
- Time spent waiting for human approval.

A 10-second request with a 200 ms database query and a 7-second model call needs a different optimization from a 10-second request waiting 6 seconds in a worker queue.

### C. Estimate model cost correctly

A simple estimate is:

\\[ C = \frac{T_i \times P_i + T_o \times P_o}{10^6} \\]

Where:

- \\(T_i\\) = billable input tokens.
- \\(T_o\\) = billable output tokens.
- \\(P_i\\) = price per million input tokens.
- \\(P_o\\) = price per million output tokens.

This is a simplified estimate. Actual billing may have different prices for cached input, reasoning tokens, image/audio tokens, batch processing, or other model-specific categories.

Maintain pricing configuration separately from trace instrumentation, version it, and distinguish estimated cost from provider-reported charges.

## 7. Trace LangGraph workflows and individual agents

This is where AI tracing becomes particularly important.

A graph may have a supervisor, planner, retrieval agent, specialist agents, tools, an evaluator, and retry loops. You need to see both the overall workflow and the individual steps.

For example:

```
langgraph.invoke
├── supervisor
│   └── llm.call
├── portfolio_agent
│   ├── llm.call
│   └── portfolio_tool
├── policy_agent
│   ├── retrieval.search
│   └── llm.call
├── evaluator
│   └── llm.call
└── retry
    └── policy_agent
```

The exact number of calls depends on your graph's routing and retry behavior.

### A. Add spans inside graph nodes

```
from opentelemetry import tracetracer = trace.get_tracer("financial-ai.langgraph")async def portfolio_agent_node(state: dict) -> dict:    with tracer.start_as_current_span(        "agent.portfolio_analysis"    ) as span:        span.set_attribute(            "agent.name",            "portfolio_analysis",        )        # Avoid placing the full state in the span.        result = await analyze_portfolio(            question=state["question"],            portfolio_id=state["portfolio_id"],        )        span.set_attribute(            "agent.result_status",            "success",        )        return {"portfolio_analysis": result}
```

Use similar instrumentation in the supervisor, retrieval node, evaluator, and tools. Instrument the LLM client separately so that a node's duration can be distinguished from the provider's actual generation time.

### B. Capture graph execution metadata

Useful attributes include:

- Workflow name and version.
- Current node name.
- Number of executed nodes.
- Retry attempt and reason.
- Selected route or agent name.
- Tool name and result status.
- Checkpoint ID or a safe reference to the persisted workflow.
- Whether the graph completed, failed, hit a step limit, or paused for human approval.

Be careful with identifiers: a checkpoint ID or conversation ID can itself be sensitive or high-cardinality. Store a safe correlation identifier and restrict access to detailed workflow state.

### C. Track rework loops

Suppose the evaluator finds that a generated answer is insufficiently grounded and routes back to retrieval.

A trace should show:

```
answer.workflow
├── retrieval.search [attempt=1]
├── llm.generate [attempt=1]
├── evaluator [grounding=insufficient]
├── retrieval.search [attempt=2]
├── llm.generate [attempt=2]
└── evaluator [grounding=acceptable]
```

This makes it possible to measure:

- Average number of rework iterations.
- Percentage of requests requiring retries.
- Extra token cost caused by retries.
- Which evaluator outcomes cause the most rework.
- Whether retries actually improve quality.

Set explicit retry budgets and maximum graph steps. Otherwise, a graph may remain technically healthy while burning tokens in an unproductive loop.

Important: tracing should observe the graph; it should not control the graph's business decisions. The workflow policy determines whether to retry, fall back, ask for human review, or fail.

## 8. Distributed tracing across FastAPI, Celery, Redis, and databases

In an enterprise application, one request may not execute in one process.

For example:

```
Client
  │
  ▼
FastAPI request (trace A)
  │
  ├── PostgreSQL
  ├── Redis cache
  │
  └── Celery task
         │
         ├── LangGraph supervisor
         ├── Embedding service
         ├── pgvector retrieval
         ├── LLM provider
         └── Result persistence
```

You want one trace, or a linked trace hierarchy, that lets you move from the HTTP request to the background job and its downstream calls.

### A. Why context propagation matters

OpenTelemetry uses a trace ID to associate spans with the same distributed operation and a span ID to identify an individual operation. Parent-child relationships tell you which component called which.

When a request crosses a process boundary, the trace context must be propagated. Without propagation, FastAPI and Celery may create unrelated traces, leaving you unable to connect the API request to the job it started.

For HTTP, instrumentations commonly propagate W3C Trace Context headers such as `traceparent`. For a message queue, propagate the context through the message headers or task metadata using the supported instrumentation.

### B. Celery propagation pattern

A simplified pattern using OpenTelemetry's propagation APIs:

```
from opentelemetry import propagate, tracefrom opentelemetry.trace import SpanKindtracer = trace.get_tracer("financial-ai.jobs")def build_trace_headers() -> dict[str, str]:    carrier: dict[str, str] = {}    propagate.inject(carrier)    return carrierasync def run_traced_job(headers: dict, job_id: str):    parent_context = propagate.extract(headers)    with tracer.start_as_current_span(        "celery.financial_analysis",        context=parent_context,        kind=SpanKind.CONSUMER,    ) as span:        span.set_attribute("job.type", "financial_analysis")        span.set_attribute("job.id", job_id)        return await perform_financial_analysis(job_id)
```

This demonstrates the propagation concept; it is not a complete Celery integration. In production, configure the supported Celery instrumentation and task publishing/consuming hooks for your installed versions. Ensure the producer injects context into task headers and the consumer extracts it correctly.

If a background job runs hours later, is retried, or is triggered by a scheduler, a new trace linked to the original trace may be more appropriate than a long-lived parent-child trace. Choose a consistent policy for long-running and asynchronous workflows.

### C. Instrument database and Redis operations

For database calls, capture:

- Operation type and database system.
- Query duration.
- Connection-pool wait time, where available.
- Timeout and error classification.
- Rows affected or returned, when safe and supported.

For Redis, capture:

- Operation name.
- Duration and error category.
- Cache hit/miss as an application-level metric.
- Connection errors, timeouts, and pool saturation.

Avoid storing full SQL statements or Redis values if they can contain sensitive information. Query parameter values should not be exported indiscriminately. Prefer query fingerprints or normalized operation names for diagnostics.

## 9. Structured logs correlated with traces

Traces tell you where the problem happened. Logs explain the application-specific circumstances.

For example, a trace may show that `retrieval.search` took 2.8 seconds. A correlated log might say that the vector query timed out after a database connection-pool wait.

Use structured logging rather than plain text.

```
import loggingfrom opentelemetry import tracelogger = logging.getLogger("financial_ai")def log_workflow_failure(    workflow_name: str,    error_code: str,):    span_context = trace.get_current_span().get_span_context()    logger.error(        "AI workflow failed",        extra={            "workflow_name": workflow_name,            "error_code": error_code,            "trace_id": (                format(span_context.trace_id, "032x")                if span_context.is_valid                else None            ),            "span_id": (                format(span_context.span_id, "016x")                if span_context.is_valid                else None            ),        },    )
```

Configure your logging formatter to emit these fields as JSON. OpenTelemetry's logging instrumentation or your logging integration may already inject trace context; avoid duplicating fields unnecessarily.

For security and privacy:

- Do not log API keys, bearer tokens, passwords, or database credentials.
- Avoid raw financial prompts and full retrieved documents in standard traces.
- Do not include complete model responses by default.
- Redact sensitive values before export.
- Restrict access to detailed traces.
- Apply retention and deletion policies.
- Ensure trace attributes and exception messages do not inadvertently contain secrets.

Trace IDs are correlation identifiers, not authentication tokens. Never use possession of a trace ID as permission to retrieve sensitive data.

## 10. Trace failures, retries, timeouts, and fallbacks

A production trace should not merely say “request failed.” It should identify the failure boundary and what happened next.

Consider this scenario:

```
llm.generate [attempt=1] → provider timeout
llm.generate [attempt=2] → rate limited
llm.fallback            → alternate model succeeded
answer.validate         → passed
```

You should be able to determine the number of attempts, the latency of each attempt, the error category, the fallback model, and the additional cost.

Recommended conventions:

- Record exceptions on failed spans.
- Set span status to error when an operation genuinely fails.
- Use bounded retries with exponential backoff and jitter.
- Record the retry reason and attempt number.
- Distinguish provider errors from application validation errors.
- Capture timeout and cancellation events.
- Avoid recording the same failure as multiple independent errors at every parent layer.

A parent workflow may be marked successful if a failed first attempt is recovered by a successful fallback. The failed child span should still show the failure. This distinction is essential for reliability analysis.

For tool calls, also record whether the tool was selected, authorized, executed, rejected, or interrupted for human approval. Do not expose hidden chain-of-thought or private model reasoning to achieve this. Record operational decisions and structured outcomes instead.

## 11. AI-specific tracing: retrieval quality and answer quality

Traditional distributed tracing tells you how the request executed. It does not automatically tell you whether the AI answer was correct.

For a RAG application, I would record separate operational and quality signals.

| Stage               | Useful information                                         |
| ------------------- | ---------------------------------------------------------- |
| Query processing    | Query class, normalized query fingerprint, detected intent |
| Embedding           | Model identifier, latency, error rate                      |
| Retrieval           | Candidate count, filtered count, top-k, retrieval strategy |
| Reranking           | Candidate count, latency, score distribution               |
| Generation          | Model identifier, token usage, finish reason               |
| Citation validation | Number of citations, citation validation outcome           |
| Evaluation          | Groundedness score, relevance score, evaluator version     |
| Final response      | Success/failure status, safety checks, total latency       |

Do not confuse a similarity score with proof that a document supports the generated answer. Retrieval relevance and answer faithfulness are related but different concepts.

For example, you may find that retrieval is fast and returns many documents, but the final answer contains unsupported claims. The problem may be chunk quality, missing evidence, prompt construction, model behavior, or a faulty evaluator—not latency.

For offline and controlled online evaluations, you can assess:

- Context precision and context recall.
- Answer relevance.
- Faithfulness to retrieved evidence.
- Citation correctness.
- Task-specific correctness against trusted ground truth.

These evaluations should be versioned alongside the prompt, model, retrieval configuration, and evaluation dataset. Online LLM-as-judge evaluations may be delayed, sampled, or asynchronous because they add latency and cost.

For regulated financial workflows, also record deterministic policy outcomes and human-review decisions. Do not rely solely on a model-generated quality score to authorize financial or compliance actions.

## 12. Metrics and dashboards: how you operate the system

A trace helps you debug one request. Metrics help you identify patterns across thousands or millions of requests.

I would organize dashboards into four groups.

Latency and throughput

API p95 latency

# 2.4 s

Illustrative

LLM p95 latency

# 1.6 s

Illustrative

Track p50, p95 and p99 latency, requests per second, queue wait time, and streaming TTFT.

Cost and token consumption

Input/output tokens per request, estimated cost per successful task, model-call count, cache hit rate, and retry overhead.

Reliability

Provider error rate, timeout rate, fallback rate, tool failures, graph step-limit exits, and database connection saturation.

Quality and safety

Grounded-answer rate, citation validation failures, human escalation rate, policy rejections, and evaluation regressions.

The values in the dashboard cards above are examples, not live measurements.

### Example: instrument an application metric

Install the Prometheus client:

```
pip install prometheus-client
```

Then record a bounded-cardinality counter and histogram:

```
from time import perf_counterfrom prometheus_client import Counter, HistogramAI_REQUESTS = Counter(    "ai_requests_total",    "AI requests by outcome",    ["workflow", "outcome"],)AI_DURATION = Histogram(    "ai_request_duration_seconds",    "End-to-end AI request duration",    ["workflow"],)async def traced_answer(question: str):    started = perf_counter()    try:        result = await answer_question(question, tenant_id="trusted-context")        AI_REQUESTS.labels(            workflow="financial_rag",            outcome="success",        ).inc()        return result    except Exception:        AI_REQUESTS.labels(            workflow="financial_rag",            outcome="error",        ).inc()        raise    finally:        AI_DURATION.labels(            workflow="financial_rag",
```

Here `answer_question` is the service function from the earlier example; the tenant argument is illustrative and should be obtained from authenticated request context in a real endpoint.

Expose metrics through an appropriate Prometheus integration or endpoint and secure it according to your deployment.

Never use trace IDs, user IDs, request IDs, raw prompts, portfolio IDs, or arbitrary model names as Prometheus labels. Unbounded label values can cause a cardinality explosion and destabilize the monitoring system. Put per-request identifiers in traces or logs instead.

## 13. Sampling: tracing every request is not always necessary

At high traffic volumes, retaining every detailed trace can be expensive.

There are three common approaches:

- Head sampling: Decide at the start of a trace whether to sample it. It is efficient, but a trace that later fails may already have been discarded.
- Tail sampling: Make the decision after observing the trace, allowing policies such as retaining errors, slow requests, or important workflows. This requires collector capacity and careful configuration.
- Always-on metrics plus sampled traces: Retain aggregate measurements for all requests and detailed traces for a subset, with deliberate exceptions for important failures.

For an enterprise AI service, I would start with a sampling policy based on traffic and cost, preserve useful error traces where possible, and use tail sampling if the infrastructure supports it. Avoid indiscriminately retaining sensitive requests just because they failed.

Sampling also needs to preserve trace completeness across services. If the API samples a request but the worker independently samples its task, the trace may become fragmented unless propagation and sampling decisions are coordinated.

## 14. Common mistakes I would look for in a production review

| Mistake                                                 | Why it is a problem                             | Better approach                                                |
| ------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------------------- |
| Logging only API duration                               | Cannot identify the slow component              | Instrument retrieval, LLM calls, tools and database operations |
| Logging full prompts by default                         | Sensitive-data exposure                         | Record safe metadata; restrict and redact detailed payloads    |
| Creating a new trace in every function                  | Breaks causal relationships                     | Preserve parent context and use nested spans                   |
| Counting all retries as separate user requests          | Inflates traffic and distorts cost metrics      | Track request-level and attempt-level metrics separately       |
| Summing parallel span durations                         | Overstates wall-clock latency                   | Analyze span timing and critical paths                         |
| Using only LLM tracing                                  | Misses queues, databases and external services  | Use distributed tracing across the complete stack              |
| Treating a successful HTTP response as a correct answer | Hides quality failures                          | Add citations, grounding and task-level evaluations            |
| Ignoring streaming                                      | Misses first-token latency and partial failures | Record TTFT, stream completion and cancellation                |
| Tracing without sampling or retention controls          | Increases cost and privacy risk                 | Apply controlled sampling, redaction and retention             |
| Building dashboards without alerts                      | Detects problems only when someone looks        | Alert on SLO violations and actionable error patterns          |

## 15. How I would implement this in your enterprise AI project

For a FastAPI + LangGraph + PostgreSQL/pgvector + Redis + Celery architecture, I would roll it out in this order:

1. Establish a trace at the API boundary.

   Instrument FastAPI, define service/environment/version attributes, propagate context, and verify that every request gets a valid trace.
2. Instrument infrastructure dependencies.

   Add supported instrumentation for SQLAlchemy, outbound HTTP, Redis, and Celery. Test trace continuity across background jobs.
3. Instrument AI operations.

   Add spans around LangGraph invocation, nodes, retrieval, embeddings, reranking, LLM calls, tool calls, evaluator loops, and human approval pauses.
4. Capture cost and quality.

   Record provider-reported token usage, model identifiers, retrieval counts, citation outcomes, and evaluation results without exposing sensitive content.
5. Establish dashboards and SLOs.

   Monitor p95/p99 latency, error rate, queue delay, cost per successful task, tool failure rate, and quality regressions.
6. Test observability itself.

   Verify traces across retries, async tasks, parallel agents, timeouts, streaming, cancellation, and failures. Confirm that secrets and private financial data do not leak into exported telemetry.

For a codebase with multiple agents and workers, I would keep instrumentation in a shared `app/observability/` package, but avoid coupling business logic directly to a particular trace backend.

A reasonable structure:

```
app/
├── observability/
│   ├── tracing.py
│   ├── metrics.py
│   ├── logging.py
│   ├── attributes.py
│   └── redaction.py
├── api/
├── agents/
├── graph/
├── retrieval/
├── llm/
├── tools/
├── services/
├── repositories/
└── workers/
```

The application code emits standard telemetry; deployment configuration controls the exporter, collector, sampling, storage, and access policies.

## 16. Staff-level interview answer

> “I implement end-to-end distributed tracing with OpenTelemetry and propagate trace context from the API through asynchronous workers, agent orchestration, retrieval, database calls, model providers, and tools. I use nested spans to capture the critical path, latency, errors, retries, and workflow decisions.
>
> For LLM operations, I track model identifiers, token usage, provider request IDs, latency, timeouts, retries, and estimated cost. For RAG, I trace embedding, retrieval, reranking, citation validation, and answer evaluation. For agentic workflows, I capture node execution, tool calls, routing decisions, rework loops, and HITL pauses.
>
> I correlate traces with structured logs and aggregate metrics. I use p95/p99 latency, error rates, token cost, queue delay, and quality metrics for alerting and capacity planning. I also apply sampling, redaction, access controls, and retention policies to prevent telemetry from becoming a data-leakage risk.
>
> Finally, I distinguish operational success from AI quality. A request can return HTTP 200 while producing an unsupported answer, so I measure grounding, citation correctness, and task-specific quality separately from latency and availability.”

The staff-level takeaway: tracing is not simply logging an LLM call. It is the ability to reconstruct a request's execution path, explain its latency and cost, diagnose failures, and connect technical behavior to AI quality—all without compromising privacy or security.