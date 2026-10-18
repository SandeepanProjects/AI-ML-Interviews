# How to Make a Complete AI System Observable — Senior/Staff AI Engineer Guide

For a production AI platform, observability means being able to answer:

- Why did this request fail?
- Why did the AI return an incorrect answer?
- Which documents did retrieval return, and why?
- Which model, prompt, tools, and agents contributed to the result?
- Where did latency and cost increase?
- Did the workflow resume correctly after a crash?
- Was the correct tenant's data used?
- Can we identify the root cause without exposing sensitive customer data?

The staff-level principle is that observability must cover the entire request lifecycle, from the API boundary to the final AI response and every dependency in between.

For your enterprise financial compliance and multi-agent RAG architecture, I would design observability across six layers:

1\. User request and API

FastAPI · authentication · tenant scope · HTTP errors

2\. Workflow and agents

LangGraph nodes · transitions · retries · checkpoints · tool calls

3\. RAG pipeline

Ingestion · embeddings · vector search · reranking · context assembly

4\. Model execution

Model version · prompt version · tokens · latency · output validation

5\. Infrastructure and data

PostgreSQL · Qdrant · Redis · queue · Kubernetes · external APIs

6\. Unified observability

Traces + metrics + structured logs + AI quality evaluations + alerts

Let's design this as if you were implementing it in a real enterprise AI service.

## 1. Understand the three pillars — and the AI-specific fourth

| Signal             | What it answers                           | Example                                                     |
| ------------------ | ----------------------------------------- | ----------------------------------------------------------- |
| Metrics            | Is the system healthy?                    | P95 latency increased from 2 to 8 seconds                   |
| Distributed traces | Where did the request spend time or fail? | Qdrant took 4 seconds                                       |
| Structured logs    | What happened at a specific stage?        | Reranker returned zero usable documents                     |
| AI evaluations     | Was the answer actually good?             | Correct documents retrieved, but answer contradicted policy |

Traditional application observability covers the first three. AI systems need additional quality and behavior measurements because a request can return HTTP 200 and still produce a dangerous or completely incorrect answer.

For example, a compliance API may have a 99.9% success rate while 8% of its answers cite the wrong policy. Infrastructure metrics alone will not reveal that failure.

## 2. Establish a consistent request identity

Every request should be traceable across FastAPI, LangGraph, retrieval, model calls, database queries, and background workers.

Use these identifiers:

- `trace_id`: correlates distributed operations for one trace.
- `request_id`: identifies the incoming API request.
- `workflow_id`: identifies a durable AI workflow, including resumptions.
- `tenant_id`: scopes the request to an authorized tenant.
- `model_version`, `prompt_version`, and `index_version`: identify the components used to produce the result.

Do not confuse these identifiers. A single workflow may run across several requests and therefore several traces.

For a resumable workflow, each execution can have a new trace ID while retaining the same workflow ID.

### Example structured log

```
{
  "event": "llm_generation_completed",
  "request_id": "req-8f21",
  "workflow_id": "wf-2901",
  "trace_id": "trace-a123",
  "stage": "generation",
  "model": "configured-model",
  "prompt_version": "compliance-v12",
  "duration_ms": 1240,
  "input_tokens": 3200,
  "output_tokens": 450,
  "outcome": "success"
}
```

This is illustrative telemetry, not a real request.

Security requirement: never log authentication tokens, API keys, raw confidential documents, or unrestricted model prompts. Tenant IDs and workflow IDs should be access-controlled and should not become unbounded metric labels.

## 3. Instrument FastAPI with OpenTelemetry

I would use OpenTelemetry as the vendor-neutral instrumentation layer and export traces, metrics, and logs to an observability backend.

A common stack is:

[See OpenTelemetry: Project Status, How to Get Started and How to Drive Adoption at CNCF Cloud Native Tel Aviv](https://images.openai.com/static-rsc-4/igH4kB511MdWdmjN0p3kDqXk3Foqt8ntG65pxkseZhKY-hBcVcov96-B8Pviy6FULIZ5NYK9uzO18YaND0uURbLLm2dPCbexjxmYJS3EQTCdSuoVoNL1pGecPvUvQ2nHe8sSTVPOuCO96DE4IfaN4f7wlmrPWv_F85CUPIDF7Xk?purpose=inline)

OpenTelemetry

Instrument the application and propagate trace context across services.

[File:Grafana logo.svg - Wikimedia Commons](https://images.openai.com/static-rsc-4/wFlazXwUNtzBWhhYsfsO-iUHvVSX5psinU3cIyRBrPH-55jhN6vA1HJ-mF2Hp100x3Tof5pLNGXUx0-pTz7u8q_JpmVgKgCX44IZUc_4hgiVbRagjB30d5GhTLYc6_wu5rfBVxWTQQ0KKNKianfSW18M10GjLFTQJ_SxujesfgQ?purpose=inline)

Grafana

Visualize metrics, service health, latency, and alerts.

[Jaeger Logo PNG Vector (SVG) Free Download](https://images.openai.com/static-rsc-4/2Oq0tVApWzeUwSHfq9G6KPgdFTmiDdqaEqyjmWb3k1Q_m27GQgyngRvAPQx1oofZmWudkJZj2dbBrctWsitVWZvUWShEbDK98O6o6K-Z_1InfJfvtXNeg5bAwXJfKZSsKJgkzVzeQkO6K8tFk-UAd-v9NdjPtT8q2Wl8VsN8WjA?purpose=inline)

Jaeger

Inspect end-to-end traces and locate slow or failing spans.

You can also use a managed platform such as Datadog, New Relic, or an OpenTelemetry-compatible backend. The important part is consistent instrumentation and correlation, not the vendor.

### Install the basic instrumentation

```
pip install \
  fastapi uvicorn \
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-otlp \
  opentelemetry-instrumentation-fastapi
```

Create `app/observability/telemetry.py`:

```
from opentelemetry import tracefrom opentelemetry.sdk.resources import Resourcefrom opentelemetry.sdk.trace import TracerProviderfrom opentelemetry.sdk.trace.export import BatchSpanProcessorfrom opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (    OTLPSpanExporter,)def configure_tracing() -> None:    resource = Resource.create({        "service.name": "financial-compliance-api",        "service.version": "1.4.0",        "deployment.environment.name": "production",    })    provider = TracerProvider(resource=resource)    exporter = OTLPSpanExporter(        # Uses OTEL_EXPORTER_OTLP_ENDPOINT and associated settings.    )    provider.add_span_processor(        BatchSpanProcessor(exporter)    )    trace.set_tracer_provider(provider)
```

The endpoint configuration should come from your deployment environment, not be hardcoded. Configure the SDK only once per process, and follow the official setup for TLS, authentication, sampling, and exporter shutdown.

Instrument FastAPI:

```
from fastapi import FastAPIfrom opentelemetry.instrumentation.fastapi import (    FastAPIInstrumentor,)from app.observability.telemetry import configure_tracingconfigure_tracing()app = FastAPI()FastAPIInstrumentor.instrument_app(app)
```

This establishes server-side HTTP spans. It does not automatically instrument every operation in your application, and database, vector-store, model-provider, and queue instrumentation may need separate integrations.

For distributed traces to cross service boundaries, propagate W3C Trace Context through supported HTTP clients and message queues. A worker should extract the trace context from the message and continue the trace where appropriate, or create a linked trace if the execution is intentionally detached.

## 4. Add spans around every important AI stage

A trace is useful only if it tells you where time was spent and what failed.

For a RAG request, I would create spans for:

- Query rewriting
- Query embedding
- Vector retrieval
- Metadata and authorization filtering
- Reranking
- Context assembly
- Each LLM call
- Output parsing and validation
- Agent execution and tool calls
- Final response construction

### Example: instrument retrieval and generation

Create `app/observability/tracing.py`:

```
from time import perf_counterfrom opentelemetry import tracetracer = trace.get_tracer("financial-compliance")async def retrieve_with_tracing(    retriever,    query: str,    tenant_scope: str,):    with tracer.start_as_current_span("rag.retrieve") as span:        started = perf_counter()        try:            documents = await retriever.retrieve(                query=query,                tenant_scope=tenant_scope,            )            span.set_attribute(                "rag.retrieval.result_count",                len(documents),            )            span.set_attribute(                "rag.retrieval.outcome",                "success",            )            return documents        except Exception as exc:            span.set_attribute(                "error.type",                type(exc).__name__,            )
```

You can instrument the LLM call similarly:

```
async def generate_with_tracing(llm, messages):    with tracer.start_as_current_span("llm.generate") as span:        started = perf_counter()        try:            response = await llm.generate(messages)            # Populate these from actual provider metadata.            span.set_attribute(                "llm.provider",                response.provider,            )            span.set_attribute(                "llm.model",                response.model,            )            span.set_attribute(                "llm.input_tokens",                response.input_tokens,            )            span.set_attribute(                "llm.output_tokens",                response.output_tokens,            )            return response        except Exception as exc:            span.set_attribute(                "error.type",                type(exc).__name__,            )            raise        finally:            span.set_attribute(
```

Here, `retriever.retrieve()` and `llm.generate()` are application interfaces; the response metadata fields are illustrative and must match your provider adapter. If your SDK exposes token usage differently, normalize it in that adapter.

Notice what we do not record in these spans: raw document contents, the full prompt, API credentials, or complete LLM responses.

For debugging retrieval, you can record a restricted set of candidate document identifiers, index version, filter version, and relevance scores in a secure diagnostic store. Avoid making sensitive document identifiers or unbounded request-specific values metric labels.

## 5. Instrument LangGraph and multi-agent workflows

In an agentic system, tracing only the outer API request is insufficient. You need to understand how the workflow reached its final decision.

For each graph node, capture:

- Node or agent name.
- Workflow ID and graph version.
- Start time, end time, and execution outcome.
- Retry count and termination reason.
- Tool name and tool outcome.
- Checkpoint or interrupt events.
- Input/output schema validation results.
- Whether the node was resumed or executed for the first time.

Example node instrumentation:

```
from opentelemetry import tracetracer = trace.get_tracer("financial-compliance.workflow")async def compliance_agent_node(state):    with tracer.start_as_current_span(        "agent.compliance_review"    ) as span:        span.set_attribute(            "workflow.id",            state["workflow_id"],        )        span.set_attribute(            "agent.name",            "compliance_review",        )        result = await run_compliance_review(state)        span.set_attribute(            "agent.outcome",            "success",        )        return {            "compliance_result": result,        }
```

This is a minimal pattern. Production wrappers should also capture errors, validate node output, and record the execution attempt.

### Observe graph-level behavior

For LangGraph, track:

| Metric                      | Why it matters                        |
| --------------------------- | ------------------------------------- |
| Workflow completion rate    | Are workflows finishing successfully? |
| Workflow duration           | Are requests getting slower?          |
| Node execution duration     | Which agent is slow?                  |
| Retry count                 | Is a node repeatedly failing?         |
| Steps per workflow          | Are agents looping unnecessarily?     |
| Interrupted workflow count  | Are human approvals accumulating?     |
| Oldest pending workflow age | Are jobs stuck?                       |
| Checkpoint write failures   | Could recovery state be lost?         |

A particularly useful signal is steps per successful workflow. If a workflow normally takes six node executions but suddenly takes 25, you may have a routing loop, a retrieval-grader rework cycle, or excessive retries.

LangGraph checkpoint events should be correlated with workflow state transitions. The checkpoint itself is durable state, not a substitute for execution traces or audit records.

## 6. Measure AI-specific metrics, not just infrastructure health

Your metrics should help distinguish system availability from AI quality.

### Metrics worth tracking

Latency

End-to-end P50/P95/P99, retrieval latency, reranking latency, model latency, queue wait, and time to first token.

Reliability

HTTP error rate, provider timeouts, schema-validation failures, retries, tool errors, and checkpoint failures.

Cost and capacity

Input/output tokens, cost per request, cost per successful task, concurrency, queue depth, and rate-limit usage.

Retrieval quality

Recall\@k, precision\@k, nDCG, empty retrieval rate, reranker changes, and document freshness.

Answer quality

Groundedness, citation correctness, answer relevance, policy compliance, abstention quality, and human-review disagreement.

### Implement metrics with OpenTelemetry

```
from opentelemetry import metricsmeter = metrics.get_meter("financial-compliance")request_counter = meter.create_counter(    "ai_requests_total",    description="AI requests by outcome",)request_duration = meter.create_histogram(    "ai_request_duration_ms",    unit="ms",    description="End-to-end AI request duration",)schema_failure_counter = meter.create_counter(    "ai_schema_validation_failures_total",    description="LLM output schema failures",)
```

Record the measurements:

```
import timeasync def process_request(request, service):    started = time.perf_counter()    outcome = "error"    try:        result = await service.run(request)        outcome = "success"        return result    except Exception:        schema_failure_counter.add(            0,            {"reason": "not_recorded"},        )        raise    finally:        duration_ms = (            time.perf_counter() - started        ) * 1000        request_counter.add(            1,            {"outcome": outcome},        )        request_duration.record(            duration_ms,            {"outcome": outcome},        )
```

In real code, the exception handler must distinguish schema failures from other failures. The zero-valued counter update above is not useful instrumentation; remove it and increment the schema-failure counter only at the actual validation failure boundary:

```
try:    result = ComplianceResult.model_validate_json(raw_output)except ValidationError:    schema_failure_counter.add(        1,        {"reason": "validation_error"},    )    raise
```

Use bounded labels such as `outcome`, `model_family`, and `stage`. Avoid `request_id`, `workflow_id`, raw queries, document IDs, or tenant IDs as metric dimensions because they create high cardinality and can leak sensitive information.

Also, token usage should come from actual provider usage metadata, not an estimate presented as an exact count. Estimates are still useful when clearly labeled.

## 7. Make RAG quality observable

This is where AI observability differs significantly from normal backend monitoring.

A RAG endpoint can return a successful response, consume the expected number of tokens, and still answer incorrectly.

You need to observe two distinct things:

1. Retrieval quality: did the system find the right evidence?
2. Generation quality: did the model use that evidence correctly?

For example, suppose a user asks whether a financial product can advertise guaranteed returns.

The retriever finds the correct policy, but the LLM says the claim is allowed. This is not a retrieval failure. It is a generation, reasoning, or policy-enforcement failure.

Conversely, if the correct policy never reaches the LLM, changing the model may not help.

### Build an evaluation pipeline

For sampled requests or a fixed evaluation dataset, record:

```
evaluation_record = {    "request_id": "req-8f21",    "retrieval_recall_at_5": 1.0,    "answer_relevance": 0.94,    "citation_correctness": 0.50,    "groundedness": 0.70,    "schema_valid": True,}
```

These scores are illustrative. The evaluation implementation must calculate them using known relevant documents, verified citations, deterministic rules, or calibrated evaluators.

Use different evaluation methods for different questions:

| Question                                  | Recommended method                                      |
| ----------------------------------------- | ------------------------------------------------------- |
| Was the expected document retrieved?      | Recall\@k against labeled relevant documents            |
| Is the citation real?                     | Deterministic document/chunk ID verification            |
| Does the cited passage support the claim? | Evidence entailment evaluation plus human audits        |
| Is the answer relevant?                   | Representative labeled dataset and calibrated evaluator |
| Did the response obey the schema?         | Deterministic schema validation                         |
| Was the decision compliant?               | Deterministic policy rules and qualified human review   |

LLM-as-judge can help evaluate semantic quality at scale, but it is itself fallible. Calibrate it against human judgments and monitor evaluator drift.

Do not block every live request on an expensive evaluation model. For production, run deterministic checks synchronously when required for safety, and run more expensive evaluations asynchronously on selected traffic, failures, and regression datasets.

## 8. Observe document ingestion and synchronization

Your observability design must also cover the ingestion pipeline.

For a frequently updated policy repository, record:

- Source document version and content hash.
- Ingestion job ID and outcome.
- Parsing and chunking version.
- Embedding model and version.
- Number of chunks created, updated, and deleted.
- Vector index update status.
- Last successful synchronization time.
- Failed or delayed ingestion jobs.
- Time from source update to searchable index.

Example:

```
{
  "event": "document_indexing_completed",
  "document_id": "policy-101",
  "source_version": "v8",
  "embedding_model_version": "embed-v2",
  "chunks_created": 24,
  "chunks_deleted": 20,
  "index_status": "success",
  "duration_ms": 1830
}
```

Again, this is an illustrative event.

One important metric is document freshness lag:

\\[ \text{Freshness lag} = \text{Index-ready time} - \text{Source-update time} \\]

If the source system has changed a policy but the index has not been updated, your answers may be stale despite healthy API latency.

Also monitor reconciliation between the source of truth and the vector index. A successful ingestion job alone does not prove that all intended documents and deletions were applied.

## 9. Make errors and logs useful for incident investigation

Structured logs should explain what happened without duplicating every piece of data already available in traces.

A practical event schema includes:

```
import logginglogger = logging.getLogger(__name__)def log_validation_failure(    request_id: str,    workflow_id: str,    error_type: str,    schema_version: str,):    logger.warning(        "llm_output_validation_failed",        extra={            "request_id": request_id,            "workflow_id": workflow_id,            "error_type": error_type,            "schema_version": schema_version,        },    )
```

Configure your logging pipeline to serialize these fields as JSON and attach the current trace ID where available.

### Security controls for observability

For a financial AI platform:

- Redact PII and financial account data before exporting telemetry.
- Never store API keys, bearer tokens, or secrets in logs or traces.
- Restrict access to prompts, retrieved chunks, and raw outputs.
- Encrypt telemetry in transit and at rest.
- Define retention and deletion policies.
- Audit access to sensitive diagnostic payloads.
- Prevent untrusted document text from becoming log instructions or executable content.
- Avoid returning internal exception details to API consumers.

For example, rather than logging the entire invalid LLM response, log the error category and sanitized validation paths. A separate, access-controlled diagnostic record may be appropriate when deeper investigation requires it.

## 10. Define SLOs and actionable alerts

Do not alert on every exception. Alert on failures that affect users, reliability, safety, or business outcomes.

An SLO is a target for service behavior over a defined window. An alert should identify a condition that requires action.

| Alert                | Example condition                                   | Response                                             |
| -------------------- | --------------------------------------------------- | ---------------------------------------------------- |
| API availability     | Error-budget burn rate exceeds threshold            | Investigate service health and dependencies          |
| Latency              | P95 exceeds the service objective                   | Inspect trace stage timings                          |
| Model provider       | Timeout or rate-limit rate rises sharply            | Check provider health, concurrency, and retry policy |
| Retrieval quality    | Recall on a fixed evaluation set falls below target | Inspect ingestion, index, filters, and ranking       |
| Output schema        | Validation failure rate exceeds baseline            | Check model, prompt, truncation, and schema changes  |
| Workflow reliability | Oldest queued job exceeds its SLO                   | Inspect worker capacity and stuck jobs               |
| Document freshness   | Index lag exceeds policy freshness objective        | Inspect ingestion and reconciliation                 |
| Security             | Cross-tenant access check fails                     | Contain immediately and investigate exposure         |

The numerical thresholds should come from your service's baseline, business requirements, and risk assessment rather than arbitrary defaults.

For high-risk compliance systems, also monitor human-review backlog, unauthorized action attempts, citation failures, and policy-evaluation disagreements.

## 11. Build dashboards around operational questions

I would create four dashboards rather than one giant dashboard.

### Dashboard A — Service health

Availability

# Success rate

By endpoint and release

Latency

# P50 / P95 / P99

By stage and dependency

Also include provider errors, database pool saturation, queue depth, and worker restarts.

### Dashboard B — AI quality

Retrieval

# Recall\@k

Labeled evaluation set

Generation

# Groundedness

Sampled and audited

Also include citation correctness, abstention behavior, schema failures, and quality by model/prompt version.

### Dashboard C — Workflow operations

Recovery

# Retry rate

By node and failure class

Backlog

# Oldest job age

Queued and awaiting approval

Also include checkpoint errors, graph step counts, interrupted workflows, lease expirations, and failed resumes.

### Dashboard D — Cost and capacity

Efficiency

# Cost / success

By task and model

Capacity

# Token throughput

By provider and model

Also include tokens per request, retries per success, cache effectiveness, and queue wait time.

The values displayed here are dashboard concepts, not measurements from your running application.

## 12. How to investigate a production issue using observability

Suppose your team reports that compliance answers have become slower and less accurate.

Here is the sequence I would follow:

1. Check the service dashboard to determine when latency and quality changed.
2. Compare the incident timeline with deployments, model changes, index updates, and configuration changes.
3. Open representative failed-request traces and identify the slowest or failing span.
4. Inspect retrieval results and confirm document freshness and authorization scope.
5. Compare the rendered prompt, model metadata, raw response, and validation outcome.
6. Check agent transitions, retries, checkpoint recovery, cache hits, and final response construction.
7. Reproduce the failure using a fixed request and controlled dependencies.
8. Apply a mitigation, verify recovery, and add a regression test.

The key is that metrics tell you where to investigate, traces tell you where execution diverged, logs explain what happened, and evaluations tell you whether the AI output was correct.

## 13. Roll out observability incrementally

You don't need to instrument everything perfectly on day one. I would implement it in this order:

1. Establish the baseline. Instrument FastAPI request duration, status codes, structured errors, and trace propagation.
2. Instrument dependencies. Add spans and metrics for PostgreSQL, Redis, Qdrant, embedding calls, model calls, and queue operations.
3. Instrument the AI workflow. Add LangGraph node spans, model/prompt/schema versions, token usage, retries, checkpoints, and tool outcomes.
4. Add quality evaluation. Create a representative, labeled RAG evaluation dataset and track retrieval and answer quality separately from availability.
5. Create SLOs and alerts. Establish measurable reliability, latency, freshness, and quality objectives, then configure actionable alerts.
6. Test observability itself. Inject failures and confirm that you can identify them from the emitted telemetry without exposing sensitive data.

For deployment, I would use OpenTelemetry Collector or an equivalent managed collection pipeline to centralize telemetry, apply redaction and filtering, and export to the selected backend. Configure sampling thoughtfully: head sampling is useful for controlling volume, while tail sampling can retain traces with errors or unusual latency. Tail sampling requires the relevant spans to reach the same collector decision point and needs adequate buffering capacity.

For AI quality, use a separate evaluation pipeline because semantic evaluations are more expensive than ordinary metrics and should not necessarily run on every live request.

## 14. What makes this staff-level rather than just adding logs?

A senior engineer adds instrumentation to components. A staff engineer designs a system in which those components can be understood together.

The distinction is:

- Correlation: one request can be followed across services and workflow resumptions.
- Causality: the team can identify the earliest incorrect or slow stage, rather than guess.
- Quality visibility: successful HTTP responses are not confused with correct AI answers.
- Operational ownership: alerts have defined actions, owners, and recovery procedures.
- Safety: observability does not expose confidential prompts, documents, credentials, or tenant data.
- Continuous verification: evaluation and regression gates catch quality degradation before or shortly after deployment.

### Interview-ready answer

> I design observability across the complete AI request lifecycle using distributed traces, structured logs, metrics, and AI-specific evaluation signals. OpenTelemetry provides vendor-neutral instrumentation and context propagation, while a central observability platform supports dashboards, trace analysis, and alerting.
>
> I instrument API boundaries, retrieval, reranking, context assembly, model calls, output validation, agent transitions, external tools, database operations, and background workers. I correlate these stages using trace IDs and stable workflow IDs, and record model, prompt, schema, and index versions so failures can be reproduced.
>
> Beyond infrastructure metrics, I monitor retrieval recall, citation correctness, groundedness, schema violations, token usage, cost per successful request, and workflow recovery. I define SLOs and alerts based on user impact, while keeping sensitive payloads out of ordinary telemetry.
>
> Finally, I validate the observability design through failure injection and controlled incident exercises. The goal is not merely to collect more logs; it is to reduce the time required to detect, isolate, explain, and resolve production AI failures.

My recommendation for your enterprise AI projects: start with OpenTelemetry, FastAPI instrumentation, explicit spans around your LangGraph and RAG stages, PostgreSQL/Qdrant/Redis monitoring, and a labeled AI evaluation dataset. Those building blocks give you a strong foundation before adding more specialized LLM tracing or evaluation platforms.