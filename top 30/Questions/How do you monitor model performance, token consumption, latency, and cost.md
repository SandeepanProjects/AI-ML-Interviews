# How do you monitor model performance, token consumption, latency, and cost in a production LLM application?

As a Senior/Staff AI Engineer, I would implement an end-to-end LLM observability and cost-monitoring system that captures every request from the moment it enters FastAPI until the final response is returned.

I would monitor four primary dimensions:

1. Model performance: Is the LLM producing accurate, relevant, grounded answers?
2. Token consumption: How many input, output, and cached tokens are consumed?
3. Latency: How long does retrieval, agent execution, and model generation take?
4. Cost: How much does each request, workflow, user, tenant, and application cost?

For your Enterprise Multi-Agent Financial Advisor Copilot, I would also track agent-level metrics, RAG retrieval quality, tool execution, and workflow failures.

## 1. High-level monitoring architecture

User request

Financial advisor question

FastAPI + LangGraph

RAG

Retrieval and reranking

Agents

Supervisor and specialists

LLM calls

Tokens and model usage

Tools

Execution and failures

OpenTelemetry instrumentation

Trace ID · Span ID · Model metadata · Usage · Duration

Metrics

Prometheus

Traces

OpenTelemetry backend

Logs

CloudWatch

Evaluations

RAGAS / MLflow

Grafana dashboards + alerts

Quality · Latency · Usage · Cost · Reliability

The most important design principle is that every request should have a trace ID, and every LLM invocation should have its own span.

That allows me to investigate exactly which agent or model call caused a latency spike or unexpected cost.

## 2. Monitor model performance

First, I distinguish between two types of model performance.

### A. Operational performance

This tells us whether the model is functioning correctly.

I monitor:

| Metric                   | What it tells us                             |
| ------------------------ | -------------------------------------------- |
| LLM request success rate | Percentage of successful model calls         |
| LLM error rate           | Provider errors, timeouts, invalid responses |
| Retry rate               | How often model calls need retries           |
| Timeout rate             | Percentage of calls exceeding timeout        |
| Fallback rate            | How often another model is selected          |
| JSON schema failure rate | Invalid structured responses                 |
| Refusal rate             | How often the model refuses requests         |

For example, if the LLM error rate suddenly increases from 1% to 15%, I would investigate:

- Provider outage.
- Rate-limit errors.
- Invalid API requests.
- Network issues.
- Model deployment or version changes.
- Context-window violations.

### B. AI quality performance

This tells us whether the model is generating useful answers.

For your RAG application, I would monitor:

| Metric             | Meaning                                       |
| ------------------ | --------------------------------------------- |
| Faithfulness       | Is the answer supported by retrieved context? |
| Answer relevancy   | Does the response address the question?       |
| Context precision  | Are retrieved chunks relevant?                |
| Context recall     | Did retrieval find the required information?  |
| Answer correctness | Does the answer match ground truth?           |
| User feedback      | Are users accepting or rejecting answers?     |

I would use RAGAS for offline evaluation and periodically sample production interactions for quality evaluation.

One important distinction: these quality metrics usually require evaluation jobs, reference answers, human feedback, or an evaluator model. They cannot all be calculated reliably from every production response in real time.

### Example: Production metrics

I would define Prometheus metrics like this:

```
from prometheus_client import Counter, Histogram, GaugeLLM_REQUESTS = Counter(    "llm_requests_total",    "Total LLM requests",    ["provider", "model", "status"],)LLM_LATENCY = Histogram(    "llm_request_duration_seconds",    "LLM request latency",    ["provider", "model"],    buckets=[        0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30    ],)LLM_RETRIES = Counter(    "llm_retries_total",    "Total LLM retries",    ["provider", "model"],)LLM_SCHEMA_FAILURES = Counter(    "llm_schema_failures_total",    "Invalid structured LLM responses",    ["model"],)
```

For every LLM invocation, I would increment the appropriate counter and record the duration.

These metrics help identify model-level operational regressions.

## 3. Monitor token consumption

Token consumption is one of the most important metrics in an LLM application because token usage directly influences inference costs.

I would separately measure:

1. Input tokens.
2. Output tokens.
3. Cached input tokens, where the provider reports them.
4. Reasoning tokens, where the provider reports them separately.
5. Embedding tokens.
6. Tokens consumed by each agent.
7. Tokens consumed by each workflow.

### Example

Suppose a financial advisor request has three LLM calls:

| Agent          | Input tokens | Output tokens |
| -------------- | ------------ | ------------- |
| Supervisor     | 1,000        | 200           |
| Research agent | 3,000        | 500           |
| Risk agent     | 2,000        | 400           |
| Total          | 6,000        | 1,100         |

Total token consumption:

\\[ T\_{total}=T\_{input}+T\_{output} \\]

\\[ T\_{total}=6000+1100=7100 \\]

But I would not only monitor total tokens.

I would also measure token consumption per agent and per workflow.

For example, if the Research agent suddenly starts consuming 10,000 input tokens instead of 3,000, that may indicate:

- Excessive retrieval context.
- Incorrect chunk size.
- Duplicate document retrieval.
- Repeated agent execution.
- Uncontrolled conversation history.
- An infinite agent loop.

### Example: Token metrics

```
INPUT_TOKENS = Counter(    "llm_input_tokens_total",    "Total LLM input tokens",    ["provider", "model", "agent"],)OUTPUT_TOKENS = Counter(    "llm_output_tokens_total",    "Total LLM output tokens",    ["provider", "model", "agent"],)CACHED_TOKENS = Counter(    "llm_cached_input_tokens_total",    "Total cached input tokens",    ["provider", "model"],)
```

Whenever a model returns usage information, I would record it.

For example, with a provider response that exposes usage:

```
usage = response.usageINPUT_TOKENS.labels(    provider="openai",    model="configured-model",    agent="research",).inc(usage.prompt_tokens)OUTPUT_TOKENS.labels(    provider="openai",    model="configured-model",    agent="research",).inc(usage.completion_tokens)
```

This is an illustrative provider-specific pattern. Actual usage fields differ by SDK and model. I would normalize them through a provider adapter rather than assuming every provider exposes identical fields.

Important: Use provider-reported usage as the billing source whenever available. Local tokenization estimates are useful for pre-request budgeting, but may not match provider billing exactly.

### Token metrics I would alert on

- Input tokens per request suddenly increase.
- Output tokens exceed configured limits.
- Average tokens per agent increase significantly.
- One tenant consumes an unexpected amount of tokens.
- A workflow repeatedly invokes the same model.
- Embedding ingestion consumes more tokens than expected.

For financial applications, I would track token consumption per tenant internally, but avoid using tenant IDs as Prometheus labels because that can create high-cardinality metrics. Detailed tenant-level usage belongs in a database or analytics store.

## 4. Monitor end-to-end latency

Latency is not just the time taken by the LLM.

In an agentic RAG application, a request might involve:

```
User Request
     |
     v
FastAPI authentication
     |
     v
Query processing
     |
     v
Vector retrieval
     |
     v
Reranking
     |
     v
Supervisor agent
     |
     v
Research agent
     |
     v
LLM generation
     |
     v
Risk calculation tool
     |
     v
Final response
```

If the total request takes 12 seconds, I need to identify which component is responsible.

### A. Latency metrics

| Metric                     | Meaning                                 |
| -------------------------- | --------------------------------------- |
| API latency                | Total request duration                  |
| Retrieval latency          | Vector search duration                  |
| Reranking latency          | Reranker execution time                 |
| LLM latency                | Model request duration                  |
| Time to first token (TTFT) | Time before first streamed output token |
| Tool latency               | External tool execution duration        |
| Agent latency              | Time spent in each agent                |
| Workflow latency           | Complete LangGraph execution time       |

I would calculate:

- P50: Median latency.
- P95: 95% of requests finish within this duration.
- P99: 99% of requests finish within this duration.

For example:

Example latency distribution

# 1.8s

P50

# 4.5s

P95

# 9.2s

P99

Illustrative numbers. P95 of 4.5 seconds means 95% of measured requests complete within 4.5 seconds.

### B. Implement latency tracking

```
import timefrom prometheus_client import HistogramWORKFLOW_LATENCY = Histogram(    "ai_workflow_duration_seconds",    "Complete AI workflow latency",    ["workflow"],)RETRIEVAL_LATENCY = Histogram(    "rag_retrieval_duration_seconds",    "RAG retrieval latency",)RERANKER_LATENCY = Histogram(    "rag_reranker_duration_seconds",    "Reranker latency",)async def execute_rag_workflow(query: str):    start = time.perf_counter()    try:        # Retrieval        retrieval_start = time.perf_counter()        documents = await retriever.search(query)        RETRIEVAL_LATENCY.observe(            time.perf_counter() - retrieval_start        )        # Reranking        rerank_start = time.perf_counter()
```

This demonstrates the basic instrumentation pattern. In production, I would also instrument individual LangGraph nodes, model calls, tools, and database operations using distributed tracing. Avoid wrapping the same operation in duplicate metric instrumentation.

### C. Why distributed tracing matters

Suppose a request takes 10 seconds.

A trace might show:

| Component      | Duration |
| -------------- | -------- |
| Authentication | 20 ms    |
| Vector search  | 100 ms   |
| Reranking      | 150 ms   |
| Supervisor     | 500 ms   |
| Research agent | 5,500 ms |
| Risk agent     | 2,000 ms |
| Final response | 1,000 ms |
| Other overhead | 730 ms   |

Now I know that the Research agent is the primary bottleneck.

I would investigate whether it is:

- Making multiple sequential LLM calls.
- Retrieving too much context.
- Waiting on an external API.
- Repeating a tool call.
- Using an unnecessarily large model.

This is much more useful than simply knowing that the API is slow.

## 5. Monitor LLM cost

Now let's discuss one of the most important Staff Engineer responsibilities: controlling inference cost.

The cost depends on:

- Model selection.
- Input tokens.
- Output tokens.
- Cached input tokens.
- Reasoning tokens, if separately billed.
- Number of LLM calls.
- Embedding generation.
- Reranking.
- Agent retries and repeated execution.

### A. Understand the cost calculation

A simplified pricing calculation is:

\\[ Cost = \frac{InputTokens}{1,000,000}\times InputRate + \frac{OutputTokens}{1,000,000}\times OutputRate \\]

Here, rates are expressed as currency per million tokens.

For example, assume a hypothetical model pricing configuration:

- Input: $2 per million tokens.
- Output: $8 per million tokens.

One request consumes:

- 6,000 input tokens.
- 1,100 output tokens.

Then:

\\[ InputCost = \frac{6000}{1,000,000}\times2 \\]

\\[ InputCost = \\$0.012 \\]

\\[ OutputCost = \frac{1100}{1,000,000}\times8 \\]

\\[ OutputCost = \\$0.0088 \\]

Illustrative request cost

Input cost

$0.0120

Output cost

$0.0088

Total

# $0.0208

Hypothetical rates for explaining the calculation, not current provider pricing.

If we process 100,000 such requests:

\\[ 100000 \times 0.0208 = \\$2080 \\]

This is only the model inference cost. It excludes infrastructure, embedding generation, reranking, storage, network transfer, and observability.

### B. Cost tracking code

I would maintain a centralized pricing configuration rather than hardcoding rates throughout the application.

`app/observability/pricing.py`

```
from dataclasses import dataclass@dataclass(frozen=True)class ModelPricing:    input_per_million: float    output_per_million: float    cached_input_per_million: float = 0.0MODEL_PRICING = {    "example-model": ModelPricing(        input_per_million=2.0,        output_per_million=8.0,        cached_input_per_million=0.5,    ),}def calculate_cost(    model: str,    input_tokens: int,    output_tokens: int,    cached_input_tokens: int = 0,) -> float:    pricing = MODEL_PRICING[model]    if min(input_tokens, output_tokens, cached_input_tokens) < 0:        raise ValueError("Token counts cannot be negative")    if cached_input_tokens > input_tokens:        raise ValueError("Cached tokens cannot exceed input tokens")    regular_input = input_tokens - cached_input_tokens
```

The example separates cached input from regular input so cached tokens are not charged twice.

A production pricing engine must also account for provider-specific pricing rules, model versions, batch pricing, regional pricing, image/audio tokens, reasoning tokens, and any other separately billed categories. Rates must be maintained with effective dates and currency.

### C. Record cost metrics

```
from prometheus_client import CounterLLM_COST = Counter(    "llm_cost_usd_total",    "Estimated cumulative LLM cost in USD",    ["provider", "model"],)def record_llm_cost(    provider: str,    model: str,    cost: float,):    if cost < 0:        raise ValueError("Cost cannot be negative")    LLM_COST.labels(        provider=provider,        model=model,    ).inc(cost)
```

This gives us cumulative cost per provider and model.

For tenant-level financial reporting, I would store individual usage events in PostgreSQL or a dedicated analytics store, rather than adding tenant IDs to Prometheus labels.

### D. Monitor cost at multiple levels

I would track:

| Level       | Metric                                  |
| ----------- | --------------------------------------- |
| Request     | Cost per request                        |
| Workflow    | Cost per complete agent workflow        |
| Agent       | Cost per supervisor/research/risk agent |
| User        | Cost per user                           |
| Tenant      | Cost per tenant                         |
| Model       | Cost per model                          |
| Application | Daily/monthly total cost                |

For example:

```
Financial Advisor Copilot
|
+-- Supervisor Agent: $120
|
+-- Research Agent: $450
|
+-- Risk Agent: $200
|
+-- Compliance Agent: $180
|
+-- Embedding Generation: $100
|
+-- Total: $1,050
```

These numbers are illustrative.

This breakdown tells us which component is responsible for the highest expenditure.

## 6. Implement a centralized LLM monitoring wrapper

This is where I would bring token tracking, latency, cost, and model reliability together.

Instead of allowing every agent to independently call the LLM provider and implement its own monitoring, I would create a centralized LLM gateway or provider adapter.

Every agent calls the same abstraction.

Supervisor / Research / Risk / Compliance agents

LLM Gateway

Provider routing · Timeout · Retry · Usage normalization

OpenAI

Anthropic

Usage and telemetry collector

Tokens · Cost · Latency · Errors · Trace context

### Suggested file structure

```
app/
│
├── llm/
│   ├── base.py
│   ├── gateway.py
│   ├── providers/
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   └── usage.py
│
├── observability/
│   ├── metrics.py
│   ├── tracing.py
│   ├── pricing.py
│   ├── usage_repository.py
│   └── monitoring_service.py
│
└── agents/
    ├── supervisor.py
    ├── research.py
    ├── risk.py
    └── compliance.py
```

### Example: Centralized monitoring service

Here is a simplified implementation of the core gateway using an injected async model client.

`app/llm/gateway.py`

```
import timefrom dataclasses import dataclassfrom typing import Anyfrom app.observability.pricing import calculate_costfrom app.observability.metrics import (    LLM_REQUESTS,    LLM_LATENCY,    LLM_RETRIES,    INPUT_TOKENS,    OUTPUT_TOKENS,    LLM_COST,)@dataclassclass LLMResult:    content: Any    model: str    input_tokens: int    output_tokens: int    cost_usd: float    latency_seconds: floatclass LLMGateway:    def __init__(        self,        provider_client,        provider: str,        model: str,    ):        self.client = provider_client        self.provider = provider        self.model = model
```

This is the core pattern, not a complete production gateway. In a real implementation, I would add:

- Provider-specific usage adapters.
- Explicit timeout handling.
- Bounded retries with exponential backoff and jitter.
- Rate-limit handling.
- Circuit breakers and controlled fallback models.
- Cached and reasoning token accounting.
- Streaming usage collection.
- Durable usage-event persistence.
- Request and workflow correlation IDs.
- Cancellation handling.
- Separate recording of failed calls that consumed billable tokens.

One subtle but important point: a request can fail after the provider has already generated and billed tokens. Therefore, I would not assume failed requests have zero cost.

Also, telemetry failures should not silently cause an otherwise successful user request to fail. I would use a non-blocking, durable usage-event pipeline with controlled buffering and reconciliation.

## 7. Monitor LangGraph agent-level consumption

This is particularly important for your multi-agent architecture.

Suppose a single user request triggers:

```
Supervisor
   |
   +---- Research Agent
   |          |
   |          +---- LLM Call 1
   |          +---- LLM Call 2
   |
   +---- Risk Agent
   |          |
   |          +---- LLM Call 3
   |
   +---- Compliance Agent
              |
              +---- LLM Call 4
```

If the entire workflow costs $0.10, I need to understand which agent generated that cost.

I would attach metadata to each invocation:

```
{    "request_id": "req-1001",    "workflow_id": "workflow-2001",    "agent_name": "research_agent",    "model": "example-model",    "input_tokens": 3000,    "output_tokens": 500,    "latency_ms": 1800,    "cost_usd": 0.01}
```

The actual request and workflow IDs should be generated or validated by the application, not supplied as trusted authorization information by the LLM.

### Example: LangGraph instrumentation

LangGraph nodes should be instrumented individually, and model calls should be instrumented by the central gateway.

```
import timeasync def research_node(state, runtime):    started = time.perf_counter()    try:        result = await llm_gateway.generate(            messages=build_research_messages(state),            agent_name="research_agent",        )        return {            "research_result": result.content,        }    finally:        duration = time.perf_counter() - started        record_agent_latency(            agent="research_agent",            duration=duration,        )
```

I would also record:

- Number of node executions.
- Number of model calls per node.
- Number of retries.
- Number of tool calls.
- Number of workflow loops.
- Workflow completion and failure rates.
- Checkpoint resume count.

This helps identify a common production problem: a graph enters an unexpected loop and repeatedly calls the LLM.

For example:

```
Expected:
Supervisor -> Research -> Risk -> Final Answer

Actual:
Supervisor -> Research -> Supervisor
           -> Research -> Supervisor
           -> Research -> Supervisor
```

The token and cost metrics should make this immediately visible.

## 8. Persist usage events for accurate cost accounting

Prometheus is excellent for monitoring and alerting, but I would not use it as the financial ledger for individual requests.

I would persist normalized usage events in PostgreSQL or a dedicated analytics pipeline.

For example:

`llm_usage_events`:

| Column        | Purpose            |
| ------------- | ------------------ |
| event_id      | Unique usage event |
| request_id    | API request        |
| workflow_id   | Agent workflow     |
| tenant_id     | Tenant accounting  |
| agent_name    | Agent responsible  |
| provider      | LLM provider       |
| model         | Model used         |
| input_tokens  | Input consumption  |
| output_tokens | Output consumption |
| cached_tokens | Cached input usage |
| cost_usd      | Calculated cost    |
| latency_ms    | Model latency      |
| status        | Success/failure    |
| created_at    | Event timestamp    |

Example SQL:

```
CREATE TABLE llm_usage_events (
    event_id UUID PRIMARY KEY,
    request_id UUID NOT NULL,
    workflow_id UUID,
    tenant_id UUID NOT NULL,

    agent_name VARCHAR(100) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(150) NOT NULL,

    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    cached_tokens BIGINT NOT NULL DEFAULT 0,

    cost_usd NUMERIC(18, 10) NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL,

    status VARCHAR(30) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

I would index the fields used for reporting:

```
CREATE INDEX idx_usage_tenant_created
ON llm_usage_events(tenant_id, created_at);

CREATE INDEX idx_usage_model_created
ON llm_usage_events(model, created_at);

CREATE INDEX idx_usage_workflow
ON llm_usage_events(workflow_id);
```

For large-scale deployments, I would partition this table by time and consider exporting events into S3 and querying them through Athena or a data warehouse.

### Example: Daily tenant cost

```
SELECT
    tenant_id,
    DATE(created_at) AS usage_date,
    SUM(cost_usd) AS total_cost,
    SUM(input_tokens) AS total_input_tokens,
    SUM(output_tokens) AS total_output_tokens,
    COUNT(*) AS total_model_calls
FROM llm_usage_events
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY tenant_id, DATE(created_at)
ORDER BY usage_date DESC, total_cost DESC;
```

This gives finance and engineering teams a useful view of consumption.

Important production detail: Use an outbox or durable event pipeline so that usage recording does not get lost if a worker crashes. Deduplicate events using a unique event ID or provider invocation ID. Also reconcile estimated application costs with provider billing reports because pricing and usage adjustments can differ.

## 9. Build Grafana dashboards

I would create separate dashboards rather than placing every metric on one screen.

Dashboard 1 — LLM operational health

# 99.4%

Request success rate

# 1.8%

Error rate

# 4.2%

Retry rate

# 0.7%

Schema failure rate

Dashboard 2 — Token consumption

# 8.5M

Input tokens / day

# 2.1M

Output tokens / day

Dashboard 3 — Latency

# 1.2s

P50

# 4.5s

P95

# 8.9s

P99

Dashboard 4 — Cost

# $240

Daily LLM spend

# $0.024

Average cost / request

All dashboard figures are illustrative, not actual application telemetry.

I would also create a model comparison dashboard:

| Model   | Requests | P95 latency | Cost/request | Quality |
| ------- | -------- | ----------- | ------------ | ------- |
| Model A | 10,000   | 2.5s        | $0.02        | 94%     |
| Model B | 10,000   | 1.4s        | $0.008       | 91%     |
| Model C | 10,000   | 3.2s        | $0.04        | 96%     |

This helps answer a business question:

Can I use a cheaper and faster model without sacrificing unacceptable quality?

I would not automatically choose the cheapest model. Model selection must account for task complexity, correctness, safety, and the business impact of errors.

## 10. Configure alerts

Dashboards help us understand what happened. Alerts help us respond before the issue becomes serious.

I would configure alerts for:

| Alert                 | Example condition                          |
| --------------------- | ------------------------------------------ |
| High LLM latency      | P95 above 5 seconds for 10 minutes         |
| High error rate       | Error rate above 5%                        |
| Token spike           | Token usage exceeds expected baseline      |
| Cost spike            | Daily cost exceeds budget                  |
| Agent loop            | Unexpectedly high model calls per workflow |
| Provider outage       | Consecutive provider failures              |
| Retrieval degradation | Retrieval quality falls below threshold    |
| Worker backlog        | Queue depth grows continuously             |
| Schema failures       | Invalid output rate rises                  |
| Security event        | Unauthorized tool execution detected       |

Example Prometheus alert rules:

```
groups:
  - name: llm-alerts

    rules:

      - alert: HighLLMLatency

        expr: |
          histogram_quantile(
            0.95,
            sum by (le, model) (
              rate(llm_request_duration_seconds_bucket[5m])
            )
          ) > 5

        for: 10m

        labels:
          severity: warning

        annotations:
          summary: "LLM P95 latency is too high"


      - alert: HighLLMErrorRate

        expr: |
          (
            sum(rate(llm_requests_total{status="error"}[5m]))
            /
            clamp_min(
              sum(rate(llm_requests_total[5m])),
              0.001
            )
          ) > 0.05

        for: 5m

        labels:
          severity: critical

        annotations:
          summary: "LLM error rate exceeds 5%"
```

These examples assume the histogram and counter metric names match the application instrumentation.

For token and cost alerts, I would evaluate rolling usage and daily budgets in a durable accounting system, then export the resulting aggregate metrics to the monitoring platform. A counter alone cannot enforce a tenant's monthly financial budget.

## 11. How would I investigate a production issue?

Let's take a realistic scenario.

Problem: The LLM application's monthly cost suddenly increases by 3x.

I would investigate systematically.

Production incident: LLM cost spike

1. Check token consumption

   Did input or output tokens increase? Identify which models and agents are responsible.
2. Check number of model calls

   Is the supervisor repeatedly invoking the research agent? Are retries excessive?
3. Check retrieval

   Are we retrieving too many chunks or duplicating context?
4. Check prompt changes

   Did a recent prompt change increase conversation history or context size?
5. Check model routing

   Did requests move from a cheaper model to a more expensive model?
6. Check tenant-level consumption

   Is one tenant generating unusual traffic or oversized workflows?
7. Apply mitigation

   Enforce token budgets, cap workflow iterations, fix routing, and reduce unnecessary context.

Other common incidents:

| Symptom                         | Likely investigation                                |
| ------------------------------- | --------------------------------------------------- |
| Latency doubles                 | Provider latency, retrieval, sequential agent calls |
| Token usage spikes              | Prompt size, context duplication, agent loops       |
| Quality decreases               | Model or prompt change, retrieval regression        |
| Cost per request increases      | Model routing, retries, output length               |
| Error rate increases            | Provider outage, rate limits, timeouts              |
| P99 increases but P50 is stable | Tail latency, queueing, dependency saturation       |

## 12. How would I reduce LLM cost without reducing quality?

I would use several optimization strategies.

1\. Model routing

Use a smaller model for simple classification or summarization and a more capable model for complex financial reasoning.

2\. Prompt optimization

Remove unnecessary instructions, duplicate context, and excessive conversation history.

3\. Retrieval optimization

Retrieve fewer but more relevant chunks. Use reranking to improve relevance rather than sending large amounts of irrelevant text.

4\. Caching

Cache appropriate repeated requests, embeddings, and retrieval results. Cache keys must include tenant and authorization scope.

5\. Limit agent iterations

Set maximum graph steps, tool calls, retries, and model invocations.

6\. Token budgets

Set maximum input and output budgets for each workflow and agent.

7\. Streaming

Stream output where appropriate to improve perceived responsiveness. Streaming does not necessarily reduce total generation cost.

8\. Batch processing

Batch embedding requests and use provider-supported batch inference for suitable offline workloads.

The important principle is to optimize cost against measured answer quality, not cost in isolation.

## 13. Recommended technology stack for your AWS deployment

For your existing architecture, this is the stack I would implement:

| Layer                         | Technology                      |
| ----------------------------- | ------------------------------- |
| API                           | FastAPI                         |
| Agent orchestration           | LangGraph                       |
| LLM abstraction               | Centralized LLM Gateway         |
| Metrics                       | Prometheus                      |
| Distributed tracing           | OpenTelemetry                   |
| Dashboards                    | Grafana                         |
| AWS infrastructure monitoring | CloudWatch                      |
| LLM evaluation                | RAGAS                           |
| Experiment tracking           | MLflow                          |
| Agent tracing                 | LangSmith or Arize Phoenix      |
| Usage accounting              | PostgreSQL                      |
| Background usage processing   | Celery / durable event pipeline |
| Deployment                    | Docker + EKS                    |

I would choose one primary tracing platform and integrate it with OpenTelemetry rather than introducing several overlapping tracing systems.

## 14. Final Senior/Staff AI Engineer interview answer

If the interviewer asks, "How do you monitor model performance, token consumption, latency, and cost?", I would answer:

"I implement centralized observability using OpenTelemetry, Prometheus, Grafana, and CloudWatch.

Every incoming request gets a correlation ID, and every LLM invocation, agent node, retrieval operation, and tool call gets a trace span.

For model performance, I monitor request success rate, error rate, retries, structured-output validity, and AI quality metrics such as faithfulness, answer relevancy, and correctness.

For token consumption, I capture provider-reported input, output, cached, and other billable token categories. I aggregate usage by model, agent, workflow, and tenant.

For latency, I measure end-to-end request duration, P50/P95/P99, time to first token, retrieval latency, reranking latency, individual agent latency, and LLM generation time.

For cost, I maintain a centralized pricing configuration and calculate cost per invocation, workflow, tenant, and model. I persist usage events in a durable accounting store and reconcile application estimates against provider billing.

I build Grafana dashboards and alerts for latency spikes, token consumption, cost anomalies, model failures, and agent loops.

Finally, I use the collected telemetry to optimize model routing, prompt size, retrieval context, caching, and workflow execution.

My goal is not simply to monitor whether the model is running. It is to understand whether the AI application is accurate, fast, reliable, secure, and economically sustainable."

The most important Staff Engineer takeaway: You should be able to trace any expensive or slow request from the API endpoint to the exact agent, model invocation, token usage, tool execution, and cost. Without that visibility, optimizing a production LLM application becomes guesswork.