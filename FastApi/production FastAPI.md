For a production **FastAPI + LLM application**, I would build observability at **three levels**:

1. **API layer** → request latency, throughput, HTTP errors
2. **LLM layer** → TTFT, generation latency, tokens, model errors
3. **Business/RAG layer** → retrieval latency, answer quality, cost per request/tenant

The architecture I'd explain in an interview is:

```text
                         FastAPI
                            │
                ┌───────────┴───────────┐
                │                       │
           Metrics                    Traces
                │                       │
                ▼                       ▼
           Prometheus             OpenTelemetry
                │                       │
                ▼                       ▼
             Grafana            Jaeger / Tempo
                                       
                         Logs
                           │
                           ▼
                  Structured Logging
```

And for the LLM specifically:

```text
Request
  │
  ├── FastAPI latency
  ├── DB latency
  ├── Redis latency
  ├── Qdrant latency
  ├── Embedding latency
  ├── Reranker latency
  ├── LLM TTFT
  ├── LLM generation latency
  ├── Input tokens
  ├── Output tokens
  ├── Model
  ├── Cost
  └── Error / retry / fallback
```

---

# 1. First define the metrics

I wouldn't just say "monitor latency."

I'd define concrete metrics.

### API metrics

```text
http_requests_total
http_request_duration_seconds
http_requests_in_flight
http_errors_total
```

I'd monitor:

```text
P50
P90
P95
P99
```

For example:

```text
P95 latency = 1.8 sec
P99 latency = 4.5 sec
```

P95/P99 are much more useful for production than just average latency.

---

# 2. LLM-specific latency

For an LLM application, I would measure:

### TTFT

**Time To First Token**

```text
Request
  │
  │
  ├──────────────► First token
  │
  └── TTFT ──────┘
```

This is extremely important for streaming applications.

### Total generation latency

```text
Request
  │
  ├──────────────► First token
  │
  │ token token token
  │
  └──────────────────► Final token
```

I'd track:

```text
llm_ttft_seconds
llm_generation_seconds
llm_total_latency_seconds
```

---

# 3. Prometheus metrics

For FastAPI, I'd expose Prometheus metrics.

Conceptually:

```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"],
)
```

Then:

```python
REQUEST_COUNT.labels(
    method="POST",
    endpoint="/chat",
    status="200",
).inc()
```

And:

```python
REQUEST_LATENCY.labels(
    endpoint="/chat"
).observe(duration)
```

Prometheus periodically scrapes:

```text
GET /metrics
```

---

# 4. Middleware for API latency

I would measure request latency centrally using middleware.

```python
import time
from fastapi import Request


@app.middleware("http")
async def metrics_middleware(
    request: Request,
    call_next,
):

    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    return response
```

This gives consistent API-level metrics.

---

# 5. Don't put high-cardinality data into Prometheus labels

This is an important production concern.

Don't do:

```python
Counter(
    "llm_requests",
    "...",
    ["user_id", "request_id", "prompt"]
)
```

because you'll create enormous cardinality.

Instead use low-cardinality labels:

```text
model
provider
endpoint
status
environment
```

Put things like:

```text
request_id
user_id
tenant_id
trace_id
```

in structured logs/traces instead.

---

# 6. Track LLM token usage

For every LLM request, I want:

```text
model
provider
input_tokens
output_tokens
total_tokens
```

For example:

```python
usage = response.usage

input_tokens = usage.prompt_tokens
output_tokens = usage.completion_tokens
total_tokens = usage.total_tokens
```

Then record metrics such as:

```text
llm_input_tokens_total
llm_output_tokens_total
llm_tokens_total
```

---

# 7. Calculate cost

Token usage alone isn't enough.

I need the price of the selected model.

For example:

```python
MODEL_PRICING = {
    "model-a": {
        "input": 0.000005,
        "output": 0.000015,
    },
    "model-b": {
        "input": 0.000001,
        "output": 0.000003,
    },
}
```

Then:

```python
def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
):

    pricing = MODEL_PRICING[model]

    input_cost = (
        input_tokens *
        pricing["input"]
    )

    output_cost = (
        output_tokens *
        pricing["output"]
    )

    return input_cost + output_cost
```

In a real system I'd store pricing/version metadata centrally because provider pricing changes.

---

# 8. Track cost by tenant

For an enterprise SaaS system, this is extremely important.

I would calculate:

```text
Total platform cost
       │
       ├── Tenant A
       ├── Tenant B
       ├── Tenant C
       └── Tenant D
```

For example:

```text
tenant_id = abc
model = primary-model
input_tokens = 20,000
output_tokens = 8,000
cost = $0.42
```

This allows:

```text
Daily cost by tenant
Monthly cost by tenant
Cost by model
Cost by endpoint
Cost by agent
Cost by RAG request
```

---

# 9. Database design for usage/cost

I would persist important usage events in PostgreSQL.

For example:

```python
class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id = mapped_column(UUID, primary_key=True)

    tenant_id = mapped_column(UUID, index=True)
    request_id = mapped_column(UUID, index=True)

    provider = mapped_column(String)
    model = mapped_column(String)

    input_tokens = mapped_column(Integer)
    output_tokens = mapped_column(Integer)

    latency_ms = mapped_column(Integer)
    ttft_ms = mapped_column(Integer)

    cost = mapped_column(Numeric)

    created_at = mapped_column(DateTime)
```

Then I can query:

```sql
SELECT
    tenant_id,
    SUM(cost)
FROM llm_usage
GROUP BY tenant_id;
```

---

# 10. Distributed tracing

Metrics tell me:

> "Latency increased."

Tracing tells me:

> "Why did latency increase?"

I'd use **OpenTelemetry**.

One request might look like:

```text
Trace
│
├── FastAPI /chat              2.8s
│
├── Redis lookup                5ms
│
├── Embedding                   80ms
│
├── Qdrant search               90ms
│
├── Reranker                   200ms
│
└── LLM generation            2.4s
```

This immediately tells me the LLM is the bottleneck.

---

# 11. Propagate a correlation ID

Every request should have a unique identifier.

```text
request_id
trace_id
```

For example:

```python
request_id = str(uuid.uuid4())
```

Put it into:

```text
HTTP response
logs
metrics context
trace
LLM usage record
```

Then I can search:

```text
request_id = abc123
```

and reconstruct the entire request.

---

# 12. Structured logging

I wouldn't use random:

```python
print("LLM failed")
```

I'd use structured logs:

```json
{
  "timestamp": "2026-08-21T12:20:00Z",
  "level": "ERROR",
  "service": "rag-api",
  "request_id": "abc123",
  "tenant_id": "tenant-1",
  "model": "primary-model",
  "error_type": "timeout",
  "retry_count": 2,
  "latency_ms": 30000
}
```

This makes logs searchable in systems such as ELK/OpenSearch or cloud logging platforms.

---

# 13. Monitor LLM errors

I'd separate error categories.

```text
llm_errors_total
    │
    ├── timeout
    ├── rate_limit
    ├── authentication
    ├── invalid_request
    ├── provider_5xx
    ├── content_policy
    └── unknown
```

Also:

```text
llm_retries_total
llm_fallback_total
llm_circuit_breaker_open_total
```

This tells me whether the provider is becoming unreliable.

---

# 14. RAG-specific observability

For your RAG API, I'd monitor retrieval separately.

```text
Question
   │
   ├── embedding latency
   │
   ├── Qdrant latency
   │
   ├── number of documents retrieved
   │
   ├── retrieval scores
   │
   ├── reranker latency
   │
   └── context tokens
```

Metrics could include:

```text
rag_retrieval_latency
rag_retrieved_documents
rag_reranking_latency
rag_context_tokens
```

Then evaluate quality separately:

```text
context precision
context recall
faithfulness
answer relevance
citation accuracy
```

That's important because:

> **A fast RAG system that retrieves the wrong documents is still a bad system.**

---

# 15. Agent observability

If the application uses LangGraph/agents, I'd track:

```text
agent_execution_total
agent_execution_latency
agent_tool_calls_total
agent_tool_failures_total
agent_retries_total
agent_steps
```

A trace might look like:

```text
User request
 │
 └── Agent
      │
      ├── Retriever
      │
      ├── Search tool
      │
      ├── Database tool
      │
      └── LLM
```

You can immediately identify:

```text
tool X is slow
tool Y fails frequently
agent is taking too many steps
LLM is being called unnecessarily
```

---

# 16. Grafana dashboard

I'd create dashboards around four categories.

### API dashboard

```text
Requests/sec
P50 latency
P95 latency
P99 latency
4xx rate
5xx rate
In-flight requests
```

### LLM dashboard

```text
Requests
TTFT
Generation latency
Tokens/sec
Input tokens
Output tokens
429 rate
Timeout rate
Fallback rate
```

### Cost dashboard

```text
Cost/hour
Cost/day
Cost/month
Cost by tenant
Cost by model
Cost by endpoint
Cost per request
Cost per successful answer
```

### RAG dashboard

```text
Embedding latency
Qdrant latency
Reranker latency
Retrieved chunks
Context tokens
Faithfulness
Context precision
Answer relevance
```

---

# 17. Alerting

I wouldn't just collect metrics—I would create alerts.

For example:

```text
P95 latency > 5 seconds for 5 minutes
                ↓
              ALERT
```

```text
5xx rate > 5%
                ↓
              ALERT
```

```text
LLM 429 rate > 10%
                ↓
              ALERT
```

```text
Daily cost > budget
                ↓
              ALERT
```

```text
LLM fallback rate > 20%
                ↓
              ALERT
```

And for AI quality:

```text
Faithfulness drops below threshold
                ↓
              ALERT
```

---

# 18. Cost optimization from observability

This is where monitoring becomes useful operationally.

Suppose Grafana shows:

```text
Model A
Cost/request = $0.08
Latency = 4 sec

Model B
Cost/request = $0.01
Latency = 1 sec
```

And 80% of requests are simple.

I could introduce a model router:

```text
                   Request
                      │
                      ▼
                 Model Router
                 /          \
                /            \
        Simple task       Complex task
             │                 │
             ▼                 ▼
         Cheap model       Powerful model
```

This can significantly reduce cost without degrading important workloads.

---

# 19. Important privacy consideration

I would **not log full prompts and retrieved documents by default**, especially in an enterprise environment.

Instead:

```text
Don't log:
❌ full confidential document
❌ full user prompt
❌ sensitive PII
❌ secrets
```

Log:

```text
✓ request_id
✓ tenant_id
✓ model
✓ token counts
✓ latency
✓ document IDs
✓ retrieval scores
✓ error category
```

If prompt logging is required for debugging, I'd use controlled access, redaction and appropriate retention policies.

---

# 20. Production architecture

Putting it all together:

```text
                         Client
                           │
                           ▼
                       FastAPI
                           │
                     ┌─────┴─────┐
                     │           │
                  Metrics      Traces
                     │           │
                     ▼           ▼
                Prometheus   OpenTelemetry
                     │           │
                     ▼           ▼
                  Grafana    Tempo/Jaeger
                     │
                     │
                     ▼
              ┌───────────────┐
              │  RAG Service  │
              └───────┬───────┘
                      │
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
      Redis        Qdrant       PostgreSQL
        │             │              │
        └─────────────┼──────────────┘
                      ▼
                  LLM Gateway
                      │
              ┌───────┼────────┐
              ▼       ▼        ▼
           Model A  Model B  Model C
                      │
                      ▼
               Usage + Cost
                      │
                      ▼
                 PostgreSQL
```

---

# Interview answer

If asked **"How would you monitor a FastAPI + LLM application for latency, errors, token usage and cost?"**, I'd answer:

> **"I'd implement observability at the API, infrastructure, LLM and RAG layers. At the FastAPI layer I'd collect request rate, P50/P95/P99 latency, in-flight requests and HTTP error rates using Prometheus. I'd use OpenTelemetry for distributed tracing so I can break one request into FastAPI, Redis, PostgreSQL, Qdrant, reranking and LLM spans. For the LLM layer I'd track TTFT, total generation latency, input and output tokens, tokens per second, timeout rate, 429s, retries and fallback rate. I'd calculate cost from token usage and model-specific pricing and persist usage records in PostgreSQL, allowing cost analysis by tenant, model, endpoint and request. For RAG I'd additionally track retrieval latency, number of retrieved chunks, reranker latency, context size and quality metrics such as context precision, recall and faithfulness. Grafana would provide dashboards and alerts for latency, error rates, provider failures and cost budgets. Finally, I would avoid logging sensitive prompts or document contents by default and use request IDs and trace IDs to correlate metrics, logs and traces."**

### The mental model to remember

```text
              OBSERVABILITY
                    │
     ┌──────────────┼───────────────┐
     ▼              ▼               ▼
  Metrics         Traces          Logs
     │              │               │
     ▼              ▼               ▼
 Prometheus     OpenTelemetry   Structured
     │              │            logging
     └──────────────┼───────────────┘
                    ▼
                 Grafana
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  Latency         Errors          Cost
                                   │
                            ┌──────┴──────┐
                            ▼             ▼
                         Tokens        Model
                         usage         pricing
```

**Senior-level takeaway:** don't monitor only whether the API is up. For an LLM system, you need to understand **where the latency comes from, why requests fail, how many tokens you're consuming, what each request/tenant costs, and whether retrieval + generation quality is actually improving.**
