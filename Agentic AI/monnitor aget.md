Monitoring AI agents in production is one of the biggest differences between a demo and a production system. Senior AI engineers don't just monitor **CPU, memory, and API latency**—they also monitor **reasoning quality, tool usage, retrieval quality, hallucinations, costs, and user satisfaction**.

A production monitoring architecture typically looks like this:

```text
                    User
                      │
                      ▼
                 API Gateway
                      │
                      ▼
              Agent Orchestrator
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
     LLM         Tool Calls        Memory
      │               │                │
      └───────────────┼────────────────┘
                      ▼
          Observability Layer
                      │
    ┌─────────┬──────────┬───────────┐
    ▼         ▼          ▼           ▼
 Metrics     Logs      Traces      Alerts
    │
    ▼
Prometheus + Grafana + OpenTelemetry
```

---

# What Should Be Monitored?

Think of monitoring in **seven layers**.

## 1. Request Metrics

Track every incoming request.

```text
Request ID
User ID
Session ID
Timestamp
Model Used
Prompt Tokens
Completion Tokens
Latency
Status
```

Example log:

```json
{
  "request_id": "req-101",
  "user_id": "user42",
  "latency_ms": 1820,
  "prompt_tokens": 1532,
  "completion_tokens": 280,
  "model": "gpt-4.1"
}
```

---

## 2. Tool Usage

Record every tool invocation.

```text
User

↓

Agent

↓

SQL Tool

↓

Search Tool

↓

Email Tool
```

Example:

```json
{
  "tool": "search_docs",
  "latency_ms": 85,
  "success": true
}
```

Questions to monitor:

* Which tools are used most?
* Which tools fail often?
* Average tool latency?
* Retry count?
* Timeout count?

Python example:

```python
import time
import logging

logger = logging.getLogger(__name__)

def monitored_tool(tool_name, func, *args):
    start = time.time()

    try:
        result = func(*args)

        logger.info({
            "tool": tool_name,
            "latency": time.time() - start,
            "status": "success"
        })

        return result

    except Exception as e:
        logger.error({
            "tool": tool_name,
            "status": "failed",
            "error": str(e)
        })
        raise
```

---

# 3. LLM Metrics

Track model performance.

Important metrics:

```text
Prompt Tokens

Completion Tokens

Latency

Cost

Model Version

Temperature

Stop Reason
```

Example

```json
{
    "model":"gpt-4",
    "latency":2.3,
    "cost":0.013,
    "tokens":2124
}
```

Useful dashboards:

```
Average latency

P95 latency

P99 latency

Tokens/request

Cost/day

Cost/user

Cost/model
```

---

# 4. Agent Reasoning

Monitor the reasoning steps.

Example

```text
User

↓

Thought

↓

Search Tool

↓

Observation

↓

SQL Tool

↓

Observation

↓

Final Answer
```

Log

```json
{
  "step":1,
  "reason":"Need order details",
  "tool":"sql"
}
```

Then

```json
{
  "step":2,
  "reason":"Need shipment",
  "tool":"shipping"
}
```

Questions

* Average reasoning steps?
* Infinite loops?
* Failed plans?
* Tool switching?

---

# 5. Retrieval Monitoring (RAG)

Monitor retrieval separately from generation.

Metrics

```
Retrieved documents

Similarity score

Context Precision

Context Recall

Hit Rate

Miss Rate
```

Example

```json
{
    "query":"refund policy",
    "top_k":5,
    "avg_similarity":0.87
}
```

If similarity suddenly drops

```
0.89

↓

0.45
```

Your embeddings or index may be stale.

---

# 6. Hallucination Monitoring

One of the hardest problems.

Suppose retrieved document says

```
Refund takes 5 days.
```

LLM answers

```
Refund takes 30 days.
```

Hallucination.

Monitor

```
Grounded answer %

Faithfulness score

Citation accuracy

Unsupported claims
```

Production systems often sample responses and periodically evaluate them using automated evaluators plus human review.

---

# 7. User Feedback

Collect user signals.

```
👍

👎

Reported Incorrect

Retry

Regenerate

Abandoned Chat
```

Questions

```
Success rate?

User satisfaction?

Conversation length?

Escalation rate?
```

---

# OpenTelemetry Integration

OpenTelemetry is the standard for distributed tracing.

Example

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def invoke_agent(query):
    with tracer.start_as_current_span("agent"):
        return run_agent(query)
```

Tool trace

```python
with tracer.start_as_current_span("search_tool"):
    docs = search(query)
```

Trace

```text
HTTP Request

↓

Agent

↓

LLM

↓

Search

↓

SQL

↓

Email

↓

Response
```

Every step gets its own latency.

---

# Prometheus Metrics

```python
from prometheus_client import Counter

requests = Counter(
    "agent_requests_total",
    "Total Requests"
)

tool_errors = Counter(
    "tool_errors_total",
    "Tool Errors"
)

requests.inc()
```

Latency

```python
from prometheus_client import Histogram

latency = Histogram(
    "agent_latency_seconds",
    "Latency"
)

with latency.time():
    agent.invoke(query)
```

---

# Grafana Dashboard

Senior AI teams usually build dashboards showing:

```
Requests/sec

Average latency

P95 latency

P99 latency

Tokens/sec

Cost/hour

Tool failures

Retrieval accuracy

Hallucination %

Memory usage

GPU utilization
```

---

# Logging

Every request should produce structured logs.

Example

```json
{
 "request":"Refund policy",
 "tool":"search",
 "latency":0.32,
 "tokens":1789,
 "cost":0.006,
 "success":true
}
```

Avoid logging sensitive user data. Mask or omit personally identifiable information (PII) and secrets.

---

# Alerting

Examples

```
LLM latency > 5 seconds

↓

Alert
```

```
Tool failures > 10%

↓

Alert
```

```
Daily cost > $500

↓

Alert
```

```
Hallucination rate ↑

↓

Alert
```

```
Retriever hit rate ↓

↓

Alert
```

---

# Production Architecture

```text
                User
                  │
                  ▼
            FastAPI Gateway
                  │
                  ▼
              Agent Service
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
     LLM        RAG         Tools
      │           │            │
      └───────────┼────────────┘
                  ▼
         OpenTelemetry Traces
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
  Prometheus    Loki       Jaeger
      │           │            │
      └───────────┼────────────┘
                  ▼
               Grafana
                  │
                  ▼
             Alertmanager
                  │
                  ▼
          Slack / PagerDuty
```

---

# LangGraph Monitoring Example

```python
from langgraph.graph import StateGraph

graph = StateGraph(...)

result = graph.invoke(input)

# Log execution metadata
print(result)
```

Combined with OpenTelemetry, each node in the graph can become a trace span, making it easy to identify slow or failing steps.

---

# What Senior AI Teams Monitor Daily

| Layer          | Key Metrics                                                |
| -------------- | ---------------------------------------------------------- |
| API            | Requests/sec, error rate, latency (P50/P95/P99)            |
| LLM            | Tokens, cost, model version, stop reason                   |
| Agent          | Reasoning steps, tool selection accuracy, loop count       |
| Tools          | Success rate, retries, timeout rate, latency               |
| RAG            | Hit rate, similarity score, context precision/recall       |
| Quality        | Hallucination rate, faithfulness, citation accuracy        |
| Users          | CSAT, thumbs up/down, conversation completion, abandonment |
| Infrastructure | CPU, memory, GPU utilization, queue depth, autoscaling     |

## A Production Observability Stack

A common production stack looks like this:

* **OpenTelemetry**: distributed traces across the agent, tools, and external APIs.
* **Prometheus**: metrics collection (latency, requests, token usage).
* **Grafana**: dashboards and alert visualization.
* **Loki** (or Elasticsearch): structured log aggregation.
* **Jaeger** or **Tempo**: trace visualization.
* **LangSmith**, **Phoenix**, or **Weights & Biases Weave**: AI-specific traces, prompt/version tracking, evaluation, and debugging.
* **MLflow** (if applicable): model versioning and experiment tracking.
* **PagerDuty** or **Slack**: automated alerts for production incidents.

This combination gives you visibility into not only whether the agent is available, but also whether it is **reasoning correctly, using the right tools, retrieving relevant context, staying within budget, and delivering high-quality responses over time**.
