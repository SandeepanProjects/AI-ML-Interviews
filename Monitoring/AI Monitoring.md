This is one of the **most frequently asked Senior AI Engineer interview topics** today.

When companies like OpenAI, Anthropic, Microsoft, AWS, NVIDIA, Databricks, or enterprise AI teams ask:

> **"How do you monitor AI systems in production?"**

they are **not** asking whether you know Prometheus or Grafana.

They are testing whether you understand:

* How to monitor an **end-to-end AI platform**
* How to detect failures before users do
* How to reduce LLM costs
* How to detect hallucinations
* How to monitor prompts
* How to trace requests across multiple services
* How to debug slow AI pipelines

---

# Why AI Monitoring is Different

A normal web application monitors:

```text
CPU
Memory
Requests
Latency
Errors
```

An AI application must also monitor:

```text
CPU
GPU
Memory
Requests
Latency
Errors
↓

Prompt

Context Length

Token Usage

Model

LLM Cost

Vector Search

Retriever Accuracy

Hallucinations

Prompt Injection

Agent Execution

Tool Calls

RAG Quality
```

AI systems have **far more moving parts**.

---

# Production AI Architecture

```
                    User

                      │

                      ▼

               API Gateway

                      │

                      ▼

                 FastAPI

                      │

             OpenTelemetry

                      │

                      ▼

             LangGraph Workflow

      ┌────────────┼─────────────┐

      ▼            ▼             ▼

  RAG Agent    SQL Agent     Web Agent

      │            │             │

      ▼            ▼             ▼

 Qdrant      PostgreSQL     MCP Tools

      └────────────┼─────────────┘

                   ▼

             Model Router

          GPT / Claude / Llama

                   ▼

            Response Validator

                   ▼

                 User
```

Every box is monitored.

---

# What Should Be Monitored?

A senior engineer divides monitoring into seven categories.

```
Infrastructure

Application

LLM

RAG

Agents

Business

Security
```

Let's go through each.

---

# 1. Infrastructure Monitoring

Questions:

* Is CPU overloaded?
* GPU utilization?
* Memory usage?
* Pod crashes?
* Disk usage?
* Network latency?

Prometheus collects these metrics.

Example:

```python
from prometheus_client import Gauge

gpu_utilization = Gauge(
    "gpu_utilization",
    "GPU Utilization"
)

gpu_utilization.set(83)
```

---

# Why Prometheus?

Interview question:

> Why Prometheus?

Junior answer:

> Collects metrics.

Senior answer:

> Prometheus is a pull-based time-series database optimized for high-cardinality operational metrics. It periodically scrapes instrumented applications, stores historical metrics, supports dimensional labels, and integrates with Kubernetes service discovery and Alertmanager.

Architecture:

```
FastAPI

↓

/metrics

↓

Prometheus

↓

Time Series DB

↓

Grafana
```

---

Example

```
requests_total 2345

latency_seconds 0.23

gpu_usage 81%

tokens_generated 250000
```

---

# Instrumenting FastAPI

```python
from prometheus_client import Counter

request_counter = Counter(
    "chat_requests_total",
    "Total Chat Requests"
)

@app.post("/chat")
def chat():

    request_counter.inc()

    return {"message": "ok"}
```

---

# 2. Visualization — Grafana

Prometheus stores numbers.

Humans need dashboards.

That's Grafana.

Interview:

> Why Grafana?

Answer:

Because engineers need real-time visualization, historical analysis, alert dashboards, and operational insights.

Example dashboard

```
Requests/sec

Latency

GPU Usage

Token Usage

Cost

Errors

Agent Failures
```

---

AI Dashboard Example

```
GPT Requests

15,432

Average Latency

1.3 sec

Average Tokens

782

Daily Cost

$246

Hallucination Rate

1.8%

Retriever Recall

94%
```

This dashboard tells you the health of the AI platform.

---

# 3. OpenTelemetry

One of the hottest interview topics.

Question:

Why OpenTelemetry?

Suppose one request goes through:

```
API

↓

Planner

↓

Retriever

↓

Qdrant

↓

LLM

↓

Validator

↓

User
```

Latency is 9 seconds.

Where?

Nobody knows.

OpenTelemetry answers that.

---

Architecture

```
Request

↓

Trace ID

↓

Planner

↓

Retriever

↓

Embedding

↓

LLM

↓

Validator
```

Every service receives the same trace ID.

---

Code

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span(
    "retrieve_documents"
):

    docs = retriever.search(query)
```

Now every request is traceable.

---

# Why Distributed Tracing?

Suppose

```
User

↓

API

↓

LangGraph

↓

RAG

↓

Qdrant

↓

LLM

↓

Redis
```

Without tracing

you only know

```
Total Time

7.2 sec
```

With tracing

```
API

120 ms

Planner

30 ms

Retriever

410 ms

Embedding

15 ms

LLM

5.8 sec

Redis

4 ms
```

Now the bottleneck is obvious.

---

# Logging

Metrics tell you **something is wrong**.

Logs tell you **why**.

Example

```python
import logging

logger = logging.getLogger()

logger.info(
    "Calling GPT-5.5"
)

logger.error(
    "Vector DB timeout"
)
```

Production logs

```
Timestamp

Trace ID

User ID

Prompt ID

Model

Latency

Error
```

Structured logs (JSON) are preferred because they are easy to search.

---

# Metrics

Every production AI system tracks:

```
Requests

Errors

Latency

GPU

Memory

Tokens

Cost

Cache Hit Ratio

Tool Success

Retriever Recall
```

Metrics should be low-cardinality and aggregated for dashboards and alerts.

---

# Alerting

Question:

What if

GPU reaches 95%?

or

Latency becomes 12 seconds?

or

Hallucination rate doubles?

Use Prometheus + Alertmanager.

Example rule

```yaml
groups:
- name: latency
  rules:
  - alert: HighLatency
    expr: request_latency_seconds > 5
    for: 2m
```

Alert Flow

```
Prometheus

↓

AlertManager

↓

Slack

↓

PagerDuty

↓

Engineer
```

Alerts should be actionable and tied to Service Level Objectives (SLOs).

---

# Latency Monitoring

Latency exists everywhere.

```
API

40 ms

Retriever

210 ms

Embedding

25 ms

LLM

4200 ms

Validation

100 ms
```

Total

```
4.6 sec
```

Always measure every stage.

---

# Token Usage Monitoring

AI cost depends on tokens.

Track

```
Prompt Tokens

Completion Tokens

Total Tokens
```

Example

```python
response = llm.invoke(prompt)

print(
    response.usage.prompt_tokens
)

print(
    response.usage.completion_tokens
)
```

Export them to Prometheus.

```
llm_prompt_tokens_total

llm_completion_tokens_total
```

---

# LLM Cost Monitoring

Interview question:

How do you monitor AI cost?

Formula

```
Cost

=

Prompt Tokens × Input Price

+

Completion Tokens × Output Price
```

Example

```python
input_cost = (
    prompt_tokens / 1_000_000
) * INPUT_PRICE

output_cost = (
    completion_tokens / 1_000_000
) * OUTPUT_PRICE

total = input_cost + output_cost
```

Track

```
Cost/User

Cost/Team

Cost/Model

Cost/Day
```

Example dashboard

```
GPT-5.5

$620/day

Claude

$180/day

Llama

$15/day
```

This helps optimize routing and budgeting.

---

# Hallucination Detection

One of the hardest production problems.

Question

How do we detect hallucinations?

Common approaches:

### 1. RAG Groundedness

Did the answer come from retrieved documents?

```
Question

↓

Retriever

↓

Answer

↓

Compare

↓

Grounded?
```

---

### 2. LLM-as-a-Judge

A second model evaluates:

```
Question

Retrieved Context

Answer
```

and scores whether the answer is supported by the context.

---

### 3. Rule-Based Validation

Example

```python
if answer not in retrieved_context:
    flag()
```

Useful for structured domains.

---

### 4. Human Feedback

Random production samples are reviewed by humans to improve prompts and retrievers.

---

# Prompt Monitoring

Interview:

Why monitor prompts?

Because prompts change.

Monitor

```
Prompt Version

Success Rate

Latency

Cost

Tokens

Hallucinations
```

Example

```python
prompt_version = "v12"

logger.info(
    {
        "prompt": prompt_version
    }
)
```

Now you can compare:

```
v11

Accuracy

92%

Cost

$0.03

v12

Accuracy

96%

Cost

$0.02
```

Prompt versioning becomes part of your release process.

---

# Complete Production Monitoring Stack

```
                User Request
                      │
                      ▼
                 FastAPI Service
                      │
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
 Structured Logs   Metrics       OpenTelemetry
        │             │              │
        ▼             ▼              ▼
 Elasticsearch   Prometheus      OTel Collector
        │             │              │
        └─────────────┼──────────────┘
                      ▼
                   Grafana
                      │
              AlertManager
                      │
          Slack / PagerDuty / Email
```

---

# Example Folder Structure

```
monitoring/
│
├── metrics.py
├── tracing.py
├── logging.py
├── alerts/
│   ├── latency.yaml
│   ├── gpu.yaml
│   └── cost.yaml
│
├── dashboards/
│   ├── ai-overview.json
│   ├── llm-cost.json
│   └── rag-health.json
│
├── middleware.py
└── exporters.py
```

---

# Senior AI Engineer Interview Answer (8–10 Minutes)

> "Monitoring an AI system requires observing not only infrastructure but also the complete inference pipeline. At the infrastructure layer, I use Prometheus to collect metrics such as CPU, GPU, memory, request rate, and latency, and Grafana to visualize trends and operational dashboards. For application observability, I instrument every service with OpenTelemetry so that a single trace follows a request across the API layer, LangGraph workflow, retriever, vector database, tool calls, and LLM, making performance bottlenecks easy to identify. I use structured logging with trace IDs to correlate logs with distributed traces. Beyond traditional metrics, I monitor AI-specific indicators such as prompt and completion token counts, latency per model, cost per request, cache hit rate, tool success rate, retrieval quality, and prompt version performance. For quality assurance, I measure groundedness using retrieved context, apply LLM-as-a-judge or rule-based validation to detect potential hallucinations, and track prompt effectiveness over time. Finally, I configure Prometheus Alertmanager to notify engineers when service-level objectives are violated, such as sustained latency, increased hallucination rates, rising costs, or repeated tool failures. This layered observability strategy enables rapid debugging, cost control, and continuous improvement of production AI systems."
