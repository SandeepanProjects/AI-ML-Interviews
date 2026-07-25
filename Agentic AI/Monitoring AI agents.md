Monitoring AI agents in production is significantly more complex than monitoring a traditional REST API because an agent reasons, plans, calls tools, retrieves information, maintains memory, and produces outputs that can vary from one execution to another.

A production AI agent should be monitored across **seven layers**:

```
                   AI Agent Monitoring

                  ┌─────────────────┐
                  │ Business Metrics│
                  └────────▲────────┘
                           │
                  ┌────────┴────────┐
                  │ Agent Metrics    │
                  └────────▲────────┘
                           │
                  ┌────────┴────────┐
                  │ Tool Monitoring │
                  └────────▲────────┘
                           │
                  ┌────────┴────────┐
                  │ LLM Monitoring  │
                  └────────▲────────┘
                           │
                  ┌────────┴────────┐
                  │ Retrieval/RAG   │
                  └────────▲────────┘
                           │
                  ┌────────┴────────┐
                  │ Infrastructure  │
                  └────────▲────────┘
                           │
                  ┌────────┴────────┐
                  │ Security        │
                  └─────────────────┘
```

---

# 1. Infrastructure Monitoring

This is identical to monitoring any distributed system.

Monitor:

* CPU usage
* Memory
* GPU utilization
* Network
* API latency
* Disk
* Kubernetes pods
* Autoscaling
* Queue length

Example Prometheus metrics

```python
from prometheus_client import Gauge

cpu_usage = Gauge("cpu_usage", "CPU usage")
memory_usage = Gauge("memory_usage", "Memory usage")

cpu_usage.set(62)
memory_usage.set(74)
```

Dashboard

```
CPU      58%
RAM      72%
GPU      81%
Pods     14
Latency  180ms
```

---

# 2. LLM Monitoring

This is unique to AI systems.

Track

* Prompt
* Response
* Model used
* Tokens
* Cost
* Latency
* Temperature
* Errors

Example

```python
import time

start = time.time()

response = llm.invoke(prompt)

latency = time.time() - start

log = {
    "model": "gpt-5.5",
    "tokens": response.usage.total_tokens,
    "latency": latency,
    "cost": response.usage.total_tokens * 0.000002
}

print(log)
```

Example output

```
Model      GPT-5.5
Latency    1.3 sec
Input      800 tokens
Output     320 tokens
Cost        ₹0.17
```

---

# 3. Agent Execution Monitoring

Track every reasoning step.

```
User

↓

Planner

↓

Retriever

↓

Calculator Tool

↓

SQL Tool

↓

LLM

↓

Answer
```

Every step should be logged.

Example

```python
class AgentTracer:

    def log_step(self, name, data):
        print({
            "step": name,
            "timestamp": time.time(),
            "data": data
        })

tracer = AgentTracer()

tracer.log_step("planner", "Need customer data")

tracer.log_step("sql", "Executed customer query")

tracer.log_step("llm", "Generated response")
```

Example trace

```
Planner
↓

Search Docs

↓

Retrieve Context

↓

SQL

↓

Calculator

↓

LLM

↓

Final Answer
```

---

# 4. Tool Monitoring

Agents rely on tools.

Track

* Number of tool calls
* Failed tool calls
* Tool latency
* Retry count
* Success rate

Example

```python
import time

def monitored_tool(func):

    def wrapper(*args, **kwargs):

        start = time.time()

        try:
            result = func(*args, **kwargs)

            print({
                "tool": func.__name__,
                "status": "success",
                "latency": time.time()-start
            })

            return result

        except Exception as e:

            print({
                "tool": func.__name__,
                "status": "failed",
                "error": str(e)
            })

            raise

    return wrapper
```

---

# 5. RAG Monitoring

Monitor retrieval quality.

Metrics

* Retrieved documents
* Similarity score
* Recall
* Precision
* Faithfulness
* Context relevance

Example

```python
docs = retriever.search(query)

for doc in docs:
    print(doc.score)
```

Example

```
Doc1 0.94

Doc2 0.91

Doc3 0.82
```

If scores suddenly drop

```
0.91

↓

0.48
```

then embeddings, indexing, or the document corpus may need investigation.

---

# 6. Business Metrics

The most important production metrics are often business outcomes.

Examples

```
Customer Satisfaction

Conversation Success

Task Completion

Lead Conversion

Bookings

Sales

Support Resolution

Escalation Rate

Refund Rate
```

Example

```
1000 conversations

↓

910 completed

↓

90 escalated

↓

91% success
```

---

# 7. Security Monitoring

Monitor

* Prompt injection attempts
* Jailbreak attempts
* PII leakage
* Toxic outputs
* Unauthorized tool usage

Example

```python
blocked_words = [
    "ignore previous instructions",
    "system prompt"
]

def detect_attack(prompt):

    for word in blocked_words:
        if word in prompt.lower():
            return True

    return False
```

---

# 8. Cost Monitoring

Track

```
Cost per request

Cost per user

Daily spend

Monthly spend

Model usage

Token usage
```

Example

```python
cost = input_tokens * 0.000001 + output_tokens * 0.000003

print(cost)
```

Dashboard

```
GPT-5.5

Requests      18,000

Tokens        45M

Today's Cost  ₹9,450
```

---

# 9. Latency Monitoring

Break latency down by stage instead of only measuring end-to-end time.

```
Request

↓

Embedding

↓

Retriever

↓

Planner

↓

LLM

↓

Tools

↓

Response
```

Example

```
Embedding      35ms

Retrieval      62ms

Planning       210ms

LLM            1.5s

SQL            80ms

Response       2.0s
```

This helps identify whether the bottleneck is the model, retrieval, or a tool.

---

# 10. Error Monitoring

Categorize failures instead of treating all errors equally.

```
LLM timeout

↓

Rate limit

↓

Tool failure

↓

Invalid JSON

↓

Hallucination

↓

Retrieval failure
```

Example

```python
try:
    result = agent.run(question)

except TimeoutError:
    logger.error("LLM timeout")

except ValueError:
    logger.error("Invalid JSON")

except Exception:
    logger.exception("Unexpected error")
```

---

# 11. End-to-End Production Architecture

```
                     User
                       │
                       ▼
                 API Gateway
                       │
                       ▼
               Agent Orchestrator
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
   LLM Service     Tool Layer      Memory Store
      │                │                │
      └────────────────┼────────────────┘
                       ▼
                 Observability Layer
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
 Structured Logs   Metrics        Distributed Traces
      │               │                │
      ▼               ▼                ▼
 Elasticsearch   Prometheus       OpenTelemetry
      │               │                │
      └───────────────┼────────────────┘
                      ▼
                  Grafana
```

---

# Recommended Observability Stack

| Layer               | Recommended Tools                             | Purpose                                                                |
| ------------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| Metrics             | Prometheus                                    | Request rate, latency, token usage, tool success rate                  |
| Dashboards          | Grafana                                       | Visualize KPIs, costs, latency, error trends                           |
| Tracing             | OpenTelemetry                                 | Trace every agent step across services                                 |
| Logging             | ELK (Elasticsearch, Logstash, Kibana) or Loki | Centralized logs for prompts, tool calls, and errors                   |
| Agent Observability | Langfuse, LangSmith, Arize Phoenix            | Inspect traces, prompts, responses, retrieval quality, and experiments |
| Error Tracking      | Sentry                                        | Capture exceptions and stack traces                                    |
| Infrastructure      | Kubernetes + node exporters                   | CPU, memory, pod health, autoscaling                                   |
| Alerting            | Prometheus Alertmanager                       | Notify on high latency, error rates, or budget thresholds              |

---

## What senior AI engineers typically monitor

In production, a senior AI engineer usually creates dashboards covering:

* **Reliability:** request rate, success rate, P50/P95/P99 latency, error rate.
* **LLM usage:** input/output tokens, token cost, model selection, cache hit rate.
* **Agent execution:** reasoning steps, tool invocation counts, tool success/failure rates, retry counts.
* **RAG quality:** retrieval latency, similarity scores, context precision, context recall, faithfulness.
* **Business KPIs:** task completion rate, user satisfaction, escalation rate, revenue or conversion impact.
* **Security:** prompt injection attempts, PII detection events, blocked requests, unauthorized tool access.
* **Infrastructure:** CPU, memory, GPU utilization, pod health, autoscaling events, queue length.

This combination provides visibility into whether the agent is **fast**, **accurate**, **cost-efficient**, **secure**, and **delivering business value**, which are the core goals of production AI observability.
