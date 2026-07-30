Monitoring **LangGraph execution** is one of the most important production topics for Senior AI Engineers.

In development, it's enough to know that a graph produced the correct answer. In production, you need to answer questions such as:

* Which node is currently running?
* How long did each node take?
* Which tool failed?
* Why did the workflow stop?
* How many tokens did each agent consume?
* Which node is the bottleneck?
* What was the total cost?
* Can I replay the execution?

These questions require **observability**, not just logging.

---

# Production Monitoring Architecture

```text
                    User Request
                          │
                          ▼
                     FastAPI API
                          │
                          ▼
                    LangGraph Graph
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
  Planner Node      Research Node       Writer Node
      │                   │                   │
      └───────────────────┼───────────────────┘
                          ▼
                 Observability Layer
      ┌──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼
   LangSmith   OpenTelemetry   Structured Logs
      │              │              │
      └──────────────┼──────────────┘
                     ▼
           Prometheus + Grafana
```

---

# What Should You Monitor?

For every graph execution:

| Metric               | Example      |
| -------------------- | ------------ |
| Workflow ID          | `thread-123` |
| Current node         | `research`   |
| Node execution time  | 420 ms       |
| Total workflow time  | 5.2 s        |
| LLM latency          | 1.8 s        |
| Tool latency         | 320 ms       |
| Number of tool calls | 5            |
| Token usage          | 4,800        |
| Estimated cost       | $0.04        |
| Retry count          | 1            |
| Errors               | Timeout      |
| Checkpoint count     | 3            |

---

# Step 1: Add Structured Logging

Instead of:

```python
print("Running research")
```

Use Python's logging module.

```python
import logging
import time

logger = logging.getLogger(__name__)

def research(state):
    start = time.time()

    logger.info("Research node started")

    # Simulate work
    result = "Research completed"

    duration = time.time() - start

    logger.info(
        "Research node finished",
        extra={
            "duration": duration,
            "node": "research"
        }
    )

    return {"research_notes": result}
```

Structured logs make searching and aggregation much easier.

---

# Step 2: Monitor Node Execution Time

A reusable decorator works well.

```python
import time

def monitor_node(func):

    def wrapper(state):

        start = time.time()

        result = func(state)

        duration = time.time() - start

        print(
            f"{func.__name__} took "
            f"{duration:.2f} sec"
        )

        return result

    return wrapper
```

Use it:

```python
@monitor_node
def writer(state):

    return {
        "report": "Done"
    }
```

Output:

```
writer took 0.83 sec
```

---

# Step 3: Trace the Entire Workflow

```text
START
  │
  ▼
Planner (120 ms)
  │
  ▼
Research (1.8 s)
  │
  ▼
Writer (700 ms)
  │
  ▼
Reviewer (400 ms)
  │
  ▼
END
```

This immediately highlights slow nodes.

---

# Step 4: Use LangSmith

Enable tracing.

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<API_KEY>"
os.environ["LANGCHAIN_PROJECT"] = "production-agent"
```

LangSmith automatically records:

* Graph execution
* Every node
* Prompt
* Response
* Tool calls
* Latency
* Token usage
* Errors

Example trace:

```text
Planner
   │
   ▼
Retriever
   │
   ▼
Writer
   │
   ▼
Reviewer
```

You can inspect every step after execution.

---

# Step 5: OpenTelemetry

For distributed systems, instrument spans.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def research(state):

    with tracer.start_as_current_span("research"):

        # retrieve documents
        return {
            "research_notes": "..."
        }
```

Each node becomes a trace span.

Useful when requests cross:

* FastAPI
* LangGraph
* Redis
* PostgreSQL
* Vector DB
* External APIs

---

# Step 6: Count Tokens

Capture usage returned by your LLM client.

```python
response = llm.invoke(prompt)

usage = response.response_metadata.get("token_usage", {})

print(usage)
```

Example:

```
Prompt tokens: 850
Completion tokens: 220
Total tokens: 1070
```

Track per node, not just per request.

---

# Step 7: Measure Tool Performance

```python
import time

def sql_tool(query):

    start = time.time()

    result = run_query(query)

    duration = time.time() - start

    print(
        f"SQL tool took {duration:.2f} sec"
    )

    return result
```

Typical metrics:

* Database latency
* API latency
* Error rate
* Timeout rate

---

# Step 8: Monitor Graph State

Sometimes the graph appears "stuck".

Log state evolution.

```python
def writer(state):

    print("Incoming State")

    print(state)

    return {
        "report": "Complete"
    }
```

Example:

```text
{
    "query": "...",
    "research_notes": "...",
    "report": ""
}
```

After execution:

```text
{
    "query": "...",
    "research_notes": "...",
    "report": "Complete"
}
```

---

# Step 9: Track Retries

```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def llm_call(prompt):
    return llm.invoke(prompt)
```

Monitor:

* Retry count
* Success after retry
* Retry latency

High retry counts often indicate provider or network problems.

---

# Step 10: Monitor Checkpoints

```text
Planner
   │
Checkpoint #1
   │
Research
   │
Checkpoint #2
   │
Writer
```

Record:

* Checkpoint creation time
* Restore time
* Resume success rate

---

# Step 11: Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "graph_requests_total",
    "Total graph executions"
)

LATENCY = Histogram(
    "graph_latency_seconds",
    "Graph execution time"
)

REQUESTS.inc()

with LATENCY.time():
    graph.invoke(state)
```

Useful metrics:

* Workflow duration
* Node duration
* Error count
* Active executions
* Queue length

---

# Step 12: Grafana Dashboard

Typical dashboard panels:

```text
-----------------------------------------
Graph Executions/min      420
Average Latency           2.8 s
P95 Latency               4.1 s
Success Rate              99.4%
LLM Cost/hour             $12.50
Retries/hour              18
Active Workflows          73
-----------------------------------------
```

You can also create per-node dashboards.

---

# Step 13: Detect Stuck Graphs

Track current node and elapsed time.

```python
import time

state["current_node"] = "research"
state["started_at"] = time.time()
```

If a node exceeds a threshold:

```python
if time.time() - state["started_at"] > 300:
    raise TimeoutError("Workflow appears stuck")
```

Production systems typically alert rather than waiting indefinitely.

---

# Step 14: Monitor Queue Depth

If you process workflows asynchronously:

```text
Queue Size

5

↓

Healthy

Queue Size

500

↓

Scale Workers
```

Queue depth is often a better autoscaling signal than CPU.

---

# End-to-End Monitoring Flow

```text
User
 │
 ▼
FastAPI
 │
 ▼
LangGraph
 │
 ▼
Planner
 │
 ▼
Research
 │
 ▼
Writer
 │
 ▼
Reviewer
 │
 ▼
Response

Every step emits:

• Logs
• Metrics
• Traces
• Token usage
• Cost
• Checkpoints
```

---

# What I Monitor in Production

| Category       | Examples                                              |
| -------------- | ----------------------------------------------------- |
| Workflow       | Success/failure, duration, active runs                |
| Node           | Latency, retries, errors                              |
| LLM            | Tokens, cost, latency, model used                     |
| Tools          | API latency, SQL latency, failures                    |
| Retrieval      | Retrieved documents, reranker latency, recall metrics |
| Infrastructure | CPU, memory, queue depth, DB connections              |
| Checkpoints    | Saves, restores, resume failures                      |

---

# Common Interview Questions

### How do you know which node is slow?

Measure execution time for every node and expose it as metrics. Traces make bottlenecks easy to identify.

---

### How do you monitor LLM cost?

Record prompt tokens, completion tokens, total tokens, model name, and estimated cost for each LLM invocation, then aggregate these metrics in dashboards.

---

### How do you debug failed workflows?

Use LangSmith or distributed traces to inspect the execution path, review node inputs and outputs, examine logs and state transitions, and replay the workflow from a checkpoint when possible.

---

### How do you monitor thousands of graphs?

Treat each execution as a trace identified by a workflow or thread ID, emit metrics for every node, aggregate them with Prometheus, visualize them in Grafana, and configure alerts for high latency, elevated error rates, excessive retries, or stuck workflows.

---

# Senior AI Engineer Interview Answer

> **In production, I monitor LangGraph at multiple levels. I instrument every node with structured logs, execution latency, retry counts, and error tracking. I enable LangSmith for graph-level traces and OpenTelemetry for distributed tracing across FastAPI, LangGraph, databases, caches, and external APIs. I export metrics such as workflow duration, node latency, token usage, cost, queue depth, and checkpoint activity to Prometheus and visualize them in Grafana. I also monitor stuck workflows using timeout thresholds and workflow IDs, allowing me to trace, replay, and debug executions efficiently at scale.**
