# LLM Observability — Complete Guide with Production Examples

LLM observability is the process of **monitoring, debugging, evaluating, and improving** AI applications running in production.

Traditional software observability focuses on:

* Logs
* Metrics
* Traces

LLM applications require much more because you also need to monitor:

* Prompt quality
* Context retrieval
* Hallucinations
* Token usage
* Cost
* Latency
* Agent decisions
* Tool calls
* Memory
* User feedback

---

# Traditional Observability vs LLM Observability

| Traditional Software | LLM Applications  |
| -------------------- | ----------------- |
| CPU                  | Token usage       |
| Memory               | Prompt quality    |
| Network              | Context retrieval |
| API latency          | Model latency     |
| Database queries     | Tool calls        |
| Exceptions           | Hallucinations    |
| Error rate           | Faithfulness      |
| Business metrics     | User satisfaction |

Traditional monitoring isn't enough.

Example:

API returns 200 OK.

Traditional monitoring says:

```
Everything is healthy.
```

Reality:

```
Model hallucinated.

Wrong answer.

Customer unhappy.
```

Only LLM observability catches this.

---

# What Should Be Observed?

```
                    User
                      │
                      ▼
                API Gateway
                      │
        ┌─────────────┼────────────┐
        ▼             ▼            ▼
 Prompt Logs      Retrieval      Memory
        │             │            │
        ▼             ▼            ▼
    LLM Request --> Tool Calls --> Agent
        │
        ▼
    LLM Response
        │
        ▼
 Evaluation
        │
        ▼
 Metrics + Traces + Dashboards
```

Everything should be observable.

---

# Main Components

```
1. Prompt

2. Context

3. LLM

4. Tools

5. Agents

6. Responses

7. Cost

8. Users

9. Feedback
```

---

# Layer 1 — Prompt Observability

Every prompt should be logged.

Example

```python
prompt = """
You are an AI assistant.

Question:
{question}

Context:
{documents}
"""

logger.info({
    "prompt": prompt,
    "model": "gpt-4.1",
    "user": user_id
})
```

Track:

* Prompt version
* Prompt length
* System prompt
* User prompt
* Variables
* Temperature
* Model
* Max tokens

---

Example log

```json
{
  "prompt_version": "v12",
  "temperature": 0.2,
  "tokens": 542,
  "model": "gpt-4.1"
}
```

---

# Why?

If accuracy suddenly drops,

you compare

```
Prompt V11

vs

Prompt V12
```

Exactly like code deployments.

---

# Layer 2 — Context Observability

For RAG systems,

log retrieved chunks.

```
Question

↓

Retriever

↓

Chunk1

Chunk2

Chunk3

↓

LLM
```

Store

```python
logger.info({

    "query": question,

    "retrieved_docs": docs,

    "scores": scores

})
```

Without this

you cannot debug retrieval.

---

Metrics

```
Recall

Precision

Hit Rate

Chunk overlap

Similarity score
```

---

Example

```
Question

↓

Retrieved

Tax PDF

HR Policy

Employee Leave
```

If answer is wrong,

was retrieval wrong?

or LLM?

Observability tells you.

---

# Layer 3 — LLM Observability

Track every request.

Example

```python
start = time.time()

response = llm.invoke(prompt)

latency = time.time() - start
```

Store

```python
{
    "model":"gpt-4.1",

    "latency":1.2,

    "input_tokens":950,

    "output_tokens":321,

    "finish_reason":"stop"
}
```

---

Important metrics

```
Latency

Input Tokens

Output Tokens

Cost

Timeout

Retries

Errors

Rate limits
```

---

# Layer 4 — Tool Observability

Suppose an agent calls

```
Search

↓

Calculator

↓

Database

↓

Email
```

Log each call.

```python
{
 "tool":"calculator",

 "arguments":"12+45",

 "duration":0.01,

 "success":True
}
```

Now you know

```
Tool latency

Failures

Retries

Wrong arguments
```

---

# Layer 5 — Agent Observability

Agents think.

```
Thought

↓

Action

↓

Observation

↓

Next Thought
```

Capture reasoning steps (where appropriate for your system design), tool selections, and state transitions rather than exposing hidden model reasoning.

Example event log:

```python
logger.info({
    "step": 3,
    "action": "SearchDocs",
    "input": "company leave policy",
    "result": "3 documents found"
})
```

Useful metrics include:

* Number of planning steps
* Tool selection frequency
* Successful task completion
* Failed tool invocations
* Loop detection
* Time spent per step

---

# Layer 6 — Response Observability

Not every successful API response is a good answer.

Track response quality.

Metrics

```
Hallucination

Faithfulness

Groundedness

Relevance

Completeness

Toxicity
```

Example

```
Ground Truth

↓

Model Answer

↓

Evaluation
```

Evaluation

```
Faithfulness = 0.95

Relevance = 0.92

Hallucination = 0.01
```

---

# Layer 7 — User Observability

Users tell you if the system works.

Collect

```
Thumbs Up

Thumbs Down

Regenerate

Copy

Share

Conversation length

Abandonment
```

Example

```python
feedback = {

"user":123,

"rating":4,

"comment":"Good answer"

}
```

---

# Layer 8 — Cost Observability

LLMs are expensive.

Track

```
Tokens

↓

Price

↓

Cost
```

Example

```python
cost = (

input_tokens * INPUT_PRICE +

output_tokens * OUTPUT_PRICE

)
```

Dashboard

```
Today's Cost

₹4,200

Average/User

₹18

Most Expensive Prompt

₹95

Largest Response

15,000 tokens
```

---

# Layer 9 — Trace Observability

One user request goes through many services.

```
User

↓

API

↓

Retriever

↓

Embedding

↓

Vector DB

↓

Reranker

↓

LLM

↓

Response
```

Use distributed tracing so every component shares a **trace ID**.

Example

```
Trace ID

abc123

↓

Retriever

↓

LLM

↓

Redis

↓

Postgres

↓

Response
```

This lets you identify which stage caused latency or errors.

---

# OpenTelemetry Example

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("llm_request"):
    response = llm.invoke(prompt)
```

Nested spans:

```python
with tracer.start_as_current_span("retrieve"):
    docs = retriever.search(query)

with tracer.start_as_current_span("rerank"):
    ranked = reranker.rank(docs)

with tracer.start_as_current_span("llm"):
    answer = llm.invoke(prompt)
```

Example trace:

```
Request (2.8 s)
├── Authentication (20 ms)
├── Retrieval (220 ms)
├── Reranking (80 ms)
├── LLM Inference (2.2 s)
└── Response Formatting (30 ms)
```

---

# Production Metrics Dashboard

A typical dashboard contains:

### System Metrics

* Requests/sec
* Success rate
* Error rate
* P50, P95, P99 latency
* Queue depth

### LLM Metrics

* Input tokens
* Output tokens
* Average response length
* Cost/request
* Cost/day
* Model usage distribution

### RAG Metrics

* Retrieval latency
* Top-k retrieval accuracy
* Average similarity score
* Context precision
* Context recall

### Agent Metrics

* Average planning steps
* Tool call count
* Tool success rate
* Loop detection events

### Quality Metrics

* Hallucination rate
* Faithfulness
* Answer relevance
* User satisfaction score

---

# Example Production Architecture

```
                    Users
                      │
                 FastAPI Gateway
                      │
            OpenTelemetry Middleware
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
 Retriever         LLM Client      Agent Engine
      │               │                │
      └───────────────┼────────────────┘
                      ▼
             Event/Metric Collector
                      │
      ┌───────────────┼───────────────────────────┐
      ▼               ▼              ▼            ▼
  Prometheus      Grafana        Jaeger      Evaluation DB
      │
      ▼
 Alerts (PagerDuty/Slack)
```

---

# Alerts You Should Configure

| Alert                | Example Threshold                        |
| -------------------- | ---------------------------------------- |
| P95 latency          | > 5 seconds                              |
| Error rate           | > 5%                                     |
| Hallucination rate   | Above normal baseline                    |
| Cost/hour            | Exceeds budget                           |
| Token spike          | 2× normal usage                          |
| Retrieval failure    | No documents retrieved for many queries  |
| Tool failure         | Success rate < 95%                       |
| Agent loops          | More than 10 steps                       |
| User dissatisfaction | Thumbs-down rate increases significantly |

---

# Popular LLM Observability Tools

| Tool             | Best For                                                  |
| ---------------- | --------------------------------------------------------- |
| OpenTelemetry    | Distributed traces across services                        |
| Prometheus       | Metrics collection                                        |
| Grafana          | Dashboards and alerting                                   |
| Jaeger           | Trace visualization                                       |
| Langfuse         | Prompt, trace, cost, and evaluation tracking for LLM apps |
| LangSmith        | LangChain application tracing and evaluation              |
| Arize Phoenix    | LLM evaluation, embeddings, and RAG debugging             |
| Helicone         | API usage, latency, and cost monitoring                   |
| Weights & Biases | Experiment tracking and model evaluation                  |
| MLflow           | Model lifecycle and experiment management                 |

---

# End-to-End Request Flow

```
User Question
      │
      ▼
API Gateway
      │
      ▼
Trace Created
      │
      ▼
Retriever
      │
      ▼
Vector Database
      │
      ▼
Reranker
      │
      ▼
Prompt Builder
      │
      ▼
LLM
      │
      ▼
Tool Calls (if any)
      │
      ▼
Response Evaluation
      │
      ▼
Metrics + Logs + Traces + Cost
      │
      ▼
Grafana / Langfuse / LangSmith Dashboards
```

## Senior AI Engineer interview points

A production-grade LLM observability strategy should answer these questions quickly:

* **What happened?** → Structured logs
* **Where did it happen?** → Distributed traces
* **How often is it happening?** → Metrics and dashboards
* **Why is answer quality changing?** → Prompt, retrieval, and evaluation metrics
* **How much does it cost?** → Token and cost tracking
* **Are users satisfied?** → Feedback and behavioral analytics

Together, these capabilities let engineering teams diagnose latency, control costs, improve answer quality, and safely operate LLM applications at scale.
