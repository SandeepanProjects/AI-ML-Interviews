Monitoring **token usage** is one of the most important aspects of operating LLM applications in production because it directly affects:

* **Cost**
* **Latency**
* **Context window usage**
* **Rate limits**
* **Capacity planning**

Almost every Senior AI Engineer interview includes questions such as:

* How do you measure token usage?
* How do you reduce LLM costs?
* How do you identify expensive prompts?
* How do you monitor token consumption in production?

---

# Why Monitor Tokens?

Suppose your AI application receives:

```text
10,000 requests/day
```

Each request:

```text
Prompt Tokens:       2,000
Completion Tokens:     500

Total:              2,500 tokens
```

Daily usage:

```text
2,500 × 10,000

=

25 Million Tokens
```

Without monitoring, you won't know:

* Which users consume the most tokens
* Which prompts are too large
* Which agent is expensive
* Which retriever returns excessive context

---

# Where Do Tokens Come From?

Every LLM request contains:

```text
             Prompt

      System Prompt

      Chat History

      Retrieved Documents

      Tool Results

      User Question

            ↓

           LLM

            ↓

     Generated Response
```

Total tokens:

```text
Prompt Tokens

+

Completion Tokens

=

Total Tokens
```

---

# Example

Request:

```text
System:
You are a helpful assistant.

History:
15 messages

Documents:
3 retrieved chunks

Question:
Explain transformers.
```

Suppose:

```text
System Prompt      100

History          1,200

Documents        2,500

Question            50

----------------------

Prompt          3,850
```

Response:

```text
Completion

650 tokens
```

Total:

```text
Prompt Tokens      3,850

Completion           650

------------------------

Total              4,500
```

---

# Monitoring with OpenAI Response Metadata

Most providers return token usage.

Example:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1"
)

response = llm.invoke(
    "Explain transformers."
)

print(response.response_metadata)
```

Example output:

```python
{
    "token_usage": {
        "prompt_tokens": 120,
        "completion_tokens": 340,
        "total_tokens": 460
    }
}
```

Retrieve values:

```python
usage = response.response_metadata["token_usage"]

print(usage["prompt_tokens"])
print(usage["completion_tokens"])
print(usage["total_tokens"])
```

---

# Logging Token Usage

```python
import logging

logger = logging.getLogger(__name__)

usage = response.response_metadata["token_usage"]

logger.info(
    f"""
Prompt={usage['prompt_tokens']}
Completion={usage['completion_tokens']}
Total={usage['total_tokens']}
"""
)
```

Logs:

```text
Prompt=450

Completion=180

Total=630
```

---

# Store Token Usage in PostgreSQL

Schema:

```sql
CREATE TABLE llm_usage (

    id SERIAL PRIMARY KEY,

    user_id VARCHAR(100),

    model VARCHAR(50),

    prompt_tokens INT,

    completion_tokens INT,

    total_tokens INT,

    latency FLOAT,

    created_at TIMESTAMP

);
```

Insert:

```python
usage = response.response_metadata["token_usage"]

record = LLMUsage(

    user_id="user123",

    model="gpt-4.1",

    prompt_tokens=usage["prompt_tokens"],

    completion_tokens=usage["completion_tokens"],

    total_tokens=usage["total_tokens"],

    latency=0.92
)

db.add(record)
db.commit()
```

Now you can answer questions like:

* Which user used the most tokens?
* Which model is most expensive?
* Which requests exceed 20k tokens?

---

# Measuring Tokens Before Calling the Model

Sometimes you need to reject oversized prompts.

Use `tiktoken`:

```bash
pip install tiktoken
```

Example:

```python
import tiktoken

encoding = tiktoken.encoding_for_model(
    "gpt-4.1"
)

def count_tokens(text):

    return len(
        encoding.encode(text)
    )
```

Usage:

```python
tokens = count_tokens(prompt)

print(tokens)
```

If:

```text
125000 tokens
```

Model limit:

```text
128000
```

You may decide to summarize or trim before calling the model.

---

# Monitor Every LangChain Call with a Callback

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenMonitor(BaseCallbackHandler):

    def on_llm_end(
        self,
        response,
        **kwargs
    ):

        usage = response.llm_output["token_usage"]

        print(
            f"""
Prompt: {usage['prompt_tokens']}
Completion: {usage['completion_tokens']}
Total: {usage['total_tokens']}
"""
        )
```

Use:

```python
response = llm.invoke(

    "Hello",

    config={
        "callbacks":[
            TokenMonitor()
        ]
    }
)
```

Every request automatically reports usage.

---

# Monitor LCEL Pipelines

Pipeline:

```python
chain = (
    prompt
    | llm
    | parser
)
```

Execution:

```python
response = chain.invoke(
    {"question":"Explain RAG"}
)
```

Record:

```text
Prompt Tokens

↓

Retriever Tokens

↓

LLM Tokens

↓

Completion Tokens
```

You can identify which stage contributes most to the total.

---

# LangGraph Token Monitoring

Each node can accumulate usage in the graph state.

```python
from typing import TypedDict

class AgentState(TypedDict):

    total_tokens: int
```

Node:

```python
def chat_node(state):

    response = llm.invoke(
        state["messages"]
    )

    usage = response.response_metadata["token_usage"]

    return {

        "messages":[response],

        "total_tokens":

            state["total_tokens"]

            + usage["total_tokens"]
    }
```

Now the workflow tracks cumulative token consumption.

---

# Alert on Large Prompts

```python
MAX_PROMPT = 6000

if usage["prompt_tokens"] > MAX_PROMPT:

    logger.warning(

        "Prompt too large"

    )
```

This helps detect prompt bloat before costs escalate.

---

# Prometheus Metrics

```python
from prometheus_client import Counter

token_counter = Counter(

    "llm_total_tokens",

    "Total LLM tokens"
)
```

Update:

```python
token_counter.inc(

    usage["total_tokens"]
)
```

Prometheus collects:

```text
llm_total_tokens
```

Grafana can visualize:

* Tokens per minute
* Tokens per user
* Tokens per model
* Daily token trends

---

# Cost Estimation

You can estimate request cost using the model's published pricing.

```python
PRICE_PER_1K = 0.002

cost = (

    usage["total_tokens"]

    /1000

) * PRICE_PER_1K

print(cost)
```

In production, keep pricing in a configuration table because prices vary by model and may change over time.

---

# Dashboard

A production dashboard typically contains:

```text
Requests/sec

Prompt Tokens/sec

Completion Tokens/sec

Average Tokens

95th Percentile Tokens

Cost per User

Cost per Team

Daily Spend

Monthly Spend
```

---

# LangSmith

When using LangSmith, every run automatically records:

```text
Prompt Tokens

Completion Tokens

Total Tokens

Latency

Model

Prompt

Tool Calls
```

This makes it easy to compare token usage across chains, agents, and models.

---

# Enterprise Architecture

```text
                    User

                      │

                      ▼

                  FastAPI

                      │

                      ▼

                 LangGraph

        ┌───────────┼────────────┐

        ▼           ▼            ▼

     Retriever      LLM         Tools

                    │

                    ▼

           Token Usage Extractor

                    │

         ┌──────────┼──────────┐

         ▼          ▼          ▼

   PostgreSQL   Prometheus   LangSmith

         ▼          ▼          ▼

                Grafana Dashboard
```

---

# What Should You Monitor?

| Metric               | Why it Matters                              |
| -------------------- | ------------------------------------------- |
| Prompt tokens        | Detect oversized prompts and context growth |
| Completion tokens    | Track verbose model outputs                 |
| Total tokens         | Overall usage and billing                   |
| Tokens per user      | Identify heavy users and enforce quotas     |
| Tokens per request   | Spot unusually expensive requests           |
| Tokens per model     | Compare model efficiency                    |
| Daily/monthly tokens | Capacity planning                           |
| Estimated cost       | Budget tracking                             |
| Prompt size trend    | Detect prompt bloat over time               |

---

# Best Practices

### Monitor every request

Never treat token usage as optional in production.

---

### Track prompt and completion separately

A large prompt and a large completion often indicate different optimization opportunities.

---

### Set alerts

Trigger alerts when prompt size, total tokens, or estimated cost exceeds expected thresholds.

---

### Aggregate by dimensions

Track usage by:

* User
* Tenant
* Model
* Endpoint
* Agent
* Workflow

This helps identify hotspots and allocate costs accurately.

---

### Correlate with latency

High token counts usually increase response time. Monitoring both together helps identify performance bottlenecks.

---

# Interview Questions

### Why monitor prompt tokens separately from completion tokens?

Prompt tokens reflect the size of the input context, while completion tokens reflect the verbosity of the model's output. They require different optimization strategies.

---

### How do you reduce token usage?

* Trim chat history.
* Summarize older conversations.
* Retrieve only relevant documents.
* Compress long documents.
* Use concise system prompts.
* Limit maximum output tokens.
* Cache repeated responses where appropriate.

---

### Where do you store token metrics?

In production, a common approach is:

* **Prometheus** for real-time metrics.
* **Grafana** for dashboards.
* **PostgreSQL** (or a data warehouse) for historical reporting and cost allocation.
* **LangSmith** for per-run traces and token details.

---

# Senior AI Engineer Interview Answer

> **In production, I monitor token usage for every LLM request. I extract prompt, completion, and total token counts from the model response metadata and log them with the user ID, model, latency, and workflow. These metrics are exported to Prometheus for real-time monitoring, visualized in Grafana, and stored in PostgreSQL for historical analysis and cost reporting. I also enforce token budgets before invoking the model, alert on oversized prompts, and optimize usage through conversation summarization, semantic retrieval, and prompt trimming. This gives complete visibility into cost, latency, and context-window utilization while preventing runaway token consumption.**
