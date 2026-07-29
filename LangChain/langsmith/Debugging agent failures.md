Debugging agent failures is one of the **most frequently discussed production topics** in Senior AI Engineer interviews.

Unlike a normal API, an AI agent has multiple components:

* LLM
* Prompt
* Tools
* Memory
* RAG
* Planning
* State Machine
* APIs

When an agent fails, **the final error rarely tells you the real cause**.

---

# What is an Agent Failure?

An agent failure occurs when the agent **does not successfully complete its objective**.

Examples:

```text
User:
Book me a flight tomorrow.

↓

Agent

↓

"I cannot help."
```

Or

```text
User

↓

Tool Called

↓

API Error

↓

Agent Stops
```

Or

```text
Retriever

↓

Wrong Context

↓

Hallucination
```

---

# Where Can an Agent Fail?

```text
                  User

                    │

                    ▼

              System Prompt
                    │
                    ▼
                Planner
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
      Memory    Retriever      Tools
        │           │            │
        ▼           ▼            ▼
     Documents   Vector DB     External APIs
                    │
                    ▼
                   LLM
                    │
                    ▼
             Output Parser
                    │
                    ▼
             Final Response
```

Every component can fail.

---

# Step 1: Identify the Failure Stage

The first question should always be:

**Where did it fail?**

```text
Request

↓

Planner?

↓

Retriever?

↓

Tool?

↓

LLM?

↓

Parser?

↓

Memory?
```

Never start by changing the prompt.

---

# Example

User:

```text
What's my account balance?
```

Trace:

```text
Planner

↓

Bank API Tool

↓

401 Unauthorized

↓

Failure
```

The problem is authentication—not the LLM.

---

# Step 2: Trace the Entire Execution

Enable tracing (LangSmith/OpenTelemetry).

Example trace:

```text
User

↓

Prompt

↓

Planner

↓

Weather Tool

↓

Response

↓

Parser

↓

Answer
```

Look for:

* Wrong prompt
* Wrong tool
* Wrong tool arguments
* Parser errors
* Timeouts
* Retries

---

# Step 3: Inspect the Prompt

Many failures originate in the prompt.

Bad prompt:

```text
Answer however you want.
```

Better:

```text
Always use the weather tool
for weather questions.

Never guess.
```

Log every prompt.

```python
logger.info(prompt)
```

---

# Step 4: Inspect Tool Selection

Example tools:

```python
@tool
def calculator(...)

@tool
def weather(...)

@tool
def search(...)
```

User:

```text
What is 25 * 30?
```

Correct:

```text
Calculator Tool
```

Failure:

```text
Search Tool
```

Wrong tool selection.

Log:

```python
logger.info(
    selected_tool
)
```

---

# Step 5: Validate Tool Inputs

Example:

```python
weather(
    city="Banaglore"
)
```

Typo:

```text
Banaglore
```

API:

```text
City Not Found
```

Always log tool arguments.

```python
def weather(city):

    logger.info(city)

    ...
```

---

# Step 6: Check Tool Output

Example:

```text
Weather API

↓

500 Internal Server Error
```

Instead of crashing:

```python
try:

    result = weather(city)

except Exception:

    ...
```

Log:

```python
logger.exception(e)
```

---

# Step 7: Verify Retrieval

Suppose:

Question:

```text
Who is the CEO?
```

Retriever returns:

```text
Vacation Policy
```

LLM answers incorrectly.

Debug:

```python
docs = retriever.invoke(question)

for doc in docs:

    print(doc.page_content)
```

Wrong retrieval—not hallucination.

---

# Step 8: Verify Memory

Conversation:

```text
My name is Alice.
```

Later:

```text
What's my name?
```

Memory:

```python
print(state["messages"])
```

If missing:

```text
Memory Bug
```

---

# Step 9: Validate Structured Output

Expected:

```json
{
  "city":"Bangalore"
}
```

Received:

```text
Bangalore
```

Parser fails.

```python
try:

    parser.invoke(output)

except Exception as e:

    logger.exception(e)
```

---

# Step 10: Detect Hallucinations

Suppose:

Retriever:

```text
No Documents
```

LLM:

```text
CEO is John.
```

Wrong.

Better:

```text
No relevant information found.
```

Add prompt:

```text
If context is insufficient,
say you don't know.
```

---

# Step 11: Monitor Retries

Sometimes tools fail temporarily.

```text
API Timeout

↓

Retry

↓

Success
```

Log:

```python
Attempt 1

Attempt 2

Attempt 3
```

---

# Step 12: Detect Infinite Loops

Bad:

```text
Planner

↓

Tool

↓

Planner

↓

Tool

↓

Planner

↓

Tool
```

Track iterations.

```python
MAX_ITERATIONS = 5

if state["steps"] >= MAX_ITERATIONS:

    stop()
```

---

# Step 13: Inspect Graph State

LangGraph:

```python
class AgentState(TypedDict):

    messages: list

    current_step: int

    tool_result: str
```

Node:

```python
def planner(state):

    print(state)

    ...
```

You can inspect state before every node.

---

# Step 14: Debug Conditional Routing

Example:

```python
def route(state):

    if state["need_tool"]:

        return "tool"

    return "answer"
```

Print routing decision.

```python
logger.info(
    state["need_tool"]
)
```

---

# Step 15: Record Latency

Example:

```python
import time

start = time.time()

response = llm.invoke(prompt)

print(
    time.time()-start
)
```

Example:

```text
Planner

120 ms

Retriever

45 ms

LLM

3.5 sec

Tool

4 sec
```

You immediately identify the bottleneck.

---

# Real Production Debugging Example

Suppose users complain:

```text
Agent gives wrong stock prices.
```

Trace:

```text
Question

↓

Planner

↓

Stock Tool

↓

429 Rate Limit

↓

Fallback Search

↓

Old Cached Result

↓

Wrong Answer
```

Root cause:

**Rate limit + stale cache**, not the LLM.

---

# LangSmith Debugging

LangSmith shows:

```text
Run

↓

Prompt

↓

Tool Calls

↓

Retriever

↓

State

↓

Output

↓

Tokens

↓

Latency
```

You can replay the same execution after changing prompts or tools.

---

# Logging Best Practices

Log:

```python
logger.info(
    {
        "user": user_id,
        "trace": trace_id,
        "prompt": prompt,
        "tool": tool_name,
        "latency": latency,
        "tokens": total_tokens
    }
)
```

Never log sensitive information such as passwords, API keys, or personal data in plain text.

---

# Failure Recovery

```python
try:

    result = tool.invoke(args)

except TimeoutError:

    result = backup_tool.invoke(args)

except Exception:

    result = "Tool unavailable"
```

Always provide graceful fallbacks.

---

# Production Debugging Checklist

| Component      | What to Check                       |
| -------------- | ----------------------------------- |
| Prompt         | Clear instructions, prompt version  |
| Planner        | Correct reasoning and routing       |
| Tool Selection | Right tool chosen                   |
| Tool Inputs    | Correct parameters                  |
| Tool Outputs   | API failures, malformed responses   |
| Memory         | Conversation state loaded correctly |
| Retriever      | Relevant documents returned         |
| Parser         | Valid structured output             |
| State          | Graph state transitions             |
| Routing        | Conditional edges behave correctly  |
| Tokens         | Prompt not exceeding budget         |
| Latency        | Slow nodes or APIs                  |
| Retries        | Excessive retry loops               |
| Logs           | Correlated with a trace ID          |

---

# Enterprise Debugging Architecture

```text
                    User
                      │
                      ▼
                   FastAPI
                      │
                 Trace ID
                      │
                      ▼
                 LangGraph
          ┌───────────┼────────────┐
          ▼           ▼            ▼
      Planner     Retriever      Tools
          │           │            │
          ▼           ▼            ▼
      LangSmith   Vector DB     External APIs
          │           │            │
          └───────────┼────────────┘
                      ▼
             OpenTelemetry
                      ▼
        Prometheus + Grafana + Logs
```

---

# Common Production Failures

| Failure       | Root Cause             | Fix                                                      |
| ------------- | ---------------------- | -------------------------------------------------------- |
| Wrong answer  | Poor retrieval         | Improve retrieval and reranking                          |
| Wrong tool    | Weak prompt or planner | Tighten tool descriptions and routing                    |
| Infinite loop | Missing stop condition | Set max iterations                                       |
| Slow response | Large prompt           | Summarize and trim context                               |
| Parser error  | Invalid JSON           | Use structured output with retries                       |
| Hallucination | Missing evidence       | Require grounded answers and detect insufficient context |
| Tool timeout  | External API           | Retry with exponential backoff and fallback              |

---

# Senior AI Engineer Interview Answer

> **When debugging an AI agent, I isolate the failure stage rather than assuming the LLM is at fault. I trace the full execution using LangSmith and OpenTelemetry, inspecting prompts, planner decisions, tool selection, tool inputs and outputs, retrieval results, graph state transitions, parser outputs, token usage, and latency. I correlate all logs with a trace ID so I can reconstruct a single request end-to-end. I also add safeguards such as retries, fallback models or tools, structured output validation, retrieval grading, and maximum iteration limits to prevent common production failures like hallucinations, tool errors, malformed outputs, and infinite loops. This systematic approach makes agent failures reproducible and much easier to diagnose.
