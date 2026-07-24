Testing an AI agent is much harder than testing normal software because:

* The **LLM is probabilistic** (responses can vary).
* Agents use **multiple tools**.
* Agents maintain **memory**.
* Agents make **multi-step decisions**.
* External APIs can fail.

Senior AI engineers typically test agents at **five levels**, similar to how backend systems are tested.

---

# Agent Testing Pyramid

```text
                E2E Tests
             (Full Agent)

          Workflow Tests
     (Planning + Multi-step Tasks)

       Tool Integration Tests
      (APIs, DB, Search, RAG)

         LLM Behavior Tests
    (Prompt + Tool Selection)

            Unit Tests
    (Individual Python Functions)
```

---

# Example Production Agent

Suppose we built this customer-support agent.

```text
User
 │
 ▼
Agent
 │
 ├── Search Knowledge Base
 ├── SQL Database
 ├── CRM API
 ├── Email API
 └── Calculator
```

Each layer should be tested independently.

---

# Level 1: Unit Testing

Test normal Python code.

Example tool

```python
# tools/calculator.py

def calculate(a, b):
    return a + b
```

Test

```python
import pytest
from tools.calculator import calculate

def test_add():
    assert calculate(2, 3) == 5

def test_zero():
    assert calculate(0, 0) == 0

def test_negative():
    assert calculate(-2, 5) == 3
```

No LLM involved.

---

# Level 2: Tool Testing

Suppose your agent has a weather tool.

```python
def weather(city):
    return api.get(city)
```

Never call the real API in unit tests.

Mock it.

```python
from unittest.mock import patch

@patch("weather.api.get")
def test_weather(mock_get):

    mock_get.return_value = {
        "temperature": 30
    }

    result = weather("Delhi")

    assert result["temperature"] == 30
```

This verifies your tool behaves correctly without depending on an external service.

---

# Level 3: Tool Selection Testing

One of the most important tests.

Question:

> Did the LLM choose the correct tool?

Available tools

```text
search_docs()

execute_sql()

calculator()

send_email()
```

User asks

```text
What is 42 × 81?
```

Expected

```text
calculator()
```

NOT

```text
search_docs()
```

Pseudo-test

```python
response = agent.invoke(
    "What is 42 * 81?"
)

assert response.used_tool == "calculator"
```

---

# Level 4: Planning Tests

Suppose the user asks:

```text
Book a meeting with Alice tomorrow.
```

Expected execution

```text
Calendar Search

↓

Find Free Slot

↓

Create Meeting

↓

Send Email
```

Test

```python
expected = [
    "calendar_search",
    "calendar_create",
    "email_send"
]

assert trace.tool_sequence == expected
```

This checks whether the agent planned the workflow correctly.

---

# Level 5: End-to-End Tests

Run the entire agent.

Example

```python
response = agent.invoke(
    "Where is my order #48291?"
)
```

Assertions

```python
assert "tracking number" in response
assert "expected delivery" in response
```

This validates the full user journey.

---

# Testing with Fake Tools

Instead of real APIs, create deterministic fake tools.

```python
class FakeSearch:

    def search(self, query):
        return {
            "documents": [
                "Refund takes 5 days."
            ]
        }
```

Agent

```python
agent = Agent(
    search=FakeSearch()
)
```

Now every test gets the same result, making failures reproducible.

---

# Mocking LLM Responses

Calling a real LLM in every test is slow and expensive.

Instead

```python
class FakeLLM:

    def generate(self, prompt):
        return "Use weather tool"
```

Test

```python
agent = Agent(llm=FakeLLM())

result = agent.run(
    "Weather?"
)

assert result.tool == "weather"
```

---

# Testing Memory

Conversation

```text
User:
My name is John.

↓

User:
What is my name?
```

Expected

```text
John
```

Test

```python
agent.invoke("My name is John")

response = agent.invoke(
    "What is my name?"
)

assert "John" in response
```

Also test memory isolation:

```text
User A

↓

Memory A

User B

↓

Memory B
```

Ensure User B cannot access User A's information.

---

# Testing Retrieval (RAG)

Knowledge Base

```text
Refund policy

Vacation policy

Insurance policy
```

Question

```text
How many vacation days?
```

Retrieved document should be

```text
Vacation policy
```

Test

```python
docs = retriever.retrieve(
    "vacation days"
)

assert docs[0].title == "Vacation Policy"
```

This isolates retrieval quality from generation.

---

# Testing Hallucinations

Prompt

```text
Answer only using retrieved documents.
```

Retrieved

```text
Python was created by Guido.
```

Question

```text
Who invented Java?
```

Correct behavior

```text
"I don't have enough information."
```

Incorrect behavior

```text
James Gosling
```

A good agent should avoid introducing unsupported facts when instructed to rely only on retrieved context.

---

# Testing Failure Recovery

Tool fails

```text
Search API

↓

Timeout
```

Expected

```text
Retry

↓

Fallback Search

↓

Graceful Error
```

Example

```python
mock_search.side_effect = TimeoutError()
```

Verify

```python
assert retry_count == 3
assert fallback_used
```

---

# Testing Infinite Loops

Some agents repeatedly call tools.

Example

```text
Search

↓

Search

↓

Search

↓

Search

↓

Search
```

Prevent this with a maximum iteration count.

```python
agent = Agent(
    max_iterations=5
)
```

Test

```python
assert trace.iterations <= 5
```

---

# Testing Cost

Track tokens.

```python
response = agent.invoke(
    question
)

assert response.input_tokens < 6000
assert response.output_tokens < 800
```

Also monitor:

* Total token usage
* Cost per request
* Average latency

These are key production metrics.

---

# Testing Latency

Measure execution time.

```python
import time

start = time.time()

agent.invoke(question)

elapsed = time.time() - start

assert elapsed < 3
```

You can set different thresholds for interactive vs. batch workloads.

---

# Testing Concurrent Users

Simulate multiple requests.

```python
from concurrent.futures import ThreadPoolExecutor

questions = [
    "Weather",
    "Stock",
    "Order",
    "Refund"
]

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(agent.invoke, questions))

assert len(results) == 4
```

Verify there is no shared-state corruption.

---

# Example with `pytest`

```python
import pytest

@pytest.mark.parametrize(
    "question,expected_tool",
    [
        ("2 + 2", "calculator"),
        ("Weather in London", "weather"),
        ("Find refund policy", "search"),
    ],
)
def test_tool_selection(agent, question, expected_tool):
    trace = agent.invoke(question)
    assert trace.used_tool == expected_tool
```

Parameterized tests help cover many scenarios with little code.

---

# What Production Teams Measure

| Category       | Example Metrics                                                     |
| -------------- | ------------------------------------------------------------------- |
| Tool Selection | Correct tool chosen (%)                                             |
| Planning       | Expected tool sequence followed                                     |
| Retrieval      | Context precision, context recall, hit rate                         |
| Grounding      | Faithfulness, hallucination rate                                    |
| Latency        | p50, p95, p99 response time                                         |
| Cost           | Tokens, cost per request                                            |
| Reliability    | Tool success rate, retry rate                                       |
| Robustness     | Recovery from API failures and invalid inputs                       |
| Memory         | Correct recall, user isolation                                      |
| Security       | Prompt injection resistance, authorization, data leakage prevention |

---

# Production Testing Pipeline

```text
Developer
    │
    ▼
Unit Tests
    │
    ▼
Tool Tests (Mocks/Fakes)
    │
    ▼
Agent Behavior Tests
    │
    ▼
Retrieval Evaluation
    │
    ▼
End-to-End Integration Tests
    │
    ▼
Load & Stress Tests
    │
    ▼
Production Monitoring
    │
    ├── Latency
    ├── Token Usage
    ├── Tool Errors
    ├── Hallucination Rate
    ├── User Feedback
    └── Success Rate
```

## Frameworks commonly used

Senior AI teams often combine several tools rather than relying on one:

* **pytest** for unit and integration tests.
* **LangSmith** for tracing agent execution, validating tool usage, and regression testing.
* **DeepEval** for automated evaluation of RAG systems and agents (faithfulness, answer relevance, hallucination detection).
* **Ragas** for retrieval quality metrics such as context precision, context recall, and answer correctness.
* **OpenTelemetry** together with observability platforms like Grafana for production monitoring.
* **Locust** or **k6** for load and stress testing concurrent agent requests.

The key principle is to **test each layer independently** (tools, planner, memory, retrieval) before running expensive end-to-end agent tests. This makes failures much easier to diagnose and keeps your test suite fast and reliable.
