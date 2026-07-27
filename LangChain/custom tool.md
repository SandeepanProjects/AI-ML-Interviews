This is one of the **most common Senior AI Engineer interview questions**.

Interviewers are testing whether you understand:

* How LLMs decide to use tools
* How tools are defined
* How an agent invokes tools
* How the tool output is fed back to the LLM
* How to build production-ready tools with error handling and security

We'll build a realistic example step by step.

---

# Problem Statement

Suppose we are building an enterprise AI assistant.

User asks:

> "What is the weather in Bangalore?"

The LLM **doesn't know live weather**.

Instead, it should:

1. Understand the request
2. Choose the Weather Tool
3. Execute it
4. Receive the result
5. Generate the final answer

Architecture:

```text
                User
                  │
                  ▼
        "Weather in Bangalore?"
                  │
                  ▼
             LangChain Agent
                  │
          Chooses Weather Tool
                  │
                  ▼
          Weather API / Function
                  │
                  ▼
        "29°C, Rain Expected"
                  │
                  ▼
             LangChain Agent
                  │
                  ▼
      "Today's weather in Bangalore..."
```

---

# Step 1: Install

```bash
pip install langchain
pip install langchain-openai
```

---

# Step 2: Create a Tool

A tool is simply a Python function decorated with `@tool`.

```python
from langchain.tools import tool

@tool
def weather(city: str) -> str:
    """
    Returns weather information.
    """

    weather_db = {
        "Bangalore": "28°C, Cloudy",
        "Delhi": "38°C, Sunny",
        "Mumbai": "31°C, Rain"
    }

    return weather_db.get(city, "Weather unavailable")
```

The docstring is important because the LLM uses it to understand **when** the tool is appropriate.

---

# Step 3: Create the LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)
```

---

# Step 4: Register the Tool

```python
tools = [weather]
```

The agent now knows it has one tool available.

---

# Step 5: Create the Agent

```python
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import hub

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
```

---

# Step 6: Invoke

```python
response = agent_executor.invoke(
    {
        "input": "What is the weather in Bangalore?"
    }
)

print(response["output"])
```

Output

```text
The current weather in Bangalore is 28°C and cloudy.
```

---

# What Actually Happens?

The interviewer expects you to explain the internal flow.

## Step A

User asks

```text
Weather in Bangalore
```

↓

LLM thinks

```text
I need weather information.

I have a tool called weather().
```

↓

Generated tool call

```json
{
  "tool": "weather",
  "arguments": {
    "city": "Bangalore"
  }
}
```

↓

Python executes

```python
weather("Bangalore")
```

↓

Returns

```text
28°C, Cloudy
```

↓

LLM receives

```text
Tool Result:

28°C, Cloudy
```

↓

LLM writes

```text
Today's weather in Bangalore is 28°C and cloudy.
```

---

# Execution Flow

```text
               User
                 │
                 ▼
     "Weather in Bangalore"
                 │
                 ▼
              Prompt
                 │
                 ▼
               GPT-4
                 │
     Should I call a tool?
                 │
          Yes
                 │
                 ▼
       weather("Bangalore")
                 │
                 ▼
         Python Function
                 │
                 ▼
        "28°C Cloudy"
                 │
                 ▼
             GPT-4
                 │
                 ▼
         Final Answer
```

---

# Example 2: Calculator Tool

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    """
    return str(eval(expression))
```

User

```text
What is (25+7)*12?
```

LLM

↓

Tool

```python
calculator("(25+7)*12")
```

↓

Returns

```text
384
```

↓

Final answer

```text
The answer is 384.
```

> **Production note:** Never use `eval()` on user input in a real application. Use a safe expression parser (such as `ast` with strict validation or a dedicated math evaluator) to avoid arbitrary code execution.

---

# Example 3: Database Tool

```python
@tool
def get_employee(employee_id: int):

    database = {
        101: "Alice",
        102: "Bob",
        103: "Charlie"
    }

    return database.get(employee_id)
```

User

```text
Who is employee 102?
```

Flow

```text
LLM

↓

Tool

↓

Database

↓

Result

↓

LLM
```

---

# Real Enterprise Example

Suppose you're building a banking assistant.

Available tools

```text
Balance Tool

Transfer Tool

Transaction Tool

Card Tool

Loan Tool
```

Architecture

```text
                      User
                        │
                        ▼
          "Transfer ₹500 to John"
                        │
                        ▼
                  LangChain Agent
                        │
         Decides Transfer Tool
                        │
                        ▼
              Transfer Service
                        │
                        ▼
          Banking Microservice
                        │
                        ▼
              Success / Failure
                        │
                        ▼
                   LangChain
                        │
                        ▼
      "Transfer completed successfully."
```

---

# Production Tool Example

Instead of returning hard-coded data, call a real API.

```python
import requests
from langchain.tools import tool

@tool
def weather(city: str):

    """
    Returns live weather.
    """

    response = requests.get(
        "https://api.weather.com",
        params={
            "city": city
        },
        timeout=5
    )

    return response.json()
```

---

# Adding Error Handling

Production code should never let exceptions propagate directly to the agent.

```python
from langchain.tools import tool
import requests

@tool
def weather(city: str) -> str:
    """Return live weather for a city."""

    try:
        response = requests.get(
            "https://api.weather.com",
            params={"city": city},
            timeout=5
        )

        response.raise_for_status()

        return response.text

    except requests.Timeout:
        return "Weather service timed out."

    except requests.RequestException:
        return "Weather service is temporarily unavailable."
```

---

# Multiple Tools

```python
tools = [
    weather,
    calculator,
    get_employee,
    search_documents,
    send_email
]
```

The agent chooses the appropriate tool based on the user's request.

```text
                User
                  │
                  ▼
      "Email John today's weather"
                  │
                  ▼
                Agent
          ┌──────────────┐
          │Weather Tool  │
          └──────┬───────┘
                 │
                 ▼
          Weather Result
                 │
                 ▼
          ┌──────────────┐
          │ Email Tool   │
          └──────┬───────┘
                 │
                 ▼
            Email Sent
```

---

# Interview Follow-up Questions

### 1. How does the LLM know which tool to call?

The tool's **name**, **description/docstring**, and **input schema** are sent to the model. The model decides whether a tool is needed and which one best matches the user's request.

---

### 2. Can the LLM call multiple tools?

Yes. For example:

```text
User:
Summarize yesterday's sales and email the report.
```

The agent can execute:

1. Query sales data
2. Summarize the results
3. Send an email

---

### 3. Can tools call other tools?

It's possible, but it's usually better to keep tools focused on a single responsibility and let the **agent orchestrate** multiple tool calls. This makes execution easier to trace, test, and recover from.

---

### 4. How do you secure tools?

* Authenticate users before allowing sensitive actions.
* Authorize based on roles (RBAC/ABAC).
* Validate and sanitize all inputs.
* Enforce rate limits and quotas.
* Log tool invocations for auditing.
* Restrict network and filesystem access where applicable.
* Require confirmation for high-risk actions (e.g., money transfers).

---

### 5. How do you observe tool usage in production?

Track metrics such as:

* Tool selection frequency
* Tool latency
* Success vs. failure rate
* Timeout count
* Retry count
* Token usage before and after tool calls
* End-to-end request latency

These metrics help identify failing tools, optimize performance, and improve agent reliability.

---

## What interviewers are really evaluating

Beyond getting the code to work, they want to know whether you can design reliable, production-grade agent systems:

* Design tools with clear, narrow responsibilities.
* Write descriptive tool schemas so the LLM selects them correctly.
* Handle failures, retries, and timeouts gracefully.
* Secure tools that interact with external systems.
* Add logging, tracing, and metrics for observability.
* Keep business logic inside services or APIs, not inside the LLM prompt.

A strong senior-level answer combines a working implementation with an explanation of the execution flow, trade-offs, and production considerations.
