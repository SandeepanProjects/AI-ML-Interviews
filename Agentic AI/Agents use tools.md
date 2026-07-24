Agents use **tools** to interact with the outside world. Without tools, an LLM can only generate text from its training data and the current prompt. With tools, it can search the web, query databases, execute code, call APIs, send emails, or control software.

---

# High-Level Architecture

```text
                 User
                   │
                   ▼
          "Book me a flight"
                   │
                   ▼
             LLM Agent
                   │
         Think / Plan / Reason
                   │
       Decide which tool to use
                   │
        ┌──────────┴───────────┐
        │                      │
        ▼                      ▼
 Search Tool              Flight API
        │                      │
        ▼                      ▼
 Search Results          Available Flights
        └──────────┬───────────┘
                   ▼
             LLM Reasons
                   │
                   ▼
         Final Response
```

The important idea:

**The LLM never executes tools directly.**

Instead:

1. LLM decides
2. Framework executes tool
3. Tool returns output
4. LLM continues reasoning

---

# Example 1: Weather Agent

User:

> What's the weather in Bangalore?

Without tools:

```
LLM:
"I don't know today's weather."
```

With tools:

```
User
   │
   ▼
LLM
   │
Decides:
Need weather API
   │
   ▼
Weather Tool
   │
Returns:
31°C Rain
   │
   ▼
LLM
   │
Produces:
"It is currently 31°C with rain."
```

---

# How does the LLM know tools exist?

Framework provides tool descriptions.

Example:

```python
tools = [
    {
        "name": "weather",
        "description": "Get current weather",
        "parameters": {
            "city": "string"
        }
    }
]
```

The LLM reads this before generating.

---

# Tool Example

```python
def weather(city):
    return {
        "temperature": 30,
        "condition": "Rain"
    }
```

---

# Agent Workflow

```
Question

↓

Reason

↓

Need Tool?

↓

Yes

↓

Call Tool

↓

Receive Output

↓

Reason Again

↓

Need Another Tool?

↓

Repeat

↓

Answer User
```

---

# Real Example

User:

```
How many people live in Paris?
```

LLM internally reasons:

```
Need search tool.
```

Framework executes:

```python
search("Paris population")
```

Returns

```
2.1 million
```

LLM answers

```
Paris has approximately 2.1 million residents.
```

---

# Example 2: SQL Database Agent

Suppose a company has employee data.

```
Employees

ID

Name

Salary

Department
```

User:

```
Who earns the highest salary?
```

Tool:

```python
def execute_sql(query):
    ...
```

Agent decides:

```sql
SELECT name
FROM employees
ORDER BY salary DESC
LIMIT 1;
```

Framework runs SQL.

Database returns

```
Alice
```

LLM:

```
Alice earns the highest salary.
```

---

# LangChain Tool Example

```python
from langchain.tools import tool

@tool
def calculator(expression: str):
    """Evaluate a math expression."""
    return eval(expression)
```

Tool description is automatically extracted.

---

Creating the agent

```python
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()

agent = initialize_agent(
    tools=[calculator],
    llm=llm
)
```

---

Running

```python
agent.invoke({
    "input": "What is (45*23)+99?"
})
```

Internal process

```
LLM

↓

Need calculator

↓

calculator("45*23+99")

↓

1134

↓

Answer
```

---

# OpenAI Tool Calling Example

Define tool

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                }
            }
        }
    }
]
```

Model response

```json
{
  "tool_calls": [
    {
      "function": {
        "name": "get_weather",
        "arguments": {
          "city": "Bangalore"
        }
      }
    }
  ]
}
```

Framework executes

```python
get_weather("Bangalore")
```

Returns

```json
{
  "temperature":30
}
```

Send back

```python
messages.append(tool_result)
```

LLM generates

```
The temperature is 30°C.
```

---

# Multi-Tool Agent

Suppose user asks:

```
Find the latest NVIDIA stock price and email it to me.
```

Available tools

```
Search

↓

Stock API

↓

Email API
```

Execution

```
User

↓

LLM

↓

Need stock API

↓

Get price

↓

Need email tool

↓

Send email

↓

Done
```

---

# Example Code

```python
from langchain.tools import tool

@tool
def stock_price(symbol):
    return "$178"

@tool
def send_email(message):
    print("Email sent:", message)
```

The agent may execute:

```
stock_price("NVDA")

↓

"$178"

↓

send_email("$178")
```

---

# Real Production AI Assistant

Imagine an enterprise customer-support agent.

Available tools:

```
CRM API

Order Database

Inventory Service

Shipping API

Payment API

Knowledge Base

RAG Search

Email Service

Slack

Calendar

Jira

GitHub

Redis Cache

SQL Database
```

Flow:

```text
Customer:
Where is my laptop order?

↓

LLM

↓

Need Order Database

↓

Fetch Order

↓

Need Shipping API

↓

Fetch Tracking

↓

Need Knowledge Base

↓

Return delivery policy

↓

Generate response
```

---

# Multi-Step Planning

User:

```
Schedule a meeting with Alice next week.
```

Available tools:

```
Calendar

↓

Email

↓

Slack
```

Execution plan:

```
Step 1
Search Alice's calendar

↓

Step 2
Find common free slot

↓

Step 3
Create meeting

↓

Step 4
Email invitation

↓

Step 5
Notify Slack
```

Each step is a separate tool invocation.

---

# How the Agent Decides Which Tool to Use

The decision comes from the LLM using the tool's **name**, **description**, and **input schema**. During generation, the model compares the user's request with the available tools.

For example:

| Tool          | Description                           |
| ------------- | ------------------------------------- |
| `search_docs` | Search internal company documentation |
| `execute_sql` | Query the employee database           |
| `weather_api` | Get current weather information       |
| `send_email`  | Send an email to a recipient          |

If the user asks "How many vacation days does an employee receive?", the model is likely to choose `search_docs`. If they ask "Who is the highest-paid engineer?", it will choose `execute_sql`.

---

# Production Agent Loop (ReAct Pattern)

Most production agents follow a reasoning loop similar to this:

```text
Receive user request
        │
        ▼
Think about the next action
        │
        ▼
Need a tool?
   │          │
  No         Yes
   │          ▼
Answer   Call tool
              │
              ▼
      Observe tool result
              │
              ▼
Think again
              │
      Repeat until finished
              │
              ▼
      Return final answer
```

This iterative "Reason → Act → Observe" cycle allows agents to solve complex tasks involving multiple systems.

---

# Best Practices for Tool Design

When building production AI agents:

* **Keep each tool focused on one responsibility** (e.g., search documents, execute SQL, send email).
* **Write clear descriptions**, since the LLM relies on them to choose the right tool.
* **Use structured input schemas** (JSON Schema or Pydantic models) so arguments are validated.
* **Handle errors gracefully** and return informative messages if a tool fails.
* **Restrict permissions** (e.g., read-only database tools unless writes are necessary).
* **Log every tool call** for observability and debugging.
* **Set timeouts and retries** for external APIs to avoid stalled agent executions.

---

## How Senior AI Engineers Build Tool-Using Agents

In production systems, tool usage is usually layered:

```text
                User Request
                     │
                     ▼
             Planner / Agent
                     │
        Chooses next tool or action
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
   RAG Search     SQL Tool     External APIs
        │            │             │
        └────────────┼─────────────┘
                     ▼
           Validate & Normalize Results
                     │
                     ▼
          LLM Synthesizes Final Answer
                     │
                     ▼
              Response to User
```

This separation keeps tools deterministic and reliable, while the LLM focuses on reasoning and language generation.

In frameworks like LangChain, LangGraph, Semantic Kernel, or OpenAI's Responses API, the overall pattern is the same: the model decides **what** action to take, the application executes the tool, and the model incorporates the tool's output into its reasoning before producing the final response.
