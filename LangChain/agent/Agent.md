This is one of the **most fundamental concepts in LangChain and LangGraph** and is almost always asked in **Senior AI Engineer interviews**.

The key idea is:

> **An LLM can reason, but it cannot interact with the outside world on its own. Tools give the agent the ability to take actions.**

---

# What is an Agent?

An **agent** is an LLM that can:

1. Understand a user's request.
2. Decide whether it needs external help.
3. Select the appropriate tool.
4. Execute the tool.
5. Observe the tool's output.
6. Decide whether another tool is needed.
7. Produce the final answer.

Unlike a simple chain, an agent **makes decisions at runtime**.

---

## Chain vs Agent

### Chain

A chain has a fixed workflow.

```text
User
  │
  ▼
Prompt
  │
  ▼
LLM
  │
  ▼
Output
```

Every request follows the same path.

---

### Agent

An agent chooses what to do.

```text
                User
                  │
                  ▼
             Agent (LLM)
          ┌──────┼────────┐
          ▼      ▼        ▼
      Search   Calculator  Database
          │      │         │
          └──────┼─────────┘
                 ▼
             Final Answer
```

The workflow changes depending on the question.

---

# Why Do We Need Tools?

Imagine asking ChatGPT:

```
What is today's weather in Bangalore?
```

Without tools:

The model only knows information from its training.

It cannot check the live weather.

---

With a weather tool:

```text
User

↓

Agent

↓

Weather API

↓

22°C

↓

LLM

↓

Final Answer
```

The tool provides fresh information.

---

# Real-World Analogy

Imagine a project manager.

The manager doesn't personally:

* write code
* run SQL queries
* search the internet

Instead, they delegate work.

```text
Manager

↓

Developer

↓

Tester

↓

Database Engineer

↓

Customer
```

The manager is the **agent**.

The specialists are the **tools**.

---

# What is a Tool?

A tool is simply a Python function with:

* a name
* a description
* input parameters
* output

Example:

```python
def calculator(expression: str) -> str:
    return str(eval(expression))
```

The LLM never executes Python itself.

Instead:

```
LLM

↓

Call calculator

↓

Python executes

↓

Return result
```

---

# LangChain Tool Example

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    return str(eval(expression))
```

The decorator converts the function into a tool the agent can use.

---

# Another Tool

```python
@tool
def get_weather(city: str):

    """Get weather for a city."""

    return "24°C"
```

Now the agent has two tools.

---

# Agent Decision Making

User:

```
What is 250 × 18?
```

The LLM reasons:

```text
Question

↓

Requires calculation

↓

Use Calculator

↓

Receive 4500

↓

Answer user
```

---

User:

```
What is the weather in Bangalore?
```

Reasoning:

```text
Question

↓

Needs weather

↓

Use Weather Tool

↓

Return answer
```

---

# Multiple Tool Calls

User:

```
What is the weather in Bangalore and convert 30°F to Celsius?
```

Agent:

```text
Question

↓

Weather Tool

↓

Calculator Tool

↓

Combine Results

↓

Answer
```

Notice:

The agent can call multiple tools.

---

# Internal Agent Loop

Internally, an agent follows a loop.

```text
Receive Question

↓

Think

↓

Need Tool?

↓

Yes

↓

Call Tool

↓

Observe Result

↓

Need Another Tool?

↓

Yes

↓

Call Tool Again

↓

Generate Final Answer
```

This loop continues until the agent decides it has enough information.

---

# Example

User:

```
What is the population of India divided by 2?
```

Agent reasoning:

```text
Need population

↓

Search Tool

↓

1.43 Billion

↓

Need math

↓

Calculator

↓

715 Million

↓

Answer
```

This demonstrates sequential tool use.

---

# Agent with Three Tools

```text
                    User

                      │

                      ▼

                 AI Agent

        ┌─────────┼─────────┐

        ▼         ▼         ▼

 Calculator   Weather    Search

        │         │         │

        └─────────┼─────────┘

                  ▼

             Final Answer
```

---

# Building a Tool

```python
from langchain.tools import tool

@tool
def square(number: int) -> int:
    """Return the square of a number."""
    return number * number
```

---

# Using the Tool

User:

```
Square 15
```

Agent reasoning:

```text
Need square

↓

Call square(15)

↓

225

↓

Return answer
```

---

# Multiple Tools Example

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

Agent:

```
Calculate (20+30)*5
```

Execution:

```text
User

↓

Agent

↓

Add Tool

↓

50

↓

Multiply Tool

↓

250

↓

Answer
```

---

# Creating an Agent

```python
from langchain.agents import create_tool_calling_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1")

tools = [
    add,
    multiply
]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
```

The agent now knows:

* available tools
* tool descriptions
* tool schemas
* how to call them

---

# Agent Execution

```python
response = agent.invoke(
    {
        "input": "What is (20+30)*5?"
    }
)
```

Internally:

```text
LLM

↓

Tool Selection

↓

Tool Execution

↓

Observation

↓

Final Answer
```

---

# Production Agent

An enterprise support agent might have tools like:

```text
                     AI Agent

      ┌──────────┬────────────┬────────────┐

      ▼          ▼            ▼

 PostgreSQL   Vector DB    Jira API

      ▼          ▼            ▼

 Customer     Documents     Tickets
```

Example:

User:

```
What is the status of ticket AI-1045?
```

Execution:

```text
Question

↓

Jira Tool

↓

Ticket Status

↓

Answer
```

Another query:

```
Summarize our refund policy.
```

Execution:

```text
Question

↓

Vector Database

↓

Relevant Documents

↓

LLM

↓

Summary
```

The same agent chooses different tools depending on the request.

---

# Agent vs Tool

| Agent                   | Tool                                   |
| ----------------------- | -------------------------------------- |
| Makes decisions         | Performs one specific action           |
| Uses reasoning          | Executes predefined logic              |
| Can call multiple tools | Cannot choose other tools              |
| Uses an LLM             | Usually ordinary Python code or an API |
| Controls workflow       | Does one task and returns a result     |

---

# Can a Tool Call Another Tool?

Generally, **no**.

The recommended architecture is:

```text
Agent

↓

Tool A

↓

Agent

↓

Tool B

↓

Agent
```

The **agent orchestrates** the workflow. Keeping tools focused on a single responsibility makes them easier to test, reuse, and monitor.

---

# Agent in LangGraph

In LangGraph, the agent is typically represented as one node in a larger workflow.

```text
          Planner

             │

             ▼

        Agent Node

             │

      ┌──────┴──────┐

      ▼             ▼

 Search Tool    SQL Tool

      │             │

      └──────┬──────┘

             ▼

        Response Node
```

The graph manages routing, retries, checkpoints, and human approvals, while the agent node focuses on reasoning and deciding which tool(s) to call.

---

# Interview Questions

### Why not let the LLM do everything?

LLMs cannot reliably:

* Access live information.
* Query enterprise databases.
* Send emails or update records.
* Execute business logic.
* Perform authenticated API calls.

Tools extend the LLM with these capabilities.

---

### What makes a good tool?

A good tool should:

* Have one clear responsibility.
* Be deterministic where possible.
* Have a descriptive name and docstring.
* Validate inputs.
* Return structured outputs when appropriate.
* Handle errors gracefully.

---

### How does the LLM know which tool to use?

The model is given:

* The tool's name.
* Its description (docstring).
* Input schema (parameters).

Using these, the LLM decides whether a tool matches the user's request. Modern models support structured tool/function calling, allowing them to emit a tool call instead of plain text.

---

# Senior AI Engineer Interview Answer

> **An agent is an LLM that can reason about a task and decide which actions to take to achieve a goal. It extends a standard LLM by using tools—specialized functions or APIs that perform actions such as searching the web, querying databases, running calculations, or calling external services. The agent follows a reasoning loop: it analyzes the request, decides whether a tool is needed, invokes the appropriate tool, observes the result, and either calls additional tools or produces the final answer. Tools are intentionally single-purpose and stateless, while the agent is responsible for orchestration and decision-making. This separation makes AI systems modular, testable, and suitable for production use.
