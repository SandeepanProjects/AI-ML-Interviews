This is **one of the most common interview questions** for LangChain, LangGraph, and AI Agents.

The short answer is:

> **The LLM decides which tool to call by comparing the user's request with the tool names, descriptions (docstrings), and input schemas that are provided in the prompt.**

There is **no hardcoded `if-else`** (unless you explicitly build one). The LLM uses its reasoning ability to choose the most appropriate tool.

Let's see exactly how this works internally.

---

# High-Level Architecture

```text
                  User Question
                        │
                        ▼
               Prompt + Available Tools
                        │
                        ▼
                     LLM
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  Generate Answer                 Call Tool
                                          │
                                          ▼
                                   Execute Tool
                                          │
                                          ▼
                                   Tool Observation
                                          │
                                          ▼
                                        LLM
                                          │
                                          ▼
                                    Final Answer
```

The LLM sees both:

* the user's question
* the list of available tools

It decides whether any tool is needed.

---

# Step 1: Define Tools

Suppose we define three tools.

```python
from langchain.tools import tool

@tool
def weather(city: str):
    """Get the current weather for a city."""
    return "24°C"


@tool
def calculator(expression: str):
    """Perform mathematical calculations."""
    return str(eval(expression))


@tool
def search(query: str):
    """Search for current information."""
    return "Search Results..."
```

Notice every tool has:

* Name
* Description (docstring)
* Input schema
* Return type

The description is extremely important.

---

# Step 2: What the LLM Actually Receives

LangChain constructs a prompt similar to this (simplified):

```text
You are an AI assistant.

You have access to these tools:

Tool:
weather

Description:
Get the current weather for a city.

Arguments:
city: string

------------------------

Tool:
calculator

Description:
Perform mathematical calculations.

Arguments:
expression: string

------------------------

Tool:
search

Description:
Search for current information.
```

Then the user's question is appended:

```text
User:

What is today's weather in Bangalore?
```

The model now reasons over both the question and the tool descriptions.

---

# Step 3: LLM Reasoning

Internally the model may reason like this:

```text
Question:
Today's weather

↓

Need current information

↓

Weather tool matches

↓

Call weather()
```

Notice:

It doesn't search tool names only.

It also understands the **description**.

---

# Example 1

User:

```text
What's today's weather?
```

Available tools:

```text
weather()

calculator()

database()
```

Reasoning:

```text
Weather question

↓

Weather tool description matches

↓

Call weather()
```

---

# Example 2

User:

```text
What is 45 × 17?
```

Reasoning:

```text
Math problem

↓

Calculator description matches

↓

Call calculator()
```

---

# Example 3

User:

```text
Who won yesterday's IPL match?
```

Reasoning:

```text
Needs current information

↓

Search tool

↓

Call search()
```

---

# Tool Selection is Semantic

Suppose the tool name is terrible.

```python
@tool
def tool_xyz(city):
    """Get current weather."""
```

The user asks:

```text
Will it rain in Delhi today?
```

The LLM can still choose it because it understands the description.

Conversely:

```python
@tool
def weather(city):
    """Calculate square roots."""
```

Now the name says "weather", but the description says "math".

The description usually carries much more weight than the name.

---

# Input Schema Matters

Consider this tool:

```python
@tool
def search(
    query: str,
    top_k: int
):
    ...
```

The schema tells the LLM:

```text
query

↓

string

top_k

↓

integer
```

User:

```text
Find top 5 AI papers
```

The model can infer:

```text
query = "AI papers"

top_k = 5
```

It fills in the arguments automatically.

---

# Function Calling

Modern models don't return plain text like:

```text
Use calculator
```

They return structured tool calls.

Example:

```json
{
  "tool": "calculator",
  "arguments": {
    "expression": "45*17"
  }
}
```

LangChain receives this structured output.

---

# LangChain Executes the Tool

```python
result = calculator(
    expression="45*17"
)
```

Output:

```text
765
```

LangChain then sends this back to the model.

```text
Observation:

765
```

---

# The Model Thinks Again

Now the prompt contains:

```text
Question:

45 × 17

Tool Result:

765
```

The model responds:

```text
The answer is 765.
```

---

# Multiple Tool Calls

Suppose the user asks:

```text
What's today's temperature in Bangalore in Fahrenheit?
```

Reasoning:

```text
Need weather

↓

Weather Tool

↓

24°C

↓

Need conversion

↓

Calculator

↓

75.2°F

↓

Answer
```

The agent chooses tools one after another.

---

# Why Doesn't the Agent Call Every Tool?

Imagine having 100 tools.

Calling all of them would be:

* Slow
* Expensive
* Wasteful

Instead:

```text
Question

↓

LLM chooses

↓

One tool

↓

Result

↓

Maybe another
```

Only the necessary tools execute.

---

# Poor Tool Descriptions Cause Wrong Decisions

Bad:

```python
@tool
def search():

    """Tool"""
```

The LLM has almost no information.

Better:

```python
@tool
def search():

    """
    Search the internet for
    recent news and events.
    """
```

Now the model understands exactly when to use it.

---

# Similar Tools

Suppose you have:

```text
Weather API

Weather Database

Weather Forecast
```

Poor descriptions:

```text
Weather tool

Weather tool

Weather tool
```

The LLM will struggle.

Better:

```text
Current weather

Historical weather

7-day forecast
```

Now the choice is much easier.

---

# Production Example

Imagine an enterprise support agent.

Available tools:

```text
CRM Tool
```

Description:

```text
Retrieve customer account information.
```

---

```text
Jira Tool
```

Description:

```text
Retrieve software tickets.
```

---

```text
Vector Search
```

Description:

```text
Search internal documentation.
```

User:

```text
Show ticket AI-123
```

Reasoning:

```text
Ticket

↓

Jira description

↓

Call Jira
```

---

User:

```text
Summarize our refund policy
```

Reasoning:

```text
Need documentation

↓

Vector Search

↓

Retrieve

↓

Answer
```

---

# What If the Agent Chooses the Wrong Tool?

This happens in real systems.

Common solutions:

### Better tool descriptions

Instead of:

```text
Database tool
```

Use:

```text
Query customer orders from PostgreSQL.
```

---

### Fewer tools

Instead of giving the model 200 tools, expose only the relevant subset.

---

### Tool Router

Use a classifier first.

```text
User

↓

Intent Classifier

↓

Finance Agent

↓

Finance Tools
```

This reduces confusion.

---

### Human Approval

For sensitive operations:

```text
Delete Customer

↓

Human Approval

↓

Execute
```

---

# How LangChain Implements This

Conceptually:

```python
tools = [
    weather,
    calculator,
    search
]

prompt = build_prompt(
    user_question,
    tools
)

response = llm.invoke(prompt)
```

If the model returns a tool call:

```python
if response.tool_call:

    result = execute_tool()

    prompt += observation

    llm.invoke(prompt)
```

The loop continues until the model produces a final answer.

---

# Interview Questions

### Is tool selection rule-based?

No.

It is **LLM-driven** by default. The model uses semantic understanding of the tool descriptions and schemas.

---

### Does the LLM see the tool code?

No.

It typically sees:

* Tool name
* Description
* Input schema
* Output schema (if provided)

It does **not** inspect the Python implementation.

---

### Can we override tool selection?

Yes.

You can:

* Use explicit routing logic.
* Restrict the available tools.
* Build separate agents with specialized toolsets.
* Use LangGraph conditional routing before invoking an agent.

---

# Senior AI Engineer Interview Answer

> **An agent decides which tool to call by reasoning over the user's request and the metadata associated with each available tool. LangChain exposes each tool's name, description, and input schema to the LLM as part of the prompt. The model semantically matches the user's intent with these tool definitions and emits a structured tool call when needed. LangChain executes the tool, returns the observation to the model, and the agent continues reasoning. In production, tool selection is improved by writing precise tool descriptions, exposing only relevant tools, grouping tools by domain, and using routing or specialized agents to reduce ambiguity.**
