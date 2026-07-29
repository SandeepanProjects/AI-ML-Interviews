Tool calling is one of the **most important concepts in modern LLMs**. Every production AI application—whether built with LangChain, LangGraph, OpenAI, Anthropic, or Azure OpenAI—relies on tool calling.

A common interview question is:

* **What is tool calling?**
* **How does tool calling work internally?**
* **How does an LLM decide to call a tool?**
* **How is tool calling different from a normal API call?**

Let's understand it from first principles.

---

# What is Tool Calling?

A **tool** is a function that performs some action.

For example:

* Search the web
* Query a database
* Call a weather API
* Execute SQL
* Send an email
* Perform calculations

A **tool call** means:

> **Instead of generating normal text, the LLM asks the application to execute one of these functions.**

---

# Without Tool Calling

Suppose the user asks:

> What is 456 × 789?

Without tools:

```text
User
  │
  ▼
LLM
  │
  ▼
Generates Answer
```

The LLM performs the calculation itself.

---

Suppose the user asks:

> What's the current weather in Bangalore?

The model cannot know live weather.

It might hallucinate.

---

# With Tool Calling

```text
User
  │
  ▼
LLM
  │
  ▼
Tool Call
  │
  ▼
Weather API
  │
  ▼
24°C
  │
  ▼
LLM
  │
  ▼
Final Answer
```

Instead of guessing, the model requests external information.

---

# Real-World Analogy

Imagine a manager.

The manager doesn't personally:

* write SQL
* check weather
* calculate taxes

Instead:

```text
Manager

↓

Accountant

↓

Database Admin

↓

Developer
```

The manager decides **who should perform the task**.

The specialists perform the work.

The manager combines the results.

That's exactly what an LLM agent does.

---

# Step-by-Step Example

User asks:

```text
What's today's weather in Bangalore?
```

---

## Step 1: User Request

```text
Question:

What's today's weather?
```

---

## Step 2: LLM Receives Available Tools

Suppose our application provides:

```text
Tool

weather(city)

Description:

Get current weather.
```

---

## Step 3: LLM Decides

Instead of replying:

```text
I think...
```

It generates a structured request.

Conceptually:

```json
{
  "tool": "weather",
  "arguments": {
    "city": "Bangalore"
  }
}
```

Notice:

This is **not** the final answer.

It's a request to execute a function.

---

## Step 4: Application Executes

Python:

```python
def weather(city):
    return "24°C"
```

Execution:

```python
weather("Bangalore")
```

Result:

```text
24°C
```

---

## Step 5: Result Goes Back to the LLM

Now the model receives:

```text
Observation:

24°C
```

---

## Step 6: Final Answer

The LLM now responds:

```text
Today's weather in Bangalore is 24°C.
```

---

# Example 1: Calculator

User:

```text
What is 25 × 30?
```

Tool:

```python
def calculator(expression):
    return eval(expression)
```

Flow:

```text
User

↓

LLM

↓

Tool Call

↓

calculator("25*30")

↓

750

↓

LLM

↓

Answer
```

---

# Example 2: Database

User:

```text
Show Alice's balance.
```

Tool:

```python
def get_customer(name):
    ...
```

Flow:

```text
User

↓

LLM

↓

Tool Call

↓

Database

↓

₹15,000

↓

LLM

↓

Final Answer
```

The LLM never directly accesses the database.

Your application executes the query.

---

# Example 3: Search

User:

```text
Who won yesterday's IPL match?
```

Flow:

```text
User

↓

LLM

↓

Search Tool

↓

Latest Score

↓

LLM

↓

Answer
```

---

# Example 4: Multiple Tools

User:

```text
What's the weather in Bangalore in Fahrenheit?
```

Execution:

```text
User

↓

LLM

↓

Weather Tool

↓

24°C

↓

LLM

↓

Calculator Tool

↓

75.2°F

↓

LLM

↓

Answer
```

The model calls more than one tool.

---

# Implementing a Tool in LangChain

## Step 1: Define the Tool

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    """
    return str(eval(expression))
```

The `@tool` decorator exposes:

* Tool name
* Description (docstring)
* Input schema
* Return type

---

## Step 2: Another Tool

```python
from langchain.tools import tool

@tool
def weather(city: str) -> str:
    """
    Get current weather for a city.
    """
    return "24°C"
```

---

## Step 3: Create the LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1"
)
```

---

## Step 4: Bind Tools

```python
llm_with_tools = llm.bind_tools(
    [calculator, weather]
)
```

Now the model knows:

* What tools exist
* When to use them
* What parameters they expect

---

## Step 5: Invoke

```python
response = llm_with_tools.invoke(
    "What's the weather in Bangalore?"
)
```

The model may not return text.

Instead it returns a tool call.

Conceptually:

```python
response.tool_calls
```

```python
[
    {
        "name": "weather",
        "args": {
            "city": "Bangalore"
        }
    }
]
```

---

## Step 6: Execute the Tool

```python
tool_call = response.tool_calls[0]

result = weather.invoke(
    tool_call["args"]
)

print(result)
```

Output:

```text
24°C
```

---

## Step 7: Return Observation

The application sends the tool result back to the model.

```python
messages = [
    response,
    {
        "role": "tool",
        "content": result
    }
]
```

Then:

```python
final = llm_with_tools.invoke(messages)
```

Now the model produces:

```text
The current weather in Bangalore is 24°C.
```

---

# Multiple Tool Calls

Suppose we have:

```python
@tool
def search(query: str):
    ...

@tool
def calculator(expr: str):
    ...
```

User:

```text
Population of India divided by 2
```

Execution:

```text
User

↓

LLM

↓

Search Tool

↓

1.43 billion

↓

LLM

↓

Calculator

↓

715 million

↓

LLM

↓

Answer
```

The application keeps executing requested tools until the model produces a normal response.

---

# How Does the LLM Know Which Tool to Use?

The LLM is given metadata like:

```text
Tool:
weather

Description:
Get current weather for a city.

Arguments:
city: string
```

The user asks:

```text
Will it rain in Delhi today?
```

The model semantically matches:

```text
Weather Question

↓

Weather Tool

↓

Call Tool
```

The Python implementation is **not** shown to the LLM.

---

# Tool Calling vs Function Calling

These terms are often used interchangeably.

| Function Calling                              | Tool Calling                                   |
| --------------------------------------------- | ---------------------------------------------- |
| Structured model output requesting a function | Broader orchestration concept                  |
| Returns function name and arguments           | Can include APIs, databases, search, SQL, etc. |
| Model emits the request                       | Application executes the action                |

In many modern frameworks, "tool calling" is the preferred term because tools can represent much more than simple Python functions.

---

# Tool Calling vs ReAct

Tool calling:

```text
LLM

↓

Tool

↓

LLM
```

ReAct:

```text
Think

↓

Tool

↓

Observe

↓

Think

↓

Tool

↓

Observe

↓

Answer
```

ReAct is a reasoning strategy that often **uses tool calling**.

---

# Production Architecture

```text
                  User

                    │

                    ▼

              FastAPI Server

                    │

                    ▼

                  LLM

                    │

         Tool Call Request

                    │

     ┌──────────────┼──────────────┐

     ▼              ▼              ▼

 Weather API   PostgreSQL     Vector DB

     ▼              ▼              ▼

      └─────────────┼──────────────┘

                    ▼

             Tool Results

                    ▼

                  LLM

                    ▼

              Final Answer
```

---

# Best Practices

### 1. Write clear descriptions

Poor:

```python
@tool
def tool():
    """Tool"""
```

Good:

```python
@tool
def search_docs(query: str):
    """
    Search internal company documentation using semantic search.
    """
```

---

### 2. Keep tools focused

Bad:

```python
everything_tool()
```

Good:

```python
search_docs()
send_email()
calculate_tax()
```

Each tool should do one thing well.

---

### 3. Return structured data

Instead of:

```python
return "Balance: ₹15,000"
```

Prefer:

```python
return {
    "customer": "Alice",
    "balance": 15000,
    "currency": "INR"
}
```

The LLM can format structured results more reliably.

---

### 4. Handle failures

Wrap external calls:

```python
from langchain.tools import tool

@tool
def weather(city: str):
    """Get current weather."""

    try:
        return call_weather_api(city)
    except Exception as e:
        return {
            "error": str(e)
        }
```

The agent can decide whether to retry or explain the error.

---

# Interview Questions

### Does the LLM execute Python code?

No. The LLM only **requests** a tool call. Your application executes the code and returns the result.

### Can the LLM call multiple tools?

Yes. It can request several tool calls in sequence, depending on the task.

### Can tool calls fail?

Yes. Production systems add retries, timeouts, fallbacks, logging, and human approval for sensitive operations.

---

# Senior AI Engineer Interview Answer

> **Tool calling allows an LLM to delegate tasks to external functions or services instead of relying solely on its internal knowledge. The application exposes tool metadata—such as names, descriptions, and input schemas—to the model. When the model determines that external information or computation is required, it emits a structured tool call containing the tool name and arguments. The application executes the tool, returns the observation to the model, and the model uses that result to continue reasoning or generate the final answer. This mechanism enables LLMs to interact with APIs, databases, search engines, vector stores, calculators, and enterprise systems in a reliable and extensible way.**
