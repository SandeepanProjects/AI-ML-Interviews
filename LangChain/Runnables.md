These are two of the **most frequently asked LangChain interview topics**. Understanding **Runnables**, **Chains**, and **Agents** is essential because almost everything in modern LangChain (LCEL) is built around the **Runnable** interface.

---

# 1. What are Runnables?

A **Runnable** is the fundamental abstraction in LangChain.

A Runnable is **any component that accepts an input, performs some work, and produces an output**.

Everything in modern LangChain implements the Runnable interface:

* Prompt Templates
* LLMs
* Chat Models
* Retrievers
* Output Parsers
* Chains
* Parallel Pipelines
* Routers

Think of a Runnable like a function.

```text
Input

↓

Process

↓

Output
```

For example:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1")

response = llm.invoke("Explain transformers.")
```

Here,

```text
Input

↓

ChatOpenAI Runnable

↓

AI Response
```

The LLM itself is a Runnable.

---

# Runnable Interface

Every Runnable supports common methods.

| Method      | Purpose                   |
| ----------- | ------------------------- |
| `invoke()`  | Execute one request       |
| `batch()`   | Execute multiple requests |
| `stream()`  | Stream tokens/events      |
| `ainvoke()` | Async execution           |
| `abatch()`  | Async batch execution     |

Example:

```python
llm.invoke("Hello")

llm.batch([
    "Explain AI",
    "Explain ML"
])

for token in llm.stream("Explain LangGraph"):
    print(token)
```

---

# Runnable Composition (LCEL)

The biggest advantage is composition.

Instead of writing:

```python
prompt = prompt.format(question)

answer = llm.invoke(prompt)

parsed = parser.invoke(answer)
```

You can compose everything.

```python
chain = prompt | llm | parser
```

Execution:

```text
Question

↓

Prompt

↓

LLM

↓

Parser

↓

Answer
```

Each component is a Runnable.

---

# Example LCEL Pipeline

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "Explain {topic}"
)

llm = ChatOpenAI(model="gpt-4.1")

parser = StrOutputParser()

chain = prompt | llm | parser

print(
    chain.invoke(
        {"topic": "LangGraph"}
    )
)
```

Pipeline:

```text
Input

↓

Prompt Runnable

↓

LLM Runnable

↓

Parser Runnable

↓

String
```

---

# RunnableParallel

Run multiple tasks simultaneously.

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain
)

result = parallel.invoke(
    {"text": article}
)
```

Execution:

```text
Document

↓

Parallel

↙          ↘

Summary   Keywords

↓

Merged Output
```

---

# RunnablePassthrough

Pass data unchanged.

```python
from langchain_core.runnables import RunnablePassthrough

chain = {
    "question": RunnablePassthrough(),
    "context": retriever
}
```

Flow:

```text
Question

↓

Passthrough

↓

Retriever

↓

Prompt
```

---

# RunnableBranch

Conditional routing.

```text
Question

↓

Router

↓

Math?

↓

Yes

↓

Calculator

------------

No

↓

LLM
```

---

# RunnableLambda

Wrap Python code.

```python
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(
    lambda x: x.upper()
)

uppercase.invoke("hello")
```

Output:

```text
HELLO
```

---

# Why LangChain Uses Runnables

Benefits:

* Standard API
* Easy composition
* Streaming
* Async support
* Parallel execution
* Retry support
* Tracing
* Type safety

---

# Runnable Hierarchy

```text
Runnable

├── Prompt

├── LLM

├── Retriever

├── Parser

├── Chain

├── Parallel

├── Branch

└── Router
```

Everything follows the same execution model.

---

# 2. Difference Between Chain and Agent

This is one of the most common interview questions.

## What is a Chain?

A Chain is a **fixed sequence of steps**.

The execution path is predetermined.

Example:

```text
Question

↓

Prompt

↓

LLM

↓

Parser

↓

Answer
```

The chain cannot decide to call a tool.

Example:

```python
chain = prompt | llm | parser

answer = chain.invoke(
    {
        "question":"Explain RAG"
    }
)
```

Execution:

```text
Always

Prompt

↓

LLM

↓

Parser
```

---

## Characteristics of Chains

* Deterministic
* Linear workflow
* No decision making
* No planning
* No reasoning loop
* Fast
* Simple

---

# What is an Agent?

An Agent is **decision-making software**.

Instead of following fixed steps:

```text
Prompt

↓

LLM

↓

Done
```

The agent decides:

* Should I search?
* Should I use SQL?
* Should I call a calculator?
* Should I ask another tool?
* Should I stop?

Execution:

```text
Question

↓

Think

↓

Choose Tool

↓

Execute Tool

↓

Observe

↓

Think Again

↓

Final Answer
```

---

# Agent Example

User:

> What is the population of India divided by 5?

Execution:

```text
Question

↓

Search Tool

↓

Population

↓

Calculator

↓

Answer
```

The path is chosen dynamically.

---

# Agent Architecture

```text
User

↓

Agent

↓

Planner

↓

Tool

↓

Observation

↓

LLM

↓

Another Tool?

↓

Final Answer
```

---

# Chain Example

```text
User

↓

Prompt

↓

GPT

↓

Answer
```

---

# Agent Example

```text
User

↓

Planner

↓

Weather Tool

↓

Calculator

↓

Translator

↓

Answer
```

---

# Real-World Example

## Chain

User:

> Summarize this PDF.

Pipeline:

```text
PDF

↓

Prompt

↓

LLM

↓

Summary
```

Simple.

---

## Agent

User:

> Analyze this company's financial report.

Execution:

```text
Question

↓

Search Financial Docs

↓

Retrieve

↓

SQL Database

↓

Calculator

↓

Generate Report
```

The agent decides which tools to use.

---

# Comparison

| Feature               | Chain   | Agent   |
| --------------------- | ------- | ------- |
| Workflow              | Fixed   | Dynamic |
| Tool selection        | ❌       | ✅       |
| Planning              | ❌       | ✅       |
| Decision making       | ❌       | ✅       |
| Reasoning loop        | ❌       | ✅       |
| Conditional execution | Limited | ✅       |
| Multiple tool calls   | ❌       | ✅       |
| Latency               | Lower   | Higher  |
| Cost                  | Lower   | Higher  |
| Complexity            | Simple  | Complex |

---

# When to Use a Chain

Use a Chain when:

* Summarization
* Translation
* Sentiment analysis
* Classification
* Fixed RAG pipeline
* Data extraction
* Report generation
* Email generation

Example:

```text
Document

↓

Prompt

↓

LLM

↓

Summary
```

---

# When to Use an Agent

Use an Agent when:

* Multiple tools are available
* The next action depends on intermediate results
* External APIs are needed
* SQL queries are required
* Web search is required
* Multi-step reasoning is needed
* Tasks cannot be expressed as a fixed pipeline

Example:

```text
Question

↓

Search

↓

Calculator

↓

SQL

↓

Answer
```

---

# Chain vs Agent vs LangGraph

| Feature                  | Chain    | Agent   | LangGraph |
| ------------------------ | -------- | ------- | --------- |
| Fixed pipeline           | ✅        | ❌       | ✅/❌       |
| Dynamic routing          | ❌        | ✅       | ✅         |
| Tool calling             | Limited  | ✅       | ✅         |
| Multi-agent              | ❌        | Limited | ✅         |
| Human approval           | ❌        | Limited | ✅         |
| Checkpointing            | ❌        | ❌       | ✅         |
| Persistent state         | ❌        | Basic   | ✅         |
| Conditional branching    | Limited  | Limited | ✅         |
| Production orchestration | Moderate | Good    | Excellent |

---

# Interview Follow-Up Questions

### 1. Is every Chain a Runnable?

Yes.

A Chain implements the Runnable interface, so you can call:

```python
chain.invoke(...)
chain.batch(...)
chain.stream(...)
```

---

### 2. Is an Agent a Runnable?

Yes.

Modern LangChain agents also expose the Runnable interface and are typically executed through an `AgentExecutor` or embedded inside LangGraph workflows.

---

### 3. Can a Chain call tools?

A traditional chain cannot **decide** to call tools dynamically.

However, a chain can contain a **predefined tool invocation** as one of its fixed steps.

---

### 4. Can an Agent contain chains?

Yes.

Example:

```text
Planner

↓

Summarization Chain

↓

Search Chain

↓

SQL Chain

↓

Final Answer
```

Agents often orchestrate multiple chains.

---

### 5. When should you avoid Agents?

Avoid agents when the workflow is deterministic.

Example:

```text
PDF

↓

Summarize

↓

Return
```

Using an agent here adds unnecessary cost and latency.

---

# Senior AI Engineer Interview Answer

A concise, strong interview response would be:

> **A Runnable is the core execution abstraction in LangChain—any component that accepts input and produces output, such as prompts, LLMs, retrievers, parsers, and chains. Runnables share a common interface (`invoke`, `batch`, `stream`, and async variants), making them easy to compose with LCEL. A Chain is a fixed, deterministic workflow where the execution path is predefined. An Agent is dynamic: it reasons about the problem, chooses tools at runtime, observes results, and may perform multiple iterations before producing a final answer. For complex, stateful workflows with branching and checkpointing, LangGraph extends these concepts into durable graph-based orchestration.
