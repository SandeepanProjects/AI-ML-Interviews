The **ReAct (Reason + Act) Agent** is one of the **most important concepts in LangChain, LangGraph, and Agentic AI**.

Almost every production AI agent today uses some variation of the ReAct pattern.

If you're interviewing for a **Senior AI Engineer** role, you should understand not just **what it is**, but **how it works internally**.

---

# What is ReAct?

**ReAct = Reason + Act**

Instead of answering immediately, the LLM alternates between:

1. **Reasoning** about the problem.
2. **Acting** by calling tools.
3. **Observing** tool outputs.
4. **Reasoning again** using those observations.
5. Repeating until it can answer.

The execution loop looks like this:

```text
User Question
      │
      ▼
   Thought
      │
      ▼
 Need Tool?
   │     │
 No      Yes
 │        │
 ▼        ▼
Answer   Tool Call
           │
           ▼
     Observation
           │
           ▼
      Thought Again
```

The key idea is:

> The model doesn't have to solve everything in one step. It can gather information, think again, and then decide what to do next.

---

# Why Was ReAct Introduced?

Suppose the user asks:

> What is the current weather in Bangalore?

A normal LLM can only answer from training data.

It cannot access today's weather.

With ReAct:

```text
Question

↓

Think

↓

Need weather information

↓

Call Weather API

↓

Receive 24°C

↓

Think again

↓

Generate answer
```

The model uses external information before responding.

---

# Real-Life Analogy

Imagine a doctor.

Without ReAct:

```text
Patient

↓

Doctor guesses diagnosis

↓

Medicine
```

With ReAct:

```text
Patient

↓

Doctor thinks

↓

Orders blood test

↓

Receives report

↓

Thinks again

↓

Diagnosis

↓

Prescription
```

The doctor alternates between **thinking** and **acting**.

That's exactly what ReAct does.

---

# The ReAct Loop

Internally:

```text
              Question

                  │

                  ▼

              Thought

                  │

                  ▼

           Choose Tool?

                  │

                  ▼

             Execute Tool

                  │

                  ▼

            Tool Observation

                  │

                  ▼

            Updated Thought

                  │

                  ▼

         Need Another Tool?

          Yes             No

           │               │

           ▼               ▼

      Execute Again     Final Answer
```

Notice the repeated cycle:

```text
Think

↓

Act

↓

Observe

↓

Think

↓

Act

↓

Observe
```

---

# Example 1: Weather Question

User:

```text
What's today's weather in Bangalore?
```

Agent reasoning:

```text
Thought:
I need current weather information.

↓

Action:
Weather API

↓

Observation:
24°C and cloudy

↓

Thought:
I now have the required information.

↓

Final Answer:
Today's weather is 24°C and cloudy.
```

---

# Example 2: Math + Search

User:

```text
Population of India divided by 2
```

Reasoning:

```text
Thought:
I don't know the current population.

↓

Search Tool

↓

Observation:
1.43 billion

↓

Thought:
Need calculation.

↓

Calculator Tool

↓

Observation:
715 million

↓

Thought:
Finished.

↓

Answer
```

Notice that **two different tools** were used.

---

# Example 3: Database Agent

User:

```text
Show customer balance for Alice.
```

Execution:

```text
Question

↓

Thought:
Need database lookup.

↓

SQL Tool

↓

Observation:
Balance = ₹15,000

↓

Thought:
Enough information.

↓

Answer
```

---

# Internal ReAct Prompt

The LLM is prompted to follow a specific pattern.

```text
Question:
What is 15 × 20?

Thought:
I should use the calculator.

Action:
calculator

Action Input:
15*20

Observation:
300

Thought:
I now know the answer.

Final Answer:
300
```

The model generates structured reasoning rather than a direct response.

---

# How LangChain Implements ReAct

Suppose we have two tools.

```python
from langchain.tools import tool

@tool
def weather(city: str):
    """Get weather information."""
    return "24°C"


@tool
def calculator(expr: str):
    """Perform calculations."""
    return str(eval(expr))
```

Create the model.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1"
)
```

Create the agent.

```python
from langchain.agents import create_react_agent

agent = create_react_agent(
    llm=llm,
    tools=[weather, calculator]
)
```

Now execute.

```python
agent.invoke(
    {
        "input":
        "What's the weather in Bangalore?"
    }
)
```

Internally, the ReAct loop begins.

---

# What Happens Internally?

Step 1

The LLM receives:

```text
Question:
What's today's weather?
```

---

Step 2

The model reasons.

```text
Thought:
Weather requires external information.
```

---

Step 3

The model emits a tool call.

```text
Action:
weather

Input:
Bangalore
```

---

Step 4

LangChain executes:

```python
weather(
    "Bangalore"
)
```

Result:

```text
24°C
```

---

Step 5

LangChain feeds that result back into the LLM.

```text
Observation:

24°C
```

---

Step 6

The model reasons again.

```text
Thought:

Now I have weather information.
```

---

Step 7

Final response.

```text
Today's weather is 24°C.
```

---

# How LangChain Controls the Loop

Conceptually, LangChain does something like this:

```python
while True:

    response = llm.invoke(prompt)

    if response.is_tool_call:

        result = tool(response.tool_input)

        prompt += result

    else:

        break
```

The real implementation is more sophisticated, but this captures the idea:

* Ask the model what to do.
* Execute tool calls.
* Feed observations back.
* Repeat until the model stops.

---

# ReAct vs Normal LLM

Without ReAct:

```text
Question

↓

LLM

↓

Answer
```

With ReAct:

```text
Question

↓

LLM Think

↓

Tool

↓

LLM Think

↓

Tool

↓

LLM Think

↓

Answer
```

---

# ReAct vs Chain

Chain:

```text
Retrieve

↓

Generate

↓

END
```

Always the same.

ReAct:

```text
Question

↓

Think

↓

Search?

↓

Maybe

↓

Calculator?

↓

Maybe

↓

Answer
```

The workflow changes dynamically.

---

# ReAct in LangGraph

In LangGraph, a ReAct agent is often represented explicitly as a graph.

```text
              User

                │

                ▼

          Agent (Think)

                │

        Tool Needed?

         │        │

        No       Yes

         │        │

         ▼        ▼

      Finish   Tool Node

                 │

                 ▼

            Observation

                 │

                 ▼

          Agent (Think Again)
```

This creates a cycle.

---

# Why ReAct Uses Cycles

Suppose the agent needs three tools.

```text
Question

↓

Search

↓

Calculator

↓

Database

↓

Answer
```

The number of tool calls is not known beforehand.

The agent loops until it decides:

> I have enough information.

This is why LangGraph is well suited for ReAct-style agents.

---

# Production Example

Imagine an enterprise support assistant.

Available tools:

* CRM lookup
* PostgreSQL
* Vector database
* Jira
* Calculator

User asks:

> What's the status of invoice INV-2024-001 and what is the outstanding balance after a 10% discount?

Execution:

```text
Question

↓

Thought

↓

Invoice Tool

↓

Observation

↓

Thought

↓

Calculator

↓

Observation

↓

Thought

↓

Answer
```

The agent combines multiple tools to complete the task.

---

# Advantages of ReAct

* Uses external knowledge.
* Handles complex, multi-step tasks.
* Chooses tools dynamically.
* Produces more reliable answers than guessing.
* Works well with APIs, databases, and retrieval systems.

---

# Limitations

Because the model controls the loop, ReAct can:

* Make unnecessary tool calls.
* Get stuck in repeated reasoning.
* Increase latency and cost.
* Loop indefinitely if not constrained.

Production systems usually add:

* Maximum iteration limits.
* Tool timeouts.
* Retry policies.
* Human approval checkpoints.
* Observability and tracing.

---

# ReAct vs Function Calling

| ReAct                               | Function Calling                                   |
| ----------------------------------- | -------------------------------------------------- |
| Multiple reasoning steps            | Usually one model response at a time               |
| Can call several tools sequentially | May require orchestration for multi-step workflows |
| Explicit Think → Act → Observe loop | Structured tool invocation mechanism               |
| Suitable for autonomous agents      | Suitable for reliable tool execution               |

Modern agent frameworks often combine **ReAct reasoning** with **LLM function/tool calling** under the hood.

---

# Interview Questions

### Why is it called ReAct?

Because the agent alternates between **Reasoning** and **Acting** until it reaches a solution.

### Why is ReAct better than a chain?

A chain has a fixed sequence of operations. A ReAct agent decides at runtime whether to use tools, which tools to use, and how many times to use them.

### Why does ReAct need memory?

Each new reasoning step depends on previous thoughts, tool calls, and observations. The agent must maintain this context throughout the loop.

---

# Senior AI Engineer Interview Answer

> **A ReAct agent implements a Reason → Act → Observe loop. Instead of producing an answer immediately, the LLM first reasons about the task, decides whether external information or computation is needed, invokes the appropriate tool, receives the observation, and reasons again using that new information. This process repeats until the agent determines it has enough context to generate a final answer. In LangChain, the framework manages the interaction between the LLM and tools, while in LangGraph the same pattern is modeled explicitly as a cyclic workflow with state flowing through reasoning and tool nodes. ReAct enables dynamic tool selection, multi-step problem solving, and integration with external systems such as APIs, databases, search engines, and vector stores.
