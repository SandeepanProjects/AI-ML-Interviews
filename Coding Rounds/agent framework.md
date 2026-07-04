A **Senior AI Engineer “Agent Framework”** question is not about LangChain-style wrappers—it’s about whether you understand how to design a **multi-step reasoning + tool-using system** with:

* Planner (decides steps)
* Executor (runs tools)
* Memory (stores context)
* Tool registry
* State machine / loop control
* Observability (logs, traces)
* Safety (timeouts, retries, limits)

Think of it as building a **mini “LLM OS runtime”**.

---

# 1. What is an Agent Framework?

An agent framework is a system where an LLM:

```text id="a1"
OBSERVE → THINK → ACT → OBSERVE → ...
```

Instead of:

```text id="a2"
User → LLM → Answer
```

We now have:

```text id="a3"
User
 ↓
Agent Loop
 ↓
Planner (LLM)
 ↓
Tool Executor
 ↓
Memory Store
 ↓
Final Answer
```

---

# 2. Core Components

A production-grade agent framework has:

### 1. LLM (Reasoning Engine)

* GPT / Claude / local model

### 2. Tool Registry

* Search
* Calculator
* Vector DB
* API calls

### 3. Memory

* Short-term (conversation)
* Long-term (vector store)

### 4. Planner

* Breaks task into steps

### 5. Executor

* Runs tools

### 6. Controller Loop

* Stops when goal is reached

---

# 3. High-Level Architecture

```text id="b1"
            User Query
                 │
                 ▼
          Agent Controller
                 │
        ┌────────┴────────┐
        ▼                 ▼
     Planner         Memory Store
        │
        ▼
   Tool Selector
        │
        ▼
   Tool Executor
        │
        ▼
     Observation
        │
        └─────── loop ───────┘
```

---

# 4. Design Goal

We want:

```python id="c1"
agent.run("Find papers on RAG and summarize them")
```

Agent should:

1. Search web / vector DB
2. Fetch documents
3. Summarize
4. Store memory
5. Return answer

---

# 5. Tool System

Each tool is just a function:

```python id="t1"
def search(query):
    return f"results for {query}"


def calculator(expr):
    return eval(expr)
```

We register them:

```python id="t2"
TOOLS = {
    "search": search,
    "calculator": calculator
}
```

---

# 6. Memory System

Simple in-memory memory:

```python id="m1"
class Memory:

    def __init__(self):
        self.store = []

    def add(self, item):
        self.store.append(item)

    def get(self):
        return self.store[-5:]
```

Production version = vector DB (semantic memory).

---

# 7. Agent State

We track execution state:

```python id="s1"
class AgentState:

    def __init__(self, query):

        self.query = query
        self.steps = []
        self.memory = []
        self.final_answer = None
```

---

# 8. Planner (LLM Simulation)

We simulate planning logic:

```python id="p1"
def planner(query):

    if "sum" in query:
        return ["search", "summarize"]

    if "calculate" in query:
        return ["calculator"]

    return ["search"]
```

In real systems → this is an LLM call.

---

# 9. Tool Executor

```python id="e1"
def execute_tool(tool_name, input_text):

    tool = TOOLS.get(tool_name)

    if not tool:
        return "Tool not found"

    return tool(input_text)
```

---

# 10. Agent Loop (Core Engine)

This is the MOST important part.

```python id="loop1"
class Agent:

    def __init__(self):
        self.memory = Memory()

    def run(self, query):

        state = AgentState(query)

        plan = planner(query)

        for step in plan:

            observation = execute_tool(step, query)

            state.steps.append((step, observation))

            self.memory.add(observation)

        state.final_answer = self.synthesize(state)

        return state.final_answer
```

---

# 11. Synthesis Step (Final Reasoning)

```python id="s1"
def synthesize(self, state):

    context = self.memory.get()

    return f"""
Based on:
{context}

Answer to: {state.query}
"""
```

---

# 12. Full Working Minimal Agent Framework

```python id="full1"
class Memory:

    def __init__(self):
        self.store = []

    def add(self, item):
        self.store.append(item)

    def get(self):
        return self.store[-5:]


def search(query):
    return f"[search results for '{query}']"


def calculator(expr):
    return str(eval(expr))


TOOLS = {
    "search": search,
    "calculator": calculator
}


class AgentState:

    def __init__(self, query):
        self.query = query
        self.steps = []
        self.memory = []
        self.final_answer = None


def planner(query):

    if any(x in query for x in ["add", "+", "sum"]):
        return ["calculator"]

    return ["search"]


def execute_tool(tool, input_text):

    return TOOLS[tool](input_text)


class Agent:

    def __init__(self):
        self.memory = Memory()

    def synthesize(self, state):

        return f"""
Steps:
{state.steps}

Memory:
{self.memory.get()}

Final answer for: {state.query}
"""


    def run(self, query):

        state = AgentState(query)

        plan = planner(query)

        for step in plan:

            result = execute_tool(step, query)

            state.steps.append((step, result))

            self.memory.add(result)

        return self.synthesize(state)
```

---

# 13. Example Run

```python id="r1"
agent = Agent()

print(agent.run("search papers on RAG"))
```

Output:

```text id="o1"
Steps:
[('search', "[search results for 'search papers on RAG']")]

Memory:
["[search results for 'search papers on RAG']"]

Final answer for: search papers on RAG
```

---

# 14. Real Production Agent Framework

A real system includes:

## 1. LLM Planner

```text id="l1"
GPT decides:

Step 1 → search
Step 2 → retrieve docs
Step 3 → summarize
```

---

## 2. Tool Router

* HTTP tools
* DB tools
* vector DB tools
* code execution tools

---

## 3. Retry + Timeout Layer

Every tool call:

* retry logic
* circuit breaker
* fallback tools

---

## 4. Memory (Two Layers)

### Short-term:

* conversation buffer

### Long-term:

* vector store (semantic memory)

---

## 5. Execution Modes

* ReAct (Reason + Act loop)
* Plan-and-execute
* Tree-of-thought agents
* Multi-agent collaboration

---

# 15. Multi-Agent Extension (Senior Level)

Instead of one agent:

```text id="m1"
Planner Agent
   ↓
Research Agent
   ↓
Tool Agent
   ↓
Critic Agent
```

Each agent has a role.

---

# 16. Production AI Architecture

```text id="p1"
User
 ↓
API Gateway
 ↓
Agent Controller
 ↓
Planner LLM
 ↓
Tool Router
 ↓
Vector DB / APIs / Code Executor
 ↓
Memory Store
 ↓
Final Response
```

---

# 17. Key Engineering Challenges

A real system must handle:

### 1. Infinite loops

→ max steps limit

### 2. Tool failure

→ retry + fallback

### 3. Latency explosion

→ parallel tool execution

### 4. Cost control

→ token + tool budget

### 5. Memory pollution

→ summarization / pruning

---

# 18. Time Complexity

Depends on loop depth:

```text id="t1"
O(steps × tool_cost)
```

Vector search + LLM dominates runtime.

---

# 19. How to Answer in a Senior AI Interview

A strong answer:

> “An agent framework is a control loop that enables an LLM to iteratively reason and act using external tools. It consists of a planner (often an LLM), a tool registry, an execution engine, and memory. The agent operates in a loop: it interprets the user goal, selects tools, executes them, stores observations in memory, and refines its plan until a termination condition is met. Production systems add safeguards like step limits, retries, timeouts, circuit breakers, and both short-term and long-term memory using vector databases. Advanced designs extend this into multi-agent systems with specialized roles such as planner, executor, and critic.”
