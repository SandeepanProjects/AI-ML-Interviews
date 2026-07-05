# What is ReAct?

**ReAct** stands for:

> **Re**ason + **Act**

It is an agent framework where an LLM alternates between:

1. **Reasoning (Thinking)**
2. **Taking Actions (Calling Tools)**
3. **Observing Results**
4. **Reasoning Again**
5. **Producing the Final Answer**

Instead of answering immediately, the model **thinks step by step and uses external tools when needed**.

---

# Why Was ReAct Introduced?

A normal LLM works like this:

```text
Question
   │
   ▼
LLM
   │
   ▼
Answer
```

This works for questions already covered by the model's training.

Example:

```
Q: What is Kubernetes?
```

The LLM already knows the answer.

---

Now consider:

```
What is Apple's stock price today?
```

The model cannot reliably know today's price because it changes.

Without tools:

```text
Question
   │
   ▼
LLM
   │
   ▼
Hallucinated Answer
```

The model guesses.

---

ReAct solves this.

```text
Question
    │
    ▼
LLM Reasoning
    │
    ▼
Search Tool
    │
    ▼
Observation
    │
    ▼
LLM Reasoning
    │
    ▼
Final Answer
```

---

# The ReAct Loop

Every ReAct agent follows the same cycle.

```text
Thought
   │
   ▼
Action
   │
   ▼
Observation
   │
   ▼
Thought
   │
   ▼
Action
   │
   ▼
Observation
   │
   ▼
Final Answer
```

This loop continues until no more tools are needed.

---

# Example

User asks:

```
What is the weather in Bangalore?
```

The LLM reasons:

```text
Thought:
I don't know current weather.

Action:
Call Weather API
```

Weather API returns:

```text
28°C
Cloudy
```

The LLM continues:

```text
Observation:
Temperature is 28°C.

Thought:
Now I have enough information.

Final Answer:
It is currently 28°C and cloudy in Bangalore.
```

Notice the LLM **did not hallucinate**.

---

# Internal Execution

```text
User
 │
 ▼
Question

"What is Tesla stock price?"

 │
 ▼

Thought

"I need live information."

 │
 ▼

Action

Search Tool

 │
 ▼

Observation

Tesla = $321.84

 │
 ▼

Thought

"I now know the answer."

 │
 ▼

Final Answer
```

---

# ReAct Prompt Format

A typical prompt looks like:

```text
Question: Who is the CEO of OpenAI?

Thought:
I should search for the latest CEO.

Action:
search("CEO of OpenAI")

Observation:
Sam Altman is the CEO.

Thought:
I now know the answer.

Final Answer:
Sam Altman is the CEO of OpenAI.
```

---

# Implementing ReAct from Scratch

Let's build a very small ReAct agent.

## Step 1: Tool

```python
def calculator(expression: str):
    return eval(expression)
```

---

## Step 2: Fake LLM

```python
class FakeLLM:

    def think(self, question):

        if "+" in question:

            return {
                "thought": "I should use calculator.",
                "action": "calculator",
                "input": question
            }

        return {
            "thought": "I already know this.",
            "action": None
        }
```

---

## Step 3: Agent

```python
class ReActAgent:

    def __init__(self):

        self.llm = FakeLLM()

    def run(self, question):

        step = self.llm.think(question)

        print("Thought:", step["thought"])

        if step["action"] == "calculator":

            result = calculator(step["input"])

            print("Observation:", result)

            return f"Final Answer: {result}"

        return "No tool required."
```

---

Run it.

```python
agent = ReActAgent()

print(agent.run("25+17"))
```

Output

```text
Thought:
I should use calculator.

Observation:
42

Final Answer:
42
```

---

# Example with Search Tool

```python
knowledge = {
    "capital of france": "Paris",
    "capital of india": "New Delhi"
}

def search(query):

    return knowledge.get(query.lower(), "Not found")
```

---

LLM

```python
class FakeLLM:

    def think(self, question):

        return {

            "thought": "Need to search.",

            "action": "search",

            "input": question
        }
```

---

Agent

```python
class Agent:

    def run(self, question):

        step = FakeLLM().think(question)

        print("Thought:", step["thought"])

        observation = search(step["input"])

        print("Observation:", observation)

        return observation
```

Run

```python
agent = Agent()

agent.run("capital of france")
```

Output

```text
Thought:
Need to search.

Observation:
Paris
```

---

# Production ReAct Architecture

In a real AI system, the flow is much richer:

```text
                   User
                     │
                     ▼
               FastAPI Endpoint
                     │
                     ▼
                 ReAct Agent
                     │
          ┌──────────┼───────────┐
          │          │           │
          ▼          ▼           ▼
      Search Tool  SQL Tool  Calculator
          │          │           │
          └──────────┼───────────┘
                     │
                     ▼
                Observation
                     │
                     ▼
              LLM Reasons Again
                     │
                     ▼
              Final Response
```

---

# Production Code Structure

```text
app/

    api/
        chat.py

    agents/
        react_agent.py

    tools/
        search_tool.py
        sql_tool.py
        calculator_tool.py

    llm/
        openai_client.py

    prompts/
        react_prompt.py

    services/
        ai_service.py
```

---

# Production-Style Agent (Simplified)

```python
class ReActAgent:

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, question):

        history = []

        while True:

            response = self.llm.generate(question, history)

            if response["type"] == "final":
                return response["answer"]

            tool_name = response["tool"]
            tool_input = response["input"]

            observation = self.tools[tool_name](tool_input)

            history.append({
                "tool": tool_name,
                "input": tool_input,
                "observation": observation
            })
```

This captures the core ReAct loop:

```text
Reason
   ↓
Select Tool
   ↓
Execute Tool
   ↓
Receive Observation
   ↓
Reason Again
```

The `history` records previous thoughts, actions, and observations so the model can make informed decisions in later steps.

---

# Why ReAct Is Better Than Plain Prompting

Suppose the user asks:

```
What is the average salary of engineers in my database?
```

A plain LLM cannot access your database.

With ReAct:

```text
Thought:
Need database access.

↓

Action:
Execute SQL query.

↓

Observation:
Average salary = $120,000.

↓

Thought:
I now have the answer.

↓

Final Answer:
The average engineer salary is $120,000.
```

The agent **uses tools instead of guessing**, making answers more accurate and enabling interaction with live systems.

---

# Senior Interview Answer

If asked, "What is ReAct?" a strong answer is:

> ReAct (Reason + Act) is an agent framework in which an LLM alternates between reasoning and tool use. Instead of generating a final answer immediately, the model reasons about the problem, selects the appropriate tool (such as search, SQL, a calculator, or an API), observes the result, and continues reasoning until it has enough information to produce a grounded answer. This approach reduces hallucinations and allows LLMs to work with real-time data and external systems.

---

## When to Use ReAct

Use ReAct when the model needs to:

* Search the web or a document corpus
* Query SQL or vector databases
* Call external APIs
* Perform mathematical calculations
* Execute Python code
* Coordinate multiple tools before answering

This is why ReAct forms the foundation of many modern AI agents built with frameworks such as **LangGraph**, **LangChain Agents**, **OpenAI Agents SDK**, and similar orchestration systems.
