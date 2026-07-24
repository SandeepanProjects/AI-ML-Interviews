Memory is what makes an AI agent feel like it is **persistent, stateful, and capable of solving long-running tasks**. Without memory, an agent is just a loop that repeatedly calls an LLM. Every iteration starts from scratch.

One of the biggest misconceptions is that the LLM itself has memory.

**It does not.**

The agent framework is responsible for storing, retrieving, and updating memory.

---

# Agent Without Memory

Suppose we build a simple agent.

```text
User:
Book a flight to Delhi.

↓

Agent:
Searching flights...

↓

User:
Also book a hotel.
```

Without memory, the agent sees only

```text
Also book a hotel.
```

It has forgotten

* destination
* travel dates
* flight information

The conversation becomes inconsistent.

---

# Agent With Memory

Now imagine

```text
User:
Book a flight to Delhi tomorrow.

↓

Memory

Destination = Delhi

Date = Tomorrow

↓

User:
Also book a hotel.
```

The agent reads memory

```text
Destination = Delhi
Date = Tomorrow
```

and automatically searches for hotels in Delhi for tomorrow.

---

# Types of Agent Memory

Real-world AI agents usually have four kinds of memory.

```text
                 Agent Memory

        ┌─────────┼──────────┬──────────┐

        ▼         ▼          ▼          ▼

 Working   Short-term   Long-term   Episodic
 Memory      Memory      Memory      Memory
```

Each serves a different purpose.

---

# 1. Working Memory

Working memory exists only during one reasoning session.

Think of it as the agent's scratchpad.

Example

```text
Question

↓

Need SQL query

↓

Execute SQL

↓

Need calculation

↓

Return answer
```

The intermediate steps are working memory.

---

Example

```python
class AgentState:

    def __init__(self):

        self.thoughts = []

        self.observations = []

        self.tool_results = []
```

Every tool updates the state.

```python
state.tool_results.append(result)
```

---

# 2. Short-Term Memory

Short-term memory stores recent conversation history.

Example

```text
User:
My name is John.

Assistant:
Nice to meet you.

User:
Where do I live?

Assistant:
...
```

Conversation history

```text
Turn 1

Turn 2

Turn 3

Turn 4
```

is stored.

---

Simple implementation

```python
class ChatMemory:

    def __init__(self):

        self.messages = []

    def add(self, role, content):

        self.messages.append({
            "role": role,
            "content": content
        })

    def get(self):

        return self.messages
```

Usage

```python
memory = ChatMemory()

memory.add(
    "user",
    "Book a flight."
)

memory.add(
    "assistant",
    "Where?"
)

print(memory.get())
```

---

# Problem

Suppose conversation reaches

```text
500 messages
```

The LLM cannot receive everything.

Context window overflow.

---

Solution

Conversation summarization.

```text
500 messages

↓

LLM Summary

↓

One paragraph

↓

Continue
```

---

Example

```python
if len(memory.messages) > 100:

    summary = summarize(memory.messages)

    memory.messages = [
        {
            "role": "system",
            "content": summary
        }
    ]
```

---

# 3. Long-Term Memory

This survives after the conversation ends.

Usually stored in

* PostgreSQL
* Redis
* Vector database
* Knowledge graph

Example

```text
User Preferences

↓

Likes Python

Lives in Bangalore

Uses AWS

Works at EY
```

Next week

Agent retrieves

```text
Previous memories
```

---

Simple implementation

```python
class LongTermMemory:

    def __init__(self):

        self.database = {}

    def save(self, user, key, value):

        self.database.setdefault(
            user,
            {}
        )[key] = value

    def load(self, user):

        return self.database.get(user, {})
```

Usage

```python
memory.save(
    "alice",
    "favorite_language",
    "Python"
)

print(memory.load("alice"))
```

---

# Real Production Storage

Instead of dictionaries

```text
Agent

↓

Memory Service

↓

Redis

↓

Postgres

↓

Vector DB
```

Different memory types use different storage.

---

# 4. Episodic Memory

Humans remember experiences.

Agents can too.

Example

```text
User

↓

Asked about Kubernetes

↓

Retrieved Docs

↓

Generated Answer

↓

User said

"This solved my issue."
```

Store

```text
Successful episode
```

Later

```text
Similar question

↓

Reuse previous strategy
```

---

# Semantic Memory

Semantic memory stores facts.

Example

```text
Python

↓

Programming language
```

or

```text
Customer

↓

Premium member
```

Unlike episodic memory, this is not tied to a specific conversation.

---

# Memory Architecture

Production systems separate memory by purpose.

```text
                     Agent
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
Working Memory   Conversation Memory   Long-Term Memory
      │                │                │
      ▼                ▼                ▼
 Python State      Redis Cache      PostgreSQL
                                        │
                                        ▼
                                   Vector Database
```

---

# Retrieval-Augmented Memory

Suppose user says

```text
Six months ago

I prefer AWS over Azure.
```

Agent stores

```text
Embedding

↓

Vector DB
```

Later

User asks

```text
Recommend cloud architecture.
```

Memory search

```text
Recommend cloud architecture

↓

Embedding

↓

Vector Search

↓

AWS Preference
```

Now the agent remembers.

---

Example

```python
memory_vector_db.add(

    text="User prefers AWS",

    metadata={
        "user":"alice"
    }
)
```

Later

```python
results = memory_vector_db.search(
    "cloud provider"
)
```

---

# Memory in LangGraph

State is passed between nodes.

```python
from typing import TypedDict

class AgentState(TypedDict):

    question: str

    documents: list

    answer: str
```

Each node updates

```python
def retrieve(state):

    docs = search(state["question"])

    return {
        "documents": docs
    }
```

The graph carries state automatically.

---

# ReAct Agent Memory

A ReAct agent stores

```text
Thought

↓

Action

↓

Observation
```

Example

```text
Thought

Need customer data

↓

Action

SQL

↓

Observation

Customer exists

↓

Thought

Need invoices

↓

Action

Invoice API
```

The reasoning chain itself becomes working memory.

---

# Production Memory Service

A real enterprise implementation separates memory into a service.

```python
class MemoryService:

    def __init__(self,
                 redis,
                 postgres,
                 vector_db):

        self.redis = redis

        self.postgres = postgres

        self.vector_db = vector_db

    def save_message(self,
                     user,
                     message):

        self.redis.append(
            user,
            message
        )

    def save_fact(self,
                  user,
                  fact):

        self.postgres.insert(
            user,
            fact
        )

    def save_embedding(
        self,
        text
    ):

        vector = embed(text)

        self.vector_db.add(
            vector,
            text
        )
```

The agent never talks directly to storage.

Everything goes through the memory service.

---

# Memory Lifecycle

```text
              User Message
                    │
                    ▼
             Working Memory
                    │
                    ▼
        Should this be remembered?
              /               \
            No                 Yes
            │                  │
            ▼                  ▼
         Discard        Store as Memory
                              │
          ┌───────────────────┼──────────────────┐
          ▼                   ▼                  ▼
     Conversation         Semantic Fact      Episodic Event
         Memory             (Postgres)        (Vector DB)
          │                   │                  │
          └───────────────────┴──────────────────┘
                              ▼
                      Memory Retrieval
                              │
                              ▼
                       Future Agent Runs
```

Notice that **not every message becomes long-term memory**. Production systems apply rules (or an LLM classifier) to decide whether something is worth remembering.

---

# Real Enterprise Example

Imagine a financial advisor agent.

### Conversation

```text
User:
I invest only in ESG funds.

↓

Agent stores

Preference:
ESG Investing
```

A month later

```text
User:
Recommend some ETFs.
```

The agent performs:

1. Load recent conversation (Redis)
2. Load user profile (PostgreSQL)
3. Retrieve relevant past preferences (Vector DB)
4. Build prompt

Prompt sent to the LLM:

```text
User preference:
Only recommend ESG investments.

Current question:
Recommend ETFs.
```

The recommendation is now personalized without the user repeating their preference.

---

# Production Memory Architecture

```text
                     User Message
                           │
                           ▼
                    Agent Controller
                           │
           ┌───────────────┼────────────────┐
           ▼               ▼                ▼
   Working Memory   Conversation Memory   Long-Term Memory
   (Current State)      (Redis)         (Postgres + Vector DB)
           │               │                │
           └───────────────┼────────────────┘
                           ▼
                    Memory Retriever
                           │
                           ▼
                  Prompt Construction
                           │
                           ▼
                           LLM
                           │
                           ▼
                   Updated Memory Store
```

---

# Senior AI Engineer Best Practices

A production-grade agent typically uses multiple memory layers together:

| Memory Type         | Typical Storage           | Purpose                                      | Lifetime           |
| ------------------- | ------------------------- | -------------------------------------------- | ------------------ |
| Working Memory      | In-process Python objects | Current reasoning state, tool outputs        | Seconds to minutes |
| Conversation Memory | Redis or session store    | Recent chat history                          | Minutes to days    |
| Semantic Memory     | PostgreSQL or graph DB    | Persistent facts and preferences             | Months to years    |
| Episodic Memory     | Vector database           | Past experiences and successful interactions | Months to years    |

The important design principle is that **the LLM never "remembers" anything by itself**. Memory is implemented by the application: it decides what to store, where to store it, when to retrieve it, and how to incorporate it into the prompt or agent state. This separation makes agent behavior predictable, scalable, and easier to debug in real-world systems.
