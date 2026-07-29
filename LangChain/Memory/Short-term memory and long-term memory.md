Short-term memory and long-term memory are **fundamentally different** in AI agents. Many interview candidates confuse them, but understanding the distinction is essential for designing production-grade conversational systems.

---

# Human Analogy

Imagine meeting someone.

### Short-Term Memory

You remember what they said **during the current conversation**.

```text
Person:
My name is Alice.

↓

10 minutes later

↓

You:
Nice to meet you, Alice.
```

After a week:

```text
You:
Sorry, I forgot your name.
```

The memory disappears.

---

### Long-Term Memory

Now suppose you save important facts.

```text
Alice

↓

Lives in Bangalore

↓

Likes Python

↓

Works at EY
```

Months later:

```text
You:
How is your Python project going?
```

You remembered across sessions.

---

# AI Memory Architecture

```text
                    User

                      │

                      ▼

              Memory Manager

          ┌───────────┴────────────┐

          ▼                        ▼

   Short-Term Memory        Long-Term Memory

          ▼                        ▼

 Recent Conversation      Persistent Knowledge

          ▼                        ▼

           Merge Relevant Context

                      │

                      ▼

                     LLM
```

---

# What is Short-Term Memory?

Short-term memory stores **the current conversation**.

Example:

```text
User:
My name is Alice.

Assistant:
Hello Alice.

User:
I work at EY.

Assistant:
Great.

User:
What's my name?
```

The agent answers because the previous messages are still available.

---

## Characteristics

* Lives only during the conversation
* Usually stored in RAM or graph state
* Fast access
* Not persistent
* Limited by context window

---

# Short-Term Memory Code (LangGraph)

State:

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: list[BaseMessage]
```

Chat node:

```python
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1")

def chatbot(state: AgentState):

    response = llm.invoke(
        state["messages"]
    )

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }
```

Execution:

```python
from langchain_core.messages import HumanMessage

state = {
    "messages": [
        HumanMessage(content="My name is Alice")
    ]
}

new_state = chatbot(state)

state["messages"].extend(new_state["messages"])
```

The conversation history grows with every interaction.

---

# Problem with Short-Term Memory

Suppose the conversation lasts 3 hours.

```text
1000 Messages

↓

Conversation History

↓

LLM
```

Problems:

* Large prompts
* High cost
* Slower responses
* Context window overflow

---

# Improving Short-Term Memory

Instead of sending everything:

```python
MAX_MESSAGES = 10

messages = messages[-MAX_MESSAGES:]
```

Keep only the most recent messages.

---

# What is Long-Term Memory?

Long-term memory stores **important information across conversations**.

Example:

Today:

```text
User:
My favorite language is Python.
```

Memory:

```text
Favorite Language

↓

Python
```

Next month:

```text
User:
Recommend a web framework.
```

The assistant retrieves:

```text
Favorite Language:
Python
```

Answer:

```text
FastAPI
```

The user didn't repeat anything.

---

# Characteristics

* Persistent
* Stored in databases
* Retrieved only when relevant
* Survives server restarts
* Shared across sessions

---

# Long-Term Memory Architecture

```text
User

↓

Extract Important Facts

↓

Embedding Model

↓

Vector Database

↓

Next Conversation

↓

Semantic Search

↓

Relevant Memory

↓

LLM
```

---

# Long-Term Memory Code

Imagine storing memories in a vector database.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

memory = "User likes Python"

embedding = model.encode(memory)
```

Store:

```python
vector_db.add(
    id="user_123_memory_1",
    embedding=embedding,
    metadata={
        "text": memory
    }
)
```

---

Later:

```python
query = "Recommend a programming framework"

query_embedding = model.encode(query)
```

Search:

```python
results = vector_db.search(
    embedding=query_embedding,
    top_k=3
)
```

Output:

```text
User likes Python
```

Now send it to the LLM.

---

# Combining Retrieved Memory

```python
prompt = f"""
Relevant user memory:

{results[0]["text"]}

Question:

Recommend a web framework.
"""
```

LLM:

```text
Since you prefer Python,
I recommend FastAPI.
```

---

# LangGraph Example

State:

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: list[BaseMessage]
    retrieved_memory: list[str]
```

Memory node:

```python
def retrieve_memory(state):

    question = state["messages"][-1].content

    memories = vector_db.search(
        question,
        top_k=3
    )

    return {
        "retrieved_memory": memories
    }
```

Chat node:

```python
def chatbot(state):

    context = "\n".join(
        state["retrieved_memory"]
    )

    prompt = f"""
User Memory:

{context}

Conversation:

{state["messages"]}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }
```

Workflow:

```text
User

↓

Retrieve Memory

↓

Merge Memory

↓

LLM

↓

Answer
```

---

# What Should Be Stored?

Not every message belongs in long-term memory.

Good examples:

```text
Favorite programming language

Company

Preferred cloud provider

Timezone

Preferred coding style

Frequently used libraries
```

Poor examples:

```text
Hello

Thanks

Good morning

Okay

Bye
```

These add little long-term value.

---

# Memory Extraction

Instead of saving everything:

Conversation:

```text
User:
I work at EY.

User:
I like Python.

User:
Hello.
```

LLM extracts:

```text
Important Facts

↓

Works at EY

Likes Python
```

Only those are persisted.

Example:

```python
def extract_memory(conversation):

    prompt = f"""
Extract long-term user facts.

Conversation:

{conversation}
"""

    return llm.invoke(prompt)
```

---

# Production Architecture

```text
                   User

                     │

                     ▼

             Recent Messages

                     │

                     ▼

         Long-Term Retriever

                     │

                     ▼

            Merge Context

                     │

                     ▼

                    LLM

                     │

                     ▼

          Extract New Memories

                     │

                     ▼

              Vector Database
```

This creates a continuous learning loop.

---

# Short-Term vs Long-Term

| Feature   | Short-Term Memory           | Long-Term Memory                      |
| --------- | --------------------------- | ------------------------------------- |
| Stores    | Current conversation        | Important user knowledge              |
| Lifetime  | Current session             | Weeks, months, years                  |
| Storage   | RAM, graph state            | PostgreSQL, Redis, Vector DB          |
| Retrieval | Always included             | Retrieved only when relevant          |
| Size      | Small                       | Large                                 |
| Cost      | Increases with conversation | Constant (retrieve only what matters) |
| Example   | Last 10 chat messages       | "User prefers Python"                 |

---

# Production Example

Suppose the user says:

```text
I work at EY.
```

### Short-Term

Current session:

```text
Message History

↓

Contains:
I work at EY.
```

If the server restarts:

```text
Gone.
```

---

### Long-Term

Conversation:

```text
I work at EY.
```

↓

Memory Extractor

↓

Store:

```text
Employment:
EY
```

Three months later:

```text
User:
Recommend an AI architecture.
```

Retriever returns:

```text
Works at EY
```

The assistant personalizes the answer without the user repeating the information.

---

# Best Practices

### Short-Term Memory

* Keep only recent messages (for example, the last 10–20 turns).
* Summarize older parts of the conversation.
* Store in LangGraph state or chat history.
* Clear when the session ends.

### Long-Term Memory

* Store only meaningful, stable facts.
* Use semantic retrieval instead of loading everything.
* Encrypt and protect sensitive data.
* Support updating or deleting outdated memories.
* Let users review and manage persisted memories when appropriate.

---

# Interview Questions

### Why not store everything as long-term memory?

Because most conversations are temporary. Storing every greeting or small talk increases storage, retrieval noise, and reduces the quality of relevant memory retrieval.

---

### Why not use only short-term memory?

Because it disappears when the session ends and is limited by the model's context window.

---

### Can both be used together?

Yes. Most enterprise AI assistants use:

* **Short-term memory** for the current conversation.
* **Long-term memory** for persistent user knowledge.
* **RAG** for external documents and enterprise data.

---

# Enterprise Architecture

```text
                           User
                             │
                             ▼
                  Current Conversation
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
     Short-Term Memory                Long-Term Memory
   (Recent chat in state)       (Vector DB / PostgreSQL)
            │                                 │
            └──────────────┬──────────────────┘
                           ▼
                   Context Builder
                           ▼
                          LLM
                           ▼
                     Final Response
                           ▼
                Memory Extraction Service
                           ▼
               Store New Long-Term Facts
```

---

# Senior AI Engineer Interview Answer

> **Short-term memory maintains the context of the current conversation by storing recent messages in the application state, allowing the LLM to answer follow-up questions naturally. Long-term memory persists important user facts across sessions in a database or vector store. Before each LLM call, the application retrieves only the most relevant long-term memories using semantic search and combines them with the recent conversation. In production, I typically use a layered memory architecture consisting of a short message window, conversation summaries for older dialogue, and long-term semantic memory backed by a vector database. This approach provides personalization while keeping token usage and latency under control.**
