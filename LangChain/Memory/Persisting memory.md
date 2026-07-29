Persisting memory means **saving conversation state outside the LLM** so that it survives:

* Server restarts
* New user sessions
* Multiple API instances
* Kubernetes pod restarts
* Horizontal scaling

Without persistence:

```text
User: My name is Alice.

↓

Server Restarts

↓

Memory Lost

↓

User: What's my name?

↓

I don't know.
```

With persistence:

```text
User

↓

Memory Manager

↓

PostgreSQL / Redis / Vector DB

↓

Next Session

↓

Load Memory

↓

LLM
```

---

# Production Memory Architecture

```text
                User

                  │

                  ▼

             FastAPI API

                  │

                  ▼

           Memory Manager

        ┌─────────┼──────────┐

        ▼         ▼          ▼

   PostgreSQL   Redis    Vector DB

        │         │          │

        └─────────┼──────────┘

                  ▼

             Build Prompt

                  ▼

                 LLM
```

Each storage system has a different purpose:

| Storage           | Purpose                                  |
| ----------------- | ---------------------------------------- |
| PostgreSQL        | Persistent chat history and user profile |
| Redis             | Fast session cache                       |
| Vector DB         | Semantic long-term memory                |
| S3/Object Storage | Large attachments/documents              |

---

# Option 1: Persist Conversation in PostgreSQL

## Database Schema

```sql
CREATE TABLE conversations (

    id SERIAL PRIMARY KEY,

    thread_id VARCHAR(100),

    role VARCHAR(20),

    message TEXT,

    created_at TIMESTAMP DEFAULT NOW()

);
```

Example data:

| thread_id | role  | message          |
| --------- | ----- | ---------------- |
| user123   | human | My name is Alice |
| user123   | ai    | Nice to meet you |
| user123   | human | I work at EY     |

---

## SQLAlchemy Model

```python
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    func,
)

from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Conversation(Base):

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)

    thread_id = Column(String)

    role = Column(String)

    message = Column(Text)

    created_at = Column(
        DateTime,
        server_default=func.now()
    )
```

---

## Save Messages

```python
from sqlalchemy.orm import Session

def save_message(
    db: Session,
    thread_id: str,
    role: str,
    message: str
):

    chat = Conversation(

        thread_id=thread_id,

        role=role,

        message=message
    )

    db.add(chat)

    db.commit()
```

Usage:

```python
save_message(
    db,
    "user123",
    "human",
    "My name is Alice"
)
```

---

## Load Memory

```python
def load_history(
    db: Session,
    thread_id: str
):

    return (

        db.query(Conversation)

        .filter(
            Conversation.thread_id == thread_id
        )

        .order_by(Conversation.id)

        .all()

    )
```

Usage:

```python
history = load_history(db, "user123")
```

Result:

```text
Human: My name is Alice

AI: Hello Alice

Human: I work at EY
```

---

# Convert Database Rows into LangChain Messages

```python
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

messages = []

for row in history:

    if row.role == "human":

        messages.append(
            HumanMessage(
                content=row.message
            )
        )

    else:

        messages.append(
            AIMessage(
                content=row.message
            )
        )
```

Now invoke the model:

```python
response = llm.invoke(messages)
```

The LLM behaves as if the conversation never ended.

---

# Option 2: Redis Session Memory

Redis is excellent for active conversations.

```python
import redis

client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)
```

Save:

```python
client.rpush(
    "chat:user123",
    "Human: My name is Alice"
)

client.rpush(
    "chat:user123",
    "AI: Hello Alice"
)
```

Load:

```python
history = client.lrange(
    "chat:user123",
    0,
    -1
)

print(history)
```

Output:

```text
[
 "Human: My name is Alice",
 "AI: Hello Alice"
]
```

Redis is very fast but is usually used as a cache rather than the system of record.

---

# Option 3: Persist Long-Term Memory in a Vector Database

Suppose the user says:

```text
I love Python.
```

Instead of storing every message, store important facts.

Generate an embedding:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embedding = model.encode(
    "User likes Python"
)
```

Store in a vector database:

```python
vector_db.add(

    id="memory1",

    embedding=embedding,

    metadata={

        "user": "user123",

        "text": "User likes Python"

    }

)
```

---

Later:

User asks:

```text
Recommend a backend framework.
```

Retrieve relevant memory:

```python
query = model.encode(
    "Recommend backend framework"
)

results = vector_db.search(
    embedding=query,
    top_k=3
)
```

Output:

```text
User likes Python
```

Now include it in the prompt:

```python
prompt = f"""
Relevant memory:

{results[0]["text"]}

Question:

Recommend a backend framework.
"""
```

---

# Option 4: LangGraph Checkpointing (Recommended)

LangGraph can automatically persist graph state between executions using a checkpointer.

State:

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: list[BaseMessage]
```

Graph:

```python
from langgraph.graph import StateGraph

builder = StateGraph(AgentState)

builder.add_node("chat", chatbot)

builder.set_entry_point("chat")
builder.set_finish_point("chat")
```

Create a checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
```

Compile:

```python
graph = builder.compile(
    checkpointer=checkpointer
)
```

Run:

```python
config = {
    "configurable": {
        "thread_id": "user123"
    }
}

graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="My name is Alice"
            )
        ]
    },
    config=config
)
```

Later:

```python
graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What's my name?"
            )
        ]
    },
    config=config
)
```

Because the same `thread_id` is used, LangGraph restores the previous state automatically.

> **Note:** `MemorySaver` is primarily for development. In production, use a durable checkpointer backed by PostgreSQL or another persistent database so state survives process restarts.

---

# Production Flow

```text
User

↓

Receive Request

↓

Load Conversation
(PostgreSQL)

↓

Retrieve Long-Term Memory
(Vector DB)

↓

Merge Context

↓

LLM

↓

Generate Response

↓

Save Conversation
(PostgreSQL)

↓

Extract Important Facts

↓

Store Embeddings
(Vector DB)
```

---

# FastAPI Example

```python
@app.post("/chat")
def chat(request: ChatRequest):

    history = load_history(
        db,
        request.user_id
    )

    messages = convert_to_messages(history)

    messages.append(
        HumanMessage(
            content=request.message
        )
    )

    response = llm.invoke(messages)

    save_message(
        db,
        request.user_id,
        "human",
        request.message
    )

    save_message(
        db,
        request.user_id,
        "ai",
        response.content
    )

    return {
        "response": response.content
    }
```

Every request:

1. Loads conversation history.
2. Calls the LLM with the accumulated context.
3. Saves both the user's message and the assistant's response.

---

# Enterprise Architecture

```text
                    User
                      │
                      ▼
                 FastAPI API
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
 Load Conversation         Retrieve User Memory
  (PostgreSQL)               (Vector DB)
          │                       │
          └───────────┬───────────┘
                      ▼
               Context Builder
                      ▼
                     LLM
                      ▼
             Assistant Response
                      ▼
      ┌───────────────┴───────────────┐
      ▼                               ▼
 Save Conversation             Extract Facts
 (PostgreSQL)                  Generate Embeddings
                                        ▼
                                   Vector DB
```

---

# Best Practices

* **Separate short-term and long-term memory**: Store the full conversation in PostgreSQL (or another durable store), but store only important user facts in the vector database.
* **Use a stable thread/session ID** so every conversation can be resumed.
* **Summarize long conversations** to keep prompts within the context window.
* **Encrypt sensitive conversation data** and enforce access controls.
* **Expire temporary session caches** in Redis while keeping durable records in PostgreSQL.
* **Avoid embedding every message**; extract and persist only meaningful long-term facts.

---

# Senior AI Engineer Interview Answer

> **In production, I persist memory using multiple storage layers. Recent conversation history is stored in PostgreSQL and keyed by a thread or session ID, allowing the application to reconstruct the chat after restarts. Frequently accessed session data is cached in Redis for low latency. Long-term user preferences and important facts are embedded and stored in a vector database so they can be retrieved semantically when relevant. When a request arrives, the application loads the conversation history, retrieves relevant long-term memories, builds the prompt, calls the LLM, and then persists both the new messages and any newly extracted long-term facts. This layered approach provides durability, scalability, and efficient context management.
