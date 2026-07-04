In a **production AI system**, **Redis**, **Qdrant**, and **PostgreSQL** solve completely different problems.

A senior AI engineer is expected to know **why all three are needed**, **who calls whom**, and **what data is stored where**.

---

# Production Architecture

```
                        User
                          │
                          ▼
                    FastAPI Gateway
                          │
                Authentication
                          │
                Request Validation
                          │
             ┌────────────┴─────────────┐
             │                          │
             ▼                          ▼
        Redis Cache              PostgreSQL
   (Fast Retrieval)          (System of Record)
             │
             │
             ▼
      Embedding Service
             │
             ▼
         Qdrant Vector DB
             │
             ▼
       Similar Documents
             │
             ▼
          LLM (GPT)
             │
             ▼
        Generated Answer
             │
             ▼
        Store Conversation
         PostgreSQL
```

Every component has a different responsibility.

---

# Why PostgreSQL?

Postgres stores structured data.

Never store vectors here.

Example tables

```
Users

id
name
email
password


Conversation

id
user_id
question
answer
created_at


Documents

id
filename
uploaded_by
upload_time


Jobs

id
status
started_at
finished_at
```

Think of PostgreSQL as

> Source of Truth

---

# Why Redis?

Redis stores

```
Cache

Session

OTP

Rate Limiter

Frequently asked questions

LLM Response Cache

Temporary Objects

Locks

Queues
```

Everything that needs

```
Ultra Fast

Temporary

Memory Based
```

---

Example

Without Redis

```
User asks

"What is Kubernetes?"

↓

Embedding

↓

Qdrant Search

↓

LLM

↓

Generate Answer

5 seconds
```

Second user asks same question.

Without Redis

Everything repeats.

With Redis

```
Redis

↓

Answer Found

↓

Return in 2ms
```

---

# Why Qdrant?

Qdrant stores

```
Embeddings
```

Not text.

Example

```
"The cat sat on the mat"

↓

Embedding Model

↓

[0.24
0.82
-0.55
...
768 numbers]

↓

Qdrant
```

---

Production Folder Structure

```
production_ai/

app/
    api/
    db/
    cache/
    vector/
    llm/
    models/
    services/
    config.py

docker-compose.yml

requirements.txt
```

---

# PostgreSQL Implementation

```
app/db/database.py
```

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL="postgresql://user:password@localhost:5432/ai"

engine=create_engine(DATABASE_URL)

SessionLocal=sessionmaker(bind=engine)
```

---

Model

```python
from sqlalchemy import Column,Integer,String

from sqlalchemy.orm import declarative_base

Base=declarative_base()


class Conversation(Base):

    __tablename__="conversation"

    id=Column(Integer,primary_key=True)

    user_question=Column(String)

    assistant_answer=Column(String)
```

Saving conversation

```python
from database import SessionLocal
from models import Conversation

db=SessionLocal()

conv=Conversation(

    user_question="What is AI?",

    assistant_answer="Artificial Intelligence..."
)

db.add(conv)

db.commit()
```

---

# Redis Implementation

Install

```bash
pip install redis
```

---

Connection

```python
import redis

redis_client=redis.Redis(

    host="localhost",

    port=6379,

    decode_responses=True
)
```

Saving Cache

```python
redis_client.set(

    "What is AI?",

    "Artificial Intelligence is..."
)
```

Reading Cache

```python
answer=redis_client.get("What is AI?")

print(answer)
```

---

Production Cache

```
Question

↓

Hash

↓

Redis

↓

Answer
```

Example

```python
import hashlib

question="What is AI?"

key=hashlib.md5(question.encode()).hexdigest()

redis_client.set(key,"cached answer")
```

---

Expiration

```python
redis_client.setex(

    key,

    3600,

    answer
)
```

Cache expires after

```
1 hour
```

---

# Qdrant Implementation

Install

```bash
pip install qdrant-client
```

---

Connection

```python
from qdrant_client import QdrantClient

client=QdrantClient(

    host="localhost",

    port=6333
)
```

---

Create Collection

```python
from qdrant_client.models import VectorParams

client.create_collection(

    collection_name="documents",

    vectors_config=VectorParams(

        size=768,

        distance="Cosine"
    )
)
```

---

Insert Vector

```python
from qdrant_client.models import PointStruct

vector=[0.1]*768

client.upsert(

collection_name="documents",

points=[

PointStruct(

id=1,

vector=vector,

payload={

"text":"Redis is in-memory database"

}

)

]

)
```

---

Searching

```python
results=client.search(

collection_name="documents",

query_vector=[0.2]*768,

limit=3
)

print(results)
```

---

# Complete Production Flow

Imagine user asks

```
Explain Kubernetes
```

---

Step 1

API receives request

```python
question="Explain Kubernetes"
```

---

Step 2

Check Redis

```python
cached=redis.get(question)

if cached:

    return cached
```

If found

Done.

---

Otherwise

Step 3

Generate embedding

```python
embedding=embedding_model.encode(question)
```

```
↓

768 numbers
```

---

Step 4

Search Qdrant

```python
docs=qdrant.search(

embedding
)
```

Returns

```
Kubernetes PDF

Architecture Doc

Interview Notes
```

---

Step 5

Prompt

```
Question

+

Retrieved Docs
```

↓

LLM

↓

Answer

---

Step 6

Store answer in Redis

```python
redis.setex(

question,

3600,

answer
)
```

Next request

Instant.

---

Step 7

Store permanently

```python
conversation=Conversation(

user_question=question,

assistant_answer=answer
)

db.add(conversation)

db.commit()
```

---

# Production Service

```python
class AIService:

    def answer(self, question):

        cached = redis.get(question)

        if cached:
            return cached

        embedding = embed(question)

        docs = qdrant.search(embedding)

        prompt = build_prompt(question, docs)

        answer = llm(prompt)

        redis.setex(question, 3600, answer)

        save_postgres(question, answer)

        return answer
```

---

# Docker Compose

```yaml
version: "3.9"

services:

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: ai
      POSTGRES_PASSWORD: password
      POSTGRES_DB: aidb
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:v1.9.0
    ports:
      - "6333:6333"
```

---

# Complete Request Lifecycle

```text
                   USER
                     │
                     ▼
              FastAPI Endpoint
                     │
                     ▼
           Check Redis Cache
                     │
           ┌─────────┴─────────┐
           │ Cache Hit         │ Cache Miss
           ▼                   ▼
     Return Answer      Create Embedding
                              │
                              ▼
                    Search Qdrant Vectors
                              │
                              ▼
                 Retrieve Relevant Documents
                              │
                              ▼
                    Build Prompt for LLM
                              │
                              ▼
                       Generate Response
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      Cache in Redis                  Persist in PostgreSQL
              │                               │
              └───────────────┬───────────────┘
                              ▼
                      Return to User
```

### Senior AI engineering best practices

In production, these components are typically abstracted behind repository or service classes rather than being accessed directly from route handlers. Use connection pooling for PostgreSQL, a shared Redis client with appropriate timeouts, and a singleton Qdrant client. Cache keys should include model/version information to avoid serving stale results after model updates. Document ingestion into Qdrant is usually asynchronous (via a queue such as Celery, Kafka, or SQS) so that uploads don't block user requests. Finally, use PostgreSQL as the authoritative source for business data, Redis only for ephemeral state and caching, and Qdrant exclusively for vector similarity search. This separation of concerns keeps the system scalable, maintainable, and resilient.
