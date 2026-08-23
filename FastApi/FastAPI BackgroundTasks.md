# FastAPI `BackgroundTasks` and Long-Running Jobs

There are two different problems:

1. **Small post-response work** → FastAPI `BackgroundTasks`
2. **Real long-running/distributed jobs** → Celery (or similar job queue)

---

# 1. What is `BackgroundTasks`?

FastAPI's `BackgroundTasks` lets you return an HTTP response first and execute some work **after the response is sent**.

Example:

```text
Client
  │
  │ POST /users
  ▼
FastAPI
  │
  ├── Create user
  │
  ├── Return HTTP 201 immediately
  │
  └── Background task
          │
          └── Send welcome email
```

## Basic example

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()


def send_email(email: str):
    print(f"Sending email to {email}")


@app.post("/users")
async def create_user(
    email: str,
    background_tasks: BackgroundTasks,
):
    # Main request work
    user = {
        "email": email
    }

    # Schedule background work
    background_tasks.add_task(
        send_email,
        email,
    )

    # Response is returned first
    return {
        "message": "User created",
        "email": email,
    }
```

Client receives:

```json
{
  "message": "User created",
  "email": "user@example.com"
}
```

Then:

```text
send_email()
```

runs in the background.

---

# 2. Background task with an async function

You can also use an async function:

```python
import asyncio

from fastapi import BackgroundTasks, FastAPI

app = FastAPI()


async def send_email(email: str):
    await asyncio.sleep(2)

    print(f"Email sent to {email}")


@app.post("/users")
async def create_user(
    email: str,
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(
        send_email,
        email,
    )

    return {
        "status": "accepted"
    }
```

---

# 3. Real-world use case

Suppose a user uploads a document.

The API should not make the user wait while you:

```text
Extract PDF
   ↓
Chunk document
   ↓
Generate embeddings
   ↓
Store in Qdrant
   ↓
Update PostgreSQL
```

For very small jobs, you could use:

```python
from fastapi import BackgroundTasks


@app.post("/documents")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
):
    document_id = "123"

    background_tasks.add_task(
        process_document,
        document_id,
    )

    return {
        "document_id": document_id,
        "status": "processing",
    }
```

The response is immediate:

```json
{
  "document_id": "123",
  "status": "processing"
}
```

---

# 4. But be careful: `BackgroundTasks` is not a job queue

This is extremely important.

`BackgroundTasks`:

```text
FastAPI Process
      │
      ├── API request
      │
      └── Background task
```

The task is still associated with the application process.

If:

```text
FastAPI crashes
```

then:

```text
Background task may be lost
```

If the application has multiple workers:

```text
                Load Balancer
                     │
           ┌─────────┴─────────┐
           │                   │
        FastAPI 1           FastAPI 2
           │                   │
         task A              task B
```

there is no durable central job queue.

So `BackgroundTasks` should **not be your primary solution for critical, long-running jobs**.

---

# 5. When should you use `BackgroundTasks`?

Good for:

```text
Send email
Send webhook
Write audit log
Update cache
Small notification
Lightweight analytics event
```

Example:

```python
@app.post("/orders")
async def create_order(
    order: OrderCreate,
    background_tasks: BackgroundTasks,
):
    new_order = create_order_in_db(order)

    background_tasks.add_task(
        send_order_notification,
        new_order.id,
    )

    return new_order
```

---

# 6. When should you NOT use `BackgroundTasks`?

Avoid it for:

```text
Large PDF processing
Video processing
ML model training
Large embedding jobs
Mass email processing
Hours-long jobs
CPU-intensive tasks
Critical jobs that must not be lost
```

For those, use a distributed job system.

---

# 7. How do you handle long-running jobs?

A production architecture is:

```text
                Client
                   │
                   ▼
             POST /jobs
                   │
                   ▼
                FastAPI
                   │
             Create Job Record
                   │
                   ▼
              PostgreSQL
                   │
                   ▼
              Job Queue
                   │
                   ▼
              Celery Worker
                   │
          ┌────────┼─────────┐
          ▼        ▼         ▼
       Processing Processing Processing
                   │
                   ▼
              Update DB
                   │
                   ▼
              Client polls
             GET /jobs/{id}
```

The API should return quickly:

```http
202 Accepted
```

Example:

```json
{
  "job_id": "abc-123",
  "status": "PENDING"
}
```

The client then calls:

```http
GET /jobs/abc-123
```

Response:

```json
{
  "job_id": "abc-123",
  "status": "PROCESSING",
  "progress": 60
}
```

Eventually:

```json
{
  "job_id": "abc-123",
  "status": "COMPLETED",
  "result": {
    "document_id": "doc-123"
  }
}
```

---

# 8. `BackgroundTasks` implementation with job status

For small jobs, you can still track status.

## Job model

```python
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int
```

For demonstration, use an in-memory dictionary:

```python
jobs = {}
```

> In production, use PostgreSQL or another durable database.

---

## Background worker function

```python
import asyncio


async def process_job(job_id: str):

    jobs[job_id]["status"] = "PROCESSING"
    jobs[job_id]["progress"] = 0

    try:

        for progress in range(0, 101, 20):

            await asyncio.sleep(1)

            jobs[job_id]["progress"] = progress

        jobs[job_id]["status"] = "COMPLETED"

    except Exception as exc:

        jobs[job_id]["status"] = "FAILED"

        jobs[job_id]["error"] = str(exc)
```

---

## Create job endpoint

```python
import uuid

from fastapi import BackgroundTasks, FastAPI

app = FastAPI()


@app.post(
    "/jobs",
    status_code=202,
)
async def create_job(
    background_tasks: BackgroundTasks,
):

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "PENDING",
        "progress": 0,
    }

    background_tasks.add_task(
        process_job,
        job_id,
    )

    return {
        "job_id": job_id,
        "status": "PENDING",
    }
```

---

## Get job status

```python
from fastapi import HTTPException


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
):

    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return {
        "job_id": job_id,
        **job,
    }
```

The flow is:

```text
POST /jobs
     │
     ▼
Create job
     │
     ▼
Return 202 Accepted
     │
     ▼
BackgroundTasks processes job
     │
     ▼
GET /jobs/{id}
     │
     ▼
Return progress
```

---

# 9. Production problem with the previous example

This:

```python
jobs = {}
```

is not production-ready.

Imagine:

```text
Worker 1
jobs = {abc: PROCESSING}
```

but the next request goes to:

```text
Worker 2
jobs = {}
```

The job appears missing.

Instead:

```text
FastAPI
   │
   ▼
PostgreSQL
   │
   ├── jobs
   │
   └── job status
```

---

# 10. Production database model

Using SQLAlchemy:

```python
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID


class Job(Base):

    __tablename__ = "jobs"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    status = Column(
        String,
        nullable=False,
        default="PENDING",
    )

    progress = Column(
        Integer,
        nullable=False,
        default=0,
    )

    result = Column(
        String,
        nullable=True,
    )

    error = Column(
        String,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )
```

The job state becomes durable.

---

# 11. Celery architecture

For actual long-running work:

```text
                     ┌──────────────┐
                     │   FastAPI    │
                     └──────┬───────┘
                            │
                            │ enqueue task
                            ▼
                     ┌──────────────┐
                     │ Redis/RabbitMQ│
                     │  Message Queue │
                     └──────┬───────┘
                            │
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
              Worker 1   Worker 2   Worker 3
                 │          │          │
                 └──────────┼──────────┘
                            │
                            ▼
                       PostgreSQL
```

FastAPI:

```text
API layer
```

Celery workers:

```text
Background processing layer
```

---

# 12. Install Celery

```bash
pip install celery redis
```

Project structure:

```text
app/
├── main.py
├── celery_app.py
├── tasks/
│   └── document_tasks.py
└── api/
    └── jobs.py
```

---

# 13. Configure Celery

`celery_app.py`:

```python
from celery import Celery


celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)
```

Here:

```text
Redis database 0
↓
Message broker

Redis database 1
↓
Result backend
```

---

# 14. Create a Celery task

```python
from app.celery_app import celery_app


@celery_app.task
def process_document(
    document_id: str,
):

    print(
        f"Processing {document_id}"
    )

    # Extract text
    # Chunk document
    # Generate embeddings
    # Store in Qdrant

    return {
        "document_id": document_id,
        "status": "COMPLETED",
    }
```

---

# 15. Send the task from FastAPI

```python
from fastapi import APIRouter

from app.tasks.document_tasks import (
    process_document,
)

router = APIRouter()


@router.post(
    "/documents/{document_id}/process",
    status_code=202,
)
async def process_document_endpoint(
    document_id: str,
):

    task = process_document.delay(
        document_id
    )

    return {
        "task_id": task.id,
        "status": "PENDING",
    }
```

The important part:

```python
task = process_document.delay(
    document_id
)
```

This does not execute the task inside FastAPI.

Instead:

```text
FastAPI
   │
   │ task message
   ▼
Redis Queue
   │
   ▼
Celery Worker
```

---

# 16. Start the Celery worker

For example:

```bash
celery -A app.celery_app.celery_app worker --loglevel=info
```

Now:

```text
FastAPI process
        │
        │ independent
        ▼
    Redis Queue
        │
        ▼
Celery Worker
```

The worker can scale independently.

---

# 17. Check Celery job status

```python
from celery.result import AsyncResult

from app.celery_app import celery_app


@app.get("/jobs/{task_id}")
async def get_job_status(
    task_id: str,
):

    task = AsyncResult(
        task_id,
        app=celery_app,
    )

    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result
        if task.ready()
        else None,
    }
```

Possible states:

```text
PENDING
STARTED
SUCCESS
FAILURE
RETRY
```

---

# 18. Retry long-running jobs

One major advantage of Celery is retries.

```python
from celery import shared_task


@shared_task(
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={
        "max_retries": 3
    },
)
def process_document(
    self,
    document_id: str,
):

    # Process document
    return {
        "document_id": document_id
    }
```

The flow:

```text
Attempt 1
    │
    ├── success → DONE
    │
    └── failure
          │
          ▼
        Retry
          │
          ▼
        Backoff
          │
          ▼
        Attempt 2
```

---

# 19. Progress tracking

For long jobs:

```python
@celery_app.task(bind=True)
def process_document(
    self,
    document_id: str,
):

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 10
        },
    )

    # Extract PDF

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 30
        },
    )

    # Chunk

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 60
        },
    )

    # Embeddings

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 100
        },
    )

    return {
        "document_id": document_id,
        "status": "COMPLETED",
    }
```

The API can expose:

```json
{
  "task_id": "123",
  "status": "PROGRESS",
  "progress": 60
}
```

---

# 20. Better production design: Database is the source of truth

For enterprise systems, I prefer:

```text
                    FastAPI
                       │
                       ▼
                  PostgreSQL
                 Job = PENDING
                       │
                       ▼
                   Celery Queue
                       │
                       ▼
                   Worker
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
      SUCCESS                    FAILURE
          │                         │
          ▼                         ▼
   Update PostgreSQL         Update PostgreSQL
```

Why?

Because:

```text
Redis/Celery
```

is primarily your task infrastructure, while:

```text
PostgreSQL
```

can hold the durable business record.

For example:

```text
jobs

id
tenant_id
user_id
job_type
status
progress
result
error
created_at
started_at
completed_at
```

---

# 21. BackgroundTasks vs Celery

| Feature                  | BackgroundTasks | Celery            |
| ------------------------ | --------------- | ----------------- |
| Runs outside API process | ❌               | ✅                 |
| Distributed workers      | ❌               | ✅                 |
| Durable queue            | ❌               | ✅                 |
| Retry support            | Manual          | Built-in          |
| Worker scaling           | No              | Yes               |
| Survives API restart     | No guarantee    | Better durability |
| Long-running jobs        | Not recommended | Yes               |
| CPU-heavy jobs           | Not recommended | Yes               |
| Simple setup             | Excellent       | More complex      |
| Small tasks              | Excellent       | Overkill          |
| Critical jobs            | Not ideal       | Better            |
| Scheduling               | Limited         | Celery Beat       |

---

# 22. The biggest difference

### `BackgroundTasks`

```text
FastAPI
   │
   ├── HTTP Request
   │
   └── Background Task
```

Same application/process environment.

### Celery

```text
FastAPI
   │
   ▼
Message Broker
   │
   ▼
Separate Worker Processes
```

Completely independent workers.

This is the key architectural difference.

---

# 23. When would I use each?

## Use `BackgroundTasks`

For:

```text
Send an email
Write audit log
Send webhook
Small cache update
Non-critical notification
```

Example:

```python
background_tasks.add_task(
    send_email,
    user.email,
)
```

---

## Use Celery

For:

```text
Document ingestion
Large PDF processing
Embedding thousands of documents
ML model training
Video processing
Large batch processing
Scheduled jobs
Retryable jobs
Critical jobs
```

For your RAG application, document ingestion is a strong example:

```text
Upload document
       │
       ▼
FastAPI stores metadata
       │
       ▼
Celery task
       │
       ├── Parse
       ├── OCR
       ├── Chunk
       ├── Embed
       └── Store in Qdrant
```

The user immediately receives:

```json
{
  "document_id": "abc",
  "status": "PROCESSING"
}
```

---

# 24. Senior-level interview answer

> **"FastAPI's `BackgroundTasks` is suitable for lightweight, non-critical work that should happen after the response, such as sending an email or writing an audit event. However, it is not a durable distributed task queue. The work is tied to the application runtime, so process crashes, deployments, and worker restarts can interrupt tasks."**
>
> **"For long-running or critical jobs, I use a job queue such as Celery with Redis or RabbitMQ. The FastAPI API creates a durable job record, enqueues the task, and immediately returns `202 Accepted` with a job ID. Independent workers process the job, support retries and scaling, and update the job status. The client polls a status endpoint or receives updates through WebSockets/SSE."**

## The interview rule to remember

```text
Small + non-critical
        ↓
BackgroundTasks


Long-running + critical + retryable + scalable
        ↓
Celery / Distributed Job Queue
```

For a **production RAG or AI system**, I would generally use a durable worker queue for document ingestion, embedding, batch evaluation, and other expensive jobs, while reserving FastAPI `BackgroundTasks` for lightweight post-response operations.
