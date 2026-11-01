# Kafka and Message Queues in Production AI Projects

This is an important topic for Senior AI Engineer and Staff AI Engineer interviews because production AI applications rarely execute everything inside a single API request.

Imagine you have built an Enterprise Multi-Agent Financial Advisor Copilot using:

- FastAPI
- LangGraph
- OpenAI
- PostgreSQL
- Qdrant
- Redis
- Celery
- AWS
- Docker and Kubernetes

Now imagine the following situations:

1. 10,000 users upload PDF documents simultaneously.
2. Each PDF requires parsing, chunking, embedding generation, and vector database storage.
3. Some LLM requests take 30–60 seconds.
4. An embedding provider becomes unavailable.
5. A worker crashes halfway through document ingestion.
6. The same document is submitted multiple times.
7. You need to process events from multiple AI services independently.

If you execute all these operations directly inside FastAPI request handlers, your application can become slow, unstable, and difficult to scale.

This is where Kafka and message queues become important.

Let's understand them from an architecture and production-code perspective.

# 1. What is a Message Queue?

A message queue is a communication mechanism that allows one application component to send a message to another component without requiring the receiver to process it immediately.

There are generally two components:

- Producer: The component that creates and sends a message.
- Consumer: The component that receives and processes the message.

For example:

A user uploads a PDF.

FastAPI receives the request and sends a message to a queue.

An independent worker receives that message and starts processing the PDF.

FastAPI

Producer — accepts PDF upload

Publish message

Message Broker / Queue

Stores pending messages

Consume message

Background Worker

Processes PDF independently

PostgreSQL / Qdrant

Stores processing status and embeddings

The important point is that FastAPI does not need to wait for the PDF processing to finish.

It can immediately return:

```
{
    "document_id": "doc_123",
    "status": "PROCESSING",
    "message": "Document uploaded successfully"
}
```

The worker processes the document in the background.

This is called asynchronous processing using a message-driven architecture.

# 2. Why do AI applications need Kafka or message queues?

Let's take a real production AI example.

Suppose 1,000 users upload documents at the same time.

Each document requires:

| Processing step      | Approximate time |
| -------------------- | ---------------- |
| PDF parsing          | 3 seconds        |
| Text cleaning        | 1 second         |
| Chunking             | 1 second         |
| Embedding generation | 8 seconds        |
| Qdrant insertion     | 2 seconds        |
| Total                | 15 seconds       |

These are illustrative timings.

If FastAPI executes all these steps synchronously, requests can remain open for a long time.

This creates several problems:

- Request timeouts.
- Increased memory usage.
- Worker exhaustion.
- Poor API responsiveness.
- Provider rate-limit errors.
- Difficulty recovering from failures.

Instead, we use a queue.

```
1,000 PDF uploads
       |
       v
    FastAPI
       |
       v
    Queue
       |
       +------------------+
       |                  |
       v                  v
   Worker 1           Worker 2
       |                  |
       v                  v
   Worker 3           Worker 4
       |                  |
       +--------+---------+
                |
                v
         Qdrant / PostgreSQL
```

Now four workers can process documents independently.

You can increase the number of workers when the workload increases.

This provides:

1. Decoupling.
2. Scalability.
3. Backpressure.
4. Failure recovery.
5. Independent processing.
6. Better resource utilization.

Important distinction: A queue does not automatically make processing faster. It separates request acceptance from execution and lets you control processing concurrency.

# 3. What is Kafka?

[Running Kafka on Production. Apache Kafka is an open-source… | by Pradyumna Khawas | WorkIndia.in | Medium](https://images.openai.com/static-rsc-4/IdgpctULbez35q_RTbMO09b-y6gGSwj0M361O82eRjmf4iacIuOlU2NJe7tJ5nG3ML_-Ramx8HPhw_f2BkJ4-7fTax9B_pG2F5mYYn7fI3Wkxu6lEMjmv6TQdRCI7p2vkYr-RROSNxhp8BCekHH4j-k8olCh620FHns5GrtDyDM?purpose=inline)

Apache Kafka

Distributed event streaming platform

[Official Kafka documentation](https://kafka.apache.org/?utm_source=chatgpt.com)

Kafka is a distributed event streaming platform used to publish, store, and consume large volumes of events.

It is particularly useful when multiple independent services need to communicate through events.

For example, in your financial advisor copilot:

- Document ingestion service publishes document events.
- Embedding service consumes document events.
- Analytics service consumes ingestion events.
- Audit service consumes processing events.
- Notification service consumes completion events.

One event can be consumed by multiple independent services.

## 3.1 How Kafka works internally

There are five important concepts you need to understand.

1\. Producer

The producer sends messages to Kafka.

Example:

Your FastAPI document upload service publishes:

```
{
    "event_id": "evt_101",
    "document_id": "doc_123",
    "tenant_id": "tenant_001",
    "event_type": "DOCUMENT_UPLOADED"
}
```

2\. Topic

A topic is a named stream of events.

For example:

```
document-uploaded
document-parsed
embedding-requested
embedding-completed
rag-query-events
ai-audit-events
```

You can think of a topic as a logical channel through which related events flow.

3\. Partition

A topic can be divided into multiple partitions.

For example:

Topic: document-uploaded

Partition 0

doc-101

doc-104

doc-107

Partition 1

doc-102

doc-105

doc-108

Partition 2

doc-103

doc-106

doc-109

Illustrative partition distribution. Each partition is an ordered log of events.

Why do we need partitions?

Because they allow Kafka to distribute events across multiple brokers and allow multiple consumers in a consumer group to process different partitions in parallel.

For example:

- Worker 1 processes partition 0.
- Worker 2 processes partition 1.
- Worker 3 processes partition 2.

A useful rule: messages with the same partition key are routed to the same partition, preserving their order within that partition.

For document processing, you might use `document_id` as the key.

4\. Consumer

A consumer reads messages from a Kafka topic and performs the required operation.

For example:

```
Kafka topic: document-uploaded

        |
        +----> Embedding Consumer
        |
        +----> Audit Consumer
        |
        +----> Analytics Consumer
```

Each consumer performs its own independent processing.

5\. Consumer Group

A consumer group is a collection of consumers working together to process a topic.

For example:

```
Topic: document-uploaded

Consumer Group: embedding-workers
    Consumer 1
    Consumer 2
    Consumer 3

Consumer Group: audit-workers
    Consumer 1
    Consumer 2
```

The embedding group processes the events independently of the audit group.

Within one consumer group, a partition is assigned to only one consumer at a time.

If you have three partitions, adding ten consumers to the same group will not give you ten-way partition-level parallelism. At most three consumers can actively own those partitions at that time.

## 3.2 How Kafka stores messages

This is one of the biggest differences between Kafka and traditional task queues.

Kafka stores messages in an append-only log.

For example:

```
Topic: document-uploaded

Offset 0 -> doc-101
Offset 1 -> doc-102
Offset 2 -> doc-103
Offset 3 -> doc-104
Offset 4 -> doc-105
```

An offset is the position of a message within a partition.

Consumers track their progress using committed offsets.

Suppose the consumer has processed offset 3.

If it crashes, it can restart and continue from its last committed position, potentially reprocessing some messages.

This is useful for recovering from failures.

Unlike a typical destructive queue, reading a Kafka event does not normally delete it. Events remain available according to the configured retention policy.

You can also replay historical events by resetting consumer offsets, provided the events are still retained.

# 4. Kafka vs RabbitMQ vs Redis vs Celery

This is an extremely common interview question.

First, understand that these technologies are not exactly interchangeable.

[Machine Learning: Why It Matters to the Future of Business | Digital Marketing Institute](https://images.openai.com/static-rsc-4/fJny5U6Fa2UNAZIOzgthQ1omNYshGgrm5Hh0AYc9gm7RQjEzJELjSN4ks8Ii5bGwyJpeNp7pFXrgtwyUrcLQE8adMp-kyR56-yE-FQ9CwOvKhADoHBdZ1hg0ARvPxTOY-gUZBpvn-YPpsnd6rISYYIk9-AzK6EgwBe113QPgJUI?purpose=inline)

Kafka

Best for high-volume event streaming, multiple independent consumers, event replay, analytics, and distributed data pipelines.

[RabbitMQ Logo & Brand Assets (SVG, PNG and vector) - Brandfetch](https://images.openai.com/static-rsc-4/dH0OOZcUHBXmCg4AeXtBX6KtUfyIG6YOYw4mrEHHnlPfqtdIWa5pQXr-QG2hQHorWkVNTA5zOvmMkbgPp8zVkeqrTYPLv4x4nVU2WQE92jTHMOyjGwrVT4hHW0k0N5MO00pz1TflQokKGGt-oJowVnS2mlQtonwHqBv6dxIGD2g?purpose=inline)

RabbitMQ

Best for traditional task queues, routing, acknowledgements, and work distribution.

[Redis Logo PNG Vector (SVG) Free Download](https://images.openai.com/static-rsc-4/epIZvTJqrn11fZoyLAql3XQvzvr05WTs4k_hS-9tdnMQyi9Nak_EpGx1d16Qeud3z3mmGKrPzkfB8l4M10kTA8OEXwDm6CtTKKvBDOawlkZbKJwF8Jjm3IiHXcBUIohDPkufesDQyxMCNKXpFH79YSUkxtDVtdaR-HKvPqdpBXw?purpose=inline)

Redis

Best for caching, fast coordination, rate limiting, and lightweight queueing with appropriate Redis mechanisms.

[Celery Demystified- Part 1: Supercharging Python with asynchronous scalable tasks | by Priyal Maheshwari | Medium](https://images.openai.com/static-rsc-4/uEtj-kMJ7tlK06FBjNTIelFWf3m57rlIsG2yyfHwVpbjEMROskDQW-hwFa-jF8hEQz29oDpXDVEKzWV8jFzn4H8bPaUPxb4Ks90BsPeetCeAubZzzw58h40g1B4WLqUlLCL0jUDxBvt0bQOvqkHM2jN4p4IKHxurX7Bto70g_gc?purpose=inline)

Celery

Best for executing Python background jobs with retries, scheduling, and worker management. It uses a broker such as Redis or RabbitMQ.

### Comparison table

| Feature                        | Kafka             | RabbitMQ                    | Redis + Celery      |
| ------------------------------ | ----------------- | --------------------------- | ------------------- |
| Primary purpose                | Event streaming   | Message brokering           | Background tasks    |
| Message retention              | Configurable      | Queue-based                 | Depends on broker   |
| Event replay                   | Yes               | Not native in normal queues | Limited             |
| Multiple independent consumers | Excellent         | Routing/fanout              | Requires design     |
| Task retries                   | Application-level | Supported patterns          | Celery supports     |
| High-volume event streams      | Excellent         | Good                        | Depends on workload |
| Python AI jobs                 | Good              | Good                        | Excellent           |

### Which one should you choose?

For your AI projects, my recommendation would be:

- PDF ingestion and embedding jobs: Celery + RabbitMQ or Redis.
- High-volume document lifecycle events: Kafka.
- LLM request caching: Redis.
- Agent workflow execution: LangGraph with durable checkpointing, often orchestrated through a task queue.
- Audit and analytics event streams: Kafka.

You do not need Kafka just because you are building an AI application.

Choose based on workload, durability requirements, event replay, throughput, and operational complexity.

# 5. Where exactly is Kafka used in an AI project?

Let's take your Enterprise Multi-Agent Financial Advisor Copilot.

This is where Kafka becomes especially interesting.

Enterprise AI platform — event-driven architecture

FastAPI Gateway

Document uploads / user queries

Kafka Event Bus

Document events · Workflow events · Audit events

Document ingestion

PDF parser → chunker → embeddings → Qdrant

AI workflow

LangGraph → planner → research → analysis

Audit service

Record document and agent events

Analytics service

Track usage, latency, tokens, costs

PostgreSQL · Qdrant · Redis · Object Storage

Let's understand six practical use cases.

## Use case 1: Document ingestion pipeline

This is probably the most important Kafka use case in an enterprise RAG system.

Suppose 5,000 financial PDFs are uploaded.

Instead of processing all documents directly in FastAPI, you publish document events to Kafka.

The pipeline looks like this:

PDF uploaded to S3

Kafka: document-uploaded

PDF parsing workers

Kafka: document-parsed

Chunking and embedding workers

Kafka: embedding-completed

Qdrant indexing workers

Document ready for RAG

Why separate these stages?

Imagine the PDF parser is working properly, but the OpenAI embedding API is temporarily unavailable.

Without a queue, the entire processing pipeline could fail.

With Kafka:

1. The parsing service completes its job.
2. It publishes `DOCUMENT_PARSED`.
3. The embedding service is temporarily unavailable.
4. Events remain in Kafka.
5. The embedding consumer recovers.
6. Processing continues from the retained events.

This is an important example of fault isolation.

## Use case 2: Asynchronous embedding generation

Embedding generation can be expensive and rate-limited.

Suppose you have 100,000 document chunks.

You should not generate embeddings for all chunks inside an HTTP request.

Instead:

```
Document chunker
      |
      v
Kafka: embedding-requested
      |
      v
Embedding consumer group
      |
      v
Batching service
      |
      v
OpenAI Embedding API
      |
      v
Qdrant
```

The consumer can:

- Batch multiple chunks.
- Apply rate limiting.
- Retry transient API failures.
- Control concurrency.
- Track failed chunks.
- Persist processing status.

This is particularly useful when embedding APIs have request and token limits.

## Use case 3: Multi-agent workflow orchestration

Suppose you have a financial research application.

A user asks:

"Analyze the risk and expected return of this investment portfolio."

Your LangGraph workflow might contain:

```
Planner Agent
     |
     v
Research Agent
     |
     v
+----------------------+
|                      |
v                      v
Risk Agent         Market Agent
|                      |
+----------+-----------+
           |
           v
     Evaluator Agent
           |
           v
      Writer Agent
           |
           v
      Final Response
```

Kafka can be used to communicate events between independently deployed services.

For example:

```
{
    "event_type": "RESEARCH_COMPLETED",
    "workflow_id": "workflow_123",
    "task_id": "task_456",
    "result_reference": "s3://results/research-456.json"
}
```

The risk analysis service consumes this event and starts its work.

However, an important architectural distinction:

Kafka is not a replacement for LangGraph.

LangGraph manages workflow state, conditional routing, agent execution, and checkpoints.

Kafka handles asynchronous event communication between services.

You can use both together.

For example:

- Kafka distributes independent research tasks.
- LangGraph manages the workflow inside the AI orchestration service.
- PostgreSQL stores business state.
- A durable LangGraph checkpointer stores workflow checkpoints.

For a single-process multi-agent workflow, you might not need Kafka at all.

## Use case 4: LLM request processing at scale

Imagine your AI platform receives 20,000 requests per minute.

Not all requests should be executed with the same priority.

You can introduce asynchronous processing.

```
User Requests
      |
      v
FastAPI
      |
      v
Kafka
      |
      +---------------------+
      |                     |
      v                     v
Priority Consumer      Normal Consumer
      |                     |
      v                     v
High priority LLM      Standard LLM
      |                     |
      +----------+----------+
                 |
                 v
          Result Storage
                 |
                 v
         Notification/API
```

Examples:

- Premium users receive higher priority.
- Long-running portfolio analysis executes asynchronously.
- Batch report generation happens in background workers.
- Expensive LLM calls are controlled through concurrency limits.

For interactive chat, however, don't automatically put every request into Kafka. Synchronous HTTP or streaming responses are often better for low-latency interactions. Use queues when the operation can run asynchronously.

## Use case 5: AI monitoring, logging, and audit events

This is particularly important in financial applications.

Whenever an AI request executes, your application can publish events such as:

```
{
    "event_type": "LLM_REQUEST_COMPLETED",
    "request_id": "req_123",
    "tenant_id": "tenant_001",
    "model": "gpt-model",
    "input_tokens": 1200,
    "output_tokens": 450,
    "latency_ms": 1800,
    "estimated_cost": 0.025,
    "status": "SUCCESS"
}
```

Different services consume these events:

| Consumer   | Responsibility                   |
| ---------- | -------------------------------- |
| Analytics  | Calculate token consumption      |
| Billing    | Calculate customer usage         |
| Monitoring | Detect latency increases         |
| Audit      | Maintain execution records       |
| Alerting   | Notify when error rates increase |

You can build dashboards for:

- Requests per minute.
- LLM token consumption.
- Model costs.
- Retrieval failures.
- Agent execution failures.
- Queue lag.
- Document processing throughput.

For sensitive financial systems, publish only the metadata required by each consumer. Avoid placing raw documents, customer data, or full prompts into broadly accessible event streams.

## Use case 6: Handling model failures and retries

Imagine your embedding provider experiences an outage.

Your embedding worker tries to generate embeddings, but receives HTTP 503.

What should happen?

Without a proper asynchronous architecture, requests may fail and work may be lost.

With Kafka, you can implement:

```
Kafka: embedding-requested
           |
           v
     Embedding Worker
           |
           v
      OpenAI API
           |
       +---+---+
       |       |
       v       v
    Success  Failure
       |       |
       v       v
    Qdrant   Retry Policy
                 |
           +-----+------+
           |            |
           v            v
        Retry       DLQ Topic
```

DLQ means Dead Letter Queue.

A DLQ stores messages that have exhausted their permitted retry attempts or require manual investigation.

For example:

```
embedding-requested
embedding-retry
embedding-dead-letter
```

Important: Kafka itself does not automatically implement a task retry policy in the way Celery does. You implement retry topics, retry scheduling, and failure handling in your consumer/application architecture.

# 6. Let's build a practical Kafka implementation using Python

Now let's understand how you would actually implement this.

We'll build a small document ingestion event pipeline using:

- FastAPI: Producer API.
- Kafka: Message broker.
- Python `aiokafka`: Kafka client.
- Worker: Consumer.
- PostgreSQL: Document status.
- Qdrant: Vector storage (the actual embedding/indexing stage).

First, install dependencies:

```
pip install fastapi uvicorn aiokafka pydantic
```

## Step 1 — Run Kafka locally

For a local development environment, you can use Docker Compose.

```
services:

  kafka:
    image: apache/kafka:latest

    ports:
      - "9092:9092"

    environment:
      KAFKA_NODE_ID: 1

      KAFKA_PROCESS_ROLES: broker,controller

      KAFKA_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093

      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092

      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER

      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093

      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1

      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1

      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
```

Run:

```
docker compose up -d
```

This is a single-broker development setup, not a production deployment. For production, use multiple brokers, replication, appropriate security, monitoring, and durable storage.

## Step 2 — Define the event schema

Create:

`app/events/schemas.py`

```
from datetime import datetime, timezonefrom uuid import UUID, uuid4from pydantic import BaseModel, Fieldclass DocumentUploadedEvent(BaseModel):    event_id: UUID = Field(default_factory=uuid4)    event_type: str = "DOCUMENT_UPLOADED"    document_id: str    tenant_id: str    storage_key: str    timestamp: datetime = Field(        default_factory=lambda: datetime.now(timezone.utc)    )    schema_version: int = 1
```

Why do we need `event_id`?

Because the same message might be delivered more than once.

We need a unique identifier to implement idempotency.

Why do we need `schema_version`?

Because event structures can change over time.

Consumers need a controlled way to handle old and new event formats.

Notice that we send an object-storage key, not the entire PDF file through Kafka.

This keeps events small and avoids unnecessarily exposing document contents.

## Step 3 — Implement the Kafka producer

Create:

`app/messaging/producer.py`

```
import jsonfrom aiokafka import AIOKafkaProducerfrom app.events.schemas import DocumentUploadedEventclass KafkaProducer:    def __init__(self, bootstrap_servers: str):        self.bootstrap_servers = bootstrap_servers        self.producer = None    async def start(self):        self.producer = AIOKafkaProducer(            bootstrap_servers=self.bootstrap_servers,            acks="all",            enable_idempotence=True,        )        await self.producer.start()    async def publish_document_event(        self,        event: DocumentUploadedEvent,    ):        if self.producer is None:            raise RuntimeError("Kafka producer not started")        payload = event.model_dump_json().encode("utf-8")        await self.producer.send_and_wait(            topic="document-uploaded",            key=event.document_id.encode("utf-8"),
```

Let's understand the important parameters.

| Parameter                 | Purpose                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| `bootstrap_servers`       | Kafka broker address                                                                              |
| `acks="all"`              | Wait for acknowledgement from all required in-sync replicas                                       |
| `enable_idempotence=True` | Protect against duplicate writes caused by producer retries within the supported producer session |
| `key`                     | Determines partition routing                                                                      |
| `send_and_wait()`         | Waits for the broker acknowledgement                                                              |

One important distinction: producer idempotence does not guarantee that your entire business operation executes only once.

Consumer-side idempotency is still required.

## Step 4 — Connect FastAPI with Kafka

Create:

`app/main.py`

```
from contextlib import asynccontextmanagerfrom fastapi import FastAPI, HTTPExceptionfrom pydantic import BaseModelfrom app.messaging.producer import KafkaProducerfrom app.events.schemas import DocumentUploadedEventkafka = KafkaProducer(    bootstrap_servers="localhost:9092")@asynccontextmanagerasync def lifespan(app: FastAPI):    await kafka.start()    yield    await kafka.stop()app = FastAPI(lifespan=lifespan)class DocumentUploadRequest(BaseModel):    document_id: str    tenant_id: str    storage_key: str
```

Run:

```
uvicorn app.main:app --reload
```

Example request:

```
POST /documents/upload
```

```
{
    "document_id": "doc_123",
    "tenant_id": "tenant_001",
    "storage_key": "documents/doc_123.pdf"
}
```

FastAPI publishes the event to Kafka and returns a queued status.

The PDF processing itself happens in another service.

Production note: This simple example assumes the document has already been stored in object storage and the database record has been created. For reliable database-to-Kafka publishing, use the transactional outbox pattern, which I'll explain below.

## Step 5 — Implement the Kafka consumer

Now create:

`app/workers/document_worker.py`

```
import asyncioimport jsonimport loggingfrom aiokafka import AIOKafkaConsumerlogger = logging.getLogger(__name__)async def process_document(event: dict):    document_id = event["document_id"]    tenant_id = event["tenant_id"]    storage_key = event["storage_key"]    logger.info(        "Processing document",        extra={            "document_id": document_id,            "tenant_id": tenant_id,        },    )    # 1. Download PDF from object storage.    # 2. Parse PDF.    # 3. Extract text.    # 4. Generate chunks.    # 5. Generate embeddings.    # 6. Store vectors in Qdrant.    # 7. Update PostgreSQL status.    await asyncio.sleep(1)
```

Run the consumer in another terminal:

```
python -m app.workers.document_worker
```

Now the complete communication looks like:

```
Client
  |
  v
FastAPI
  |
  v
Kafka Producer
  |
  v
document-uploaded topic
  |
  v
Kafka Consumer
  |
  v
Document Worker
  |
  v
PDF Parsing
  |
  v
Chunking
  |
  v
Embedding Generation
  |
  v
Qdrant
```

This is a working architectural skeleton, but the consumer above is intentionally simplified. In particular, a production consumer needs explicit handling for poison messages, retries, duplicate delivery, and offset management. Simply leaving a failed offset uncommitted can cause the same failing message to block later messages in its partition.

# 7. What happens if a Kafka consumer crashes halfway through processing?

This is one of the most important interview questions.

Imagine:

```
Kafka receives document event
          |
          v
Consumer starts processing
          |
          v
PDF parsing completed
          |
          v
Chunking completed
          |
          v
Embedding generation completed
          |
          v
Consumer crashes before Qdrant insertion
```

What happens?

When the consumer restarts, Kafka can redeliver the event if its offset has not been committed.

But there is a problem.

What if the consumer had already inserted embeddings into Qdrant and crashed before committing the Kafka offset?

The event will be processed again.

Now you may generate duplicate embeddings or duplicate records.

This is why production AI systems need idempotent processing.

## 7.1 Implement idempotency

An operation is idempotent if executing it multiple times produces the same final result as executing it once.

For example:

```
First execution:
Document doc_123 indexed.

Second execution:
Document doc_123 already indexed.
Skip or safely update.
```

You can maintain a processing table in PostgreSQL.

```
CREATE TABLE processed_events (
    event_id UUID PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    processed_at TIMESTAMPTZ
);
```

Your worker checks whether the event has already been processed.

A simplified implementation:

```
async def process_event_once(    event_id: str,    document_id: str,    repository,    ingestion_service,):    if await repository.is_processed(event_id):        return {            "status": "ALREADY_PROCESSED"        }    await ingestion_service.ingest(        document_id=document_id    )    await repository.mark_processed(        event_id=event_id    )    return {        "status": "SUCCESS"    }
```

However, this code alone has a race condition: two workers or a retry could both pass the initial check.

A production implementation should use a unique database constraint and an atomic claim/lease mechanism, and should make downstream writes idempotent too.

For example, Qdrant points can use deterministic IDs derived from:

```
tenant_id + document_id + document_version + chunk_index
```

This makes retries safer because the same chunk updates the same logical vector point rather than creating a new random point every time.

Remember: Kafka generally provides at-least-once processing when you commit offsets after processing. Exactly-once business effects across Kafka, PostgreSQL, an embedding provider, and Qdrant require additional design. Kafka's transactions alone cannot make all those external systems transactional.

# 8. What is a Dead Letter Queue (DLQ)?

Suppose a particular PDF is corrupted.

Your worker tries to process it.

It fails repeatedly.

If you keep retrying indefinitely, the same bad event can consume worker resources and block progress.

Instead, use a retry strategy.

```
document-uploaded
        |
        v
   Worker
        |
        v
   Processing
        |
    +---+---+
    |       |
    v       v
 Success  Failure
    |       |
    v       v
 Complete  Retry
             |
         Max attempts?
             |
         +---+---+
         |       |
         v       v
        Yes      No
         |       |
         v       v
        DLQ    Retry later
```

Example retry policy:

| Attempt | Delay       |
| ------- | ----------- |
| 1       | Immediate   |
| 2       | 5 seconds   |
| 3       | 30 seconds  |
| 4       | 2 minutes   |
| 5       | Send to DLQ |

These are illustrative delays. Production retry schedules should use exponential backoff, jitter, and error classification.

Not every failure should be retried.

| Error                  | Action                |
| ---------------------- | --------------------- |
| HTTP 429               | Retry with backoff    |
| HTTP 503               | Retry                 |
| Network timeout        | Retry                 |
| Invalid PDF            | DLQ                   |
| Invalid tenant ID      | Reject / DLQ          |
| Authentication failure | Alert and investigate |
| Invalid event schema   | Quarantine / DLQ      |

One more important detail: in Kafka, a DLQ is usually another topic. You should preserve the original event, error reason, attempt count, and source topic/partition/offset.

# 9. What is the Transactional Outbox Pattern?

This is a very important Staff Engineer concept.

Imagine FastAPI needs to do two things:

1. Save document information in PostgreSQL.
2. Publish an event to Kafka.

Consider this situation:

```
await postgres.save_document(document)await kafka.publish(event)
```

What if PostgreSQL succeeds but Kafka is unavailable?

The document is saved, but no event is published.

Your document will remain stuck in the uploaded state.

This is called the dual-write problem.

### Solution: Transactional Outbox

Instead of directly publishing to Kafka inside the request, save the document and an outbox event in the same PostgreSQL transaction.

```
FastAPI
   |
   v
PostgreSQL transaction
   |
   +----------------------+
   |                      |
   v                      v
Save document        Save outbox event
   |                      |
   +----------+-----------+
              |
        COMMIT transaction
              |
              v
       Outbox Publisher
              |
              v
            Kafka
              |
              v
       Document Consumer
```

Example SQL:

```
BEGIN;

INSERT INTO documents (
    document_id,
    tenant_id,
    status
)
VALUES (
    'doc_123',
    'tenant_001',
    'UPLOADED'
);

INSERT INTO outbox_events (
    event_id,
    event_type,
    aggregate_id,
    payload,
    published
)
VALUES (
    gen_random_uuid(),
    'DOCUMENT_UPLOADED',
    'doc_123',
    '{"document_id":"doc_123"}',
    FALSE
);

COMMIT;
```

A separate publisher reads unpublished events and sends them to Kafka.

After Kafka acknowledges the message, the publisher marks the event as published.

If the publisher crashes after sending but before marking the event published, it might send the event again.

Therefore, the consumer must still be idempotent.

This pattern is extremely useful for document ingestion, payment-related workflows, compliance systems, and AI workflow orchestration.

# 10. How do you integrate Kafka with LangGraph?

Let's make this practical.

Suppose your Financial Advisor Copilot has a workflow that takes 45 seconds to complete.

You don't want the FastAPI request to remain open for the entire process.

You can use this architecture:

FastAPI

Receives portfolio analysis request

Kafka: analysis-requested

Workflow Worker

Consumes event and starts LangGraph

LangGraph

Planner → Research → Risk → Evaluator → Writer

PostgreSQL

Stores workflow status and final result

Kafka: analysis-completed

Notification / Result API

The API immediately returns:

```
{
    "workflow_id": "wf_123",
    "status": "QUEUED"
}
```

The client can poll:

```
GET /workflows/wf_123
```

Or receive notifications using WebSockets or server-sent events.

### Important: What if LangGraph crashes?

Kafka can redeliver the triggering event, but that does not automatically resume the graph from its last completed node.

For that, you need LangGraph durable checkpointing.

For example:

```
config = {    "configurable": {        "thread_id": "wf_123"    }}
```

Use a persistent LangGraph checkpointer, such as PostgreSQL-backed checkpoint storage.

Then:

- Kafka manages workflow-start and completion events.
- LangGraph checkpoints preserve workflow state.
- PostgreSQL stores application-level workflow status.
- Idempotent nodes and tools prevent duplicate external side effects.

When a worker restarts, it can invoke the graph using the same thread ID and recover its saved state, subject to the graph's interrupt and checkpoint semantics.

# 11. How does Kafka help when an AI application handles millions of requests?

Imagine your application receives 1 million events per day.

You cannot depend on a single worker to process everything.

You need horizontal scaling.

Kafka provides partition-based parallelism.

For example:

Horizontal scaling architecture

Multiple FastAPI instances

Kafka cluster

Partition 0

Partition 1

Partition 2

Worker 1

Consumer

Worker 2

Consumer

Worker 3

Consumer

Distributed AI processing

When traffic increases:

1. Increase Kafka partitions if the topic needs more parallelism.
2. Increase the number of consumers.
3. Increase worker replicas in Kubernetes.
4. Monitor consumer lag.
5. Scale based on queue backlog and processing throughput.

For example, Kubernetes can scale the embedding worker deployment based on Kafka consumer lag using an autoscaling mechanism such as KEDA.

However, remember:

- More consumers cannot exceed available partition parallelism within one consumer group.
- More partitions do not automatically make an LLM provider accept more requests.
- Embedding and LLM concurrency must respect provider rate limits.
- Qdrant and PostgreSQL must also have sufficient capacity.

Kafka solves event distribution and buffering. It does not eliminate downstream bottlenecks.

# 12. Important production concepts you must know

These are frequently asked in senior-level interviews.

| Concept         | Meaning                                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Backpressure    | Slow down or limit producers/consumers when downstream systems cannot keep up                                           |
| Consumer lag    | Difference between the latest available offset and consumer progress                                                    |
| Partition key   | Determines which partition receives an event                                                                            |
| Replication     | Keeps copies of partition data across brokers                                                                           |
| Offset commit   | Records consumer progress                                                                                               |
| At-least-once   | A message can be processed more than once                                                                               |
| At-most-once    | A message can be lost after being consumed but before successful processing                                             |
| Exactly-once    | Kafka supports transactional processing within Kafka boundaries; external business effects need additional coordination |
| DLQ             | Stores events that cannot be processed successfully                                                                     |
| Schema Registry | Helps manage and validate event schema evolution                                                                        |

### What is consumer lag?

Suppose:

```
Latest Kafka offset: 10,000

Consumer committed offset: 7,000
```

Your consumer has approximately 3,000 messages of backlog, subject to the precise offset and lag metric convention.

This means messages are arriving faster than they are being processed.

In an AI application, this can happen because:

- LLM provider latency increases.
- Embedding API rate limits are reached.
- Workers crash.
- Qdrant becomes slow.
- Too few consumers are running.

I would monitor consumer lag and processing latency and scale or throttle accordingly.

# 13. Kafka vs Celery: What should you use in your AI project?

This is especially important because your financial advisor copilot already uses FastAPI, Celery, Redis, and LangGraph.

I would not replace everything with Kafka.

I would divide responsibilities.

Recommended architecture

FastAPI

Celery + Redis/RabbitMQ

Background PDF processing

Embedding tasks

Scheduled jobs

Retries

Kafka

Document lifecycle events

Audit streams

Usage analytics

Cross-service events

LangGraph

Agent orchestration and durable workflow checkpoints

PostgreSQL · Qdrant · Redis · S3

My preferred approach:

- Use Celery when you need Python background task execution with retries and scheduling.
- Use Kafka when multiple independent services need to consume and replay high-volume business events.
- Use LangGraph to orchestrate multi-agent workflows.
- Use Redis for caching, rate limiting, and coordination.

Avoid introducing all four technologies if the application does not require them. Every additional distributed component adds operational complexity.

# 14. Common mistakes in production AI systems

As a Staff Engineer, I would pay attention to these mistakes:

1. Publishing entire PDFs or huge prompts into Kafka instead of object-storage references.
2. Assuming Kafka automatically guarantees exactly-once business processing.
3. Not implementing consumer idempotency.
4. Committing offsets before processing completes.
5. Retrying permanent failures indefinitely.
6. Not monitoring consumer lag.
7. Having too few partitions for required parallelism.
8. Ignoring schema evolution.
9. Publishing sensitive customer data into shared topics.
10. Using Kafka as a replacement for LangGraph checkpointing.

# 15. Interview-ready answer: How is Kafka used in production AI projects?

Here is how I would explain it in a Senior AI Engineer interview:

"Kafka is used in production AI applications to build event-driven, asynchronous, and scalable architectures.

For example, in an enterprise RAG application, when a user uploads a PDF, FastAPI stores the document in object storage and publishes a document-uploaded event to Kafka.

The document ingestion service consumes that event and performs PDF parsing, text extraction, chunking, and embedding generation.

After embedding generation, it publishes another event, which is consumed by the indexing service to store vectors in Qdrant.

Kafka helps decouple these services so that document parsing, embedding generation, indexing, analytics, and audit processing can scale independently.

I also use Kafka for AI workflow events, model usage tracking, audit logging, and asynchronous multi-agent task distribution.

For reliability, I implement consumer-side idempotency, offset management, retry topics, dead-letter queues, schema versioning, and the transactional outbox pattern.

I monitor consumer lag, event processing latency, throughput, and failure rates.

In a multi-agent architecture, Kafka handles communication between independently deployed services, while LangGraph handles agent orchestration, conditional routing, state management, and checkpoint recovery.

I would choose Kafka for high-throughput event streaming and replay, whereas Celery with Redis or RabbitMQ is often more suitable for conventional Python background jobs."

# 16. Test your understanding — Senior AI Engineer interview practice

Knowledge check

0/5 answered

1\. What happens if a Kafka consumer crashes after inserting embeddings into Qdrant but before committing the offset?

Kafka deletes the message

The message may be processed again; use idempotent writes

Kafka automatically rolls back Qdrant

The producer sends the event again

2\. Why would you use Kafka instead of directly calling an embedding API from FastAPI?

Kafka makes embeddings more accurate

Kafka removes token costs

Kafka decouples processing and provides buffering and independent scaling

Kafka replaces the embedding model

3\. What is the purpose of a Kafka consumer group?

Encrypting messages

Allowing consumers to share processing of partitions

Storing PDF documents

Generating embeddings

4\. What does a Dead Letter Queue do?

Stores permanently failed or quarantined messages

Stores successful LLM responses

Replaces Kafka partitions

Caches embeddings

5\. What is the difference between Kafka and LangGraph?

They are identical technologies

Kafka generates embeddings; LangGraph stores PDFs

Kafka manages event streams; LangGraph orchestrates AI workflows

LangGraph replaces Kafka in all cases

Submit answers

What I recommend learning next: Implement a complete asynchronous RAG ingestion pipeline using FastAPI + Kafka + PostgreSQL + Qdrant, including transactional outbox, retry topics, DLQ, idempotency, and crash recovery. That is a strong production-level project to discuss in a Staff AI Engineer interview.