# How do you synchronize embeddings when documents are frequently updated or deleted?

Senior AI Engineer / Staff Engineer interview answer

In a production RAG system, I would implement an event-driven, version-aware document ingestion and embedding synchronization pipeline.

My objective would be to ensure that whenever a document is created, updated, or deleted:

1. The latest document content is reflected in the vector database.
2. Old embeddings are never accidentally used.
3. Deleted documents cannot be retrieved.
4. Failed embedding jobs can resume safely.
5. Duplicate events do not create duplicate embeddings.
6. Concurrent updates do not overwrite newer versions.
7. All changes remain consistent across PostgreSQL, object storage and Qdrant.

I would not simply run a background job that re-embeds every document periodically. That approach becomes expensive and difficult to maintain at enterprise scale.

Let's design this as a production system using FastAPI + PostgreSQL + Redis + Celery + Qdrant + OpenAI embeddings.

## 1. First understand the synchronization problem

Suppose a financial organization has uploaded a document:

Document: Investment Policy.pdf

Version 1

Indexed

```
Maximum equity allocation: 40%
```

Document updated

Version 2 pending

```
Maximum equity allocation: 50%
```

Potential inconsistency

PostgreSQL contains version 2, but Qdrant still contains embeddings generated from version 1.

A user asks:

"What is the maximum equity allocation?"

The RAG system retrieves the old embedding and responds:

"40%"

This is a stale embedding problem.

Now consider deletion:

- User deletes Investment Policy.pdf.
- PostgreSQL marks it deleted.
- Qdrant still contains its vectors.
- RAG retrieves information from a deleted document.

This is a data consistency and security problem, not merely an embedding problem.

## 2. Architecture I would implement

I would separate document management from asynchronous embedding synchronization.

## Document synchronization architecture

FastAPI

Upload / Update / Delete APIs

PostgreSQL

Document metadata, version, status, outbox events

Transactional Outbox

Reliable event publication

Celery workers

Asynchronous ingestion and synchronization

Document processing

Parse → Chunk → Embed

Version management

Validate → Activate → Retire

Object storage

Original documents

Qdrant

Versioned embeddings

RAG retrieval applies tenant, document-version and publication-state filtering.

The important design decision is that PostgreSQL is the source of truth for document state, while Qdrant is the derived retrieval index.

## 3. Design the document lifecycle

I would maintain an explicit document state machine.

| Status     | Meaning                            |
| ---------- | ---------------------------------- |
| UPLOADED   | Original document received         |
| PROCESSING | Parsing and chunking               |
| EMBEDDING  | Embeddings being generated         |
| INDEXING   | Vectors being stored               |
| ACTIVE     | Version is available for retrieval |
| FAILED     | Processing failed                  |
| DELETING   | Removal in progress                |
| DELETED    | Document removed from retrieval    |

The lifecycle looks like this:

UPLOADED

PROCESSING

EMBEDDING → INDEXING

ACTIVE

Updates create a new version; deletions move through DELETING to DELETED.

I would never consider a document active just because its embedding job started.

It becomes active only after the new version has been successfully indexed and validated.

## 4. Database design: versioning is critical

I would maintain separate tables for documents, document versions and synchronization events.

### Documents table

```
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,

    active_version INTEGER,
    status VARCHAR(30) NOT NULL,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP
);
```

### Document versions table

```
CREATE TABLE document_versions (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id),

    version INTEGER NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,

    status VARCHAR(30) NOT NULL,
    chunk_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(document_id, version)
);
```

### Outbox events table

```
CREATE TABLE outbox_events (
    id UUID PRIMARY KEY,
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,

    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

Why do I need these tables?

- `documents`: Maintains the latest active document version.
- `document_versions`: Tracks every ingestion and embedding version.
- `outbox_events`: Guarantees that database changes and event creation happen in the same transaction.

Important Staff Engineer decision: I would use the transactional outbox pattern instead of directly publishing a Celery task inside the API request.

Imagine PostgreSQL commits the document update, but Redis goes down before the Celery task is published.

Without an outbox, the database says the document has changed, but embedding synchronization never starts.

With an outbox, a separate publisher can retry the unpublished event.

## 5. Document update workflow

Suppose version 1 is already indexed and a user uploads version 2.

I would follow these steps:

1. Receive the new document.
2. Calculate its SHA-256 content hash.
3. Check whether the content has actually changed.
4. Create a new document version in PostgreSQL.
5. Create an outbox event in the same database transaction.
6. Publish the event to the worker.
7. Parse the new document.
8. Generate chunks.
9. Generate embeddings.
10. Upsert vectors into Qdrant using version-specific IDs.
11. Verify the indexed chunk count.
12. Atomically update the active version in PostgreSQL.
13. Retire old vectors asynchronously.

The critical part is step 12.

I would not delete version 1 before version 2 is successfully indexed.

This avoids making a document unavailable during an update.

### Version-aware vector IDs

Suppose we have:

```
Document ID: doc_123
Version: 1
Chunks: 100
```

The new upload creates:

```
Document ID: doc_123
Version: 2
Chunks: 120
```

I would use version-specific vector identifiers:

```
doc_123:v1:chunk_0
doc_123:v1:chunk_1
...
doc_123:v1:chunk_99
```

And:

```
doc_123:v2:chunk_0
doc_123:v2:chunk_1
...
doc_123:v2:chunk_119
```

This is important because a document can have a different number of chunks after an update.

If I reuse the old IDs without carefully managing removed chunks, old vectors can remain in Qdrant.

## 6. Implement the embedding synchronization service

Let's implement the main components.

### A. Content hashing

I would avoid reprocessing documents if their content has not changed.

```
import hashlibdef calculate_content_hash(content: bytes) -> str:    return hashlib.sha256(content).hexdigest()
```

For example:

```
old_hash = calculate_content_hash(old_content)new_hash = calculate_content_hash(new_content)if old_hash == new_hash:    print("No document changes detected")
```

In production, I would also include parser version, chunking configuration and embedding model version in the indexing fingerprint.

A document's bytes can remain unchanged while its indexing configuration changes.

### B. Document chunking

```
from langchain_text_splitters import (    RecursiveCharacterTextSplitter)class DocumentChunker:    def __init__(self):        self.splitter = RecursiveCharacterTextSplitter(            chunk_size=400,            chunk_overlap=60,            separators=[                "\n\n",                "\n",                ". ",                " ",                ""            ]        )    def chunk(self, text: str):        return self.splitter.split_text(text)
```

In a real enterprise pipeline, I would also preserve:

- Document ID
- Version
- Tenant ID
- Page number
- Section heading
- Chunk index
- Content hash

### C. Generate embeddings in batches

```
from openai import AsyncOpenAIclass EmbeddingService:    def __init__(self, client: AsyncOpenAI):        self.client = client    async def generate_embeddings(        self,        texts: list[str]    ) -> list[list[float]]:        response = await self.client.embeddings.create(            model="text-embedding-3-small",            input=texts        )        return [            item.embedding            for item in sorted(                response.data,                key=lambda x: x.index            )        ]
```

For production, I would batch requests, control concurrency, enforce provider rate limits and retry transient failures.

### D. Store vectors in Qdrant

Here is an important implementation detail.

Each vector must carry enough metadata to identify exactly which document version it belongs to.

```
from qdrant_client import AsyncQdrantClientfrom qdrant_client.models import (    PointStruct,    Filter,    FieldCondition,    MatchValue)import uuidclass QdrantRepository:    def __init__(        self,        client: AsyncQdrantClient,        collection_name: str    ):        self.client = client        self.collection_name = collection_name    async def upsert_chunks(        self,        document_id: str,        tenant_id: str,        version: int,        chunks: list[str],        embeddings: list[list[float]]    ):        if len(chunks) != len(embeddings):            raise ValueError(                "Chunk and embedding counts do not match"            )        points = []        for index, (chunk, embedding) in enumerate(
```

Why use deterministic IDs?

Because Celery provides at-least-once execution in typical deployments. A worker might execute the same task more than once after a timeout or crash.

If the same version and chunk are processed again, the same point ID is generated. Qdrant upsert replaces the existing point instead of creating another copy.

This makes the operation idempotent.

## 7. Build the asynchronous ingestion worker

I would use Celery to process document synchronization outside the FastAPI request lifecycle.

The following illustrates the orchestration. The repository and service dependencies are injected into the worker application during startup.

```
from celery import Celerycelery_app = Celery(    "document_ingestion",    broker="redis://localhost:6379/0",    backend="redis://localhost:6379/1")@celery_app.task(    bind=True,    autoretry_for=(ConnectionError, TimeoutError),    retry_backoff=True,    retry_backoff_max=300,    retry_jitter=True,    max_retries=5)def synchronize_document(    self,    document_id: str,    version: int,    event_id: str):    import asyncio    return asyncio.run(        ingestion_worker.process(            document_id=document_id,            version=version,            event_id=event_id        )    )
```

The worker's processing method should follow this logic:

```
class IngestionWorker:    def __init__(        self,        document_repository,        parser,        chunker,        embedding_service,        vector_repository    ):        self.documents = document_repository        self.parser = parser        self.chunker = chunker        self.embeddings = embedding_service        self.vectors = vector_repository    async def process(        self,        document_id: str,        version: int,        event_id: str    ):        # 1. Verify event has not already completed.        if await self.documents.event_completed(event_id):            return {"status": "already_processed"}        # 2. Read immutable document version.        document = await self.documents.get_version(            document_id,            version        )        if document is None:            return {"status": "document_version_not_found"}
```

There is one important concurrency detail here.

Imagine:

- Version 2 starts processing.
- Version 3 starts processing.
- Version 3 finishes first.
- Version 2 finishes later.

Version 2 must never overwrite version 3 as the active version.

Therefore, `activate_version()` must perform a conditional database update, checking that the version is still the latest requested version and that the document is not deleted.

For example:

```
UPDATE documents
SET active_version = :version,
    status = 'ACTIVE'
WHERE id = :document_id
  AND requested_version = :version
  AND deleted_at IS NULL;
```

If zero rows are updated, the version has been superseded or the document has been deleted.

This prevents out-of-order worker completion from publishing stale data.

## 8. How would you handle document deletion?

This is particularly important in financial, healthcare and enterprise systems.

Suppose a user deletes:

```
Document ID: doc_123
Version: 2
```

I would not depend only on deleting vectors from Qdrant.

Why?

Because Qdrant might be temporarily unavailable. If the deletion request fails, the old vectors could still be retrieved.

I would use a tombstone-first deletion strategy.

### Deletion workflow

1. Mark the document as deleted in PostgreSQL.
2. Set `deleted_at` and status to `DELETING`.
3. Immediately exclude it from retrieval.
4. Publish a deletion event using the transactional outbox.
5. Delete all associated vectors from Qdrant.
6. Verify deletion.
7. Mark the document as `DELETED`.
8. Retain an audit record according to the organization's retention policy.

### Qdrant deletion implementation

```
from qdrant_client.models import (    Filter,    FieldCondition,    MatchValue)class QdrantRepository:    # Existing constructor and upsert_chunks omitted here.    async def delete_document(        self,        document_id: str,        tenant_id: str    ):        await self.client.delete(            collection_name=self.collection_name,            points_selector=Filter(                must=[                    FieldCondition(                        key="document_id",                        match=MatchValue(                            value=document_id                        )                    ),                    FieldCondition(                        key="tenant_id",                        match=MatchValue(                            value=tenant_id                        )                    )                ]            ),            wait=True        )
```

Notice that I am deleting by document ID and tenant ID, not by individual chunk IDs.

This removes all versions of the document.

For a high-volume system, I would also maintain deletion progress and periodically reconcile documents marked `DELETING` against Qdrant.

### Why tombstones matter

Consider this situation:

| System        | Document status          |
| ------------- | ------------------------ |
| PostgreSQL    | DELETED                  |
| Qdrant        | Old vectors still exist  |
| RAG retrieval | Must not return document |

The PostgreSQL deletion state becomes authoritative.

Even if Qdrant cleanup fails, retrieval must reject the deleted document.

This is a security boundary, not an eventual cleanup optimization.

## 9. How do you prevent stale embeddings from being retrieved?

This is probably the most important follow-up question in a senior AI interview.

I would use version-aware retrieval and validate returned candidates against authoritative document state.

For example:

```
PostgreSQL:

doc_123
active_version = 3

Qdrant:

doc_123, version = 1
doc_123, version = 2
doc_123, version = 3
```

Only version 3 should be used.

The simplest approach is to include version and tenant filters in Qdrant queries.

However, there is a subtle distributed consistency issue: the database might activate version 3 while Qdrant still contains old versions.

Therefore, I would combine two controls:

1. Query using available tenant and document-version filters.
2. Validate every returned candidate against the current authoritative document state before passing it to the LLM.

Example:

```
class RetrievalService:    def __init__(        self,        vector_store,        document_repository    ):        self.vector_store = vector_store        self.documents = document_repository    async def retrieve(        self,        query: str,        tenant_id: str,        top_k: int = 10    ):        candidates = await self.vector_store.search(            query=query,            tenant_id=tenant_id,            limit=top_k        )        # Batch-fetch authoritative document states.        document_ids = list({            item.payload["document_id"]            for item in candidates        })        states = await self.documents.get_active_versions(            tenant_id=tenant_id,            document_ids=document_ids        )        valid_results = []
```

Here, `get_active_versions()` should use one batched database query, not one SQL query per candidate.

For high-throughput systems, I would also consider maintaining an authorization-aware active-version index or cache with carefully controlled invalidation.

Important: If post-validation removes many candidates, retrieve additional candidates to fill the result set. Otherwise, stale vectors can indirectly reduce retrieval recall.

## 10. Handle the transactional outbox correctly

Let's examine how the update event reaches the Celery worker reliably.

The API should update document metadata and insert an outbox event within the same PostgreSQL transaction.

Conceptually:

```
async def request_document_update(    session,    document_id,    new_version,    content_hash):    async with session.begin():        document = await get_document_for_update(            session,            document_id        )        if document.content_hash == content_hash:            return {"status": "unchanged"}        document.requested_version = new_version        document.status = "PROCESSING"        event = OutboxEvent(            aggregate_id=document_id,            event_type="DOCUMENT_UPDATED",            payload={                "document_id": str(document_id),                "version": new_version            }        )        session.add(event)    return {"status": "accepted"}
```

A separate outbox publisher polls unpublished events and dispatches Celery tasks.

```
async def publish_pending_events(    outbox_repository,    event_publisher):    events = await outbox_repository.claim_pending(        limit=100    )    for event in events:        await event_publisher.publish(            event_type=event.event_type,            payload=event.payload,            event_id=str(event.id)        )        await outbox_repository.mark_published(            event.id        )
```

`claim_pending()` should use row-level locking such as `FOR UPDATE SKIP LOCKED` so multiple publishers do not claim the same event simultaneously.

There is another important failure scenario:

- Publisher sends the event to Celery.
- Publisher crashes before marking the event published.
- Another publisher sends the same event again.

This is expected in an at-least-once delivery architecture.

Therefore:

- Use unique event IDs.
- Maintain event processing records.
- Make workers idempotent.
- Use deterministic Qdrant point IDs.
- Make activation conditional.

I would never assume exactly-once delivery across PostgreSQL, Redis and Qdrant.

## 11. How do you handle frequent document updates at scale?

Imagine an enterprise application with:

- 1 million documents.
- 100,000 document updates per day.
- Multiple tenants.
- Different embedding models.
- Thousands of concurrent users.

Re-embedding everything after every change would be wasteful.

I would introduce the following optimizations.

| Optimization          | Purpose                          |
| --------------------- | -------------------------------- |
| Content hashing       | Skip unchanged documents         |
| Incremental ingestion | Process only changed documents   |
| Batch embeddings      | Reduce API overhead              |
| Async workers         | Scale processing independently   |
| Queue partitioning    | Avoid processing bottlenecks     |
| Idempotency           | Prevent duplicate indexing       |
| Versioned vectors     | Safely support updates           |
| Retry with backoff    | Recover transient failures       |
| Dead-letter queue     | Isolate permanently failing jobs |
| Reconciliation jobs   | Detect missing or stale vectors  |

### Incremental chunk synchronization

Suppose the original document has 100 chunks.

After an update, it has 105 chunks.

Rather than assuming all chunks are identical, I would compare chunk hashes.

```
import hashlibdef chunk_hash(text: str) -> str:    return hashlib.sha256(        text.encode("utf-8")    ).hexdigest()def compare_chunks(old_chunks, new_chunks):    old_hashes = {        chunk_hash(chunk)        for chunk in old_chunks    }    new_hashes = {        chunk_hash(chunk)        for chunk in new_chunks    }    added = new_hashes - old_hashes    removed = old_hashes - new_hashes    unchanged = old_hashes & new_hashes    return {        "added": added,        "removed": removed,        "unchanged": unchanged    }
```

This is a simplified content comparison.

For a robust implementation, I would preserve chunk identity using stable section identifiers and occurrence-aware hashes, rather than relying on text hashes alone. Identical paragraphs can appear more than once in a document.

There are two possible strategies:

Strategy A: Full version reindexing

Generate all vectors for the new version and activate it after successful indexing.

- Simpler.
- Safer.
- Easier to recover.
- More embedding cost.

Strategy B: Incremental vector updates

Only regenerate changed chunks and reuse unchanged embeddings when the embedding model, preprocessing and chunk content are identical.

- Lower embedding cost.
- More complex version management.
- Requires careful handling of chunk identity and references.

For financial policy documents, I would initially choose full version reindexing for correctness. I would introduce incremental embedding reuse only after measuring the cost and validating its correctness.

## 12. What if the embedding model changes?

This is another important production scenario.

Suppose initially I use:

```
text-embedding-3-small
```

Later, I decide to migrate to:

```
text-embedding-3-large
```

The embedding dimensions are different, so I cannot simply mix both embedding types in the same vector configuration.

I would implement a blue-green indexing strategy.

Embedding model migration

Current

Collection A

text-embedding-3-small

# Active

Production retrieval

Migration

Collection B

text-embedding-3-large

# Building

Backfill and evaluation

Validate retrieval quality and operational performance

Switch production retrieval to Collection B

Migration complete

My migration steps would be:

1. Create a new Qdrant collection with the new vector dimensions.
2. Reprocess existing document versions.
3. Generate new embeddings.
4. Populate the new collection.
5. Compare retrieval Recall\@K, Precision\@K and answer correctness.
6. Run shadow queries against both collections.
7. Switch production traffic.
8. Keep the old collection temporarily for rollback.
9. Remove the old collection after the rollback window.

I would never mix embeddings from incompatible models in the same vector space.

## 13. Build a reconciliation mechanism

Even with reliable event processing, distributed systems can develop inconsistencies.

For example:

| PostgreSQL           | Qdrant                      | Problem              |
| -------------------- | --------------------------- | -------------------- |
| Version 3 active     | Version 3 exists            | Correct              |
| Version 3 active     | Version 2 only              | Missing vectors      |
| Deleted              | Vectors exist               | Stale data           |
| Version 4 processing | Version 4 partially indexed | Incomplete ingestion |

I would build a scheduled reconciliation job.

Its responsibilities:

- Find active documents with missing vectors.
- Find incomplete indexing operations.
- Find vectors belonging to deleted documents.
- Find vectors belonging to obsolete versions.
- Retry failed synchronization operations.
- Generate operational alerts.

Example:

```
async def reconcile_documents(    document_repository,    vector_repository):    documents = await document_repository.get_active_documents(        limit=500    )    for document in documents:        expected_count = document.chunk_count        actual_count = await vector_repository.count_version(            document_id=str(document.id),            version=document.active_version,            tenant_id=str(document.tenant_id)        )        if actual_count != expected_count:            await document_repository.mark_for_reindex(                document.id,                document.active_version            )
```

This is a simplified reconciliation example. In production, I would use paginated scans, bounded concurrency, repair-job idempotency and checksums or expected point IDs, not just count comparisons.

## 14. Production monitoring

I would monitor these metrics:

| Metric                       | What it tells me                     |
| ---------------------------- | ------------------------------------ |
| Embedding generation latency | Embedding provider performance       |
| Indexing latency             | Time to make a new version available |
| Pending outbox events        | Event processing backlog             |
| Failed ingestion jobs        | Processing reliability               |
| Stale vector count           | Synchronization health               |
| Deleted vector count         | Deletion cleanup backlog             |
| Embedding cost per document  | Operational cost                     |
| Reindexing success rate      | Pipeline reliability                 |
| Active-version mismatch      | Data consistency issue               |

I would define an SLO for update propagation.

For example:

Example production SLO

# 99%

Of successful document updates become searchable within 5 minutes.

# 0

Stale or deleted document versions returned to authorized RAG queries.

Illustrative targets, to be adjusted based on business requirements.

I would use OpenTelemetry for tracing, Prometheus for metrics and Grafana for operational dashboards.

## 15. Important failure scenarios interviewers may ask

| Scenario                            | My solution                                       |
| ----------------------------------- | ------------------------------------------------- |
| Redis goes down                     | Transactional outbox retains unpublished events   |
| Celery worker crashes               | Retry using idempotent jobs                       |
| Embedding API times out             | Exponential backoff with jitter                   |
| Qdrant is unavailable               | Retry indexing; don't activate incomplete version |
| Same update event executes twice    | Event deduplication and deterministic point IDs   |
| Version 2 finishes after version 3  | Conditional version activation                    |
| Document is deleted during indexing | Tombstone check before activation                 |
| Some chunks are missing             | Validate expected point IDs and reconcile         |
| Document is uploaded unchanged      | Content hash prevents unnecessary reprocessing    |
| Old vectors remain after deletion   | Retrieval validation plus asynchronous cleanup    |

## 16. How I would explain this in a senior AI engineer interview

Interview-ready answer

"I would implement an event-driven, version-aware embedding synchronization architecture.

Whenever a document is uploaded or updated, I would calculate its content hash, create a new immutable document version in PostgreSQL and publish a synchronization event through a transactional outbox.

Celery workers would asynchronously parse the document, generate chunks, create embeddings and upsert vectors into Qdrant.

I would use deterministic vector IDs and document-version metadata to make indexing idempotent.

Importantly, I would not remove the previous version immediately. I would first ensure that the new version has been completely indexed and validated, then atomically update the active version in PostgreSQL. Retrieval would validate document versions against authoritative metadata to prevent stale embeddings from being returned.

For deletion, I would use a tombstone-first approach. I would immediately mark the document as deleted in PostgreSQL, exclude it from retrieval and asynchronously remove all its vectors from Qdrant.

I would also implement retries, dead-letter queues, reconciliation jobs, distributed tracing and monitoring to identify synchronization failures.

At scale, I would introduce incremental embedding reuse, batching, queue partitioning and separate worker pools.

My primary design principle is that PostgreSQL is the source of truth, Qdrant is a derived index, and every synchronization operation must be idempotent, recoverable and version-aware."

## 17. Follow-up questions to prepare

These are particularly important for Staff AI Engineer interviews:

1. How do you prevent two workers from processing the same document simultaneously?
2. How do you handle a document update while an older embedding job is still running?
3. How do you guarantee that deleted documents are never retrieved?
4. How would you synchronize embeddings across multiple tenants?
5. How do you migrate from one embedding model to another without downtime?
6. How do you implement incremental embedding updates?
7. How would you design an outbox pattern with PostgreSQL and Celery?
8. How do you recover from a Qdrant outage without losing document updates?

My recommendation: Master questions 2, 3, 5 and 7. They test distributed systems, concurrency, data consistency and production architecture—not just basic RAG knowledge. Those are the areas that distinguish a senior AI engineer from someone who has only implemented a basic RAG pipeline.