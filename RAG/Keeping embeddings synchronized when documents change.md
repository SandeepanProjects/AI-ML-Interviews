# Keeping embeddings synchronized when documents change frequently — Staff AI Engineer approach

In a production RAG system, the hard part isn't generating embeddings. It's ensuring that the vectors used for retrieval represent the latest valid document content, while handling concurrent updates, failures, deletions, and permission changes safely.

For example, imagine a financial institution updates an investment policy every few minutes. If your RAG system retrieves embeddings from an older policy, the LLM can generate a confident but outdated answer, even if the rest of your pipeline is functioning correctly.

I would design synchronization around five guarantees:

1. Freshness: detect document changes quickly.
2. Correctness: never publish a partially indexed version.
3. Ordering: an older worker must not overwrite a newer version.
4. Recoverability: missed events and worker failures must be recoverable.
5. Security: deleted or unauthorized documents must not remain retrievable.

## 1. Architecture I would use

Document source

SharePoint, S3, internal policy repository, or another authoritative source

Change detection

Events / webhooks + periodic reconciliation

PostgreSQL + transactional outbox

Desired version, active version, durable work items, ACL state

Celery ingestion workers

Fetch → parse → chunk → embed → write → validate

PostgreSQL

Authoritative version and publication state

Qdrant

Versioned vector points and metadata

Retrieval-time validation

Current version + tenant + authorization + deletion checks

The key distinction is that PostgreSQL records which version is valid, while Qdrant stores the derived vectors. The presence of a vector in Qdrant does not prove that the vector is current.

## 2. Detect changes using events and reconciliation

I would use two complementary mechanisms.

Event-driven detection provides low latency. A source system might emit:

```
{
  "event_id": "evt-1082",
  "document_id": "policy-123",
  "operation": "updated",
  "source_version": "v42"
}
```

The event handler should:

1. Authenticate and validate the event.
2. Deduplicate it using `event_id`.
3. Fetch the latest authoritative version when necessary.
4. Record the desired version in PostgreSQL.
5. Create a durable ingestion work item.
6. Acknowledge the event only after it has been durably recorded.

Do not assume events arrive in order. Version 42 might be processed after version 43. The worker must always check whether the version it is processing is still the desired version.

Periodic reconciliation catches missed events, broken webhooks, and abandoned jobs. It compares source versions against the document registry and schedules repairs.

For a large repository, use incremental change tokens or source-specific change feeds. Full scans can remain a periodic safety check rather than running on every update.

## 3. Separate document identity, desired version, and active version

A single `processed = true` flag is insufficient. It cannot tell you whether the successfully processed version is still current.

A simplified PostgreSQL model:

```
CREATE TABLE documents (
    tenant_id UUID NOT NULL,
    document_id UUID NOT NULL,
    source_id TEXT NOT NULL,
    desired_version_id UUID,
    active_version_id UUID,
    source_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    acl_version BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, document_id),
    UNIQUE (tenant_id, source_id)
);

CREATE TABLE document_versions (
    version_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    document_id UUID NOT NULL,
    source_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    status TEXT NOT NULL,
    expected_chunk_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

These fields serve different purposes:

- `desired_version_id`: the version we want to index.
- `active_version_id`: the version currently eligible for retrieval.
- `source_deleted`: whether the source has withdrawn the document.
- `acl_version`: tracks authorization changes independently of content.

For example, if version 42 fails embedding, the system can still identify version 41 as the last successfully published version. Whether version 41 is allowed to remain available depends on the business's freshness and compliance policy.

## 4. Use content hashes to avoid unnecessary embedding

Not every file update requires a new embedding.

```
import hashlibdef calculate_content_hash(normalized_text: str) -> str:    return hashlib.sha256(        normalized_text.encode("utf-8")    ).hexdigest()
```

A robust ingestion service compares:

- The source version or ETag, if available.
- A hash of the normalized content.
- The embedding model and chunking configuration.
- Relevant metadata and ACL versions.

If the normalized content and embedding configuration are unchanged, reuse the existing vectors instead of paying to embed everything again.

For instance, changing PDF metadata without changing its extracted text may not require re-embedding. Changing the document's permissions, however, still requires an authorization update even if its content hash is identical.

For especially large documents, a further optimization is to hash individual chunks and reuse embeddings for unchanged chunks. That optimization needs careful chunk identity management: inserting a paragraph near the beginning can shift every subsequent chunk boundary.

## 5. Build new vectors before replacing old vectors

This is the most important part of the design.

Suppose the old policy has 20 chunks and the new policy has 12. If you overwrite chunks 0–11 and forget to remove chunks 12–19, stale content remains searchable.

Instead, assign every ingestion generation a unique version ID and create deterministic point IDs for its chunks.

```
from uuid import UUID, uuid5NAMESPACE = UUID("d2f9718b-57bd-4c82-9b7c-4d872e0b8a10")def make_point_id(    tenant_id: str,    document_id: str,    version_id: str,    chunk_index: int,) -> str:    identity = (        f"{tenant_id}:{document_id}:"        f"{version_id}:{chunk_index}"    )    return str(uuid5(NAMESPACE, identity))
```

The same version and chunk always produce the same point ID. If a Celery task retries, it upserts the same point rather than creating a duplicate.

A versioned point should include metadata like:

```
payload = {    "tenant_id": tenant_id,    "document_id": document_id,    "version_id": version_id,    "chunk_index": chunk_index,    "content_hash": content_hash,    "acl_version": acl_version,}
```

### Publication lifecycle

1. Fetch and validate the authoritative source version.
2. Parse, normalize, and chunk the document.
3. Generate embeddings in bounded batches.
4. Upsert version-specific vectors into Qdrant.
5. Validate completeness: expected chunk count, vector dimensions, metadata, and successful writes.
6. Conditionally activate the version in PostgreSQL only if it is still the desired version and the document has not been deleted.
7. Garbage-collect old vectors asynchronously after publication and any required grace period.

This is a blue-green style publication pattern: construct the replacement first, then switch to it.

PostgreSQL and Qdrant do not share a transaction, so the switch cannot be atomic across both databases. The design instead ensures that all required Qdrant writes are complete before PostgreSQL declares the version active.

## 6. Prevent older workers from overwriting newer versions

Imagine two tasks are running:

- Worker A is embedding version 42.
- Version 43 arrives.
- Worker B finishes version 43 first.
- Worker A finishes later.

Without a conditional publication check, Worker A could roll the document back to version 42.

The publication step should update the active pointer only if the target version is still the desired version.

```
UPDATE documents
SET active_version_id = :version_id,
    updated_at = NOW()
WHERE tenant_id = :tenant_id
  AND document_id = :document_id
  AND desired_version_id = :version_id
  AND source_deleted = FALSE;
```

If the affected row count is zero, that version was superseded or the document was deleted. It must not become active.

This conditional database update is a correctness mechanism. A Redis lock can reduce duplicate work, but should not replace the version check: locks can expire while a worker is paused.

## 7. Make ingestion durable and retry-safe

For an enterprise system, I would combine PostgreSQL, a transactional outbox, and Celery.

When a source change is recorded, one PostgreSQL transaction creates or updates the desired version and adds an outbox event. A dispatcher publishes pending outbox records to Celery.

This prevents the classic failure where the database commits a new version but the application crashes before enqueueing the task.

Celery tasks should be idempotent and retry transient errors with bounded exponential backoff and jitter. A task should record its progress, and stuck jobs should be recoverable by a scheduler.

An embedding worker might follow this pattern:

```
async def process_document_version(    request,    source,    registry,    parser,    chunker,    embedder,    vector_store,):    document = await source.fetch(        request.document_id,        version=request.source_version,    )    if document.version != request.source_version:        # The source changed after this task was created.        # Do not publish an obsolete snapshot.        await registry.schedule_latest_version(            request.tenant_id,            request.document_id,        )        return    parsed = await parser.parse(document.content)    chunks = chunker.split(parsed)    if not chunks:        raise ValueError("Document has no indexable content")    vectors = []    for batch in batched(chunks, size=64):        vectors.extend(await embedder.embed(batch))    if len(vectors) != len(chunks):        raise RuntimeError("Embedding output is incomplete")
```

The interfaces above are illustrative; `activate_if_current`, `verify_version`, and cleanup scheduling need explicit database transactions, failure handling, and monitoring in the actual implementation.

## 8. Treat deletion and permission changes as separate critical paths

If a document is deleted, don't wait for a full re-embedding cycle.

1. Record a durable tombstone.
2. Mark the document deleted or suppressed.
3. Make retrieval exclude it immediately through authoritative state or a reliable denylist.
4. Enqueue Qdrant cleanup.
5. Retry cleanup until complete.

A delayed update event must not resurrect a deleted document. Re-creation should be accepted only after checking the latest authoritative source state.

For permission changes, update authorization independently of the content pipeline. Retrieval should enforce tenant and document-level permissions, and fail closed if the required authorization state cannot be verified. Do not rely solely on ACL metadata that might be stale in Qdrant.

For a financial or compliance system, the policy may need to suppress the old version immediately, even if the replacement is not yet indexed. A temporary “no answer available” is safer than serving a withdrawn policy.

## 9. Handle embedding-model changes separately

Even if document content hasn't changed, changing the embedding model or chunking configuration can invalidate the existing index.

Track at least:

- Embedding model and pinned revision.
- Vector dimensions and normalization.
- Chunker version and chunk size/overlap configuration.
- Any transformations used to create the embedding input.

For a model migration, build a parallel Qdrant collection or separate named vector representation, re-embed and evaluate the corpus, then switch retrieval to the validated representation. Don't silently mix vectors from incompatible embedding spaces.

## 10. Measure freshness and consistency

I would monitor these production signals:

| Metric                                     | Why it matters                    |
| ------------------------------------------ | --------------------------------- |
| Source-change-to-publication lag, p95/p99  | Measures freshness                |
| Pending and stuck ingestion jobs           | Detects pipeline backlog          |
| Retry and dead-letter rates                | Detects persistent failures       |
| Desired version versus active version      | Detects stale documents           |
| Expected versus actual chunk count         | Detects partial indexing          |
| Orphan and obsolete vector count           | Detects cleanup failures          |
| Retrieval relevance and stale-content rate | Measures user-visible correctness |
| Unauthorized retrieval attempts            | Detects security failures         |

I would also test out-of-order events, duplicate deliveries, partial Qdrant writes, worker crashes, concurrent updates, deletion during ingestion, ACL revocation, and embedding-provider outages.

## Recommended design for your financial RAG platform

For your FastAPI + PostgreSQL + Qdrant + Redis + Celery stack, I would use:

- PostgreSQL as the authoritative version registry.
- Source events plus periodic reconciliation.
- A transactional outbox for durable work dispatch.
- Celery workers with idempotent writes and bounded retries.
- Version-specific Qdrant points with deterministic IDs.
- Conditional activation based on the latest desired version.
- Immediate retrieval suppression for deletions and permission revocations.
- Asynchronous garbage collection of obsolete vectors.
- Retrieval evaluation and synchronization-lag monitoring.

Interview-ready summary: “I treat embedding synchronization as a distributed consistency problem. I detect changes using events and reconciliation, maintain desired and active document versions in PostgreSQL, and process ingestion through durable, idempotent workers. I build version-specific vectors, validate completeness, and publish only if the version is still current. Deletes and ACL revocations are enforced at retrieval time immediately, while vector cleanup is retryable. This prevents stale versions from becoming active even when events arrive out of order, workers crash, or the vector database is temporarily unavailable.”