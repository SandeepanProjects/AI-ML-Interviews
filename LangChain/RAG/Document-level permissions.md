# Document-level permissions, updates, and deletions in production RAG

In an enterprise RAG system, document security and lifecycle management are as important as retrieval quality. You can have excellent embeddings, reranking, and agents, but if a user can retrieve another department's confidential document—or an old policy remains searchable after deletion—the system is not production-ready.

For your financial compliance platform, I would design this around three principles:

1. Authorization is enforced before evidence reaches the LLM. The model never decides whether a user may access a document.
2. A document update is a versioned, coordinated workflow. PostgreSQL, the vector index, keyword index, caches, and object storage must not drift silently out of sync.
3. Deletion is a lifecycle operation, not just a SQL `DELETE`. Every derived copy, embedding, cache entry, and searchable index entry must be accounted for.

I'll use FastAPI, PostgreSQL, pgvector, Redis, and a background worker as the reference architecture. The same principles apply to Qdrant and other vector databases.

## 1. Architecture: where permissions are enforced

Client request

Question + optional document filters

Authentication

Validate JWT / session; derive user and tenant from trusted identity

Authorization service

Resolve document ACLs, group membership, roles, classification, and applicable policy

Secure retrieval boundary

Apply authorized scope in PostgreSQL/vector queries before returning evidence

RAG pipeline

Retrieve → rerank → assemble context → generate answer with permitted evidence only

The critical boundary is the retrieval service. Filtering the final answer after the LLM has already seen confidential chunks is too late.

There are multiple layers worth enforcing:

- Application-level authorization, for understandable policy decisions.
- Database-level constraints and optionally PostgreSQL row-level security (RLS), as defense in depth.
- Vector-store filters, if using a separate vector database.
- Cache authorization and invalidation.
- Object-storage authorization for original documents and image assets.

Never trust `tenant_id`, `user_id`, or allowed roles supplied by the client. Derive them from a validated identity and server-side access policy. A user-supplied `document_id` is a resource request to authorize, not proof of permission.

# 2. Model document permissions explicitly

A common beginner implementation stores this on a chunk:

```
{
  "document_id": "policy-123",
  "tenant_id": "tenant-a",
  "allowed_roles": ["compliance"]
}
```

That can work for simple role-based access, but it becomes inadequate when you need individual grants, groups, revocations, inherited permissions, or auditability.

I prefer a document-centric ACL model in PostgreSQL. Chunks refer to a document; the document and its ACL are the source of truth.

## 2.1 Example schema

```
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'pending',
            'active',
            'updating',
            'deleting',
            'deleted',
            'failed'
        )
    ),
    current_version INTEGER NOT NULL DEFAULT 0,
    classification TEXT NOT NULL,
    created_by UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

CREATE TABLE document_acl (
    document_id UUID NOT NULL REFERENCES documents(id),
    principal_type TEXT NOT NULL CHECK (
        principal_type IN ('user', 'group', 'role')
    ),
    principal_id TEXT NOT NULL,
    permission TEXT NOT NULL CHECK (
        permission IN ('read', 'write', 'admin')
    ),
    granted_by UUID NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        document_id,
        principal_type,
        principal_id,
        permission
    )
);

CREATE TABLE document_versions (
    document_id UUID NOT NULL REFERENCES documents(id),
    version INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    object_uri TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, version)
);

CREATE TABLE rag_chunks (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    document_id UUID NOT NULL REFERENCES documents(id),
    document_version INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    page_number INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX rag_chunks_document_version_idx
    ON rag_chunks (document_id, document_version);

CREATE INDEX document_acl_principal_idx
    ON document_acl (
        principal_type,
        principal_id,
        permission
    );
```

This is a starting schema, not a complete migration. In production, add foreign keys for versions, indexes based on measured query patterns, an immutable audit-event table, and constraints preventing invalid lifecycle transitions. For large multi-tenant deployments, also consider tenant-aware partitioning where workload measurements justify it.

A few design decisions matter:

- `documents` defines the current lifecycle state.
- `document_acl` stores explicit grants.
- `document_versions` tracks immutable source versions.
- `rag_chunks` stores derived retrieval content and its version.
- The application must not treat a chunk as authorized merely because it exists in the vector index.

## 2.2 Decide how permissions combine

You need a clearly defined policy. For example:

- A user can read a document if they have a direct user grant, an applicable group grant, or an applicable role grant.
- The user must belong to the same tenant as the document.
- The document must be active and its version eligible for retrieval.
- An explicit deny, if your authorization model supports denies, must override applicable allows.
- A revoked permission must stop granting access without waiting for re-embedding.

Do not accidentally combine access rules with `OR` conditions that bypass tenant or classification constraints. Tenant scope is a mandatory constraint around all permission alternatives.

# 3. Enforce document-level permissions at retrieval time

Imagine a user asks:

> “What does the advisory fee policy say about premium clients?”

The RAG service must first establish which documents that user is allowed to read. Only then should it perform similarity ranking over eligible chunks.

## 3.1 Build a trusted authorization context

The API can construct a context from a verified JWT and the server-side identity/authorization service.

```
from dataclasses import dataclassfrom uuid import UUID@dataclass(frozen=True)class AuthContext:    user_id: UUID    tenant_id: UUID    group_ids: tuple[str, ...]    role_ids: tuple[str, ...]# Illustrative: `identity` must come from a verified token# and trusted server-side membership lookup.async def build_auth_context(identity, membership_service):    groups, roles = await membership_service.get_memberships(        user_id=identity.user_id,        tenant_id=identity.tenant_id,    )    return AuthContext(        user_id=identity.user_id,        tenant_id=identity.tenant_id,        group_ids=tuple(groups),        role_ids=tuple(roles),    )
```

Do not decode a JWT without verifying its signature, issuer, audience, and expiry. Also, if group membership can be revoked while a JWT remains valid, do not rely indefinitely on old group claims: resolve membership from a trusted source or use a revocation/refresh strategy.

## 3.2 Filter by ACL inside the database query

The following PostgreSQL query illustrates direct user, group, and role grants. It also restricts retrieval to the active document version.

```
SELECT
    c.id,
    c.document_id,
    c.document_version,
    c.page_number,
    c.content,
    1 - (c.embedding <=> :query_embedding) AS similarity
FROM rag_chunks AS c
JOIN documents AS d
  ON d.id = c.document_id
 AND d.tenant_id = c.tenant_id
WHERE d.tenant_id = :tenant_id
  AND d.status = 'active'
  AND c.document_version = d.current_version
  AND EXISTS (
      SELECT 1
      FROM document_acl AS a
      WHERE a.document_id = d.id
        AND a.permission IN ('read', 'write', 'admin')
        AND (
            (
                a.principal_type = 'user'
                AND a.principal_id = CAST(:user_id AS TEXT)
            )
            OR (
                a.principal_type = 'group'
                AND a.principal_id = ANY(:group_ids)
            )
            OR (
                a.principal_type = 'role'
                AND a.principal_id = ANY(:role_ids)
            )
        )
  )
ORDER BY c.embedding <=> :query_embedding
LIMIT :top_k;
```

This is illustrative SQL; driver-specific array binding may need adjustment. Also, allowing `write` or `admin` to imply `read` is a policy decision, not a universal rule. Define that behavior explicitly.

Notice what the query does:

1. Requires the authenticated tenant.
2. Requires the document to be active.
3. Requires the chunk to belong to the current version.
4. Checks for a matching ACL grant.
5. Ranks only the eligible candidates.

A crucial detail: the ACL table above implements allow grants only. If you support explicit denies, inherited ACLs, document classification ceilings, or legal holds, those must be included in the authorization policy. Do not bolt on deny rules as an unrelated post-filter and assume all cases are covered.

## 3.3 Add PostgreSQL row-level security for defense in depth

RLS can protect document rows even if a developer accidentally forgets an application-level tenant predicate.

A simplified tenant isolation policy:

```
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

CREATE POLICY documents_tenant_isolation
ON documents
USING (
    tenant_id =
    NULLIF(current_setting('app.tenant_id', true), '')::UUID
);
```

Set the tenant context inside each database transaction:

```
from sqlalchemy import textasync def set_tenant_context(session, tenant_id):    await session.execute(        text(            "SELECT set_config("            "'app.tenant_id', :tenant_id, true)"        ),        {"tenant_id": str(tenant_id)},    )
```

The final `true` makes the setting transaction-local, which is important with pooled connections. Run the query and tenant-context setup within the same transaction.

This example only isolates tenant rows. It does not enforce document-level ACLs on its own, and the equivalent RLS policies must also be designed for `rag_chunks`, ACLs, versions, and other protected tables.

For RLS to provide meaningful defense, use a restricted application database role. Superusers and roles with `BYPASSRLS` can bypass policies; table ownership and connection configuration must also be handled carefully.

### What if the vector store is Qdrant instead of pgvector?

Store security metadata on every point, such as:

```
{
  "tenant_id": "tenant-a",
  "document_id": "policy-123",
  "document_version": 4,
  "status": "active"
}
```

Use a server-generated Qdrant filter for tenant, version, and lifecycle eligibility. For document-level ACLs, either materialize a carefully maintained access scope into payloads or retrieve authorized document IDs from the canonical authorization store and constrain the vector query accordingly.

Do not trust the client to send arbitrary Qdrant filters. And because ACL changes can race with an in-flight query, revalidate the returned evidence against the canonical authorization source before it enters the LLM context. That revalidation is a security check, not a substitute for pre-filtering.

# 4. How to handle document updates

An update is more complex than replacing a file. You must keep the original file, extracted text, chunks, embeddings, keyword index, caches, and document version consistent.

Consider a policy document moving from version 4 to version 5.

A naive workflow does this:

1. Delete version 4 chunks.
2. Parse the new file.
3. Generate embeddings.
4. Insert version 5 chunks.

What happens if embedding generation fails after step 1? The document becomes unavailable. What if only some chunks are inserted? The index contains incomplete evidence.

A stronger design uses versioned ingestion and an atomic activation step.

## 4.1 Update lifecycle

1\. Accept the update

Authorize the writer, verify file type and size, compute a content hash, and persist the source file.

2\. Create a new version

Allocate version 5, create an ingestion job, and retain version 4 as the currently active version.

3\. Process in the background

Extract text, tables, and images; chunk; embed; validate counts and provenance; populate the new version's index records.

4\. Validate the new version

Check completeness, embedding dimensions, extraction errors, duplicate chunks, and source references.

5\. Activate the version

Atomically update the canonical current-version pointer and lifecycle state in PostgreSQL.

6\. Reconcile and clean up

Invalidate caches, update external indexes, retire old records according to retention policy, and retry failed cleanup jobs.

There are two reasonable availability policies:

- Keep serving the old version until the new version is ready. This avoids downtime, but you must ensure the old version is still legally and operationally eligible.
- Immediately stop serving the old version. This is necessary when an urgent policy revocation or security incident makes continued use unacceptable. Mark the document ineligible immediately and allow the new version to become active only after validation.

The policy depends on the business meaning of the update. A routine correction and a revoked compliance policy should not automatically follow the same path.

## 4.2 Make ingestion asynchronous with an outbox

FastAPI should normally accept the update, persist the job, and return a job ID rather than performing a lengthy PDF parse and embedding operation in the HTTP request.

For reliable job dispatch, use the transactional outbox pattern:

```
CREATE TABLE outbox_events (
    id UUID PRIMARY KEY,
    aggregate_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ
);
```

In one PostgreSQL transaction:

1. Insert the new document version with status `processing`.
2. Insert an outbox event such as `document.version.requested`.
3. Commit the transaction.

A separate dispatcher publishes pending events to Celery or another queue. If the dispatcher crashes, the event remains in the outbox and can be retried. Workers must be idempotent because an event may be delivered more than once.

A simplified service operation:

```
import jsonfrom uuid import uuid4from sqlalchemy import textasync def request_document_update(    session,    *,    document_id,    auth,    content_hash,    object_uri,):    # `authorize_write` must check tenant and document ACL.    await authorize_write(        session,        document_id=document_id,        auth=auth,    )    async with session.begin():        # Lock the canonical record to serialize competing updates.        doc = (            await session.execute(                text("""                    SELECT id, current_version, status                    FROM documents                    WHERE id = :document_id                      AND tenant_id = :tenant_id                    FOR UPDATE                """),                {                    "document_id": document_id,                    "tenant_id": auth.tenant_id,                },            )
```

This illustrates the transaction boundary, not a complete drop-in implementation. In particular:

- `authorize_write` must enforce authorization within a safe transaction strategy, with concurrent ACL revocations considered.
- `session.begin()` requires a clean transaction state; configure the session lifecycle accordingly.
- Version allocation should be robust against failed prior versions and competing jobs. In a mature implementation, use a dedicated monotonic version allocator or unique constraints plus retry logic.
- A document can have several ingestion attempts for the same version. Track attempt IDs or job IDs separately instead of treating a version number as an attempt identifier.
- Object storage writes are not part of the PostgreSQL transaction. Use staging object keys and cleanup/reconciliation for abandoned uploads.

## 4.3 Build new chunks without exposing them

The worker should insert chunks tagged with the new version. Retrieval should continue to use only the canonical `current_version`.

```
async def persist_staged_chunks(    session,    *,    document_id,    tenant_id,    version,    chunks,):    async with session.begin():        for chunk in chunks:            await insert_chunk(                session,                document_id=document_id,                tenant_id=tenant_id,                version=version,                content=chunk.content,                embedding=chunk.embedding,                page_number=chunk.page_number,            )        await validate_staged_version(            session,            document_id=document_id,            version=version,            expected_chunk_count=len(chunks),        )
```

The helpers are repository methods that must be implemented with bound parameters and appropriate uniqueness constraints. In a high-volume pipeline, use batch inserts instead of one database round trip per chunk.

For a pgvector-only implementation, staging chunks and switching the canonical pointer can happen within coordinated PostgreSQL transactions. For a separate vector database, PostgreSQL and the vector store cannot share one atomic transaction, so you need a staged-index workflow, activation marker, and reconciliation strategy.

## 4.4 Activate with a compare-and-swap update

Do not allow two concurrent update workers to overwrite one another's active version.

Before activation, validate that the staged version is complete and that it is still the version intended to be activated. Then update the pointer using a guarded transaction.

```
UPDATE documents
SET current_version = :new_version,
    status = 'active',
    updated_at = now()
WHERE id = :document_id
  AND tenant_id = :tenant_id
  AND current_version = :expected_old_version
  AND status IN ('active', 'updating');
```

Check that exactly one row was updated. If zero rows were updated, another update or deletion may have won the race; the worker must stop and reconcile instead of activating stale work.

In the same transaction, mark the new version as active and the previous version as superseded. The actual allowed state transitions should be encoded and validated consistently.

Important distinction: switching the PostgreSQL pointer is not the same as atomically switching an external vector index. For Qdrant, you can stage new points under a version-specific namespace or payload, then coordinate activation through a canonical manifest. Every query must obey the active manifest, and a reconciliation worker must handle partial failures.

# 5. How to handle document deletion

Deletion is often the harder lifecycle operation because a document can exist in multiple places:

- Original PDF in object storage.
- Extracted text and table rows.
- Embeddings in pgvector or Qdrant.
- Keyword-search index.
- Parent documents and child chunks.
- Image descriptions and OCR results.
- Redis caches and previously assembled RAG context.
- Logs, evaluation datasets, and backup copies.

The system needs a deletion policy that accounts for all of these, including legal holds and retention requirements.

## 5.1 Logical deletion first, physical cleanup second

For a normal authorized deletion, I recommend a tombstone workflow:

1. Authorize the deletion request.
2. Mark the document as `deleting` in the canonical database.
3. Make all retrieval queries exclude it immediately.
4. Write a durable deletion event in the same database transaction.
5. Purge derived indexes, cached answers, and stored assets asynchronously.
6. Verify cleanup and mark the lifecycle operation complete.
7. Record an audit event without retaining unnecessary deleted content.

This separates two different guarantees:

- Access revocation: the document must no longer be returned by retrieval.
- Physical erasure: the content and its derived copies must be removed according to the applicable retention policy.

The first must happen promptly and cannot depend on a background worker eventually succeeding.

## 5.2 Implement a tombstone transaction

```
from uuid import uuid4import jsonfrom sqlalchemy import textasync def request_document_deletion(    session,    *,    document_id,    auth,):    await authorize_delete(        session,        document_id=document_id,        auth=auth,    )    async with session.begin():        result = await session.execute(            text("""                UPDATE documents                SET status = 'deleting',                    deleted_at = COALESCE(deleted_at, now()),                    updated_at = now()                WHERE id = :document_id                  AND tenant_id = :tenant_id                  AND status <> 'deleted'                RETURNING id            """),            {                "document_id": document_id,                "tenant_id": auth.tenant_id,            },        )        deleted_record = result.first()
```

The `authorize_delete` operation must verify the tenant and deletion permission. In production, coordinate authorization and mutation so a concurrent ACL revocation or document deletion cannot create a time-of-check/time-of-use gap. The update and outbox event must commit together.

Once the transaction commits, retrieval must reject the document because the canonical document status is no longer `active`.

The example uses `deleted_at` as a tombstone timestamp, but for strict lifecycle management I would also keep a separate deletion job record with its own status, attempt count, timestamps, and failure reason. That makes retries and operational monitoring much easier.

## 5.3 Delete from pgvector

A cleanup worker can remove chunks by document ID:

```
DELETE FROM rag_chunks
WHERE document_id = :document_id;
```

If you retain versions separately, delete the relevant chunks by document and version according to the retention policy.

Then remove version records and original objects only when permitted by retention, legal hold, and audit requirements.

For Qdrant, use a server-side delete-by-filter for the document ID, scoped to the correct tenant and collection. For an external keyword index, issue the equivalent delete-by-document operation.

Do not assume that a successful PostgreSQL deletion also deletes Qdrant points, cached responses, image objects, or source files. Each system needs an explicit cleanup step.

## 5.4 Make cleanup idempotent

A worker may run twice because a queue redelivers a task, a process crashes after deletion but before acknowledging the job, or a network timeout leaves the result uncertain.

Cleanup should therefore be safe to repeat.

```
async def cleanup_document(    *,    document_id,    postgres_repository,    vector_index,    keyword_index,    cache,    object_store,):    # Each operation should be safe to retry.    await cache.invalidate_document(document_id)    await keyword_index.delete_document(document_id)    await vector_index.delete_document(document_id)    await postgres_repository.delete_chunks(document_id)    # This method must check legal holds and retention policy.    await object_store.delete_document_assets_if_permitted(        document_id    )
```

This is a conceptual orchestration example. In real deployments, wrap each external operation in bounded retries and record its completion independently. Do not simply mark the job successful because one operation succeeded.

For example, the vector index could be unavailable while PostgreSQL cleanup succeeds. The deletion job should remain incomplete, retry the vector cleanup, and continue to block the document from retrieval during the entire process.

A deletion job record might track:

| Step                     | Status                      |
| ------------------------ | --------------------------- |
| Canonical tombstone      | Complete                    |
| Redis/cache invalidation | Complete                    |
| Vector index cleanup     | Retry pending               |
| Keyword index cleanup    | Complete                    |
| Object storage cleanup   | Waiting for retention check |
| Final verification       | Pending                     |

This gives operations teams a way to prove which cleanup steps have completed and which remain outstanding.

# 6. Cache invalidation and stale-context risks

A common enterprise mistake is to secure the vector query but leave an unsafe Redis cache in front of it.

Imagine a compliance officer retrieves a confidential policy. The system caches the answer:

```
"What is the policy's exception?" → "The exception is ..."
```

Later, the officer's permission is revoked. If another user can retrieve that cached answer by question alone, the authorization boundary has failed.

## 6.1 Design cache keys around authorization

Avoid this:

```
cache_key = f"rag:{question}"
```

That key ignores the tenant, user scope, document version, and permission changes.

A safer pattern is to include a stable authorization-scope fingerprint and a version that changes when permissions change:

```
import hashlibimport jsondef make_rag_cache_key(    *,    tenant_id: str,    question: str,    filters: dict,    authz_version: str,    retrieval_version: str,) -> str:    payload = {        "tenant_id": tenant_id,        "question": question.strip(),        "filters": filters,        "authz_version": authz_version,        "retrieval_version": retrieval_version,    }    canonical = json.dumps(        payload,        sort_keys=True,        separators=(",", ":"),    )    digest = hashlib.sha256(        canonical.encode("utf-8")    ).hexdigest()    return f"rag:v2:{tenant_id}:{digest}"
```

The `authz_version` must be sourced from a trusted authorization service or authorization database. It must change when relevant permissions change. A user ID alone is not enough if group memberships or document grants can be revoked.

Even with this key, reauthorize cached evidence before returning it, or use a robust invalidation strategy that prevents revoked evidence from being served. Cache keys reduce collisions; they do not independently guarantee access control.

For strict security, I generally prefer caching retrieval candidates or reusable non-sensitive computation with explicit authorization checks, rather than treating a generated answer as universally reusable.

Also account for in-flight requests: a request may have retrieved evidence just before a permission revocation. Define the revocation consistency requirement for your application and revalidate authorization at the appropriate boundary before returning sensitive output.

# 7. What changes when you use Qdrant, Celery, and LangGraph?

For your enterprise financial compliance platform, I would assign responsibilities like this:

| Component          | Responsibility                                                             |
| ------------------ | -------------------------------------------------------------------------- |
| FastAPI            | Authentication, authorization, update/delete APIs, job status              |
| PostgreSQL         | Canonical documents, ACLs, versions, lifecycle state, audit events, outbox |
| pgvector or Qdrant | Searchable embeddings; never the sole authority for permissions            |
| Redis              | Cache and transient coordination; not the source of truth for ACLs         |
| Celery             | Background ingestion, re-indexing, cleanup, retries                        |
| Object storage     | Original documents, extracted assets, staged files                         |
| LangGraph          | Orchestrates reasoning and review workflows using authorized evidence      |
| OpenTelemetry      | Traces API requests, workers, retrieval, and cleanup failures              |

Two important rules:

- LangGraph should receive only authorized evidence. Do not rely on a prompt or agent to filter unauthorized documents after retrieval.
- Celery tasks must be idempotent. Workers may execute more than once; duplicate ingestion or deletion events must not corrupt the active document state.

For an in-flight LangGraph workflow, also consider checkpoints and persisted state. A workflow might have loaded a confidential chunk before the permission was revoked. On resume, revalidate authorization before reusing persisted evidence. If the source document has been deleted or is no longer authorized, invalidate or remove the affected evidence from the workflow state according to your retention requirements.

## 8. Testing the security and lifecycle behavior

I would create automated integration tests for at least these scenarios:

## Production readiness checklist

0 of 10

Cross-tenant isolation

A user from tenant B cannot retrieve tenant A's document.

Direct ACL grant

An explicitly authorized user can retrieve the document.

Revoked ACL

A removed user/group grant no longer allows retrieval or cached access.

Version activation

New chunks are not searchable before activation; only the active version is returned.

Failed ingestion

An embedding or parsing failure does not expose a partial new version.

Deletion tombstone

A deleting document is immediately excluded from all retrieval paths.

Retry safety

Repeated cleanup events do not recreate or corrupt deleted data.

Cache isolation

A cached answer cannot bypass changed permissions or document lifecycle state.

Worker race

A stale update worker cannot reactivate a document after deletion.

RLS bypass attempts

Application queries cannot access another tenant's records when a filter is omitted.

Copy checklist

These tests should run against real PostgreSQL/RLS and your actual vector backend, not only mocked repositories. Include concurrent update/delete tests, queue redelivery, and fault injection where external services fail halfway through cleanup.

## 9. Staff-level design decisions

At staff level, the important work is defining consistency and failure behavior across services.

| Decision                   | Recommended approach                                                      |
| -------------------------- | ------------------------------------------------------------------------- |
| Source of truth            | PostgreSQL for ACLs, lifecycle state, and active version                  |
| Permission revocation      | Enforce at retrieval; version/invalidate relevant caches                  |
| Document updates           | Stage new version, validate, then activate with a guarded transaction     |
| External index consistency | Transactional outbox, idempotent workers, reconciliation                  |
| Deletion                   | Immediate tombstone, asynchronous physical cleanup, verifiable completion |
| Security defense in depth  | Application authorization plus carefully configured database RLS          |
| Auditability               | Record who changed permissions, activated versions, or requested deletion |
| Recovery                   | Durable job state, retries, dead-letter handling, and operational alerts  |

There is no universal transaction that atomically commits PostgreSQL, Redis, object storage, and Qdrant together. A robust design acknowledges that limitation and provides immediate logical safety, durable progress, idempotent cleanup, and reconciliation.

For a regulated system, I would also define the deletion and retention contract with the security, legal, and compliance teams. Some records may need to be retained for audit or legal hold, while their normal retrieval access is revoked. Physical deletion from backups may follow a separate retention schedule, and the system must be clear about that distinction.

The interview takeaway is:

> “I use a canonical document registry with explicit ACLs and versioned lifecycle state. Authorization is enforced at the retrieval boundary and reinforced at the database layer. Updates use staged ingestion and guarded version activation. Deletion uses a tombstone to revoke access immediately, followed by durable asynchronous cleanup across all derived stores. An outbox, idempotent workers, cache invalidation, audit logs, and reconciliation handle cross-service failures. This prevents unauthorized retrieval and avoids pretending that multiple independent storage systems share one atomic transaction.”