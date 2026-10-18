# How to prevent cross-tenant document retrieval in a production RAG system

As a senior or staff AI engineer, I treat this as a multi-tenant data isolation and authorization problem, not just a vector-search filtering problem.

Imagine a financial institution has two tenants:

- Tenant A: `bank_a`
- Tenant B: `bank_b`

Tenant A asks: “What are the current lending policies?”

If Qdrant returns a highly relevant document belonging to Tenant B, the system must prevent that document from reaching the LLM. Otherwise, Tenant A could receive confidential information from another organization—even if the final answer doesn't explicitly cite the document.

The critical principle is:

Enforce tenant isolation before retrieval results enter the application’s trusted context, and enforce it again at every data-access boundary. Never rely on the LLM to protect tenant data.

## 1. Defense-in-depth architecture

Authenticated user

JWT or OIDC identity

Authentication + tenant authorization

Resolve tenant from trusted identity and membership records

Tenant-scoped retrieval service

Mandatory filters; no caller-controlled tenant override

Qdrant

Tenant filter on every search

PostgreSQL

Tenant-scoped records and optional RLS

Authorization validation

Validate every candidate before it enters the prompt

LLM generation

Receives only authorized evidence

I would enforce isolation in at least four places:

1. Identity and tenant resolution.
2. Vector database search.
3. Application and database authorization.
4. Context construction and downstream tool execution.

This way, a bug in one layer is less likely to expose another tenant's data.

## 2. Never trust the tenant ID sent by the client

One of the most common security mistakes is accepting the tenant ID directly from a request body.

Unsafe example:

```
from pydantic import BaseModelfrom uuid import UUIDclass SearchRequest(BaseModel):    tenant_id: UUID  # Dangerous if trusted without authorization    question: str
```

A user could change `tenant_id` in the HTTP request and attempt to retrieve another tenant's documents.

Instead, resolve the tenant from the authenticated identity and verify the user's membership. For example, a user may belong to multiple organizations, so the selected organization must be checked against trusted membership records.

```
from dataclasses import dataclassfrom uuid import UUIDfrom fastapi import Depends, HTTPException@dataclass(frozen=True)class TenantContext:    user_id: UUID    tenant_id: UUID    roles: frozenset[str]async def get_tenant_context(    principal=Depends(authenticate_user),    membership_service=Depends(get_membership_service),) -> TenantContext:    # The principal must come from validated authentication.    # The active tenant must be resolved from trusted membership.    tenant_id = await membership_service.resolve_authorized_tenant(        user_id=principal.user_id,        requested_tenant=principal.selected_tenant,    )    if tenant_id is None:        raise HTTPException(status_code=403, detail="Access denied")    return TenantContext(        user_id=principal.user_id,        tenant_id=tenant_id,        roles=frozenset(principal.roles),    )
```

This is illustrative code: `authenticate_user` must validate the token's signature, issuer, audience, expiry, and other applicable claims. `resolve_authorized_tenant` must verify active membership and any relevant account restrictions.

The important distinction is:

- Authentication: Who is the user?
- Tenant authorization: Which organization's data may the user access?
- Document authorization: Which documents within that organization may the user access?

A valid JWT alone does not prove that the user is authorized for every document.

## 3. Enforce tenant filtering inside Qdrant

Every vector point should contain mandatory tenant metadata.

For example:

```
{    "tenant_id": "tenant-a-uuid",    "document_id": "policy-123",    "version_id": "version-42",    "chunk_index": 0,    "classification": "internal"}
```

Create an index for the tenant payload field as part of your collection setup. For a UUID represented as a string, a keyword payload index is commonly appropriate.

```
from qdrant_client import QdrantClient, modelsclient = QdrantClient(url="http://localhost:6333")client.create_payload_index(    collection_name="financial_documents",    field_name="tenant_id",    field_schema=models.PayloadSchemaType.KEYWORD,)
```

In a real deployment, create indexes through an idempotent migration or provisioning step rather than attempting to recreate them on every request.

### Secure vector search

```
from qdrant_client import modelsasync def search_tenant_documents(    *,    tenant: TenantContext,    query_vector: list[float],    limit: int = 10,):    tenant_filter = models.Filter(        must=[            models.FieldCondition(                key="tenant_id",                match=models.MatchValue(                    value=str(tenant.tenant_id)                ),            )        ]    )    response = await qdrant_client.query_points(        collection_name="financial_documents",        query=query_vector,        query_filter=tenant_filter,        limit=limit,        with_payload=True,    )    # Defense in depth: validate the ownership of returned candidates.    results = []    for point in response.points:        payload = point.payload or {}        if payload.get("tenant_id") != str(tenant.tenant_id):            raise SecurityError("Cross-tenant vector returned")        results.append(point)
```

The code assumes an asynchronous Qdrant client is configured as `qdrant_client`; use the matching API for your installed client version.

Notice that the tenant filter is constructed inside the trusted retrieval service. The caller cannot pass an arbitrary filter that removes the tenant restriction.

Never do this:

```
# Unsafe: the client can choose the tenant.await search_documents(    tenant_id=request.tenant_id,    question=request.question,)
```

Instead, make the tenant context a required parameter derived from authorization. The retrieval service should always add its own tenant constraint, even when the caller supplies additional filters such as document type or date range.

### What about metadata filters being wrong or missing?

Treat missing tenant metadata as a security failure. Do not interpret it as a public document.

For ingestion, validate that every point has a tenant ID, document ID, version ID, and other required fields before upserting. Quarantine malformed records rather than indexing them.

For higher isolation requirements, use separate collections or separate vector databases for tenants. A shared collection with mandatory filters can work well, but it requires stronger testing and operational controls.

## 4. Protect PostgreSQL with tenant isolation too

Filtering Qdrant is not sufficient. Your application may also fetch document text, metadata, conversation history, citations, or permissions from PostgreSQL.

Every query must be tenant-scoped.

```
SELECT document_id, title, content
FROM documents
WHERE tenant_id = :tenant_id
  AND document_id = :document_id;
```

Never fetch a record by `document_id` alone and assume its UUID is secret or unguessable. IDs are identifiers, not authorization controls.

### Add PostgreSQL Row-Level Security (RLS)

For sensitive multi-tenant systems, I would consider PostgreSQL RLS as a second enforcement layer.

```
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

CREATE POLICY documents_tenant_isolation
ON documents
USING (
    tenant_id =
    NULLIF(current_setting('app.tenant_id', true), '')::uuid
)
WITH CHECK (
    tenant_id =
    NULLIF(current_setting('app.tenant_id', true), '')::uuid
);
```

At the beginning of each database transaction, set the tenant context:

```
from sqlalchemy import textasync def get_tenant_documents(    session,    tenant: TenantContext,):    async with session.begin():        await session.execute(            text("SELECT set_config('app.tenant_id', :tenant_id, true)"),            {"tenant_id": str(tenant.tenant_id)},        )        result = await session.execute(            text("SELECT document_id, title FROM documents")        )        return result.mappings().all()
```

The third argument to `set_config` makes the setting transaction-local. This is important with pooled connections: tenant context must not leak between requests.

Important production details:

- Use a restricted application database role; don't run the application as a superuser.
- PostgreSQL superusers and roles with `BYPASSRLS` can bypass RLS. Table owners also normally bypass RLS unless `FORCE ROW LEVEL SECURITY` is enabled.
- Apply equivalent policies to all tenant-owned tables, not just `documents`.
- Keep explicit tenant predicates in application queries even when RLS is enabled.
- Ensure every tenant-scoped database operation runs inside the correct transaction and context.

RLS is defense in depth, not a substitute for correct authentication. If the application can arbitrarily set the tenant context, its authorization layer must still prevent unauthorized tenant selection.

## 5. Validate authorization before constructing the LLM prompt

A subtle but serious failure happens when retrieval is correctly filtered but a later stage accidentally fetches documents by ID without checking tenant ownership.

For example:

1. Qdrant returns authorized chunks.
2. The application collects document IDs.
3. A database query fetches full document content using only those IDs.
4. The LLM receives content that was not authorized.

The secure approach is to validate ownership and permissions at every boundary where data is loaded.

```
async def build_authorized_context(    *,    tenant: TenantContext,    candidates: list,    document_repository,    authorization_service,):    context_chunks = []    for candidate in candidates:        payload = candidate.payload or {}        if payload.get("tenant_id") != str(tenant.tenant_id):            raise SecurityError("Tenant mismatch")        document_id = payload.get("document_id")        if not document_id:            continue        # Query using BOTH tenant and document ID.        document = await document_repository.get_for_tenant(            tenant_id=tenant.tenant_id,            document_id=document_id,        )        if document is None:            continue        allowed = await authorization_service.can_read(            user_id=tenant.user_id,            tenant_id=tenant.tenant_id,            document_id=document_id,        )        if not allowed:            continue
```

This example illustrates the checks; production code should batch database and authorization lookups rather than issuing one request per candidate.

There are three independent checks here:

- Tenant ownership.
- User/document-level permission.
- Active document version.

A document can belong to the correct tenant but still be restricted to a particular department, user, or role.

Also, avoid using the vector payload's text blindly. Depending on your design, fetch canonical chunk content from an authorized repository or verify that the indexed content belongs to the active version.

## 6. Secure caches, conversation memory, and agent tools

Cross-tenant leaks don't only happen in vector search.

| Component           | Common mistake                             | Protection                                                                     |
| ------------------- | ------------------------------------------ | ------------------------------------------------------------------------------ |
| Redis cache         | Key uses only the question                 | Include tenant, user/permission scope, query, and relevant version identifiers |
| Conversation memory | Reuse history from another tenant          | Scope memory by tenant, user, and conversation                                 |
| RAG citations       | Load source by document ID only            | Authorize every citation lookup                                                |
| LangGraph state     | Resume a checkpoint under another tenant   | Bind checkpoint/thread ownership to the authenticated tenant and user          |
| Agent tools         | Tool accepts arbitrary tenant ID           | Derive tenant context from trusted execution context                           |
| SQL generation      | Generated SQL accesses unrestricted tables | Use a restricted database role, enforce tenant isolation, validate queries     |
| Logs and traces     | Record complete document text              | Redact sensitive content and restrict observability access                     |

For example, a cache key might be:

```
import hashlibdef rag_cache_key(    tenant_id: str,    authorization_scope_version: str,    query: str,) -> str:    query_hash = hashlib.sha256(        query.encode("utf-8")    ).hexdigest()    return (        f"rag:{tenant_id}:"        f"{authorization_scope_version}:{query_hash}"    )
```

The authorization scope version must change when relevant permissions change. For systems with document-level permissions, include the relevant ACL version or use a cache invalidation strategy that reliably accounts for ACL updates. A tenant ID alone does not protect against users within the same tenant who have different permissions.

For LangGraph, do not treat a client-provided thread ID as proof of ownership. Verify that the thread and its checkpoint belong to the authenticated tenant before loading or resuming the workflow.

For agents, tenant identity should be part of a trusted execution context passed to every tool. The LLM should never be able to override that context by generating arguments such as `"tenant_id": "tenant-b"`.

## 7. What if a cross-tenant document still appears in search results?

I would fail closed rather than silently pass the result to the LLM.

For example, if Qdrant returns a point with the wrong tenant ID:

1. Do not add its text to the prompt.
2. Emit a security metric and a redacted audit event.
3. Investigate whether the problem is bad metadata, an incorrect filter, a shared-cache bug, or an authorization failure.
4. Consider disabling the affected retrieval path if the isolation guarantee cannot be trusted.
5. Repair or quarantine affected vectors and verify the collection before restoring service.

Don't log the full document text, embedding, access token, or sensitive query merely to diagnose the issue. Log identifiers and relevant security metadata under appropriate access controls.

A mismatch should not be treated as an ordinary low-quality retrieval result.

## 8. Test tenant isolation like a security boundary

I would add automated tests at the API, retrieval, repository, and workflow levels.

| Test                                                     | Expected result                                                    |
| -------------------------------------------------------- | ------------------------------------------------------------------ |
| Tenant A searches for a common query                     | Only authorized Tenant A documents are returned                    |
| Tenant A submits Tenant B's document ID                  | No content is disclosed                                            |
| Tenant A modifies the tenant ID in the request           | Request is rejected or the unauthorized tenant selection is denied |
| Qdrant returns a malformed point without tenant metadata | Point is rejected; security event is recorded                      |
| Redis contains a result cached for Tenant B              | Tenant A cannot receive it                                         |
| User loses document access                               | Subsequent retrieval and citation lookups deny access              |
| An old LangGraph checkpoint belongs to another tenant    | Resume is denied                                                   |
| A tool attempts to change the trusted tenant context     | Override is rejected                                               |
| A document is deleted or its version is superseded       | It is no longer eligible for retrieval                             |

For the most important tests, use two seeded tenants with deliberately similar documents. Assert not just that the final answer is correct, but that the unauthorized text never enters the retrieval context sent to the LLM.

## 9. What isolation strategy should you choose?

Option A — Shared Qdrant collection with mandatory tenant filters

Good default

Cost-efficient and operationally simpler. Appropriate when tenant filters are enforced centrally, every data path is tested, and authorization checks are reliable.

Option B — Separate collections per tenant

Provides a stronger organizational boundary and can simplify tenant-specific retention or index configuration. However, collection selection must still be authorized, and operational complexity grows with the number of tenants.

Option C — Separate database or infrastructure per tenant

Appropriate when contractual, regulatory, or risk requirements demand stronger isolation. It increases deployment, monitoring, capacity, and maintenance costs.

For a multi-tenant financial RAG platform, I would usually start with a shared collection plus mandatory server-side filters, PostgreSQL tenant isolation, and strict retrieval validation. For highly sensitive tenants or contractual isolation requirements, I would evaluate separate collections or infrastructure.

## Interview-ready answer

> “I prevent cross-tenant retrieval through defense in depth. The tenant is derived from a validated identity and authorized membership, never trusted from the request body. Every Qdrant query includes a mandatory server-side tenant filter, and every PostgreSQL query is tenant-scoped, with RLS as an additional control where appropriate. Before constructing the LLM context, I validate tenant ownership, document-level permissions, and active document versions. I also scope Redis caches, LangGraph checkpoints, citations, and agent tools by tenant and authorization context. If any ownership check fails, the document is excluded and a security event is recorded. Finally, I test adversarial tenant-ID changes and verify that unauthorized document content never reaches the LLM.”

Staff-level takeaway: tenant isolation must be enforced end to end—from identity, to retrieval, to cached data, to workflow state, to the final prompt. A Qdrant filter is essential, but it is not the entire security architecture.