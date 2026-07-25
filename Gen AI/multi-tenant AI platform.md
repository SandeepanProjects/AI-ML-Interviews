A **multi-tenant AI platform** is an AI system where **one deployment serves multiple customers (tenants)** while keeping each customer's **data, users, models, prompts, API keys, and usage completely isolated**.

Examples of multi-tenant AI platforms include enterprise AI SaaS products where companies such as banks, hospitals, retailers, and law firms all use the same backend but cannot access each other's data.

---

# What is a Tenant?

A tenant is typically:

* One company
* One organization
* One customer
* One workspace

Example:

```
AI Platform

├── Tenant A (Google)
│      Users
│      Documents
│      Vector DB
│      Models
│
├── Tenant B (Microsoft)
│      Users
│      Documents
│      Vector DB
│
└── Tenant C (Amazon)
       Users
       Documents
       Vector DB
```

All tenants use the same application, but their resources remain isolated.

---

# Why Multi-Tenancy?

Without multi-tenancy:

```
Customer A
   ↓
Dedicated AI Server

Customer B
   ↓
Dedicated AI Server

Customer C
   ↓
Dedicated AI Server
```

Problems:

* Expensive
* Hard to maintain
* Difficult to scale
* Separate deployments for every customer

Instead:

```
               AI Platform

      ┌──────────────────────┐
      │ Authentication        │
      │ Billing              │
      │ RAG                  │
      │ Agents               │
      │ APIs                 │
      └─────────┬────────────┘
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
   Tenant A  Tenant B  Tenant C
```

One platform serves everyone securely.

---

# High-Level Architecture

```
                     Internet
                         │
                  Load Balancer
                         │
                    API Gateway
                         │
                  Authentication
                         │
                  Tenant Resolver
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Agent Service            RAG Service
             │                       │
             └──────────┬────────────┘
                        ▼
               PostgreSQL Database
                        │
                        ▼
                    Redis Cache
                        │
                        ▼
                    Vector Database
```

---

# Core Components

## 1. Authentication

Every request must identify:

* User
* Organization
* Roles
* Permissions
* Tenant

Example JWT:

```json
{
    "user_id": "u123",
    "tenant_id": "company_xyz",
    "role": "admin"
}
```

Every API extracts the tenant:

```python
def get_tenant(jwt):
    return jwt["tenant_id"]
```

---

# 2. Tenant Middleware

Every request is tagged with its tenant.

```python
from fastapi import Request

async def tenant_middleware(request: Request, call_next):

    tenant = request.headers["X-Tenant-ID"]

    request.state.tenant = tenant

    return await call_next(request)
```

Every downstream service uses:

```
request.state.tenant
```

---

# 3. PostgreSQL Design

Every table includes a tenant identifier.

```
Users

id

tenant_id

name

email
```

```
Documents

id

tenant_id

title

content
```

```
Chats

id

tenant_id

conversation
```

Query:

```sql
SELECT *
FROM documents
WHERE tenant_id = 'company_xyz';
```

Never query without filtering by tenant.

---

# 4. Row-Level Security (RLS)

PostgreSQL Row-Level Security can enforce isolation even if an application bug occurs.

```
Tenant A

↓

Database

↓

Returns only Tenant A rows
```

Example policy:

```sql
CREATE POLICY tenant_isolation
ON documents
USING (tenant_id = current_setting('app.tenant_id'));
```

The application sets:

```sql
SET app.tenant_id='company_xyz';
```

This adds a database-level safeguard.

---

# 5. Vector Database Isolation

Each tenant's embeddings should be isolated.

```
Qdrant

Collection

↓

company_a

↓

Vectors
```

```
company_b

↓

Vectors
```

Or use payload filtering:

```json
{
  "tenant_id": "company_a"
}
```

Search:

```python
search(
    query_vector,
    filter={
        "tenant_id": "company_a"
    }
)
```

---

# 6. Redis Caching

Always namespace cache keys.

Bad:

```
chat:123
```

Good:

```
company_a:chat:123

company_b:chat:123
```

Example:

```python
key = f"{tenant_id}:chat:{chat_id}"
```

---

# 7. Agent Memory

Memory must also be tenant-aware.

```
Memory

↓

Tenant

↓

User

↓

Conversation
```

Key:

```
tenant:user:conversation
```

---

# 8. Storage

```
S3

↓

company_a/

documents/

images/
```

```
company_b/

documents/
```

Use tenant-specific prefixes or buckets and enforce access controls.

---

# 9. Model Configuration

Different tenants may require different models.

```
Tenant A

GPT-5.5
```

```
Tenant B

Claude
```

```
Tenant C

Llama
```

Configuration example:

```json
{
    "tenant": "company_a",
    "model": "gpt-5.5",
    "temperature": 0.2
}
```

The platform selects the appropriate model at runtime.

---

# 10. Rate Limiting

Per-tenant limits prevent one customer from impacting others.

```
Tenant A

1000 RPM
```

```
Tenant B

500 RPM
```

```
Tenant C

200 RPM
```

Track usage by tenant ID rather than only by IP address.

---

# 11. Observability

Every log, metric, and trace should include the tenant.

```python
logger.info(
    "retrieval_complete",
    extra={
        "tenant": tenant_id,
        "user": user_id,
        "latency_ms": 180,
    },
)
```

Example dashboard:

```
Tenant A

Requests      18,000

Latency       190 ms

Cost          ₹12,000
```

```
Tenant B

Requests      2,100

Latency       140 ms

Cost          ₹1,850
```

---

# 12. Security

Each tenant has:

* Separate API keys
* RBAC (roles such as Admin, Editor, Viewer)
* Optional SSO (SAML/OIDC)
* Encryption at rest and in transit
* Audit logs
* Data retention policies

Every tool invocation should verify that the authenticated user belongs to the active tenant.

---

# Production Architecture

```
                   Users
                     │
              Load Balancer
                     │
               API Gateway
                     │
           Authentication (JWT/SSO)
                     │
              Tenant Resolver
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
   Agent Service  RAG Service  Admin Service
        │            │             │
        └──────┬─────┴─────────────┘
               ▼
         PostgreSQL (RLS)
               │
         Redis (Tenant Keys)
               │
      Vector DB (Tenant Filter)
               │
      Object Storage (Tenant Prefix)
               │
     LLM Providers / Self-Hosted Models
               │
        Observability Stack
 (OpenTelemetry + Prometheus + Grafana)
```

---

# Database Isolation Strategies

| Strategy                                            | Isolation | Cost      | Typical Use                                            |
| --------------------------------------------------- | --------- | --------- | ------------------------------------------------------ |
| Shared database, shared tables (`tenant_id` column) | Moderate  | Low       | SaaS with many small tenants                           |
| Shared database, separate schema per tenant         | High      | Medium    | Mid-sized enterprise SaaS                              |
| Separate database per tenant                        | Very High | High      | Finance, healthcare, government                        |
| Dedicated infrastructure per tenant                 | Maximum   | Very High | Customers with strict compliance or custom deployments |

---

# Real-World Example: Enterprise AI Copilot

Imagine you build an AI Copilot platform used by three companies:

| Component    | Company A       | Company B       | Company C         |
| ------------ | --------------- | --------------- | ----------------- |
| Users        | 500             | 120             | 2,000             |
| Documents    | Private HR docs | Legal contracts | Financial reports |
| Vector Index | Tenant-filtered | Tenant-filtered | Tenant-filtered   |
| LLM          | GPT-5.5         | Claude          | Self-hosted Llama |
| Cache        | `company_a:*`   | `company_b:*`   | `company_c:*`     |
| Rate Limit   | 2,000 RPM       | 500 RPM         | 5,000 RPM         |

Although all three companies use the same platform, every request carries a `tenant_id`, every database query, cache lookup, vector search, and audit log is scoped to that tenant, ensuring strong isolation while sharing the underlying infrastructure.

This is the architecture commonly used by production AI SaaS platforms because it balances **security**, **scalability**, **cost efficiency**, and **operational simplicity**.
