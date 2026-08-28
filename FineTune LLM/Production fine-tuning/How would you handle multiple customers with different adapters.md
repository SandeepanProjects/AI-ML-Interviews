# How would you handle multiple customers with different LoRA adapters?

This is a **multi-tenant LoRA serving** problem.

Suppose your SaaS application has three customers:

```text
Customer A → Customer-specific LoRA A
Customer B → Customer-specific LoRA B
Customer C → Customer-specific LoRA C
```

But you don't want:

```text
Customer A → Full 8B model
Customer B → Full 8B model
Customer C → Full 8B model
```

That would be expensive.

Instead:

```text
                    Shared Base Model
                         (GPU)
                           │
           ┌───────────────┼────────────────┐
           │               │                │
           ▼               ▼                ▼
      Customer A       Customer B       Customer C
       LoRA A           LoRA B           LoRA C
```

The architecture is:

```text
Base Model
+
Tenant-Specific LoRA Adapter
=
Tenant-Specific AI Behavior
```

---

# 1. The correct production architecture

A production system might look like this:

```text
                         Client
                           │
                           ▼
                    API Gateway
                           │
                    Authentication
                           │
                           ▼
                     JWT Token
                           │
                           ▼
                  Extract tenant_id
                           │
                           ▼
                   Tenant Service
                           │
                           ▼
                 Adapter Authorization
                           │
                           ▼
                   Adapter Registry
                           │
                           ▼
                  LoRA Inference Server
                           │
                    Shared Base Model
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
              LoRA A    LoRA B    LoRA C
```

The key principle is:

> **The client should not decide which LoRA adapter it can use. The server determines the adapter from the authenticated customer's identity.**

---

# 2. Tenant → adapter mapping

For example:

```python
TENANT_ADAPTERS = {
    "customer_a": "customer_a_adapter",
    "customer_b": "customer_b_adapter",
    "customer_c": "customer_c_adapter",
}
```

Then:

```python
def get_adapter_for_tenant(
    tenant_id: str
) -> str:

    adapter = TENANT_ADAPTERS.get(
        tenant_id
    )

    if adapter is None:

        raise ValueError(
            "No adapter configured for tenant"
        )

    return adapter
```

Flow:

```text
JWT
 │
 ▼
tenant_id = customer_b
 │
 ▼
Adapter Registry
 │
 ▼
customer_b_adapter
```

---

# 3. Don't use client-controlled adapter names

Bad API:

```json
{
    "adapter": "customer_a_adapter",
    "prompt": "Give me customer information"
}
```

Why is this dangerous?

A malicious customer could send:

```json
{
    "adapter": "customer_b_adapter"
}
```

and potentially access another customer's AI behavior.

Instead:

```json
{
    "prompt": "Give me customer information"
}
```

The backend extracts the tenant from authentication:

```text
Authenticated User
       │
       ▼
tenant_id = customer_b
       │
       ▼
Backend lookup
       │
       ▼
customer_b_adapter
```

---

# 4. Example JWT authentication

A JWT might contain:

```json
{
    "sub": "user_123",
    "tenant_id": "customer_b",
    "role": "user"
}
```

Define a user:

```python
from pydantic import BaseModel


class CurrentUser(BaseModel):

    user_id: str

    tenant_id: str

    role: str
```

Dependency:

```python
from fastapi import (
    Depends,
    HTTPException
)


async def get_current_user():

    # In real production:
    # 1. Read Authorization header
    # 2. Verify JWT signature
    # 3. Validate expiration
    # 4. Extract claims

    return CurrentUser(
        user_id="user_123",
        tenant_id="customer_b",
        role="user"
    )
```

Your endpoint:

```python
@app.post("/generate")
async def generate(
    request: GenerateRequest,
    user: CurrentUser = Depends(
        get_current_user
    )
):

    adapter = get_adapter_for_tenant(
        user.tenant_id
    )

    response = await inference_service.generate(
        adapter=adapter,
        prompt=request.prompt
    )

    return {
        "response": response
    }
```

Notice:

```text
Client
  ↓
Sends prompt only
  ↓
Backend
  ↓
Reads tenant from JWT
  ↓
Selects authorized adapter
```

---

# 5. Use a database instead of a Python dictionary

For production:

```text
tenant_adapter_mapping
```

Example schema:

```sql
CREATE TABLE tenant_adapters (

    id UUID PRIMARY KEY,

    tenant_id UUID NOT NULL,

    adapter_name VARCHAR(255) NOT NULL,

    adapter_version VARCHAR(50) NOT NULL,

    adapter_uri TEXT NOT NULL,

    base_model VARCHAR(255) NOT NULL,

    status VARCHAR(50) NOT NULL,

    created_at TIMESTAMP NOT NULL,

    updated_at TIMESTAMP NOT NULL
);
```

Example:

| tenant_id  | adapter_name | version | status     |
| ---------- | ------------ | ------- | ---------- |
| customer_a | support_a    | v1      | production |
| customer_b | finance_b    | v3      | production |
| customer_c | support_c    | v2      | production |

A query:

```python
async def get_tenant_adapter(
    session,
    tenant_id: str
):

    result = await session.execute(

        select(
            TenantAdapter
        ).where(

            TenantAdapter.tenant_id
            ==
            tenant_id,

            TenantAdapter.status
            ==
            "production"
        )
    )

    return result.scalar_one_or_none()
```

This gives you:

```text
Customer
   ↓
Database
   ↓
Adapter metadata
```

---

# 6. Add an adapter registry

A good production design separates:

```text
Customer information
```

from:

```text
Model information
```

Architecture:

```text
Tenant Database

Customer A
    │
    └── adapter_id = adapter_101


Adapter Registry

adapter_101
    │
    ├── name
    ├── version
    ├── base_model
    ├── storage location
    ├── checksum
    └── deployment status
```

Example model:

```python
class AdapterMetadata:

    def __init__(
        self,
        adapter_id,
        tenant_id,
        version,
        base_model,
        uri
    ):

        self.adapter_id = adapter_id

        self.tenant_id = tenant_id

        self.version = version

        self.base_model = base_model

        self.uri = uri
```

Example:

```text
adapter_id: support_a_v2

tenant_id: customer_a

base_model:
meta-llama/Llama-3.1-8B

uri:
s3://model-registry/customer-a/v2

version:
v2
```

---

# 7. Verify base model compatibility

This is important.

You cannot blindly attach any LoRA adapter to any base model.

For example:

```text
Adapter trained on:
Llama-3.1-8B
```

should not automatically be loaded into:

```text
Mistral-7B
```

Your registry should verify:

```python
def validate_adapter(
    adapter,
    base_model
):

    if (
        adapter.base_model
        !=
        base_model
    ):

        raise ValueError(
            "Incompatible adapter"
        )
```

Production metadata should ideally include:

```text
Base model ID
Base model revision
Tokenizer revision
Adapter framework/version
LoRA configuration
Rank
Target modules
Checksum
```

---

# 8. Serving architecture for 10 customers

For a small number of customers:

```text
                       GPU Server

                Shared Base Model
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     Customer A     Customer B     Customer C
     Adapter        Adapter        Adapter
```

All adapters can potentially remain available.

But what happens with:

```text
1,000 customers?
```

Loading every adapter permanently may consume too much memory.

---

# 9. Dynamic adapter loading

For many customers:

```text
                    Request
                       │
                       ▼
                  tenant_id
                       │
                       ▼
                 Adapter Registry
                       │
               ┌───────┴────────┐
               │                │
               ▼                ▼
          Adapter loaded?       No
               │                 │
              Yes                ▼
               │            Load adapter
               │                 │
               └────────┬────────┘
                        ▼
                    Inference
```

Pseudo-code:

```python
class AdapterManager:

    def __init__(self):

        self.loaded_adapters = {}


    async def get_adapter(
        self,
        adapter_metadata
    ):

        adapter_id = (
            adapter_metadata.adapter_id
        )

        if (
            adapter_id
            in self.loaded_adapters
        ):

            return self.loaded_adapters[
                adapter_id
            ]


        adapter = await self.load_adapter(
            adapter_metadata
        )

        self.loaded_adapters[
            adapter_id
        ] = adapter

        return adapter
```

But you need a memory limit.

---

# 10. Use an LRU adapter cache

Suppose your GPU/server can hold only:

```text
50 active adapters
```

Use:

```text
Least Recently Used cache
```

Example:

```python
from collections import OrderedDict


class AdapterCache:

    def __init__(
        self,
        max_adapters: int = 50
    ):

        self.max_adapters = (
            max_adapters
        )

        self.cache = OrderedDict()


    def get(
        self,
        adapter_id: str
    ):

        if (
            adapter_id
            not in self.cache
        ):

            return None


        self.cache.move_to_end(
            adapter_id
        )

        return self.cache[
            adapter_id
        ]


    def put(
        self,
        adapter_id: str,
        adapter
    ):

        if (
            adapter_id
            in self.cache
        ):

            self.cache.move_to_end(
                adapter_id
            )


        self.cache[
            adapter_id
        ] = adapter


        if (
            len(self.cache)
            >
            self.max_adapters
        ):

            removed_adapter_id, _ = (
                self.cache.popitem(
                    last=False
                )
            )

            print(
                "Evicted:",
                removed_adapter_id
            )
```

Flow:

```text
Adapter Cache

[A] [B] [C] [D]

Request → A
A becomes recently used

Cache full
      │
      ▼
Remove least recently used adapter
```

**Important:** In real GPU inference, don't casually implement this only as a Python dictionary. The inference server must safely manage adapter loading/unloading and GPU memory.

---

# 11. Avoid race conditions during adapter loading

Imagine 100 requests arrive for a new customer:

```text
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼──► Load Adapter X
Request N ─┘
```

Bad implementation:

```python
if adapter not in cache:
    load_adapter()
```

All 100 requests might try to load the same adapter.

Use a per-adapter lock:

```python
import asyncio


class AdapterManager:

    def __init__(self):

        self.cache = {}

        self.locks = {}


    async def get_adapter(
        self,
        adapter_id
    ):

        if (
            adapter_id
            in self.cache
        ):

            return self.cache[
                adapter_id
            ]


        if (
            adapter_id
            not in self.locks
        ):

            self.locks[
                adapter_id
            ] = asyncio.Lock()


        async with self.locks[
            adapter_id
        ]:

            # Check again after lock

            if (
                adapter_id
                in self.cache
            ):

                return self.cache[
                    adapter_id
                ]


            adapter = await self.load_adapter(
                adapter_id
            )

            self.cache[
                adapter_id
            ] = adapter

            return adapter
```

This prevents a **cache stampede**.

---

# 12. Version adapters

Never use only:

```text
customer_a_adapter
```

Use versions:

```text
customer_a_adapter:v1
customer_a_adapter:v2
customer_a_adapter:v3
```

Database:

```text
Customer A

Production → v2
Candidate  → v3
Rollback   → v1
```

Example:

```python
class AdapterDeployment:

    tenant_id: str

    production_version: str

    candidate_version: str
```

Request:

```text
Customer A
   │
   ▼
Adapter Registry
   │
   ▼
customer_a:v2
```

---

# 13. Canary deployment for adapters

Suppose you trained:

```text
Customer A adapter v3
```

Don't immediately send 100% traffic to it.

Instead:

```text
90% → v2
10% → v3
```

Example:

```python
import random


def select_adapter_version():

    value = random.random()

    if value < 0.10:

        return "v3"

    return "v2"
```

Monitor:

```text
Error rate
Latency
Customer feedback
Hallucination rate
Task success rate
```

If v3 is better:

```text
100% → v3
```

If v3 fails:

```text
Rollback → v2
```

---

# 14. Data isolation is critical

LoRA adapters can encode customer-specific terminology, style, and potentially sensitive learned patterns.

You need isolation at multiple levels:

```text
┌──────────────────────────────┐
│ Authentication               │
├──────────────────────────────┤
│ Tenant Identification        │
├──────────────────────────────┤
│ Adapter Authorization        │
├──────────────────────────────┤
│ Data Isolation               │
├──────────────────────────────┤
│ RAG Namespace Isolation      │
├──────────────────────────────┤
│ Logging / Audit Isolation    │
└──────────────────────────────┘
```

For example:

```text
Customer A request
        │
        ├── Customer A LoRA
        │
        └── Customer A Vector DB
```

Never:

```text
Customer A
     │
     ▼
Customer B adapter ❌

Customer B vector data ❌
```

---

# 15. Add RAG isolation

Often the best architecture is:

```text
                  Customer Request
                         │
                         ▼
                    Authenticate
                         │
                         ▼
                     tenant_id
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
          LoRA Router             RAG Router
              │                      │
              ▼                      ▼
       Tenant LoRA Adapter      Tenant Vector DB
              │                      │
              └──────────┬───────────┘
                         ▼
                      LLM
```

Example:

```python
def get_tenant_resources(
    tenant_id
):

    return {

        "adapter":
            get_adapter_for_tenant(
                tenant_id
            ),

        "vector_namespace":
            f"tenant:{tenant_id}"
    }
```

Then:

```python
documents = vector_db.search(

    query=query,

    namespace=(
        resources[
            "vector_namespace"
        ]
    )
)
```

This ensures:

```text
Customer A
      ↓
Customer A adapter
      +
Customer A documents
```

---

# 16. Example production FastAPI flow

```python
from fastapi import (
    FastAPI,
    Depends
)

app = FastAPI()


@app.post("/chat")
async def chat(

    request: ChatRequest,

    user: CurrentUser = Depends(
        get_current_user
    )
):

    # 1. Identify tenant

    tenant_id = user.tenant_id


    # 2. Get authorized deployment

    deployment = await adapter_registry.get_active(
        tenant_id
    )


    # 3. Validate adapter compatibility

    validate_adapter(
        deployment.adapter,
        BASE_MODEL_VERSION
    )


    # 4. Retrieve tenant-specific context

    documents = await rag_service.search(

        tenant_id=tenant_id,

        query=request.message
    )


    # 5. Build prompt

    prompt = build_prompt(

        documents=documents,

        user_message=request.message
    )


    # 6. Send to inference server

    response = await inference_client.generate(

        model=deployment.adapter_id,

        prompt=prompt,

        max_tokens=500
    )


    # 7. Audit

    await audit_service.log(

        tenant_id=tenant_id,

        adapter_version=
            deployment.version,

        request_id=
            request.request_id
    )


    return {
        "response": response
    }
```

This is much closer to a real production design.

---

# 17. How would I handle 10,000 customers?

I would **not** put all 10,000 adapters into one GPU.

I would use:

```text
                     API Gateway
                           │
                           ▼
                      Auth Service
                           │
                           ▼
                      Tenant Router
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
        Adapter Pool 1  Adapter Pool 2  Adapter Pool 3
             │             │              │
             ▼             ▼              ▼
          GPU Nodes      GPU Nodes      GPU Nodes
```

Each pool:

```text
Base Model
+
Frequently used adapters
+
Dynamic adapter cache
```

The routing layer could shard adapters:

```text
tenant_id hash
     │
     ▼
GPU Pool
```

Example:

```python
import hashlib


def get_shard(
    tenant_id: str,
    num_shards: int
):

    value = hashlib.md5(
        tenant_id.encode()
    ).hexdigest()

    number = int(
        value,
        16
    )

    return (
        number
        %
        num_shards
    )
```

Example:

```text
Customer A → Pool 2
Customer B → Pool 1
Customer C → Pool 3
```

This improves cache locality.

---

# 18. What should you monitor?

### General inference metrics

```text
P50 latency
P95 latency
P99 latency
Requests/second
Tokens/second
Error rate
GPU utilization
GPU memory
Queue length
```

### Adapter-specific metrics

```text
Adapter load time
Adapter cache hit rate
Adapter eviction rate
Requests per adapter
Adapter errors
Adapter version
```

### Tenant metrics

```text
Requests per customer
Cost per customer
Tokens per customer
Rate-limit violations
Errors per customer
```

Example:

```python
metrics = {

    "tenant_id":
        tenant_id,

    "adapter_id":
        adapter_id,

    "adapter_version":
        version,

    "latency_ms":
        latency_ms,

    "input_tokens":
        input_tokens,

    "output_tokens":
        output_tokens
}
```

---

# Interview-ready answer

> **For multiple customers with different LoRA adapters, I would use a multi-tenant architecture where the base model is shared and each customer has its own authorized LoRA adapter. The customer identity comes from authentication, typically a JWT, and the backend—not the client—maps the tenant ID to the correct adapter and adapter version.**
>
> **I would store this mapping in an adapter registry containing the adapter ID, version, base-model compatibility, storage location, checksum, and deployment status. For a small number of adapters, the serving layer can keep them loaded. For thousands of customers, I would use dynamic loading with bounded adapter caching, prevent concurrent load stampedes, and shard tenants across inference pools.**
>
> **I would also isolate tenant RAG data, authorization, logging, and metrics. Adapter versions would support canary deployments and rollback. Finally, I would use a LoRA-aware inference server rather than manually switching adapters in a shared FastAPI model instance.**

## The key architecture

```text
Authenticate Customer
        ↓
Extract tenant_id
        ↓
Authorize tenant
        ↓
Adapter Registry
        ↓
Select correct adapter + version
        ↓
Tenant-specific RAG retrieval
        ↓
Shared Base Model + LoRA
        ↓
Return response
```

The most important production principle is:

> **One shared base model for efficiency, but strict tenant-level adapter selection and data isolation for security.**
