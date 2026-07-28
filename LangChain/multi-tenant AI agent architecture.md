This is one of the **most important Staff AI Engineer and AI Architect system design interview questions**.

Large enterprises rarely build AI systems for a single customer. Instead, they build **multi-tenant AI platforms** where hundreds or thousands of organizations share the same infrastructure while keeping their data, models, and configurations isolated.

Examples include:

* Microsoft Copilot
* Salesforce Agentforce
* ServiceNow AI
* Atlassian Rovo
* Google Vertex AI Agent Builder
* Enterprise internal AI platforms

Interviewers want to evaluate whether you understand:

* Multi-tenancy
* Data isolation
* Tenant-specific RAG
* Authentication and authorization
* Agent orchestration
* Scalability
* Cost optimization
* Observability
* Security
* Deployment architecture

---

# Problem Statement

Design an AI platform where:

* 10,000+ companies use the same platform
* Every company has its own:

  * users
  * documents
  * vector database
  * prompts
  * LLM settings
  * agents
  * API keys
* No company can access another company's data.

---

# High-Level Architecture

```text
                         Internet
                             │
                     Load Balancer
                             │
                    API Gateway (Auth)
                             │
                    Tenant Resolver
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     Agent Service      Retrieval API     Admin API
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                     Agent Orchestrator
                             │
      ┌──────────────┬───────────────┬──────────────┐
      ▼              ▼               ▼              ▼
 Planner Agent   Tool Agent    SQL Agent    Search Agent
      │              │               │              │
      └──────────────┴───────────────┴──────────────┘
                             │
         ┌───────────────────┼────────────────────┐
         ▼                   ▼                    ▼
     PostgreSQL         Vector Store         Redis Cache
```

---

# Tenant Isolation

Every request includes a tenant identifier.

Example:

```http
POST /chat

Headers:

Authorization: Bearer JWT

X-Tenant-ID: ey_india
```

The backend resolves:

```text
Tenant

↓

Configuration

↓

Resources

↓

Execute Agent
```

Never trust a tenant ID supplied by the client alone. Derive the tenant from the authenticated identity or verify that the authenticated user belongs to the requested tenant.

---

# Authentication Flow

```text
User

↓

Identity Provider

↓

JWT

↓

API Gateway

↓

Tenant Resolution

↓

RBAC

↓

Agent
```

JWT example

```json
{
  "user_id":"123",
  "tenant_id":"ey_india",
  "role":"manager"
}
```

---

# Database Design

A common schema:

```sql
tenants
--------
id
name
plan
status

users
--------
id
tenant_id
email
role

documents
--------
id
tenant_id
filename

conversations
--------
id
tenant_id
user_id

agent_configs
--------
tenant_id
model
temperature
system_prompt
```

Every business table contains a `tenant_id`.

---

# Vector Database Isolation

Option 1 (recommended for many SaaS platforms)

```text
Qdrant

↓

Collection

↓

tenant_a

tenant_b

tenant_c
```

Option 2

```text
Single Collection

↓

Metadata Filter

tenant_id=tenant_a
```

Example metadata

```json
{
    "tenant_id":"ey_india",
    "department":"finance",
    "document":"policy.pdf"
}
```

Retrieval

```python
retriever.search(
    query,
    filter={
        "tenant_id":"ey_india"
    }
)
```

This prevents cross-tenant retrieval.

---

# Agent Configuration

Every tenant may choose different settings.

```text
Tenant A

GPT-4.1

Temperature 0

Finance Prompt

----------------

Tenant B

GPT-4.1-mini

Temperature 0.3

Legal Prompt
```

Schema

```python
class TenantConfig:

    model: str

    temperature: float

    system_prompt: str

    tools: list
```

The orchestrator loads the configuration before execution.

---

# Agent Orchestrator

```text
User Question

↓

Tenant Resolver

↓

Load Tenant Config

↓

Planner

↓

Retriever

↓

Tools

↓

LLM

↓

Response
```

The orchestrator remains generic while behavior is driven by tenant configuration.

---

# Multi-Agent Design

```text
                 Orchestrator
                      │
      ┌───────────────┼──────────────┐
      ▼               ▼              ▼
 Research Agent   SQL Agent    Tool Agent
      │               │              │
      └───────────────┼──────────────┘
                      ▼
                 Aggregator
```

Different tenants can enable or disable agents.

---

# Caching

Avoid repeated work.

```text
User

↓

Redis

↓

Cache Hit?

↓

Yes

↓

Return

↓

No

↓

LLM
```

Cache key

```text
tenant_id

+

user question

+

model

+

prompt version
```

Including prompt and model versions avoids serving stale responses after configuration changes.

---

# Memory

Each tenant has isolated conversations.

```text
Tenant A

Conversation 1

Conversation 2

------------------

Tenant B

Conversation 1
```

Example

```sql
conversation_id

tenant_id

user_id

messages
```

---

# LLM Routing

Different tenants may have different plans.

```text
Free Plan

↓

GPT-4.1-mini

---------------

Enterprise

↓

GPT-4.1
```

Router

```python
def select_model(plan):

    if plan == "enterprise":
        return "gpt-4.1"

    return "gpt-4.1-mini"
```

---

# Horizontal Scaling

```text
                 Load Balancer
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
   Agent Pod 1     Agent Pod 2     Agent Pod 3
       │                │                │
       └────────────────┼────────────────┘
                        ▼
               Shared Databases
```

Pods remain stateless.

---

# Kubernetes

```text
Ingress

↓

API

↓

Deployment

↓

Agent Pods

↓

HPA

↓

Autoscaling
```

Scale based on:

* CPU
* Memory
* Request rate
* Queue length
* LLM concurrency

---

# Queue-Based Execution

Long-running jobs should be asynchronous.

```text
User

↓

API

↓

Kafka / RabbitMQ

↓

Worker

↓

LLM

↓

Database
```

Examples:

* Large document ingestion
* Embedding generation
* Batch summarization
* Report generation

---

# Observability

Monitor:

```text
Tenant

↓

Latency

↓

Tokens

↓

Cost

↓

Errors

↓

Retries
```

Dashboard

```text
Top Tenants

Highest Cost

Slowest Requests

Most Used Models

Error Rate
```

Track metrics per tenant to support billing, capacity planning, and troubleshooting.

---

# Security

```text
JWT

↓

RBAC

↓

Tenant Filter

↓

PII Detection

↓

Audit Log
```

Best practices:

* Encrypt data at rest and in transit.
* Rotate secrets using a secrets manager.
* Validate tool permissions.
* Apply rate limits per tenant.
* Keep immutable audit logs for sensitive actions.

---

# Production Deployment

```text
                     Users
                       │
                  CDN / WAF
                       │
                  Load Balancer
                       │
                  API Gateway
                       │
             Authentication Service
                       │
                Tenant Resolver
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
   Agent API      Retrieval API   Admin API
        │              │               │
        └──────────────┼───────────────┘
                       ▼
                LangGraph Orchestrator
                       │
      ┌────────────┬────────────┬────────────┐
      ▼            ▼            ▼            ▼
 Planner      Retrieval      SQL Tool     Web Tool
      │            │            │            │
      └────────────┼────────────┴────────────┘
                   ▼
          PostgreSQL   Vector DB   Redis
                   │
                   ▼
             LLM Provider(s)
```

---

# Cost Optimization

Instead of sending every request to the largest model:

```text
Simple FAQ

↓

GPT-4.1-mini

------------------

Complex Analysis

↓

GPT-4.1
```

Additional strategies:

* Cache repeated responses
* Batch embedding requests
* Use semantic caching
* Limit maximum context size
* Route retrieval-only requests away from expensive models

---

# Disaster Recovery

```text
Region A

↓

Failure

↓

Traffic Manager

↓

Region B

↓

Continue Service
```

Recommended practices:

* Multi-region deployment
* Database replication
* Regular backups
* Infrastructure as Code
* Health checks and automated failover

---

# Interview Follow-Up Questions

## 1. How do you prevent data leakage between tenants?

Use multiple layers:

* Tenant-aware authentication
* Authorization (RBAC/ABAC)
* Database filtering by `tenant_id`
* Vector store metadata filters or isolated collections
* Separate caches or tenant-scoped cache keys
* End-to-end testing for isolation

---

## 2. Should every tenant have a separate database?

It depends on requirements.

| Strategy                      | Best for                   |
| ----------------------------- | -------------------------- |
| Shared database + `tenant_id` | Most SaaS platforms        |
| Separate schema per tenant    | Medium isolation           |
| Separate database per tenant  | Highly regulated customers |

---

## 3. How would you scale to 100,000 tenants?

* Stateless API services
* Kubernetes autoscaling
* Distributed caches
* Sharded databases
* Partitioned vector stores
* Asynchronous ingestion pipelines
* Multi-region deployment

---

## 4. How would you monitor tenant costs?

Track per tenant:

* Input/output tokens
* Model used
* API calls
* Embedding requests
* Tool invocations
* Estimated spend

Set alerts or quotas when usage exceeds plan limits.

---

## 5. How would you support tenant customization?

Store configuration separately from code:

* System prompts
* Model selection
* Enabled tools
* Agent graph
* Retrieval settings
* Safety policies
* UI branding (if applicable)

Load these dynamically during request processing.

---

# Production-Grade Multi-Tenant AI Platform

```text
                         Internet
                              │
                         CDN / WAF
                              │
                        Load Balancer
                              │
                         API Gateway
                              │
                Authentication + RBAC
                              │
                      Tenant Resolver
                              │
                     LangGraph Router
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   Planner Agent       Retrieval Agent        Tool Agent
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                  Tenant Configuration Service
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
     PostgreSQL          Vector Store          Redis
                              │
                              ▼
                 Model Router (Multi-LLM)
                              │
               GPT-4.1 / GPT-4.1-mini / Local Model
                              │
                              ▼
                    Observability & Billing
                  (Tracing, Metrics, Audit)
```

This architecture is scalable because services are **stateless**, tenant context is resolved at the edge, data access is always tenant-scoped, expensive operations are cached or queued, and every request is observable. It also supports tenant-specific models, prompts, tools, and retrieval policies without requiring separate deployments for each customer.
