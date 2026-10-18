# What if the LLM provider becomes unavailable? — Senior AI Engineer approach

As a senior or staff AI engineer, I treat LLM provider unavailability as a dependency failure that can affect the availability of the entire AI application.

For example, your financial-advisor copilot uses OpenAI for planning, RAG answer generation, and financial research. If the provider becomes unavailable, your FastAPI application might still be healthy, PostgreSQL might still be working, and Qdrant might still return relevant documents—but any workflow requiring the LLM could fail.

The goal is not simply to retry the request. The goal is to contain the failure, preserve the parts of the system that still work, and recover safely without compromising security or answer quality.

## 1. What happens when the provider goes down?

Your AI application

FastAPI · LangGraph · RAG · Agent tools

Primary LLM provider unavailable

Connection errors · timeouts · 5xx · rate limits

Still operational

Authentication, PostgreSQL, document storage, deterministic business rules

Potentially degraded

LLM generation, planning, agent decisions, LLM-based reranking and evaluation

The impact depends on your architecture:

- Simple LLM API: Requests requiring generation return errors or time out.
- RAG application: Retrieval can succeed, but the final answer cannot be generated.
- Multi-agent system: The planner may fail, preventing downstream agents from executing.
- Asynchronous workflows: Jobs may fail or accumulate in queues.
- Streaming API: The user may receive partial output before the stream fails.
- LLM-dependent ingestion: New documents might not be embedded or indexed if embedding also depends on the unavailable provider.

An important distinction: a provider outage, a rate-limit response, and an invalid API key are different failure modes. They should not all trigger the same retry and fallback behavior.

## 2. Identify which failure you are dealing with

Before implementing recovery, classify the failure. A `429` does not always mean the provider is down, and a timeout does not prove that the provider is unavailable.

| Failure          | Typical interpretation                                       | Response                                                   |
| ---------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Connection error | DNS, network, TLS, or provider connectivity issue            | Check connectivity; bounded retry or failover              |
| Timeout          | Request did not finish within the deadline                   | Check outcome if needed; retry safely or fail over         |
| HTTP 500/502/503 | Internal error, gateway failure, or temporary unavailability | Bounded retry; consider circuit breaker and fallback       |
| HTTP 429         | Rate limit, temporary throttling, or quota/billing limit     | Inspect error code; throttle or remediate the actual cause |
| HTTP 401/403     | Credentials or permissions problem                           | Alert and fix configuration; don't blindly retry           |
| HTTP 400         | Invalid request or unsupported parameters                    | Fix the request; failover usually won't help               |

Provider error details matter more than status codes alone. OpenAI's error documentation distinguishes temporary rate limiting and model overload from exhausted credits, usage limits, and authentication failures.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://developers.openai.com\&sz=32)

OpenAI API

+2



## 3. Design provider failover into the architecture

For a production AI platform, I would avoid scattering provider-specific fallback logic throughout every agent. Instead, introduce a model gateway or provider-routing layer.

FastAPI / LangGraph

Business workflow and agent orchestration

LLM Gateway

Deadlines · retry budget · circuit breaker · routing policy

Primary provider

Healthy

Preferred route

Secondary provider

Standby

Approved and tested

Existing application services remain independent

PostgreSQL · Qdrant · Redis · audit logs

The gateway should own the policy, but each workflow must decide whether failover is safe.

For example, if the primary model fails during a read-only financial research summary, the gateway may route the request to an approved secondary model. However, if an agent was in the middle of executing a trade or another external action, the workflow must first reconcile whether the action already happened.

A secondary provider is not automatically equivalent to the primary. Validate its model capabilities, structured-output support, tool-calling behavior, context window, data-handling terms, latency, cost, and answer quality before enabling it. Cross-provider failover also does not automatically transfer an ongoing agent session or its saved state.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://developers.openai.com\&sz=32)

OpenAI Developers

+1



## 4. Implement a provider adapter and fallback router

Here is a practical Python pattern for a FastAPI/LangGraph application. The example retries the primary once after a transient failure, then tries the secondary provider within a deadline.

First, define a common interface:

```
from typing import Protocolclass LLMProvider(Protocol):    async def generate(self, prompt: str) -> str:        ...
```

Each provider gets its own adapter implementing `generate()`. The business workflow doesn't need to know whether the response came from provider A or provider B.

Next, implement routing:

```
import asyncioimport logginglogger = logging.getLogger(__name__)class ProviderUnavailable(Exception):    passclass LLMGateway:    def __init__(        self,        primary: LLMProvider,        secondary: LLMProvider,    ):        self.primary = primary        self.secondary = secondary    async def generate(self, prompt: str) -> str:        try:            async with asyncio.timeout(5):                return await self.primary.generate(prompt)        except (TimeoutError, ProviderUnavailable):            logger.warning(                "Primary LLM failed; attempting fallback"            )        try:            async with asyncio.timeout(4):                return await self.secondary.generate(prompt)        except (TimeoutError, ProviderUnavailable) as exc:            logger.error("All LLM providers unavailable")            raise ProviderUnavailable(                "AI generation is temporarily unavailable"            ) from exc
```

This demonstrates the routing pattern, not a complete production implementation.

In real code:

- Provider adapters must map actual SDK errors into typed, classified failures.
- Do not classify every exception as provider unavailability.
- Do not automatically retry authentication errors, invalid requests, or exhausted billing limits.
- Include time spent in retries and queueing within a single end-to-end deadline.
- Add circuit breakers and a retry budget.
- Ensure only one retry layer is active, including any retries performed by the SDK.
- Track which provider and model actually served each request.

The five-second and four-second timeouts are illustrative. Set them from your service-level objectives and measured latency.

## 5. Add a circuit breaker

Why do we need a circuit breaker if we already have fallback?

Suppose the primary provider is down. Every incoming request attempts the primary, waits five seconds, fails, and then calls the secondary. During a traffic spike, thousands of requests repeat this behavior.

A circuit breaker stops repeatedly calling a provider that is known to be unhealthy.

CLOSED

Normal traffic

Normal operation

Requests go to the primary. If failures cross a configured threshold, open the circuit.

OPEN

Skip primary

Failure containment

New requests bypass the primary and go to the secondary, or receive a controlled unavailable response.

HALF-OPEN

Probe recovery

Recovery verification

Allow a limited number of test requests. Close the circuit when health recovers; reopen it if failures continue.

Use a tested circuit-breaker implementation rather than building complex state management from scratch. In Kubernetes deployments with multiple replicas, remember that a local circuit breaker in one process does not automatically share health state with the other replicas. A centralized gateway or shared state may be appropriate, depending on the operational requirements.

Also, don't open a circuit merely because a user supplied an invalid prompt. Count appropriate provider-health failures, not every possible error.

## 6. What happens to LangGraph and multi-agent workflows?

This is particularly important in an enterprise multi-agent platform.

Imagine this workflow:

```
Planner
   |
   +--> Research Agent
   |
   +--> Portfolio Analysis Agent
   |
   v
Evaluator
   |
   v
Final Answer
```

If the LLM provider fails during the planner step, the graph might never start its workers. If it fails during the evaluator step, research may already have completed.

I would implement the following behavior:

- Retry transient failures: Only within a bounded retry policy.
- Checkpoint completed work: Persist important workflow state so recovery doesn't require repeating every successful step.
- Resume safely: Restart from the failed stage only when the graph and its side effects support that recovery.
- Limit retries and graph steps: Prevent infinite rework loops during an outage.
- Persist long-running jobs: Keep job status in durable storage rather than depending on an HTTP connection remaining open.
- Reconcile tool actions: Never blindly repeat a non-idempotent action because an LLM call timed out.

For example, if the research agent already fetched market data, store that result in the workflow state. If the final-answer model call fails, the workflow may reuse the stored research after the provider recovers.

However, a checkpoint does not automatically guarantee safe resumption. You must design idempotent steps, persist important state, and explicitly handle partially completed side effects.

## 7. What if both providers are unavailable?

A robust system must have a policy for total LLM failure. Do not let requests hang indefinitely or keep generating retries.

| Application capability                           | Behavior during outage                                                      |
| ------------------------------------------------ | --------------------------------------------------------------------------- |
| Authentication and authorization                 | Continue normally                                                           |
| Read existing portfolios or reports              | Continue if the operation doesn't require an LLM                            |
| Search stored documents                          | Continue retrieval, but do not pretend an LLM-generated answer is available |
| Generate new financial advice                    | Return a clear temporary-unavailability response                            |
| Background ingestion                             | Queue safely or pause LLM-dependent stages                                  |
| Pending agent workflows                          | Preserve state and retry later within policy                                |
| Financial transactions or other external actions | Require deterministic controls and action-state reconciliation              |

An API could return HTTP `503 Service Unavailable` for a synchronous request that cannot be completed:

```
from fastapi import HTTPException@app.post("/advisor/ask")async def ask_advisor(request: AdvisorRequest):    try:        answer = await gateway.generate(request.question)        return {"status": "completed", "answer": answer}    except ProviderUnavailable:        raise HTTPException(            status_code=503,            detail={                "code": "llm_temporarily_unavailable",                "message": (                    "AI generation is temporarily unavailable. "                    "Please retry shortly."                ),            },            headers={"Retry-After": "5"},        )
```

Here, `app`, `gateway`, and `AdvisorRequest` are assumed to be defined elsewhere. The `Retry-After` value is illustrative; production systems should choose it based on their recovery policy.

For asynchronous work, return a job identifier and a pending or degraded status instead of holding the request open. Never return a fabricated successful answer simply to maintain a 200 response.

## 8. How do you monitor and recover?

I would track:

- Primary and secondary provider success rates.
- Connection errors, timeouts, 5xx responses, and classified 429 responses.
- Circuit-breaker state and duration.
- Fallback rate and secondary-provider capacity.
- Retry attempts per logical request.
- End-to-end latency and failed workflow count.
- Pending jobs, oldest queue age, and recovery backlog.

When the primary appears healthy again, do not immediately send all production traffic back to it. Use controlled probes, gradually shift traffic, and watch for renewed failures. Make sure the secondary has enough capacity to serve the workload during failover.

For actual incidents, check the provider's status page and error codes, and correlate those with your own traces. A healthy provider dashboard doesn't rule out problems in your network, credentials, quotas, or regional route.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://developers.openai.com\&sz=32)

OpenAI Developers

+1



## 9. Staff-level interview answer

> "If an LLM provider becomes unavailable, I treat it as a dependency outage and first determine the scope and failure type. I distinguish transient connection failures, timeouts and server errors from authentication issues, malformed requests, rate limiting, and exhausted quotas.
>
> At the architecture level, I use a provider abstraction and centralized routing policy. For eligible transient failures, I use bounded retries with exponential backoff and jitter, strict deadlines, and a circuit breaker. If the primary remains unavailable, I fail over to a validated secondary provider whose model capabilities, security requirements, capacity, and output quality have been tested.
>
> For multi-agent workflows, I checkpoint completed work, persist state, enforce step limits, and resume only from a safe recovery point. I reconcile side effects before replaying any tool action. If all providers are unavailable, I preserve durable jobs, keep independent application functions operational, and return a clear degraded-service response rather than fabricating an answer.
>
> Finally, I monitor provider health, fallback rates, latency, errors, and queue depth, and restore primary traffic gradually when recovery is verified. My objective is to maintain availability where possible while preserving correctness, security, and predictable recovery."

The key principle is: provider failover is not just switching model endpoints. It is a controlled recovery strategy for the entire AI workflow.