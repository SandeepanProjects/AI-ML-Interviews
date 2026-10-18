# How to Handle Cost and Model Failure in Production AI Systems

As a senior or staff AI engineer, I would not solve cost and model failures by simply switching to a cheaper model or retrying the same request. I would design the AI platform to control spending, detect failures, recover safely, preserve answer quality, and remain available when a model provider is degraded or unavailable.

For an enterprise system such as a financial advisor copilot or compliance-review platform, this is especially important because a failed request can affect availability, accuracy, auditability, and potentially financial decisions.

I would design the solution around six capabilities:

1. Cost budgets and token controls — prevent unexpected spending.
2. Model routing — choose a suitable model based on task complexity, risk, latency, and price.
3. Failure detection — distinguish transient errors from permanent failures.
4. Retries, timeouts, and fallbacks — recover without creating retry storms.
5. Graceful degradation — return a safe partial result or explain that the service is temporarily unavailable.
6. Observability and evaluation — measure whether cost optimizations preserve quality.

## 1. Understand the two problems separately

Problem A: Cost is too high

Symptoms: token usage increases, expensive models are used for simple requests, agents make repeated calls, prompts grow, or traffic spikes.

Solution: budgets, model routing, context reduction, caching, bounded workflows, and cost monitoring.

Problem B: Model or provider fails

Symptoms: timeouts, HTTP 429 rate limits, HTTP 5xx errors, invalid structured output, empty responses, or degraded answer quality.

Solution: bounded retries, circuit breakers, provider/model fallbacks, schema validation, deadlines, and safe failure handling.

These problems are connected. For example, retrying a failing model can increase cost, while switching every request to a premium model can hide quality problems but increase the bill dramatically.

## 2. Design the production architecture first

A practical architecture separates business orchestration from model-provider details.

Client / API Gateway

Authentication · rate limits · request ID

AI Request Orchestrator

Deadline · request budget · task classification

Cost Controller

Token caps · spending limits · usage accounting

Model Router

Task tier · quality · latency · provider health

Model Gateway

Timeouts · retries · circuit breaker · fallback

Model A

Primary

Model B

Fallback

Safe response

If recovery fails

Observability and evaluation

Cost · latency · errors · quality · traces · alerts

The key architectural decision is to avoid scattering provider-specific retry and fallback logic across individual agents. Put shared resilience policies in a model gateway, while business workflows decide whether a partial result is acceptable.

## 3. Handle cost: build a budget system, not just a cheaper-model strategy

### 3.1 Understand where the money goes

For an LLM request, approximate cost is:

\\[ C\_{\text{request}} = \frac{T\_{\text{in}}P\_{\text{in}}}{10^6} + \frac{T\_{\text{out}}P\_{\text{out}}}{10^6} \\]

Where:

- \\(T\_{\text{in}}\\): billable input tokens, including applicable cached-input tokens.
- \\(T\_{\text{out}}\\): billable output tokens.
- \\(P\_{\text{in}}\\): input price per million tokens.
- \\(P\_{\text{out}}\\): output price per million tokens.

Actual provider billing can have separate rates for cached tokens, reasoning tokens, tools, and other features. Use the selected model's current pricing and actual usage metadata.

Consider an illustrative model with input pricing of $2 per million tokens and output pricing of $8 per million tokens.

| Metric               | Value  |
| -------------------- | ------ |
| Input tokens         | 10,000 |
| Output tokens        | 2,000  |
| Input cost           | $0.020 |
| Output cost          | $0.016 |
| Total estimated cost | $0.036 |

If the system handles 100,000 such requests per month, the model cost alone is approximately $3,600/month.

For an agentic workflow, the cost may be much higher because one user request can trigger multiple LLM calls, retrieval operations, reranking, and tool execution.

### 3.2 Establish a cost budget at multiple levels

I would implement budgets at four levels:

- Per request: maximum model spend, tokens, number of model calls, and workflow steps.
- Per user or tenant: prevent one tenant from consuming the entire shared allocation.
- Per feature: separate budgets for summarization, RAG, compliance analysis, and agent workflows.
- Per organization and billing period: alert on spending thresholds and enforce an agreed hard limit.

Example policy:

| Control                                | Illustrative limit |
| -------------------------------------- | ------------------ |
| Maximum model calls per request        | 6                  |
| Maximum output tokens per call         | 1,500              |
| Maximum total model tokens per request | 12,000             |
| Per-request model budget               | $0.10              |
| Agent workflow steps                   | 10                 |
| Request deadline                       | 20 seconds         |

These are example values, not universal defaults. Production limits should come from measured workloads, quality requirements, service-level objectives, and your pricing model.

Important distinction: a token cap is not the same as a cost cap. Different models and token types have different prices. Enforce both when necessary.

### 3.3 Route requests to the cheapest model that meets the quality requirement

Do not send every request to the most powerful model.

Tier 1

Simple tasks → smaller, cheaper model

Classification, query rewriting, metadata extraction, straightforward summaries, and simple structured outputs.

Tier 2

Moderately complex tasks → balanced model

RAG answer generation, document comparison, multi-step extraction, and common business reasoning.

Tier 3

High-complexity tasks → stronger model

Ambiguous compliance interpretation, complex financial reasoning, or difficult synthesis requiring stronger evaluation results.

The routing decision should consider more than task complexity:

- Quality measured on representative evaluation datasets.
- Sensitivity and risk of the task.
- Input and output token estimates.
- Current latency, availability, and provider rate limits.
- Whether the model supports required structured output or tool calling.
- Tenant policy and data-handling requirements.

A smaller model is only a cost optimization if it still meets the task's quality threshold.

### 3.4 Implement model routing in Python

The following is a simplified example of a routing policy. Model identifiers are illustrative; use your actual configured model names.

```
from dataclasses import dataclassfrom enum import Enumclass TaskType(str, Enum):    CLASSIFICATION = "classification"    RAG_ANSWER = "rag_answer"    COMPLIANCE_REVIEW = "compliance_review"@dataclass(frozen=True)class ModelPolicy:    primary: str    fallback: str    max_output_tokens: int    max_attempts: intPOLICIES = {    TaskType.CLASSIFICATION: ModelPolicy(        primary="small-model",        fallback="balanced-model",        max_output_tokens=300,        max_attempts=2,    ),    TaskType.RAG_ANSWER: ModelPolicy(        primary="balanced-model",        fallback="strong-model",        max_output_tokens=1200,        max_attempts=2,    ),    TaskType.COMPLIANCE_REVIEW: ModelPolicy(        primary="strong-model",        fallback="approved-compliance-model",        max_output_tokens=1800,        max_attempts=2,    ),}def choose_policy(task: TaskType) -> ModelPolicy:    return POLICIES[task]
```

In production, keep this configuration externalized and versioned. The compliance fallback must itself be approved and evaluated for the task. A fallback model is not automatically a safe substitute just because it is available.

### 3.5 Control agent loops

Suppose your financial advisor agent uses:

- One planning call.
- Three specialist-agent calls.
- Two retrieval-grading calls.
- One final synthesis call.
- Two retries.

That can become nine or more model invocations for one user request.

The solution is to enforce a workflow-level budget, not merely a per-call token limit.

```
from dataclasses import dataclassclass BudgetExceeded(Exception):    pass@dataclassclass WorkflowBudget:    max_calls: int = 8    max_output_tokens: int = 6000    calls_used: int = 0    output_tokens_reserved: int = 0    def reserve_call(self, output_token_limit: int) -> None:        if self.calls_used + 1 > self.max_calls:            raise BudgetExceeded("Model call limit reached")        if (            self.output_tokens_reserved + output_token_limit            > self.max_output_tokens        ):            raise BudgetExceeded("Output token budget reached")        self.calls_used += 1        self.output_tokens_reserved += output_token_limit
```

Call `reserve_call()` before each model invocation.

This example counts reserved output-token capacity, not actual token consumption or money. After each response, reconcile the reservation with actual provider usage and the applicable rates.

In a real distributed system, the budget must be maintained atomically in a shared store when multiple workers process the same workflow. A Python object alone cannot enforce a tenant-wide budget across multiple processes.

For agent workflows, also limit:

- Maximum graph steps and retries.
- Maximum parallel fan-out.
- Maximum tool executions.
- Maximum retrieval context size.
- Maximum wall-clock duration.

If the budget is exhausted, the workflow should stop predictably. It must not start another model call just to explain that it has run out of budget.

## 4. Reduce cost without reducing answer quality

Model selection is only one optimization. I would examine the entire request path.

| Technique                      | What it improves                     | Important trade-off                                    |
| ------------------------------ | ------------------------------------ | ------------------------------------------------------ |
| Prompt compression             | Input token cost                     | Must preserve essential instructions                   |
| Context filtering              | RAG token cost                       | Can remove relevant evidence if too aggressive         |
| Semantic caching               | Repeated-query cost and latency      | Requires careful user, tenant, and freshness isolation |
| Batch processing               | Throughput and sometimes cost        | May increase individual request latency                |
| Smaller models                 | Per-token cost                       | Quality may decline                                    |
| Parallel independent tasks     | Latency                              | May increase peak cost                                 |
| Reduce unnecessary agent calls | Cost and latency                     | Could remove useful verification                       |
| Limit output tokens            | Output cost                          | Can truncate useful answers                            |
| Reuse retrieved evidence       | Avoid repeated retrieval and grading | Requires correct invalidation and access checks        |

### A. Use semantic caching carefully

For example, a cached answer for a public FAQ might be reusable. A financial answer based on a specific customer's portfolio usually is not reusable across customers.

A safe cache key should account for the relevant combination of:

- Tenant and authorization scope.
- Normalized query or semantic query identity.
- Model and prompt version.
- Retrieval corpus or document version.
- Relevant user-specific context.
- Output schema and task configuration.

Never use a global semantic cache that allows one tenant to receive another tenant's private answer. Recheck authorization before serving cached content.

### B. Avoid unnecessary model calls

Do not call an LLM for deterministic tasks.

Examples:

- Arithmetic and portfolio percentage calculations → deterministic Python or a validated financial calculation service.
- Permission checks → server-side authorization logic.
- Schema validation → Pydantic or JSON Schema.
- Duplicate detection → deterministic hashes where appropriate.
- Simple classification → a cheaper model only if a model is actually needed and meets the evaluation threshold.

This is often more effective than switching between expensive models.

### C. Measure cost per successful outcome

Do not optimize only for cost per model call.

A model that costs $0.01 per attempt but succeeds only 70% of the time may require retries or human intervention. Another model at $0.02 with a 95% success rate could be cheaper per successful task.

A useful metric is:

\\[ \text{Cost per successful task} = \frac{\text{Total AI and recovery cost}} {\text{Number of successful tasks}} \\]

Include fallback calls, failed attempts, evaluations, and other relevant costs in the numerator.

## 5. Handle model failures: classify the error before recovering

A common production mistake is treating every error the same way:

```
try:    response = await model.generate(prompt)except Exception:    response = await another_model.generate(prompt)
```

This is inadequate because some errors are transient, some require changing the request, and others should never be retried.

### 5.1 Build an explicit failure policy

| Failure                     | Typical example                         | Recommended action                                                                |
| --------------------------- | --------------------------------------- | --------------------------------------------------------------------------------- |
| Rate limit                  | HTTP 429                                | Respect `Retry-After`, retry with bounded backoff, or use an approved alternative |
| Temporary provider failure  | HTTP 500, 502, 503, 504                 | Retry selectively; use fallback when appropriate                                  |
| Network timeout             | Request exceeds deadline                | Retry only if budget and deadline allow                                           |
| Invalid credentials         | HTTP 401 or 403                         | Fail fast, alert, fix configuration or permissions                                |
| Invalid request             | HTTP 400                                | Fix request or schema; retrying unchanged is usually pointless                    |
| Context too large           | Input exceeds model limit               | Reduce or restructure context; then retry once if safe                            |
| Malformed structured output | Invalid JSON                            | Validate, optionally repair or retry within budget                                |
| Poor answer quality         | Unsupported claims or missing citations | Diagnose retrieval and generation; use a bounded correction path                  |
| Safety refusal              | Model refuses a request                 | Respect the refusal; do not use another model to bypass safety controls           |
| Provider outage             | Repeated failures across requests       | Open circuit breaker and route to a qualified alternative or fail safely          |

The exact status codes and error categories depend on the provider. For example, not every 5xx error is transient, and not every timeout means the provider failed to execute the request.

### 5.2 Use bounded retries with exponential backoff and jitter

For a retryable failure, a common delay strategy is:

\\[ d_n = \min(d\_{\max}, d_0 \times 2^n) + J \\]

Where \\(J\\) is a random jitter component.

Jitter prevents many concurrent requests from retrying at exactly the same time. In production, account for provider-specific `Retry-After` instructions, the overall deadline, and the maximum retry count.

Never retry indefinitely. Every retry consumes time, capacity, and potentially money.

### 5.3 Know when not to retry

- Invalid credentials: retrying the same request will not repair the credential.
- Invalid schema or unsupported model parameters: correct the request first.
- Exhausted budget: do not start another model invocation.
- User cancellation or expired deadline: stop unnecessary work.
- Non-idempotent external actions: do not blindly repeat them.
- Safety refusal: do not use model switching to circumvent it.

A timeout is especially important: the provider may have processed the request even if your application never received the response. This is one reason to distinguish a model-generation retry from retrying a payment, trade, or other external side effect.

## 6. Implement a resilient model gateway in Python

Here is a reusable design for FastAPI-based AI applications. The code demonstrates:

- A provider abstraction.
- Explicit retryable exceptions.
- A request deadline.
- Bounded retry attempts.
- Primary-to-fallback model switching.
- Output-token limits.
- Failure reporting.

It is an illustrative implementation; provider-specific API adapters, distributed budget reservations, circuit-breaker state, and production telemetry should be added around it.

### Step 1: Define the model contract

```
# app/ai/contracts.pyfrom dataclasses import dataclassfrom typing import Protocol@dataclass(frozen=True)class GenerationRequest:    prompt: str    model: str    max_output_tokens: int    temperature: float = 0.0@dataclass(frozen=True)class TokenUsage:    input_tokens: int    output_tokens: int@dataclass(frozen=True)class GenerationResult:    text: str    usage: TokenUsage    model: strclass ModelProvider(Protocol):    async def generate(        self,        request: GenerationRequest,    ) -> GenerationResult:        ...
```

The rest of the application depends on `ModelProvider`, not directly on an individual vendor's SDK. You can therefore implement adapters for multiple approved providers without changing your business workflow.

### Step 2: Define explicit failure types

```
# app/ai/errors.pyclass ModelError(Exception):    """Base exception for model gateway failures."""class RetryableModelError(ModelError):    """Temporary error that may succeed on retry."""class PermanentModelError(ModelError):    """Error requiring request/configuration correction."""class ModelDeadlineExceeded(ModelError):    """The request deadline has expired."""class ModelBudgetExceeded(ModelError):    """The request has exhausted its permitted budget."""class InvalidModelOutput(ModelError):    """The model response failed required validation."""
```

Provider adapters should map their SDK exceptions into these categories. Do not assume every SDK exception is safe to retry.

### Step 3: Implement retry and fallback

```
# app/ai/gateway.pyimport asyncioimport randomfrom dataclasses import dataclassfrom time import monotonicfrom app.ai.contracts import (    GenerationRequest,    GenerationResult,    ModelProvider,)from app.ai.errors import (    ModelBudgetExceeded,    ModelDeadlineExceeded,    PermanentModelError,    RetryableModelError,)@dataclass(frozen=True)class GatewayPolicy:    primary_model: str    fallback_model: str | None = None    max_attempts_per_model: int = 2    base_delay_seconds: float = 0.25    max_delay_seconds: float = 2.0    request_timeout_seconds: float = 12.0    max_output_tokens: int = 1500class ModelGateway:    def __init__(        self,        providers: dict[str, ModelProvider],        policy: GatewayPolicy,    ):        self.providers = providers        self.policy = policy    async def generate(        self,        prompt: str,        *,        deadline: float,        output_token_limit: int | None = None,    ) -> GenerationResult:        now = monotonic()        if now >= deadline:            raise ModelDeadlineExceeded("Request deadline expired")        output_limit = min(            output_token_limit or self.policy.max_output_tokens,            self.policy.max_output_tokens,        )        if output_limit < 1:            raise ModelBudgetExceeded("No output token budget available")        models = [self.policy.primary_model]        if self.policy.fallback_model:            models.append(self.policy.fallback_model)        last_error: Exception | None = None        for model in models:            provider = self.providers.get(model)            if provider is None:                # Configuration errors should be caught at startup.                raise PermanentModelError(                    f"No provider configured for model: {model}"                )            for attempt in range(self.policy.max_attempts_per_model):                remaining = deadline - monotonic()                if remaining <= 0:                    raise ModelDeadlineExceeded(                        "Request deadline expired"                    ) from last_error                timeout = min(                    remaining,                    self.policy.request_timeout_seconds,                )                request = GenerationRequest(                    prompt=prompt,                    model=model,                    max_output_tokens=output_limit,                )                try:                    async with asyncio.timeout(timeout):                        result = await provider.generate(request)                    if not result.text.strip():                        raise PermanentModelError(                            "Model returned empty output"                        )                    return result                except asyncio.CancelledError:                    # Preserve caller cancellation.                    raise                except TimeoutError as exc:                    # A provider adapter may need to distinguish a                    # timeout from an asyncio timeout in its SDK.                    last_error = RetryableModelError(                        "Model call timed out"                    )                    last_error.__cause__ = exc                except RetryableModelError as exc:                    last_error = exc                except PermanentModelError:                    # Do not repeat an unchanged invalid request.                    raise                if attempt + 1 < self.policy.max_attempts_per_model:                    delay_cap = min(                        self.policy.max_delay_seconds,                        self.policy.base_delay_seconds * (2 ** attempt),                    )                    delay = random.uniform(0, delay_cap)                    remaining = deadline - monotonic()                    if remaining <= delay:                        raise ModelDeadlineExceeded(                            "Insufficient time for retry"                        ) from last_error                    await asyncio.sleep(delay)                # If this model keeps failing, try the next configured                # model, provided the deadline still permits it.        raise RetryableModelError(            "Primary and fallback models failed"        ) from last_error
```

Example use:

```
from time import monotonicfrom app.ai.gateway import GatewayPolicy, ModelGatewaypolicy = GatewayPolicy(    primary_model="balanced-model",    fallback_model="approved-fallback",    max_attempts_per_model=2,    request_timeout_seconds=8,    max_output_tokens=1200,)gateway = ModelGateway(    providers=provider_adapters,    policy=policy,)async def answer_question(prompt: str):    deadline = monotonic() + 15    return await gateway.generate(        prompt,        deadline=deadline,        output_token_limit=1000,    )
```

`provider_adapters` represents the application’s configured mapping of model identifiers to provider adapters.

### What this implementation does not solve by itself

A senior engineer should explicitly identify these limitations before deploying it:

1. Provider-specific retries: `Retry-After`, connection failures, and SDK-specific error codes must be handled by the adapter or retry policy.
2. Budget accounting: the example caps output tokens but does not reserve or reconcile monetary spend across requests.
3. Circuit breaking: repeated failures still cause requests to try the provider. Add a shared circuit breaker.
4. Quality validation: a non-empty response is not necessarily correct.
5. Fallback compatibility: the alternative model must support the required schema, context size, safety rules, and quality level.
6. Streaming: streaming needs separate handling for failures after response tokens have already been sent to the client.
7. Overall deadline: the outer API and workflow must enforce the same deadline so later stages do not keep running after the gateway returns.

Also, some providers return HTTP 200 with a refusal or malformed output. HTTP success alone is not proof that the task succeeded.

## 7. Add a circuit breaker to protect the system during an outage

Retries help when a failure is temporary. They are counterproductive when a provider is continuously failing.

A circuit breaker has three states:

## CLOSED

Normal traffic flows to the provider. Failures are counted.

## OPEN

After the failure threshold, requests skip the unhealthy provider and use an approved fallback or fail safely.

## HALF-OPEN

After a cooldown, a small number of probe requests test recovery. Success closes the circuit; failure reopens it.

For example, if a provider has five consecutive transient failures, you might open the circuit for 30 seconds. Those values must be tuned to your traffic and service-level objectives.

Important implementation details:

- Count qualifying provider failures, not application validation errors or user mistakes.
- Use a sliding window or rolling failure ratio when traffic is high.
- Keep circuit state shared across workers if all workers need a consistent view.
- Prevent a burst of half-open probe requests.
- Maintain separate health state for different providers or model routes.
- Do not send all fallback traffic to another provider without checking its capacity.

A circuit breaker reduces wasted calls, latency, and retry amplification. It does not fix the underlying provider problem.

## 8. Fallback is a business decision, not merely a technical one

A fallback model must preserve the guarantees of the task.

Consider a financial compliance platform that reviews marketing content against internal policy documents.

Primary model fails

Approved fallback with equivalent capabilities

Retry within budget, validate the result, and record which model produced it.

Fallback has not passed compliance evaluation

Do not silently substitute it for a compliance decision. Return a controlled unavailable or pending-review result.

For high-risk tasks, the fallback should satisfy the same essential constraints:

- Approved model and provider.
- Required data residency and privacy requirements.
- Validated structured output.
- Acceptable quality on representative test cases.
- Equivalent tool restrictions and safety controls.
- Compatible context limits and latency requirements.

If the fallback cannot meet these requirements, degrade gracefully rather than fabricate confidence.

For example:

```
from enum import Enumclass ReviewStatus(str, Enum):    COMPLETED = "completed"    PENDING_REVIEW = "pending_review"    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"def handle_review_failure() -> dict:    return {        "status": ReviewStatus.PENDING_REVIEW.value,        "decision": None,        "reason": (            "Automated review could not be completed reliably. "            "Human review is required."        ),    }
```

This is a safe application-level response, not a model-generated substitute. Your API should return an appropriate HTTP status or business status based on its contract.

For example, a transient outage may be represented as HTTP 503, while an accepted asynchronous job awaiting review may return HTTP 202. The choice depends on whether work has actually been durably accepted.

## 9. Validate output quality before declaring success

There are two independent questions:

1. Did the model call technically succeed?
2. Did the model produce a result that is valid and useful for the task?

For structured output, validate the response using a schema.

```
from pydantic import BaseModel, Fieldclass ComplianceFinding(BaseModel):    rule_id: str    risk_level: str    explanation: str    evidence_quotes: list[str] = Field(min_length=1)class ComplianceReview(BaseModel):    findings: list[ComplianceFinding]    requires_human_review: bool
```

After parsing the model output, add domain validation:

- Does every referenced rule ID exist?
- Does each quoted evidence passage exist in the retrieved documents?
- Is the risk level in the permitted enumeration?
- Does the output contain required fields?
- Does the output obey the requested tenant and document scope?
- Are any claims unsupported by evidence?

A schema-valid answer can still be factually wrong. For a RAG workflow, verify that citations correspond to actual retrieved evidence. For financial calculations, independently verify numerical outputs using deterministic code.

### What if the answer is invalid?

Use a bounded recovery sequence:

1. Attempt schema parsing.
2. If the failure is repairable, perform one constrained correction or regeneration attempt.
3. Revalidate the result.
4. If validation still fails, try a qualified fallback if budget and deadline permit.
5. If the result remains invalid, return a safe failure or send it for human review.

Do not let the evaluator repeatedly call the model until it approves an answer. Evaluation and correction also consume money and can introduce new errors.

## 10. Enforce cost and deadlines at the FastAPI boundary

The model gateway is one layer. The API must also enforce limits before expensive work begins.

A typical request path is:

1. Authenticate the caller.
2. Resolve tenant and authorization scope.
3. Apply rate limits.
4. Check quota and budget availability.
5. Establish a request deadline.
6. Run the AI workflow under those limits.
7. Reconcile actual usage.
8. Record success, failure, latency, and cost.

A simplified endpoint:

```
from time import monotonicfrom fastapi import APIRouter, HTTPExceptionfrom pydantic import BaseModel, Fieldfrom app.ai.errors import (    ModelBudgetExceeded,    ModelDeadlineExceeded,    RetryableModelError,)router = APIRouter()class QuestionRequest(BaseModel):    question: str = Field(min_length=1, max_length=8000)@router.post("/ai/answer")async def answer_question(request: QuestionRequest):    deadline = monotonic() + 15.0    # Production steps omitted here:    # authenticate user, authorize tenant, apply rate limits,    # reserve a shared monetary budget, and establish a trace ID.    try:        result = await answer_service.answer(            question=request.question,            deadline=deadline,        )        return {            "answer": result.text,            "model": result.model,            "usage": {                "input_tokens": result.usage.input_tokens,                "output_tokens": result.usage.output_tokens,            },        }    except ModelBudgetExceeded:        raise HTTPException(            status_code=429,            detail="AI usage budget exhausted",        )    except ModelDeadlineExceeded:        raise HTTPException(            status_code=504,            detail="AI request deadline exceeded",        )    except RetryableModelError:        raise HTTPException(            status_code=503,            detail="AI service temporarily unavailable",            headers={"Retry-After": "5"},        )
```

This endpoint is an illustrative skeleton. In production, `answer_service` must enforce the deadline across all stages, the budget must be checked atomically, and the API must distinguish tenant quota exhaustion from temporary provider throttling. The error mapping and retry instructions should follow the API contract.

Also, never expose raw provider errors, internal prompts, credentials, or sensitive document contents to clients.

## 11. Make cost and failure observable

As a staff-level engineer, I would not consider this complete until the system can answer:

- Which model is costing us the most?
- Which tenants consume the most budget?
- Are retries increasing spending?
- Is the primary model unhealthy?
- Is the fallback improving availability without reducing quality?
- Are failures caused by the provider, application, retrieval pipeline, or invalid output?
- What is the cost per successful business task?

### Essential metrics

| Metric                                 | Why it matters                             |
| -------------------------------------- | ------------------------------------------ |
| `llm_input_tokens_total`               | Tracks input consumption                   |
| `llm_output_tokens_total`              | Tracks generated output                    |
| `llm_cost_estimated_total`             | Estimates spend by model, task, and tenant |
| `llm_request_duration_seconds`         | Measures latency                           |
| `llm_request_failures_total`           | Identifies error patterns                  |
| `llm_retries_total`                    | Detects retry amplification                |
| `llm_fallback_total`                   | Shows dependency degradation               |
| `llm_budget_exceeded_total`            | Identifies budget pressure                 |
| `llm_output_validation_failures_total` | Measures invalid results                   |
| `ai_task_success_total`                | Measures successful business outcomes      |

Track model, provider, task type, and environment as metric dimensions. Keep high-cardinality values such as raw prompts, user IDs, and request IDs in appropriately protected logs or traces instead of metric labels.

For financial or compliance systems, also record the model and prompt versions, retrieval references, evaluation outcome, and fallback reason in an access-controlled audit trail.

### Example alerting policies

These are illustrative thresholds to tune against your baseline.

Cost alert

Notify when daily spend exceeds the forecast or a configured budget threshold.

Reliability alert

Alert when the rolling failure rate exceeds the service's error budget or availability target.

Retry amplification alert

Alert when retry attempts per successful task rise sharply.

Quality alert

Alert when citation validity, groundedness, structured-output validity, or task success falls below its accepted threshold.

Cost dashboards should reconcile estimates with provider billing records. Token-based estimates alone may not capture all billable features.

## 12. Test failure scenarios before production

I would use unit tests, integration tests, load tests, and fault injection.

| Test scenario                    | Expected behavior                                                  |
| -------------------------------- | ------------------------------------------------------------------ |
| Primary model returns 429        | Respect rate-limit instructions and retry within limits            |
| Primary model returns 503        | Retry selectively, then use approved fallback                      |
| Both models fail                 | Return a controlled failure; do not fabricate an answer            |
| Model returns invalid JSON       | Reject output and use bounded correction                           |
| Budget is exhausted              | Stop before starting another expensive call                        |
| Request deadline expires         | Cancel unnecessary work and stop retries                           |
| Provider returns a slow response | Enforce timeout and preserve API responsiveness                    |
| Traffic spikes                   | Rate limits, concurrency limits, and backpressure protect capacity |
| Fallback is low quality          | Reject it or route the task to human review                        |
| Agent enters a loop              | Enforce step and call limits                                       |
| Same external action is retried  | Prevent duplicate side effects using idempotency controls          |

A useful test for the gateway is to mock the provider so that the primary fails twice and the fallback succeeds:

```
import pytestfrom app.ai.errors import RetryableModelErrorclass FailingProvider:    async def generate(self, request):        raise RetryableModelError("Simulated outage")class WorkingProvider:    async def generate(self, request):        from app.ai.contracts import (            GenerationResult,            TokenUsage,        )        return GenerationResult(            text="Answer based on verified evidence.",            usage=TokenUsage(                input_tokens=100,                output_tokens=20,            ),            model=request.model,        )@pytest.mark.asyncioasync def test_gateway_falls_back_to_healthy_model():    from time import monotonic    from app.ai.gateway import GatewayPolicy, ModelGateway    gateway = ModelGateway(        providers={            "primary": FailingProvider(),            "fallback": WorkingProvider(),        },        policy=GatewayPolicy(            primary_model="primary",            fallback_model="fallback",            max_attempts_per_model=2,            request_timeout_seconds=2,        ),    )    result = await gateway.generate(        "Summarize the verified evidence.",        deadline=monotonic() + 10,    )    assert result.model == "fallback"    assert result.text
```

This test verifies a basic recovery path. Your full suite should also verify retry counts, deadlines, cancellation, budget enforcement, output validation, circuit-breaker transitions, and the behavior when every provider fails.

## 13. How I would roll this out in an enterprise platform

I would introduce these controls incrementally rather than changing every model and workflow at once.

1. Baseline current behavior. Measure token usage, cost per successful task, latency percentiles, error rates, retry counts, and quality.
2. Add centralized model adapters and the gateway. Standardize timeouts, exception mapping, usage accounting, and structured output handling.
3. Introduce budgets and deadlines. Begin with conservative limits, then tune them using production traffic and evaluation results.
4. Add resilience controls. Implement bounded retries, circuit breakers, concurrency limits, and tested fallback routes.
5. Evaluate cheaper model routes offline. Run representative evaluation datasets, including difficult and high-risk cases.
6. Canary the changes. Route a small share of eligible traffic through the new policy and compare cost, latency, failure rate, and answer quality.
7. Expand with rollback controls. Increase traffic only if quality and reliability remain within defined thresholds. Keep model routing configurable and versioned.

For multi-agent systems, I would apply these controls both at the individual model-call level and at the overall workflow level. A planner, specialist agents, retrieval graders, and final writer should all share a request-level budget and deadline.

## 14. Senior AI engineer interview answer

In production, I handle model cost and model failures through a centralized model gateway, request-level budgets, intelligent routing, and resilience policies.

First, I establish cost controls at multiple levels: per request, tenant, feature, and organization. I track input and output token usage, estimate the cost of each model call, and reconcile that estimate with provider billing. I also impose limits on output tokens, model calls, agent steps, parallel execution, and request duration.

Second, I use model routing based on task complexity, measured quality, latency, risk, and cost. Simple classification and extraction tasks can use smaller models, while complex reasoning or compliance tasks use models that have passed the required evaluation criteria. I optimize for cost per successful task, not just cost per token.

Third, I classify model failures. Rate limits and transient server errors may justify bounded retries with exponential backoff and jitter. Invalid requests and authentication failures require correction rather than repeated retries. I enforce deadlines and use circuit breakers to stop sending traffic to unhealthy providers.

Fourth, I implement fallback models through a provider-independent gateway. A fallback must satisfy the same essential quality, security, data-handling, and output-contract requirements. If no qualified model is available, the system returns a controlled failure or routes the task for human review.

Fifth, I validate model output instead of assuming HTTP success means business success. I use structured schemas, citation validation, domain-specific checks, and deterministic calculations where possible. In agentic workflows, I enforce shared budgets and step limits to prevent runaway loops and unexpected cost.

Finally, I instrument token usage, cost, latency, failures, retries, fallback frequency, and task quality. I test outages, timeouts, rate limits, malformed outputs, and budget exhaustion, then deploy changes through canary releases with rollback mechanisms.

The goal is to minimize cost while maintaining the required quality, availability, security, and auditability—not to minimize model spending at the expense of correctness.

The key staff-level principle is this: cost control, reliability, and answer quality must be designed together. A cheap model that produces incorrect answers, or a resilient retry policy that multiplies spending, is not a successful production design.