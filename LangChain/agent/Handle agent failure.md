# How to handle agent failure in LangGraph — senior/staff AI engineer approach

In a production multi-agent AI system, agent failure should be an expected operating condition, not an exceptional situation that brings down the entire application.

Consider your financial compliance review platform. The performance agent may time out, the policy retrieval agent may return no evidence, the risk agent may produce invalid JSON, or the evaluator may repeatedly reject the generated report.

A robust system needs to determine:

- Should the failed operation be retried?
- Should it fall back to another model or retrieval strategy?
- Can the workflow continue with partial results?
- Must the review stop and request human intervention?
- How do we prevent duplicate side effects when retrying or resuming?

The answer depends on the failure type and the business risk. For compliance workflows, an unavailable specialist must never silently become a successful review.

## 1. Classify failures before designing recovery

Transient infrastructure failure

Examples: HTTP 429, temporary network error, provider 503, connection reset.

Response: Retry with exponential backoff and jitter; honor `Retry-After`; use a circuit breaker if failures persist.

Model or output failure

Examples: malformed JSON, schema violations, unsupported citations, incomplete output.

Response: Validate, perform bounded repair or regeneration, then reject the output if it remains invalid.

Retrieval or tool failure

Examples: no relevant policies, database timeout, tool permission denial.

Response: Distinguish a legitimate empty result from a failed query. Retry transient failures; never bypass authorization to obtain more results.

Workflow or reasoning failure

Examples: infinite rework loop, planner oscillation, contradictory agent findings.

Response: Enforce step budgets, detect conflicts, route to a controlled terminal state, and request human review when necessary.

The important engineering decision is to distinguish a retryable failure from a business failure. A timeout may be temporary; a missing mandatory policy document is not something you can fix by blindly retrying the same LLM call.

## 2. Use the right recovery strategy for each failure

A senior engineer does not retry every error. Retries can make outages worse, waste tokens, or repeat dangerous operations.

| Failure                           | Recovery strategy                          | Continue workflow?                        |
| --------------------------------- | ------------------------------------------ | ----------------------------------------- |
| Temporary provider/network error  | Bounded retry with backoff                 | Usually                                   |
| HTTP 429 or 503                   | Respect rate limits; retry within deadline | Usually                                   |
| Invalid structured output         | Bounded validation/repair attempt          | Only if valid afterward                   |
| Missing policy evidence           | Alternate retrieval or request review      | Only if the required evidence is obtained |
| Permission denied                 | Stop that operation; log and report safely | Only through an authorized path           |
| Contradictory specialist findings | Validate, reconcile, or escalate           | Depends on policy                         |
| Repeated evaluator rejection      | Enforce rework limit; request human review | Not as an approved result                 |
| Unknown programming error         | Capture diagnostics; fail safely           | Usually not without investigation         |

LangGraph supports node-level retry policies, timeouts, and error handlers in current versions. Retries occur before the error handler, which can route execution to a recovery node. Check the installed version because these node-level timeout and error-handler APIs require LangGraph 1.2 or later.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 3. Implement bounded retries for transient failures

Suppose your policy retrieval agent calls PostgreSQL or an external search service. A temporary connection failure is a reasonable candidate for retry.

```
from typing_extensions import TypedDictfrom langgraph.graph import StateGraph, START, ENDfrom langgraph.types import RetryPolicyclass RetrievalState(TypedDict, total=False):    query: str    evidence: list[dict]    status: strdef retrieve_policy_evidence(state: RetrievalState) -> dict:    # Replace with your async repository/service call.    # Raise a retryable exception for a transient failure.    evidence = policy_repository.search(state["query"])    return {        "evidence": evidence,        "status": "retrieved" if evidence else "no_evidence",    }builder = StateGraph(RetrievalState)builder.add_node(    "retrieve",    retrieve_policy_evidence,    retry_policy=RetryPolicy(        max_attempts=3,       # Includes the initial attempt        initial_interval=1.0,        backoff_factor=2.0,        max_interval=8.0,        jitter=True,    ),)
```

The retry intervals increase after failures, with jitter to reduce synchronized retries across workers. The exact behavior depends on the installed LangGraph version and configured policy.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference

+1



### Production considerations

- Retry only exceptions that are plausibly transient.
- Do not retry authorization failures or invalid business requests.
- Honor provider `Retry-After` guidance when applicable.
- Set client-side connection and read timeouts as well.
- Keep retry attempts within the overall request deadline.
- Avoid stacking aggressive retries in the HTTP client, SDK, Celery worker, and graph simultaneously. Their combined retry counts can multiply.

For example, three attempts in the graph combined with three attempts in an HTTP client can produce up to nine underlying requests for one logical operation.

## 4. Add a fallback when retries are exhausted

Retries are not recovery by themselves. If the primary model or retrieval service continues to fail, your graph needs a defined next step.

For example, a primary policy search may use hybrid vector retrieval, while a fallback uses a separately implemented keyword search. The fallback must preserve tenant and document permissions; it must not weaken access controls.

```
from typing_extensions import TypedDictfrom langgraph.graph import StateGraph, START, ENDfrom langgraph.errors import NodeErrorfrom langgraph.types import Command, RetryPolicyclass EvidenceState(TypedDict, total=False):    query: str    evidence: list[dict]    status: str    failure_reason: strdef primary_search(state: EvidenceState) -> dict:    # Can raise a transient exception.    evidence = hybrid_policy_search(state["query"])    return {        "evidence": evidence,        "status": "retrieved" if evidence else "no_evidence",    }def primary_search_failed(    state: EvidenceState,    error: NodeError,) -> Command:    # Do not expose raw exception details to the end user.    # Internal logs should record sanitized diagnostic context.    return Command(        update={            "status": "primary_search_failed",            "failure_reason": "Primary retrieval unavailable",        },        goto="fallback_search",
```

`hybrid_policy_search` and `keyword_policy_search` are illustrative application services. They need to be implemented, and both must enforce the same access-control rules.

The error handler runs after the primary node exhausts its retry attempts. The `Command` updates the state and routes execution to the fallback node.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



Important: this example handles failure in the primary search node. If the fallback also fails, it returns an explicit degraded status rather than inventing evidence. In a real compliance workflow, downstream nodes must inspect that status and must not treat it as a successful review.

## 5. Handle timeouts as first-class failures

An agent can hang even if it never throws an exception. Examples include an unresponsive model provider, a blocked network call, or an external tool waiting indefinitely.

For async nodes in LangGraph 1.2 or later, use a timeout policy:

```
from langgraph.types import TimeoutPolicy, RetryPolicybuilder.add_node(    "analyze_claim",    analyze_claim_async,    timeout=TimeoutPolicy(        run_timeout=30,        idle_timeout=10,    ),    retry_policy=RetryPolicy(        max_attempts=2,        initial_interval=1.0,        backoff_factor=2.0,        jitter=True,    ),)
```

Here, `run_timeout` caps the duration of an attempt, while `idle_timeout` limits the time without observable progress. A timeout can trigger the node's retry policy, and an exhausted retry sequence can trigger its error handler.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



A node timeout is not a substitute for end-to-end deadlines. Your FastAPI request, HTTP client, database operations, and background worker should also have explicit timeouts. Cancellation of external work must be handled carefully: a timed-out request does not necessarily mean the remote system did not complete the operation.

## 6. Handle invalid LLM output separately from infrastructure failures

Suppose the performance agent returns:

```
{
  "severity": "extreme",
  "approved": true,
  "policy_id": "FAKE-999"
}
```

This is not necessarily a network failure. It is a contract violation: the severity is invalid, the agent is attempting to set an unauthorized approval field, and the policy reference may be fabricated.

The correct response is to validate the output and permit only a bounded repair attempt.

```
from typing import Literalfrom pydantic import BaseModel, ConfigDict, Field, ValidationErrorclass AgentFinding(BaseModel):    model_config = ConfigDict(extra="forbid", frozen=True)    severity: Literal["low", "medium", "high", "critical"]    status: Literal["supported", "unsupported", "uncertain"]    rationale: str = Field(min_length=1, max_length=3000)    policy_ids: tuple[str, ...]def parse_agent_output(raw: dict, allowed_policy_ids: set[str]):    finding = AgentFinding.model_validate(raw)    if not set(finding.policy_ids).issubset(allowed_policy_ids):        raise ValueError("Agent referenced policy evidence outside its scope")    return findingdef analyze_with_bounded_repair(model, prompt, allowed_policy_ids):    last_error = None    for attempt in range(2):  # Initial generation + at most one repair        raw = model.invoke(prompt)        try:            return parse_agent_output(raw, allowed_policy_ids)        except (ValidationError, ValueError) as exc:            last_error = exc            prompt = (                f"{prompt}\n"                f"Your previous output violated the contract: {exc}. "
```

This is a simplified pattern; production code should also normalize model responses, redact sensitive exception content, and use a structured-output mechanism supported by the selected model.

Notice that the repair loop is bounded. If validation still fails, the error propagates to the surrounding recovery policy.

A staff-level distinction:

- Transient exception: retry the operation.
- Schema violation: attempt bounded repair or reject the result.
- Unsupported claim: request better evidence or mark it uncertain.
- Permission violation: block the operation.
- Business disagreement: reconcile findings or escalate.

These failures should not all be routed through the same generic retry handler.

## 7. Handle parallel agent failures without losing trustworthy results

Imagine three independent agents running in parallel:

- Performance analysis succeeds.
- Risk analysis succeeds.
- Fee analysis times out.

Should the entire workflow fail?

The answer depends on whether the missing result is mandatory. For a financial compliance review, you may preserve the first two results, but you must not mark the overall review complete if fee analysis is required.

A robust architecture gives every specialist a structured outcome:

```
from typing import Literalfrom pydantic import BaseModel, ConfigDict, Fieldclass AgentOutcome(BaseModel):    model_config = ConfigDict(extra="forbid", frozen=True)    agent_name: Literal["performance", "risk", "fees"]    status: Literal["success", "failed", "skipped"]    findings: tuple[dict, ...] = ()    error_code: str | None = Nonedef aggregate_outcomes(    outcomes: list[AgentOutcome],    required_agents: set[str],) -> dict:    by_agent = {outcome.agent_name: outcome for outcome in outcomes}    missing = required_agents - set(by_agent)    failed = {        name for name in required_agents        if name in by_agent and by_agent[name].status != "success"    }    if missing or failed:        return {            "review_status": "incomplete",            "needs_human_review": True,            "missing_agents": sorted(missing),            "failed_agents": sorted(failed),        }    return {        "review_status": "ready_for_validation",
```

This aggregation function is independent of the graph. In the actual LangGraph workflow, each specialist should return a result or a structured failure outcome, and a fan-in node should aggregate them according to the required-agent policy.

For a fail-closed compliance workflow, a missing mandatory specialist means `incomplete`, not `approved`. Partial results are useful for diagnosis and human review, but they do not override required checks.

## 8. Make recovery safe with checkpointing and idempotency

One of the hardest production issues is a failure after a side effect has happened but before the workflow records success.

For example:

1. An agent asks a service to publish a report.
2. The service publishes it successfully.
3. The worker crashes before the graph checkpoints the completed step.
4. The workflow resumes and tries to publish the report again.

A retry can duplicate the action unless the external operation is idempotent.

Use a stable idempotency key derived from the workflow and logical action:

```
import hashlibdef make_idempotency_key(    workflow_id: str,    action: str,    resource_id: str,) -> str:    value = f"{workflow_id}:{action}:{resource_id}"    return hashlib.sha256(value.encode()).hexdigest()def publish_report_safely(    *,    workflow_id: str,    report_id: str,    publishing_service,):    key = make_idempotency_key(        workflow_id,        "publish_compliance_report",        report_id,    )    return publishing_service.publish(        report_id=report_id,        idempotency_key=key,    )
```

The publishing service must actually honor the idempotency key, ideally with a durable uniqueness constraint or equivalent atomic deduplication. Generating the key alone does not guarantee idempotency.

For long-running workflows, use a durable LangGraph checkpointer and a stable `thread_id`. On resume, revalidate permissions and the current state of external resources. Interrupt-based human approval also requires care: LangGraph can restart the interrupted node from its beginning when resumed, so side effects before an `interrupt()` must be designed to tolerate replay.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 9. Route failures to human review when automation cannot recover

For your compliance platform, I would define terminal workflow statuses explicitly:

| Status               | Meaning                                                       |
| -------------------- | ------------------------------------------------------------- |
| `completed`          | Required checks passed and validation succeeded               |
| `incomplete`         | One or more required checks did not finish                    |
| `needs_human_review` | Evidence, findings, or business decisions need human judgment |
| `failed`             | The workflow could not safely produce a usable result         |

Do not use a single `success: false` flag for every situation. It makes it difficult for API clients, dashboards, retry workers, and human reviewers to determine what should happen next.

A human-in-the-loop node can pause the workflow when automated recovery is exhausted:

```
from langgraph.types import interruptdef request_compliance_review(state: dict) -> dict:    decision = interrupt({        "reason": "Automated review could not safely complete",        "review_id": state["review_id"],        "failure_summary": state["failure_summary"],        "proposed_status": "needs_human_review",    })    # Validate the resumed decision in trusted application code.    validated_decision = validate_reviewer_decision(decision)    return {        "review_status": validated_decision["status"],        "reviewer_id": validated_decision["reviewer_id"],    }
```

`validate_reviewer_decision` is application-specific. It must verify the reviewer's identity, authorization, allowed decision values, and the review's current version. Never accept an arbitrary client-supplied reviewer ID as proof of authority.

When the graph resumes, recheck that the reviewer is still authorized and that the review has not changed since approval was requested. LangGraph's interrupt mechanism supports this pause-and-resume pattern when used with a checkpointer.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 10. Observe failures in production

I would instrument every agent execution with:

- Workflow ID, node name, tenant-safe correlation ID, and model identifier.
- Attempt number and retry reason.
- Duration, timeout count, and provider error category.
- Token usage and estimated cost.
- Output-validation failures and fallback usage.
- Partial-result and human-escalation rates.

Track metrics such as:

| Metric                      | Why it matters                                |
| --------------------------- | --------------------------------------------- |
| Agent failure rate          | Identifies unreliable nodes or dependencies   |
| Retry recovery rate         | Shows whether retries actually help           |
| Fallback activation rate    | Detects primary-service degradation           |
| Invalid output rate         | Measures model/contract reliability           |
| Human escalation rate       | Indicates unresolved cases or weak automation |
| End-to-end p95 latency      | Reveals the cost of retries and recovery      |
| Duplicate side-effect count | Detects idempotency problems                  |

Log structured error categories rather than unrestricted prompts, raw sensitive documents, or secrets. In a financial domain, observability must not become a second channel for leaking protected data.

## 11. Recommended architecture

Execute agent node

Validate output and classify error

Retry

Transient error

Fallback

Alternate strategy

Escalate

Unsafe to continue

Validated aggregation and final status

The critical design decision is that recovery is a separate workflow concern, not something delegated entirely to the failed agent. An agent can suggest how to recover, but trusted graph logic and application services decide whether that recovery is allowed.

## 12. Interview-ready answer

> “I handle agent failures through layered fault tolerance. I classify errors into transient infrastructure failures, output-validation failures, business failures, authorization violations, and unexpected errors. I use bounded retries with exponential backoff and jitter for transient failures, per-node timeouts to prevent hung executions, and fallback strategies when a primary dependency remains unavailable. After retries are exhausted, I route to a recovery handler or a controlled terminal state. For parallel agents, I preserve valid partial results but prevent missing mandatory checks from being treated as success. I also use checkpointing, idempotency keys, structured observability, and human-in-the-loop escalation. In compliance workflows, the system fails closed whenever it cannot establish that the required checks have completed safely.”

That is the staff-level mindset: design for failure, bound recovery, preserve trustworthy work, and never turn uncertainty into false success.