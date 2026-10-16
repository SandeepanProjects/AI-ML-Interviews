# How to prevent an agent from corrupting shared state in LangGraph

At senior/staff AI engineer level, the core principle is:

Do not let every agent freely mutate the entire workflow state. Give each agent the minimum state it needs, validate its output, and control how that output is merged into shared state.

This becomes especially important in multi-agent systems using LangGraph, where a planner, supervisor, retrieval agent, compliance agent, and evaluator may all contribute to the same workflow.

Consider your financial compliance review platform. Multiple agents independently review a marketing claim:

- The retrieval agent finds relevant policy evidence.
- The performance agent checks return-related claims.
- The risk agent checks risk disclosures.
- The fee agent checks fee disclosures.
- The evaluator validates the findings.
- The supervisor aggregates the results.

What happens if the performance agent accidentally overwrites the risk agent's findings? Or an LLM returns `"approved": true` even though the evidence does not support that decision?

Those are shared-state integrity problems. They require architectural controls, not just better prompts.

## 1. Understand the main failure modes

1\. Parallel write conflicts

Two agents write to the same state field. Without an appropriate reducer, LangGraph may raise an invalid concurrent update error; with an unsafe reducer, one result may overwrite another.

2\. Unauthorized field mutation

A specialist agent changes fields it should not control, such as `tenant_id`, `permissions`, or `approved`.

3\. Invalid or hallucinated output

An LLM returns malformed JSON, an invalid status, unsupported policy references, or unexpected fields.

4\. Mutable-object aliasing

Two components share a mutable dictionary or list. One component modifies it in place, unexpectedly affecting another component's data.

5\. Retry and replay corruption

A retried agent appends duplicate findings, repeats a side effect, or replays an outdated result after a checkpoint resume.

The solution is a combination of typed state, ownership boundaries, controlled reducers, output validation, immutable evidence, and explicit approval transitions.

## 2. Design state ownership before writing agents

A common mistake is to define one large dictionary and allow every node to write into it.

### Fragile design

```
class SharedState(TypedDict):    tenant_id: str    user_permissions: list[str]    policy_evidence: list[dict]    performance_findings: list[dict]    risk_findings: list[dict]    fee_findings: list[dict]    approved: bool    final_report: str
```

This schema describes the data, but it does not enforce which agent is allowed to modify each field. `TypedDict` primarily helps with static typing; it is not a runtime security boundary.

### Better design: separate state by responsibility

Trusted workflow context

Tenant, user identity, permissions, request ID

Read-only task input

Claim text, authorized policy evidence, task ID

Performance

Own findings

Risk

Own findings

Fees

Own findings

Trusted aggregator and validator

Validates schemas, provenance, conflicts and required checks

Approval service

Owns final approval and publication decisions

The important architectural boundary is that specialist agents produce findings, not authoritative approval decisions. The approval service decides whether a review can be approved based on validated findings and business rules.

LangGraph nodes return partial state updates, and each state key has a reducer that controls how those updates are applied. Without an explicit reducer, the default behavior is replacement; concurrent writes to the same single-value field can fail.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 3. Use separate output schemas and validate every agent

Use Pydantic models to validate the structure and allowed values of agent outputs. Do not ask the LLM to return the whole workflow state.

```
from typing import Literalfrom pydantic import BaseModel, ConfigDict, Fieldclass ComplianceFinding(BaseModel):    model_config = ConfigDict(extra="forbid", frozen=True)    finding_id: str = Field(min_length=1, max_length=100)    specialist: Literal["performance", "risk", "fees"]    severity: Literal["low", "medium", "high", "critical"]    claim: str = Field(min_length=1, max_length=2_000)    rationale: str = Field(min_length=1, max_length=4_000)    policy_ids: list[str] = Field(min_length=1, max_length=20)    status: Literal["supported", "unsupported", "uncertain"]class SpecialistResult(BaseModel):    model_config = ConfigDict(extra="forbid", frozen=True)    task_id: str    specialist: Literal["performance", "risk", "fees"]    findings: list[ComplianceFinding] = Field(max_length=100)
```

### Why these constraints matter

- `extra="forbid"` rejects unexpected fields instead of silently accepting them.
- `Literal` restricts values to approved categories.
- Field bounds constrain the size of the response.
- `frozen=True` prevents ordinary field reassignment on the validated model.
- `task_id` helps associate the result with the intended task.

However, schema validation does not prove that a finding is true. A model can produce a perfectly valid object with fabricated policy IDs. You must separately verify policy references against the authorized evidence supplied to the agent.

Also, a frozen Pydantic model is not automatically deeply immutable: nested mutable objects such as lists can still be modified. If deep immutability matters, use immutable nested types such as tuples and frozen models, or serialize the validated result into an immutable representation before sharing it.

## 4. Build a safe parallel LangGraph workflow

Here is a more complete implementation pattern. The three specialist agents write to different state fields, the aggregator validates their results, and only the trusted aggregator can set the final review status.

The specialist functions below are deterministic stand-ins for your LLM-backed agents. You can replace their bodies with model calls while retaining the same validation boundaries.

```
from typing import Literalfrom typing_extensions import TypedDictfrom pydantic import BaseModel, ConfigDict, Fieldfrom langgraph.graph import StateGraph, START, END# ---------- Validated agent output ----------class Finding(BaseModel):    model_config = ConfigDict(extra="forbid", frozen=True)    finding_id: str = Field(min_length=1, max_length=100)    specialist: Literal["performance", "risk", "fees"]    severity: Literal["low", "medium", "high", "critical"]    claim: str = Field(min_length=1, max_length=2_000)    rationale: str = Field(min_length=1, max_length=4_000)    policy_ids: tuple[str, ...] = Field(min_length=1, max_length=20)    status: Literal["supported", "unsupported", "uncertain"]class SpecialistResult(BaseModel):    model_config = ConfigDict(extra="forbid", frozen=True)    task_id: str    specialist: Literal["performance", "risk", "fees"]    findings: tuple[Finding, ...] = Field(max_length=100)# ---------- Shared workflow state ----------class ReviewState(TypedDict, total=False):    task_id: str    claim_text: str    authorized_policy_ids: list[str]    # Each specialist owns one result field.    performance_result: dict    risk_result: dict    fees_result: dict    # Written by the aggregator, not by specialists.    findings: list[dict]    review_status: str    validation_errors: list[str]# ---------- Specialist agents ----------# In production, replace the stub logic with LLM calls.# Validate the raw model output before returning it.def performance_agent(state: ReviewState) -> dict:    raw = {        "task_id": state["task_id"],        "specialist": "performance",        "findings": [            {                "finding_id": "perf-001",                "specialist": "performance",                "severity": "high",                "claim": state["claim_text"],                "rationale": "The return claim requires supporting evidence.",                "policy_ids": ["PERF-001"],                "status": "uncertain",            }        ],    }    result = SpecialistResult.model_validate(raw)    return {"performance_result": result.model_dump(mode="json")}def risk_agent(state: ReviewState) -> dict:    raw = {        "task_id": state["task_id"],        "specialist": "risk",        "findings": [            {                "finding_id": "risk-001",                "specialist": "risk",                "severity": "medium",                "claim": state["claim_text"],                "rationale": "Risk disclosure must be checked against policy.",                "policy_ids": ["RISK-001"],                "status": "uncertain",            }        ],    }    result = SpecialistResult.model_validate(raw)    return {"risk_result": result.model_dump(mode="json")}def fees_agent(state: ReviewState) -> dict:    raw = {        "task_id": state["task_id"],        "specialist": "fees",        "findings": [],    }    result = SpecialistResult.model_validate(raw)    return {"fees_result": result.model_dump(mode="json")}# ---------- Trusted aggregator ----------def aggregate_results(state: ReviewState) -> dict:    raw_results = [        state["performance_result"],        state["risk_result"],        state["fees_result"],    ]    authorized_ids = set(state["authorized_policy_ids"])    seen_finding_ids: set[str] = set()    all_findings: list[dict] = []    errors: list[str] = []    expected_specialists = {"performance", "risk", "fees"}    for raw in raw_results:        # Validate again at the trust boundary.        result = SpecialistResult.model_validate(raw)        if result.task_id != state["task_id"]:            errors.append("Result belongs to a different task.")            continue        if result.specialist not in expected_specialists:            errors.append("Unexpected specialist.")            continue        for finding in result.findings:            if finding.specialist != result.specialist:                errors.append("Finding specialist does not match its result.")                continue            if finding.finding_id in seen_finding_ids:                errors.append("Duplicate finding ID.")                continue            if not set(finding.policy_ids).issubset(authorized_ids):                errors.append("Finding references an unauthorized policy ID.")                continue            seen_finding_ids.add(finding.finding_id)            all_findings.append(finding.model_dump(mode="json"))    # Fail closed if the evidence/result contract is invalid.    if errors:        status = "needs_human_review"    elif any(        f["severity"] in {"high", "critical"}        or f["status"] == "uncertain"        for f in all_findings    ):        status = "needs_human_review"    else:        status = "ready_for_approval"    return {        "findings": all_findings,        "validation_errors": errors,        "review_status": status,    }# ---------- Graph assembly ----------builder = StateGraph(ReviewState)builder.add_node("performance_agent", performance_agent)builder.add_node("risk_agent", risk_agent)builder.add_node("fees_agent", fees_agent)builder.add_node("aggregate_results", aggregate_results)builder.add_edge(START, "performance_agent")builder.add_edge(START, "risk_agent")builder.add_edge(START, "fees_agent")# Fan-in: run only after all three specialist nodes finish.builder.add_edge(    ["performance_agent", "risk_agent", "fees_agent"],    "aggregate_results",)builder.add_edge("aggregate_results", END)graph = builder.compile()result = graph.invoke({    "task_id": "review-123",    "claim_text": "The fund delivers superior returns with low risk.",    "authorized_policy_ids": ["PERF-001", "RISK-001"],})print(result["review_status"])print(result["findings"])
```

### What this design protects

1. Parallel write isolation: each specialist writes a separate result field, so they do not race to overwrite one shared `findings` list.
2. Typed outputs: malformed output and unexpected fields are rejected.
3. Task integrity: results from a different task are rejected.
4. Provenance checks: findings must reference policy IDs included in the authorized evidence set.
5. Centralized aggregation: specialists cannot directly publish the final aggregated report through their returned state updates.

There are important limitations to understand. `TypedDict` and the graph schema are not a permission system. A node that receives the entire state can still read fields it should not need, and Python code can still return unexpected keys unless you validate the update boundary. For stronger isolation, pass agents a minimal task input through a node-specific input schema or an isolated subgraph, and use a trusted wrapper to validate the keys and values returned by each agent before they are accepted into the parent workflow.

Also, `ready_for_approval` is deliberately not the same as `approved`. Actual approval should be a separate authorized step with the required business rules and, where applicable, human sign-off.

## 5. Understand reducers: they can protect state—or corrupt it

LangGraph reducers determine how multiple updates to a state field are merged.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



Consider this common pattern:

```
import operatorfrom typing import Annotatedfrom typing_extensions import TypedDictclass State(TypedDict):    findings: Annotated[list[dict], operator.add]
```

If two parallel agents return:

```
# Agent A{"findings": [{"id": "A", "severity": "high"}]}# Agent B{"findings": [{"id": "B", "severity": "low"}]}
```

the reducer can combine both findings. That is useful, but it is not a complete integrity guarantee.

`operator.add` does not automatically:

- Deduplicate repeated findings.
- Validate finding contents.
- Prevent a finding from being associated with the wrong task.
- Resolve conflicting conclusions.
- Enforce agent-specific ownership.

For more advanced workflows, you can implement a reducer that merges findings by a stable identifier and rejects conflicting updates. Reducers should be deterministic, and their behavior must remain correct when updates are replayed or combined in different batches.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference

+1



For security-sensitive findings, I generally prefer separate agent-owned result fields followed by a validation-and-aggregation node. This makes ownership and conflict handling easier to reason about than an unrestricted shared list.

## 6. Protect against agents changing trusted fields

Never rely on an LLM prompt such as “Do not modify the tenant ID” as the security boundary.

Instead, keep trusted identity and authorization information outside the agent's writable output. In a production application, a node wrapper can enforce an allowlist:

```
from collections.abc import Callablefrom typing import Anydef enforce_update_contract(    node: Callable[..., dict[str, Any]],    *,    allowed_keys: set[str],) -> Callable[..., dict[str, Any]]:    def wrapped(state: dict[str, Any]) -> dict[str, Any]:        update = node(state)        if not isinstance(update, dict):            raise TypeError("Agent must return a dictionary update.")        unexpected = set(update) - allowed_keys        if unexpected:            raise ValueError(                f"Agent attempted unauthorized state update: "                f"{sorted(unexpected)}"            )        return update    return wrapped
```

For example, a risk agent might be allowed to write only `risk_result`. If it attempts to return `{"tenant_id": "another-tenant"}`, the wrapper rejects the update.

This wrapper is a useful application-level control, but it is not a substitute for limiting what the node can read. For stronger isolation:

- Give each node only the input fields it needs.
- Keep tenant identity and permissions in trusted runtime context.
- Derive database access rules from authenticated server-side identity.
- Validate every external tool call independently.
- Use a centralized state-update boundary for untrusted agent outputs.

The trusted workflow layer—not the LLM—should own authorization, approval, and publication.

## 7. Handle retries, checkpointing, and concurrent execution

Shared-state corruption can also occur even when every schema is correct.

| Risk                                     | Recommended control                                              |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Same finding appended after a retry      | Stable finding IDs and deduplication                             |
| Database write repeated after a resume   | Idempotency key and database uniqueness constraint               |
| Stale agent result overwrites newer work | Task/version IDs and optimistic concurrency checks               |
| Two workers update the same record       | Transactional update or compare-and-swap                         |
| Revoked access while workflow is paused  | Reauthorize before resuming sensitive work                       |
| Agent produces an invalid result         | Validate at the boundary and route to a controlled failure state |

LangGraph's checkpointing and replay mechanisms make idempotency especially important for side effects. A checkpoint helps recover workflow progress; it does not make an external database write or tool call automatically safe to repeat.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 8. Production testing checklist

Before deploying a multi-agent workflow, I would test at least these cases:

State-integrity test plan

0 of 9

A specialist tries to update another specialist's state field.

An agent returns an unknown field or invalid severity.

Two agents return the same finding ID.

Two agents return contradictory findings.

A result references a policy ID outside the authorized evidence set.

A node is retried and attempts to create the same database record twice.

A checkpoint is resumed after the user's permissions have changed.

One parallel specialist fails while the others succeed.

A final approval is attempted while findings remain uncertain.

## 9. Interview-ready answer

> “I prevent shared-state corruption in LangGraph by defining explicit state ownership, minimizing each agent's input and writable fields, and validating every agent output with strict schemas. For parallel execution, I avoid having multiple agents overwrite the same field; instead, I collect isolated results and merge them through a trusted aggregation node. I use reducers only when their merge semantics are well-defined, and I handle duplicate IDs, conflicting results, retries, and checkpoint replay explicitly. Authorization and final approval remain in trusted application services rather than being delegated to the LLM. Finally, I test adversarial updates and concurrency failures and monitor state-validation errors in production.”

The staff-level takeaway is that shared state should be treated as a controlled data contract, not as a dictionary that every agent is free to modify. Schema validation protects structure; ownership protects boundaries; reducers protect merge semantics; and trusted services protect business and security invariants.