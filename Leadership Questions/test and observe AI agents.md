# How to test and observe AI agents — Senior/Staff Engineer approach

In a production multi-agent AI system, I treat testing and observability as two complementary engineering disciplines:

- Testing tells us whether an agent behaves correctly under known conditions.
- Observability tells us what an agent actually did, why it did it, and where it failed when running in a real environment.

For an enterprise RAG or financial compliance platform using FastAPI, LangGraph, PostgreSQL, Qdrant, Redis, and LLMs, we need to observe more than latency and exceptions. We must understand the agent's decisions, tool calls, retrieved evidence, state changes, retries, cost, and final output quality.

## 1. Define an agent's contract first

Before writing tests or dashboards, define what each agent is allowed to do and what it must produce.

For example, consider a `PolicyAnalysisAgent` that receives retrieved policy evidence and identifies potential compliance violations.

Input contract

Document ID, authorized tenant context, relevant policy passages, policy versions.

Execution contract

Only approved tools; bounded steps and time; no unauthorized data access; untrusted document text is treated as data.

Output contract

Structured findings, risk categories, evidence IDs, confidence or review status, and explicit handling of insufficient evidence.

This contract gives us concrete test assertions. Without it, we may only test whether the LLM returned a response—not whether the agent did the right thing.

## 2. Test agents at five levels

| Level               | What I verify                            | Example                                                          |
| ------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| 1. Contract tests   | Input/output schema and invariants       | Every finding references permitted evidence                      |
| 2. Behavioral tests | Decisions under known scenarios          | No evidence → no unsupported finding                             |
| 3. Tool tests       | Tool selection, arguments, permissions   | Agent cannot query another tenant                                |
| 4. Workflow tests   | Agent interactions and state transitions | Failed specialist triggers defined fallback                      |
| 5. Evaluation tests | Real-world quality and regressions       | Relevant evidence, grounded answers, correct risk classification |

Use deterministic fakes for most tests. Use real model calls for a smaller, controlled evaluation suite. This separation makes the test suite faster, cheaper, and reproducible.

## 3. Build an agent test harness

I would create a reusable test harness so every agent can be tested with the same structure rather than writing completely different tests for every specialist.

A useful project layout is:

```
app/
├── agents/
│   ├── policy_agent.py
│   ├── risk_agent.py
│   └── research_agent.py
├── contracts/
│   └── findings.py
├── observability/
│   ├── tracing.py
│   ├── metrics.py
│   └── audit.py
└── workflow/
    └── graph.py

tests/
├── agents/
│   ├── test_policy_agent.py
│   ├── test_risk_agent.py
│   └── test_tool_permissions.py
├── evaluation/
│   ├── policy_cases.jsonl
│   └── test_agent_quality.py
└── integration/
    └── test_agent_observability.py
```

### Step 1: Define a strict output contract

```
# app/contracts/findings.pyfrom typing import Literalfrom pydantic import BaseModel, Fieldclass Finding(BaseModel):    rule_id: str    risk: Literal["low", "medium", "high", "critical"]    explanation: str    evidence_ids: list[str] = Field(min_length=1)class AgentResult(BaseModel):    findings: list[Finding]    needs_human_review: bool    status: Literal[        "completed",        "insufficient_evidence",        "needs_review",    ]
```

The schema prevents malformed structures from silently flowing into downstream agents. It does not prove that the findings are factually correct; that requires separate behavioral and quality tests.

### Step 2: Test behavior using a fake model

For a test, I want complete control over the model response.

```
# tests/agents/test_policy_agent.pyimport pytestfrom pydantic import ValidationErrorfrom app.contracts.findings import AgentResultdef validate_agent_output(raw_output, allowed_evidence_ids):    result = AgentResult.model_validate(raw_output)    for finding in result.findings:        unknown_ids = (            set(finding.evidence_ids) - allowed_evidence_ids        )        if unknown_ids:            raise ValueError(                f"Unsupported evidence IDs: {unknown_ids}"            )    return resultdef test_valid_finding_references_retrieved_evidence():    raw = {        "status": "completed",        "needs_human_review": True,        "findings": [            {                "rule_id": "POL-101",                "risk": "high",                "explanation": "The claim lacks required support.",                "evidence_ids": ["chunk-17"],            }        ],    }    result = validate_agent_output(        raw,        allowed_evidence_ids={"chunk-17"},    )    assert result.findings[0].risk == "high"    assert result.needs_human_review is Truedef test_agent_cannot_invent_evidence_ids():    raw = {        "status": "completed",        "needs_human_review": False,        "findings": [            {                "rule_id": "POL-101",                "risk": "high",                "explanation": "Unsupported claim.",                "evidence_ids": ["made-up-chunk"],            }        ],    }    with pytest.raises(ValueError):        validate_agent_output(            raw,            allowed_evidence_ids={"chunk-17"},        )def test_invalid_risk_category_is_rejected():    raw = {        "status": "completed",        "needs_human_review": False,        "findings": [            {                "rule_id": "POL-101",                "risk": "extreme",                "explanation": "Example.",                "evidence_ids": ["chunk-17"],            }        ],    }    with pytest.raises(ValidationError):        validate_agent_output(            raw,            allowed_evidence_ids={"chunk-17"},        )
```

Run:

```
pytest -q tests/agents/test_policy_agent.py
```

These tests verify output invariants. To test the actual agent, inject a fake model into its implementation and assert that the agent calls it with the expected authorized context, handles malformed responses, and obeys retry limits.

### Step 3: Test the decision, not just the schema

Create a small behavioral test matrix:

| Input scenario                  | Expected behavior                                  |
| ------------------------------- | -------------------------------------------------- |
| Strong supporting evidence      | Identify the applicable rule and cite the evidence |
| No supporting evidence          | Return insufficient evidence or escalate           |
| Conflicting policy passages     | Flag ambiguity or apply a defined precedence rule  |
| Malicious instructions in a PDF | Treat them as untrusted content                    |
| Model output is malformed       | Bounded repair or explicit failure                 |
| Tool returns unauthorized data  | Reject it; do not pass it to the model             |
| Model provider times out        | Respect deadline and retry policy                  |

For high-risk decisions, define acceptance criteria with domain experts and a reviewed reference dataset. Do not rely on the LLM grading its own output as the only judge of correctness.

## 4. Observe each agent using distributed tracing

This is where observability becomes essential.

Imagine the user's request produces a bad compliance report. The workflow contains a planner, a retrieval agent, a policy-analysis agent, and a risk-review agent.

Without tracing, you see only: `Review failed`.

With tracing, you can investigate:

- Did the planner choose the wrong agent?
- Did retrieval return irrelevant or stale policy documents?
- Did the policy agent ignore relevant evidence?
- Did the risk agent receive incomplete state?
- Did a retry overwrite a previous result?
- Did the LLM return invalid structured output?
- Did the request exceed its deadline?

### Use one trace for one execution, with nested spans

Trace: compliance review

1 trace

`workflow.execute`

Total duration · outcome · workflow version

`agent.planner`

Decision · step count · model latency

`retrieval.search`

Filters · result count · retrieval latency

`agent.policy_analysis`

Input/output validation · LLM calls · tool calls

`agent.risk_review`

Risk result · retry count · validation outcome

The trace tells you where time was spent and which operation failed. Logs provide detailed event context, while metrics tell you whether a problem affects one request or the whole service.

For implementation, I would use OpenTelemetry instrumentation and a tracing backend such as Jaeger, Grafana Tempo, or an appropriately configured managed platform. If using LangSmith with LangGraph, it can also provide useful graph-level run and node traces.

### What every agent span should capture

| Field                           | Why it matters                                          |
| ------------------------------- | ------------------------------------------------------- |
| `workflow_id`                   | Connect events to a durable workflow                    |
| `agent.name`                    | Identify the executing agent                            |
| `agent.version`                 | Compare behavior across deployments                     |
| `model.name`                    | Identify the model used                                 |
| `prompt.version`                | Detect prompt regressions                               |
| `attempt`                       | Understand retries                                      |
| `duration_ms`                   | Diagnose slow agents                                    |
| `input_tokens`, `output_tokens` | Estimate usage and cost                                 |
| `tool.name`                     | Understand tool usage                                   |
| `output.valid`                  | Measure contract violations                             |
| `outcome`                       | Distinguish success, failure, timeout, and interruption |

Use a trace ID to correlate spans, and a durable workflow ID to correlate work across queue deliveries or resumed executions. If a workflow resumes in a later process, do not assume it will always share the original in-memory execution context; preserve explicit correlation metadata.

Security rule: do not put raw documents, unrestricted prompts, access tokens, or sensitive model responses into trace attributes. Prefer IDs, versions, lengths, hashes where appropriate, and redacted summaries. Any content capture for debugging should be explicitly controlled, access-restricted, and retention-limited.

## 5. Instrument agent execution in code

Here is a small reusable OpenTelemetry wrapper. It captures the agent's duration and outcome without logging the document content.

```
# app/observability/agent_trace.pyfrom time import perf_counterfrom typing import Callable, Anyfrom opentelemetry import tracefrom opentelemetry.trace import Status, StatusCodetracer = trace.get_tracer("enterprise-ai.agents")def run_observed_agent(    *,    agent_name: str,    agent_version: str,    workflow_id: str,    operation: Callable[[], Any],):    with tracer.start_as_current_span(        f"agent.{agent_name}"    ) as span:        span.set_attribute("agent.name", agent_name)        span.set_attribute("agent.version", agent_version)        span.set_attribute("workflow.id", workflow_id)        started = perf_counter()        try:            result = operation()            span.set_attribute("agent.outcome", "success")            span.set_status(Status(StatusCode.OK))            return result
```

Example usage:

```
result = run_observed_agent(    agent_name="policy_analysis",    agent_version="v3",    workflow_id=workflow_id,    operation=lambda: policy_agent.run(agent_input),)
```

This is a basic instrumentation boundary. In a production implementation, I would also instrument the model client, retrieval client, and tools individually. If an agent makes three LLM calls, one outer span alone is not enough to distinguish the slow call from the other two.

Configure the OpenTelemetry SDK and exporter at application startup. Ensure trace context propagates through async calls and queue messages. Do not create a new unrelated trace for every nested operation when the operations belong to the same execution.

## 6. Monitor agent-specific metrics

A workflow can be healthy at the API level while one agent is silently degrading. I therefore create metrics per agent role.

I would monitor four categories.

A. Reliability

Success rate, timeout rate, exception rate, schema failure rate, retry exhaustion, fallback rate.

B. Performance

p50/p95/p99 duration, queue wait, model latency, retrieval latency, tool latency, total workflow latency.

C. Cost and capacity

Tokens per task, model calls per task, estimated cost, concurrent executions, rate-limit events.

D. Quality and safety

Groundedness, citation validity, task accuracy, unauthorized tool attempts, human override rate.

### Implement metrics with Prometheus-compatible instrumentation

```
# app/observability/metrics.pyfrom prometheus_client import Counter, HistogramAGENT_RUNS = Counter(    "ai_agent_runs_total",    "Agent executions by outcome",    ["agent", "outcome"],)AGENT_DURATION = Histogram(    "ai_agent_duration_seconds",    "Agent execution duration",    ["agent"],    buckets=(        0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60    ),)AGENT_OUTPUT_INVALID = Counter(    "ai_agent_invalid_outputs_total",    "Agent outputs rejected by validation",    ["agent", "reason"],)
```

Record each execution using a stable set of labels:

```
from time import perf_counterdef record_agent_execution(agent_name, operation):    started = perf_counter()    try:        result = operation()        AGENT_RUNS.labels(            agent=agent_name,            outcome="success",        ).inc()        return result    except TimeoutError:        AGENT_RUNS.labels(            agent=agent_name,            outcome="timeout",        ).inc()        raise    except Exception:        AGENT_RUNS.labels(            agent=agent_name,            outcome="error",        ).inc()        raise    finally:        AGENT_DURATION.labels(            agent=agent_name,        ).observe(perf_counter() - started)
```

These examples can be used together with the tracing wrapper, although I would avoid duplicating instrumentation responsibilities in the final implementation.

Important production rule: do not use `workflow_id`, `user_id`, document IDs, prompts, or arbitrary exception strings as metric labels. Their high cardinality can overwhelm the metrics backend. Put request-specific details in controlled traces or structured logs instead.

For token usage and cost, instrument the model client, where actual usage metadata is available. Do not estimate every request from text length if the provider returns authoritative token counts.

## 7. Use structured logs to explain what happened

Metrics show a trend. Traces show the execution path. Structured logs explain the events and decisions.

For an agent, I would emit events such as:

```
{
  "event": "agent.completed",
  "workflow_id": "wf-123",
  "trace_id": "trace-abc",
  "agent_name": "policy_analysis",
  "agent_version": "v3",
  "model": "configured-model",
  "attempt": 1,
  "input_evidence_count": 5,
  "output_findings_count": 2,
  "output_valid": true,
  "duration_ms": 842,
  "outcome": "success"
}
```

This is an example of the log schema, not a real execution.

Useful event names include:

- `agent.started`
- `agent.completed`
- `agent.failed`
- `agent.retry_scheduled`
- `agent.tool_denied`
- `agent.output_validation_failed`
- `agent.human_review_requested`
- `agent.checkpoint_saved`

I would log identifiers and safe metadata by default. For detailed reasoning, prefer a concise, structured decision record—such as the selected route, permitted tool name, validation result, and evidence IDs—rather than storing hidden chain-of-thought or unrestricted raw prompts.

## 8. Evaluate the quality of the agent's decisions

This is the part many teams miss.

An agent can have:

- 99.9% execution success,
- valid JSON on every response,
- low latency,
- and still produce bad decisions.

Operational success is not the same as AI quality.

For each agent, I define an evaluation suite specific to its responsibility.

| Agent              | Evaluation                                                         |
| ------------------ | ------------------------------------------------------------------ |
| Planner            | Correct task decomposition, allowed routing, bounded steps         |
| Retrieval agent    | Relevant evidence, correct metadata filters, freshness             |
| Policy agent       | Correct rule application, supported citations                      |
| Risk agent         | Risk classification against expert-reviewed labels                 |
| Tool-using agent   | Correct tool selection and arguments, permission compliance        |
| Final answer agent | Completeness, groundedness, output contract, no unsupported claims |

### Example evaluation record

```
evaluation_case = {    "case_id": "policy-case-014",    "expected_rule_ids": ["POL-101"],    "expected_risk": "high",    "required_evidence_ids": ["policy-chunk-17"],    "must_escalate": True,}
```

Run the agent on this case and compare the result against the reviewed expectations.

For classification tasks, measure precision, recall, and F1 where appropriate. For retrieval, use Recall\@k and ranking metrics. For generated explanations, evaluate citation support and factual correctness, ideally with human-reviewed reference judgments.

I would run these evaluations:

1. On every prompt, agent logic, or retrieval configuration change.
2. Before changing the model or embedding model.
3. On a scheduled basis against a stable benchmark.
4. Against sampled production cases, using an approved privacy-safe evaluation process.

Track metrics by agent version, model, prompt version, and dataset version. A global average can hide a regression in one specialist agent or one class of documents.

## 9. Test and observe tool usage separately

In an agentic system, the agent is not just generating text; it is choosing actions. Therefore, tool behavior needs its own tests and telemetry.

For every tool call, record:

- Tool name and tool version.
- Whether authorization passed or failed.
- Validated argument schema.
- Execution duration and outcome.
- Retry count and whether the operation was idempotent.
- Whether the result was validated before returning to the agent.

Never treat an LLM's decision to call a tool as authorization to execute it.

For example, an agent may request `get_customer_portfolio(customer_id=...)`. The application must independently check that the authenticated principal is permitted to access that customer before the tool queries the database.

A useful test is:

```
import pytestdef test_agent_cannot_access_another_tenant():    principal = {        "tenant_id": "tenant-A",        "permissions": ["portfolio:read"],    }    requested_resource = {        "tenant_id": "tenant-B",        "customer_id": "customer-42",    }    with pytest.raises(PermissionError):        authorize_portfolio_read(            principal=principal,            resource=requested_resource,        )
```

Here, `authorize_portfolio_read` represents your server-side authorization policy. The test should exercise the actual policy implementation, not a mock that always denies access.

I would also test malformed arguments, unauthorized tools, repeated tool calls, destructive operations, timeouts, and tools returning unexpected data.

## 10. Observe interactions between multiple agents

For a multi-agent system, individual agent metrics are not sufficient. You need workflow-level visibility.

Suppose your architecture is:

Supervisor / Planner

Selects tasks and permitted agents

Policy agent

Evidence and rule analysis

Risk agent

Risk classification

Aggregator / Evaluator

Checks completeness, contradictions, evidence, and review requirements

I would observe:

- Which agents were invoked and in what order.
- Whether the supervisor selected an allowed route.
- Whether parallel agents received immutable, appropriate inputs.
- Whether the aggregator received all expected outputs.
- Whether specialist agents disagreed.
- Whether an evaluator triggered a rework loop.
- How many cycles occurred before completion or termination.
- Whether the final result preserved evidence provenance.

### Prevent state corruption

In LangGraph, agents commonly share workflow state. I would test that each agent returns only its permitted state updates and that parallel agents do not overwrite unrelated fields.

For example, give each specialist a typed output contract and merge results through a controlled reducer or aggregation node. Test that conflicting updates are rejected or resolved according to an explicit rule.

### Observe rework loops

Track metrics such as:

- `workflow_agent_handoffs_total`
- `workflow_rework_cycles`
- `workflow_max_steps_reached_total`
- `workflow_agent_failures_total`
- `workflow_missing_outputs_total`

A rising rework-cycle count may indicate poor retrieval, a broken prompt, conflicting agent contracts, or an overly strict evaluator. A rising handoff count may indicate that the planner is over-decomposing simple tasks.

Set explicit step, time, and retry budgets so that an agent cannot loop indefinitely.

## 11. Build dashboards and alerts around agent health

I would create four dashboards rather than trying to put every metric into one panel.

Agent reliability

Success rate, timeout rate, malformed output, retry exhaustion, tool denials, fallback rate.

Latency and cost

Per-agent p95 latency, provider latency, queue delay, tokens per workflow, cost per successful task.

AI quality

Evaluation pass rate, citation validity, retrieval relevance, unsupported claims, human override rate.

Workflow and safety

Step-limit breaches, interrupted-workflow age, failed resumes, authorization failures, audit gaps.

Example alerts:

| Alert                                        | Why it matters                                |
| -------------------------------------------- | --------------------------------------------- |
| Agent timeout rate exceeds its SLO           | Provider, network, or workload degradation    |
| Schema rejection rate rises after deployment | Prompt or model regression                    |
| Citation support rate drops                  | Grounding or retrieval regression             |
| Rework cycles increase sharply               | Planner, agent contract, or evaluator problem |
| Pending human reviews exceed the allowed age | Workflow operations issue                     |
| Checkpoint persistence fails                 | Resumption may no longer be reliable          |
| Cross-tenant access is attempted             | Security investigation required               |

Set thresholds using baselines and service-level objectives rather than arbitrary universal numbers. For example, alert on a sustained increase in error rate or latency and on any confirmed authorization bypass. Not every failed authorization attempt is an incident, but it should be observable and appropriately audited.

## 12. How I investigate a failing agent in production

Suppose users report that the policy agent has started generating incorrect compliance findings.

I would investigate in this order:

1. Find affected traces. Compare successful and failed executions by agent version, model, prompt version, and time window.
2. Inspect the inputs. Verify the agent received the correct tenant context, policy versions, and retrieved evidence IDs.
3. Inspect tool execution. Check authorization decisions, arguments, tool results, errors, and retries.
4. Inspect the model boundary. Check model latency, token usage, response format, refusal status, and validation failures.
5. Evaluate quality. Replay affected cases against a versioned evaluation dataset and compare the current output with a known-good baseline.
6. Mitigate and verify. Roll back the prompt or model configuration, disable a faulty route, or apply an approved fallback. Then rerun regression tests before restoring normal traffic.

I would not immediately conclude that the model is at fault. The root cause could be stale embeddings, incorrect metadata filtering, a missing state field, a bad tool result, or an aggregation bug.

The trace helps locate the failure boundary; evaluation helps establish whether the agent's behavior is actually wrong.

## 13. A practical implementation order

For your enterprise financial AI platform, I would implement this incrementally:

1. Contracts: typed input/output schemas and explicit agent permissions.
2. Deterministic tests: valid output, invalid output, insufficient evidence, failure, and retry cases.
3. Tracing: workflow span, per-agent spans, model calls, retrieval, and tool execution.
4. Metrics and structured logs: reliability, latency, cost, validation failures, and safe event metadata.
5. AI evaluation: reviewed test cases for grounding, citation correctness, and domain-specific decisions.
6. Integration and security tests: tenant isolation, actual tool authorization, checkpoint recovery, and duplicate delivery.
7. Dashboards and release gates: SLO alerts, model/prompt regression checks, and safe rollback procedures.

### Interview-ready answer

> “I treat agent testing and observability as separate but connected capabilities. I first define an agent contract covering its inputs, permitted tools, output schema, and failure behavior. I use deterministic unit and behavioral tests for routing, tool selection, state updates, schema validation, retries, and security boundaries. I then run integration tests and versioned AI evaluations to measure retrieval relevance, groundedness, citation correctness, and task-specific accuracy.
>
> For observability, I instrument the workflow and each agent with OpenTelemetry spans, structured logs, and low-cardinality metrics. I correlate model calls, retrieval, tool execution, retries, and state transitions with the workflow ID and trace context. I monitor reliability, latency, token cost, quality, and safety separately.
>
> When an agent degrades, I inspect its trace to locate the failure boundary, replay the case against a known evaluation dataset, identify the root cause, and verify the fix with regression tests. My goal is to make every agent’s behavior measurable, testable, secure, and recoverable—not merely to confirm that it returned a response.”

The staff-level principle is: an agent is not production-ready just because it can reason and call tools. Its behavior must be constrained by contracts, validated by tests, observable at runtime, evaluated for quality, and recoverable when dependencies fail.