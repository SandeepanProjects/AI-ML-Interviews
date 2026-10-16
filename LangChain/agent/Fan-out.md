# Fan-out / Fan-in in LangGraph — Senior/Staff AI Engineer Guide

Fan-out / fan-in is a workflow pattern where one task branches into multiple independent tasks that can run in parallel, and then their results are combined at a synchronization point.

In LangGraph, this is useful when an AI workflow can break a large task into independent subtasks, execute them concurrently, and aggregate the results into one final response.

For example, in your financial market intelligence platform, a user asks:

> “Evaluate this company’s investment risk. Analyze its financial performance, market conditions, and regulatory risks, then provide a recommendation.”

Instead of asking one LLM to analyze everything sequentially, a graph can run three specialist agents in parallel:

- Financial analyst — revenue, margins, cash flow, debt.
- Market analyst — competition, industry trends, market conditions.
- Risk analyst — regulatory exposure, operational risks, downside scenarios.

A final synthesis node collects all three results and produces one evidence-based assessment.

## 1. Understand the pattern visually

START / Planner

Understand the question and establish the analysis scope

Fan-out

Dispatch independent analysis tasks

Finance

Revenue, cash flow, debt

Market

Competition, trends

Risk

Regulation, downside

Fan-in / Aggregation

Wait for the required branches, validate and combine their results

Final assessment

One synthesized answer with supporting evidence and limitations

The central benefit is parallelism without losing coordinated control of the workflow.

A key distinction: fan-out is not the same as simply calling three functions sequentially. And fan-in is not merely concatenating strings; in a production AI workflow it often involves validation, deduplication, conflict resolution, confidence assessment, and evidence-aware synthesis.

## 2. When should you use fan-out / fan-in?

Use it when the branches are sufficiently independent and the result needs multiple perspectives.

| Scenario           | Parallel branches                                                    |
| ------------------ | -------------------------------------------------------------------- |
| Financial analysis | Financials, market, regulatory risk                                  |
| Compliance review  | Performance claims, fee disclosures, prohibited language             |
| Enterprise RAG     | Search policy documents, product documentation, historical decisions |
| Research agent     | Company research, competitor research, macroeconomic research        |
| Code review        | Security, correctness, performance                                   |
| Customer support   | Account policy, troubleshooting, relevant knowledge-base search      |

Avoid fan-out when every task depends on the previous task's result. For example, if an agent must first identify a policy clause before deciding which exception rule to inspect, that dependency is sequential unless the system can safely predefine independent searches.

Also, parallel execution is not automatically faster. If each branch calls the same rate-limited model, saturates a shared database, or performs redundant retrieval, the total latency or cost may increase.

# 3. Implement fixed fan-out / fan-in in LangGraph

Let's start with the simplest production-style example: three known specialist agents run in parallel, and a synthesis node combines their outputs.

Install the core dependencies:

```
pip install -U langgraph langchain-openai pydantic
```

Set `OPENAI_API_KEY` in your environment. The example below uses an LLM for each specialist and a final synthesizer.

## 3.1 Complete example

```
# financial_analysis_graph.pyfrom typing import TypedDictfrom langgraph.graph import StateGraph, START, ENDfrom langchain_openai import ChatOpenAIfrom pydantic import BaseModel, Field# --------------------------------------------------# 1. Define typed output contracts# --------------------------------------------------class AnalysisResult(BaseModel):    summary: str    findings: list[str] = Field(default_factory=list)    risks: list[str] = Field(default_factory=list)    evidence: list[str] = Field(default_factory=list)class FinalAssessment(BaseModel):    executive_summary: str    key_findings: list[str]    key_risks: list[str]    missing_evidence: list[str]    recommendation: str# --------------------------------------------------# 2. Define the shared graph state# --------------------------------------------------class FinancialAnalysisState(TypedDict, total=False):    company: str    question: str    # Each parallel node writes to a different state key.    financial_analysis: dict    market_analysis: dict    regulatory_analysis: dict    final_assessment: dict# --------------------------------------------------# 3. Initialize the model# --------------------------------------------------llm = ChatOpenAI(    model="gpt-4.1-mini",    temperature=0,    timeout=30,    max_retries=2,)# --------------------------------------------------# 4. Define independent specialist nodes# --------------------------------------------------async def financial_analyst(    state: FinancialAnalysisState,) -> dict:    prompt = f"""    Analyze the financial performance of {state['company']}.    User question: {state['question']}    Focus on:    - Revenue and margin trends    - Cash flow and debt    - Financial resilience    - Missing data    Do not invent financial figures.    If source data is not provided, explicitly say so.    """    result = await llm.with_structured_output(        AnalysisResult    ).ainvoke(prompt)    return {"financial_analysis": result.model_dump()}async def market_analyst(    state: FinancialAnalysisState,) -> dict:    prompt = f"""    Analyze the market and competitive position of    {state['company']}.    User question: {state['question']}    Focus on:    - Industry conditions    - Competitive threats    - Market opportunities    - Missing evidence    Do not invent facts or market statistics.    """    result = await llm.with_structured_output(        AnalysisResult    ).ainvoke(prompt)    return {"market_analysis": result.model_dump()}async def regulatory_analyst(    state: FinancialAnalysisState,) -> dict:    prompt = f"""    Analyze regulatory and operational risks for    {state['company']}.    User question: {state['question']}    Focus on:    - Regulatory exposure    - Compliance concerns    - Operational risks    - Missing evidence    Do not claim a regulation applies unless supported.    """    result = await llm.with_structured_output(        AnalysisResult    ).ainvoke(prompt)    return {"regulatory_analysis": result.model_dump()}# --------------------------------------------------# 5. Fan-in: synthesize all completed branches# --------------------------------------------------async def synthesize(    state: FinancialAnalysisState,) -> dict:    prompt = f"""    Produce an evidence-conscious assessment of    {state['company']}.    User question:    {state['question']}    Financial analysis:    {state['financial_analysis']}    Market analysis:    {state['market_analysis']}    Regulatory analysis:    {state['regulatory_analysis']}    Requirements:    - Combine overlapping findings.    - Identify conflicting conclusions.    - Distinguish supported facts from assumptions.    - Do not invent missing evidence.    - State limitations clearly.    - Do not provide personalized investment advice.    Return an executive summary, key findings, key risks,    missing evidence, and a concise recommendation.    """    result = await llm.with_structured_output(        FinalAssessment    ).ainvoke(prompt)    return {"final_assessment": result.model_dump()}# --------------------------------------------------# 6. Construct the graph# --------------------------------------------------builder = StateGraph(FinancialAnalysisState)builder.add_node("financial_analyst", financial_analyst)builder.add_node("market_analyst", market_analyst)builder.add_node("regulatory_analyst", regulatory_analyst)builder.add_node("synthesize", synthesize)# Fan-out: launch three independent branches.builder.add_edge(START, "financial_analyst")builder.add_edge(START, "market_analyst")builder.add_edge(START, "regulatory_analyst")# Fan-in: wait for all three branches.builder.add_edge(    [        "financial_analyst",        "market_analyst",        "regulatory_analyst",    ],    "synthesize",)builder.add_edge("synthesize", END)graph = builder.compile()# --------------------------------------------------# 7. Execute# --------------------------------------------------async def main():    result = await graph.ainvoke({        "company": "Example Financial Services",        "question": (            "Evaluate its financial resilience, market "            "position, and major regulatory risks."        ),    })    print(result["final_assessment"])if __name__ == "__main__":    import asyncio    asyncio.run(main())
```

This example demonstrates graph orchestration, not verified financial research: the prompt doesn't provide company financial statements or a web-search tool. For a real assessment, supply authorized source evidence to the specialist nodes and require claims to reference it.

## 3.2 Understand the execution sequence

1. LangGraph receives the initial state: company and question.
2. The financial, market, and regulatory nodes become runnable in the same graph superstep.
3. LangGraph schedules the independent nodes; their asynchronous model calls can overlap.
4. Each node returns a partial state update. Each writes to its own key, so their updates do not conflict.
5. The multi-source edge waits for all three upstream nodes to complete successfully before running `synthesize`.
6. The synthesizer reads all three results and writes `final_assessment`; then the graph reaches `END`.

One important nuance: a LangGraph parallel fan-in edge is a barrier. If one branch raises an unhandled exception, the graph does not simply treat that branch as a successful empty result. You need an explicit failure-handling strategy if partial results are acceptable.

Also, LangGraph schedules parallel graph nodes; it is not a guarantee that all work executes simultaneously on separate CPU cores. Actual concurrency depends on the runtime, async implementation, model client, and service limits.

## 3.3 Why use separate state keys?

In the example, each specialist writes to a distinct key:

```
return {"financial_analysis": result.model_dump()}
```

```
return {"market_analysis": result.model_dump()}
```

```
return {"regulatory_analysis": result.model_dump()}
```

This avoids a common concurrent-state bug: multiple nodes trying to overwrite the same `analysis` field in the same graph step.

If all three nodes returned `{"analysis": ...}` without a reducer, LangGraph could raise an `InvalidUpdateError` because multiple nodes wrote to one state key without a defined merge rule.

When you need multiple branches to append to a shared collection, use a reducer.

# 4. Dynamic fan-out using `Send`

The previous example is ideal when the number of branches is known in advance.

But consider a different request:

> “Research the top 20 companies in this portfolio and evaluate their market, financial, and regulatory risks.”

Or a compliance request:

> “Review every claim in this marketing document against the relevant policies.”

You may not know the number of tasks until runtime. This is where LangGraph's `Send` API becomes useful.

`Send` lets a node dynamically dispatch work to another node with a task-specific state. It is a key building block for map-reduce workflows.

For example:

- A planner produces 20 company tasks.
- A worker node analyzes one company per task.
- LangGraph executes the dispatched worker tasks with configurable concurrency.
- A reducer/fan-in node combines the results.

## 4.1 Dynamic fan-out / fan-in example

```
# dynamic_fanout_graph.pyimport operatorfrom typing import Annotated, TypedDictfrom pydantic import BaseModel, Fieldfrom langchain_openai import ChatOpenAIfrom langgraph.graph import StateGraph, START, ENDfrom langgraph.types import Sendllm = ChatOpenAI(    model="gpt-4.1-mini",    temperature=0,    timeout=30,    max_retries=2,)class CompanyAnalysis(BaseModel):    company: str    summary: str    risks: list[str] = Field(default_factory=list)    missing_evidence: list[str] = Field(default_factory=list)class PortfolioSummary(BaseModel):    overall_summary: str    major_risks: list[str]    missing_evidence: list[str]class PortfolioState(TypedDict, total=False):    companies: list[str]    question: str
```

### What is `operator.add` doing here?

This state declaration is critical:

```
analyses: Annotated[list[dict], operator.add]
```

Each worker returns one result:

```
return {"analyses": [result.model_dump()]}
```

LangGraph merges the list updates through the reducer. Conceptually:

\\[ [\text{analysis A}] + [\text{analysis B}] + [\text{analysis C}] = [\text{A}, \text{B}, \text{C}] \\]

Without a reducer, parallel writes to the same state field can conflict.

Important: `operator.add` concatenates lists. It does not deduplicate results, resolve conflicting findings, or establish a deterministic business ordering. If those properties matter, use stable task IDs and a dedicated reducer or aggregation node.

### What does `Send` accomplish?

This code:

```
Send(    "analyze_company",    {        "company": company,        "question": state["question"],    },)
```

creates an execution of `analyze_company` with task-specific input.

The worker's input schema is different from the overall graph state. It doesn't need the entire portfolio state; it only needs the company and the question.

This is useful for scalable architectures because you can dispatch work items independently without giving every worker unnecessary state.

### What happens when the input list is empty?

The example's dispatch function returns an empty list. A production graph should define empty-input behavior explicitly instead of assuming that the aggregation node will always execute as desired.

A robust design validates the task list before fan-out or routes empty input to a dedicated node that returns a clear “no work items” result.

# 5. Fan-out/fan-in vs. planner–supervisor–executor

These patterns are related, but they solve different problems.

| Pattern                 | Primary responsibility                                                             |
| ----------------------- | ---------------------------------------------------------------------------------- |
| Fan-out / fan-in        | Run independent tasks in parallel and collect their results                        |
| Planner–executor        | Plan work and execute tasks based on that plan                                     |
| Supervisor              | Decide which specialist should act next, inspect results, and route follow-up work |
| Map-reduce              | Apply a worker to many items, then aggregate the outputs                           |
| Evaluator / rework loop | Assess output quality and retry or revise when needed                              |

A supervisor may decide to initiate fan-out, but fan-out itself does not require an LLM planner or supervisor.

For your financial compliance platform, I would use a hybrid architecture:

Planner / Supervisor

Identify applicable policies, determine review scope, and select required checks

Fan-out to specialists

Performance claims · Fees · Risk disclosures · Prohibited language

Fan-in / deterministic aggregation

Merge findings, deduplicate citations, apply risk rules, identify conflicting evidence

Evaluator

Check evidence support, completeness, policy version, and output schema

Conditional routing

Pass → human approval or final report; fail → bounded rework loop

In a regulated workflow, I would not let the final LLM invent the overall risk rating by averaging the opinions of four agents. Use deterministic aggregation rules for policy severity and required evidence, then use the LLM to explain the resulting findings.

# 6. Production considerations

The examples above illustrate the graph mechanics. Before deploying them at enterprise scale, I would address the following.

## 6.1 Bound concurrency

Launching hundreds of tasks simultaneously can hit model rate limits, exhaust database connections, or increase cost dramatically.

Use graph runtime concurrency controls where supported by your LangGraph version, and add provider-level rate limiting and backpressure. For example, the invocation configuration supports a recursion limit and a configurable maximum concurrency for graph execution:

```
result = await graph.ainvoke(    input_state,    config={        "recursion_limit": 25,        "max_concurrency": 8,    },)
```

Test the behavior against your installed LangGraph version and workload. Concurrency limits are not a substitute for provider rate limits, per-tenant quotas, or worker-level connection-pool limits.

## 6.2 Handle partial failures deliberately

Suppose the market analyst succeeds, the financial analyst times out, and the regulatory analyst succeeds.

You need an explicit policy:

- Fail closed: do not produce a final report until all required analyses succeed.
- Partial result: synthesize successful branches while explicitly marking failed analyses as unavailable.
- Retry: retry transient failures within a bounded budget.
- Fallback: use a secondary provider or non-LLM retrieval method where appropriate.

Do not silently turn a failed analysis into an empty list of risks. “No risks found” and “risk analysis failed” mean different things.

## 6.3 Keep aggregation deterministic where possible

The LLM synthesizer can consolidate evidence, but critical decisions should be validated by application code.

For example:

```
REQUIRED_BRANCHES = {    "financial",    "market",    "regulatory",}def validate_required_results(    completed: set[str],) -> None:    missing = REQUIRED_BRANCHES - completed    if missing:        raise ValueError(            f"Required analyses are missing: {sorted(missing)}"        )
```

In a real graph, track branch status and structured results explicitly. Validate that each required branch completed successfully before allowing the final assessment to proceed.

For dynamic tasks, assign each work item a stable task ID, persist its status, and ensure retries don't produce duplicate findings. When result order matters, sort by task ID or use a defined domain-specific ordering instead of relying on completion order.

## 6.4 Add checkpointing for durable workflows

For long-running research, human approval, or multi-step compliance reviews, configure a persistent LangGraph checkpointer and a stable `thread_id`. This lets the graph resume from saved state after interruptions, depending on the checkpoint and deployment configuration.

Remember that a checkpoint stores workflow state; it does not automatically guarantee that external side effects such as sending notifications or writing to an external system happen exactly once. Use idempotency keys and durable job records for those operations.

## 6.5 Measure the right things

Track:

- End-to-end latency and per-branch latency.
- Branch success and timeout rates.
- LLM token usage and cost per branch.
- Retry rate and failure recovery time.
- Number of tasks per fan-out.
- Evidence coverage and citation correctness.
- Aggregation failures and conflicting findings.

A useful performance metric is:

\\[ T\_{\text{parallel}} \approx \max(T_1,T_2,\ldots,T_n) + T\_{\text{aggregation}} \\]

This is an idealized estimate. Real latency includes scheduling, model queueing, retrieval, network calls, retries, and synchronization overhead.

Parallelism helps most when the branches are independent, have comparable resource constraints, and are not all bottlenecked by the same external service.

# 7. Common interview questions

Q: How do you implement fan-out/fan-in in LangGraph?

I use multiple outgoing edges for known parallel branches and a multi-source edge to synchronize them. For runtime-determined tasks, I use `Send` to dispatch work to a worker node. If multiple workers write to the same state key, I define an appropriate reducer such as `operator.add` for list accumulation.

Q: Why do you need a reducer?

LangGraph needs to know how concurrent state updates are merged. Independent branches can write to separate keys without a shared reducer. If multiple branches update one key, a reducer defines how those updates combine and prevents conflicting writes.

Q: Is fan-in always an LLM call?

No. It can be deterministic Python code, SQL, a reducer, an LLM synthesis node, or a combination. For financial compliance, I favor deterministic aggregation for required checks and severity rules, with an LLM used for evidence-grounded explanations.

Q: What is the difference between `Send` and ordinary graph edges?

Ordinary edges connect predefined nodes in the graph. `Send` dynamically schedules a target node with a specific input, making it suitable for map-reduce workflows where the number of tasks varies at runtime.

Q: How do you prevent a failed branch from corrupting the final result?

Track branch status explicitly, use bounded retries and timeouts, distinguish failed work from valid empty results, and validate required outputs before aggregation. Permit partial reports only when the product requirements allow them and clearly disclose missing analyses.

## My recommendation

For your enterprise AI projects, master both implementations:

1. Fixed parallel branches for known specialist workflows.
2. Dynamic `Send` + reducers for variable-size research, document review, and map-reduce tasks.

Then add bounded concurrency, structured outputs, failure handling, checkpointing, observability, and deterministic validation. That combination is much closer to production LangGraph engineering than simply launching multiple agents and concatenating their responses.