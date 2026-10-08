Yes. At a **Senior/Staff AI Engineer** level, you should think of **Planner–Supervisor–Executor** not as three “agents,” but as three different **control responsibilities** in a production workflow.

The most important distinction is:

> **Planner decides what needs to be done. Supervisor decides what should happen next. Executor actually performs the work.**

They solve different problems.

---

# 1. The big picture

Imagine a user asks your Financial AI platform:

> “Analyze this investment proposal, compare it against our investment policy, identify risks, calculate the fee impact, and give me a recommendation with citations.”

A production system should not simply send that entire prompt to one LLM.

Instead:

```text
                         User
                           │
                           ▼
                    ┌─────────────┐
                    │   Planner   │
                    │             │
                    │ What work   │
                    │ is required?│
                    └──────┬──────┘
                           │
                           ▼
                    Execution Plan
                           │
                           ▼
                    ┌─────────────┐
                    │ Supervisor  │
                    │             │
                    │ What should │
                    │ happen next?│
                    └──────┬──────┘
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
        Research Agent  Risk Agent    Fee Agent
             │             │              │
             ▼             ▼              ▼
           Tools         Tools          Tools
             │             │              │
             └─────────────┼──────────────┘
                           ▼
                      Evaluator
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
              PASS                 REWORK
                 │                   │
                 ▼                   └──→ Supervisor
             Finalizer
                 │
                 ▼
               User
```

This is much closer to a real enterprise AI architecture.

---

# 2. Why do we need all three?

Consider a traditional application.

You might have:

```text
Request
   ↓
Service
   ↓
Repository
   ↓
Database
```

The flow is mostly deterministic.

AI workflows are different.

The system may need to determine:

* which information is needed
* which agents should run
* whether agents can run in parallel
* whether another retrieval is required
* whether the answer is good enough
* whether a human must approve it
* whether an agent should retry
* whether a different model should be used
* whether a tool call is authorized

That's where orchestration patterns become important.

---

# 3. Planner

The **Planner** answers:

> **“What tasks need to be performed to solve this problem?”**

It creates a structured execution plan.

For example:

```text
User request:

"Analyze investment ABC and determine whether it complies
with our investment policy."
```

Planner might produce:

```json
{
  "goal": "Analyze investment ABC",
  "tasks": [
    {
      "id": "research",
      "type": "market_research",
      "description": "Gather current investment information"
    },
    {
      "id": "policy",
      "type": "policy_retrieval",
      "description": "Retrieve applicable investment policies"
    },
    {
      "id": "risk",
      "type": "risk_analysis",
      "depends_on": ["research", "policy"]
    },
    {
      "id": "compliance",
      "type": "compliance_analysis",
      "depends_on": ["research", "policy"]
    },
    {
      "id": "recommendation",
      "type": "recommendation",
      "depends_on": ["risk", "compliance"]
    }
  ]
}
```

The Planner is essentially producing a **workflow DAG**.

---

# 4. Planner is not the Supervisor

This distinction is extremely important in interviews.

### Planner

Determines:

```text
What work should be done?
```

### Supervisor

Determines:

```text
What should happen NOW?
```

For example:

```text
Planner:

1. Research market
2. Retrieve policy
3. Analyze risk
4. Analyze compliance
5. Generate recommendation
```

Then the Supervisor observes the current state.

If research hasn't completed:

```text
Supervisor → Research Agent
```

After research:

```text
Supervisor → Policy Retrieval Agent
```

Once both are available:

```text
Supervisor → Risk Agent
Supervisor → Compliance Agent
```

If risk analysis fails:

```text
Supervisor → Retry Risk Agent
```

If evaluator rejects the result:

```text
Supervisor → Rework
```

So:

> **Planner creates the plan; Supervisor executes and adapts the plan.**

---

# 5. Executor

The Executor is the component that actually performs a task.

For example:

```text
Supervisor
     │
     ▼
Risk Agent
     │
     ├── retrieve_documents()
     ├── calculate_metrics()
     ├── call_llm()
     └── produce RiskAssessment
```

The Executor shouldn't decide the entire workflow.

Its responsibility is:

> **Execute one assigned unit of work correctly and return structured output.**

This separation is critical.

---

# 6. Production architecture

A more realistic Staff-level system looks like this:

```text
                       ┌──────────────────┐
                       │      Client      │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │    API Gateway   │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │     FastAPI      │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Application      │
                       │ Service          │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ LangGraph /      │
                       │ Workflow Engine  │
                       └────────┬─────────┘
                                │
                ┌───────────────┴──────────────┐
                │                              │
                ▼                              ▼
        ┌──────────────┐              ┌────────────────┐
        │   Planner    │              │  State Store   │
        └──────┬───────┘              │ PostgreSQL     │
               │                      └────────────────┘
               ▼
        Execution Plan
               │
               ▼
        ┌──────────────┐
        │  Supervisor  │
        └──────┬───────┘
               │
       ┌───────┼────────┐
       │       │        │
       ▼       ▼        ▼
    Research  Risk     Compliance
     Agent    Agent      Agent
       │       │        │
       └───────┼────────┘
               │
               ▼
          ┌──────────┐
          │Evaluator │
          └────┬─────┘
               │
        ┌──────┴───────┐
        ▼              ▼
       PASS           FAIL
        │              │
        ▼              ▼
     Finalizer       Rework
        │              │
        │              └────→ Supervisor
        ▼
      Result
```

---

# 7. The most important component: State

A production multi-agent system should **not rely on conversation messages alone**.

You need explicit workflow state.

For example:

```python
from typing import TypedDict


class WorkflowState(TypedDict):
    workflow_id: str

    user_request: str

    plan: list[dict]

    current_task: str | None

    research_result: dict | None

    risk_result: dict | None

    compliance_result: dict | None

    evaluator_result: dict | None

    retry_count: int

    requires_human_approval: bool

    human_decision: str | None

    final_answer: dict | None
```

This state becomes the shared context of the workflow.

---

# 8. Planner implementation

A production Planner should return **structured output**, not free-form text.

For example:

```python
from pydantic import BaseModel


class Task(BaseModel):
    id: str
    type: str
    description: str
    depends_on: list[str] = []


class ExecutionPlan(BaseModel):
    goal: str
    tasks: list[Task]
```

Then:

```python
class Planner:

    def __init__(self, llm):
        self.llm = llm

    async def create_plan(
        self,
        request: str
    ) -> ExecutionPlan:

        response = await self.llm.generate_structured(
            prompt=f"""
            Create an execution plan for:

            {request}

            Only create tasks from the allowed task types:
            - research
            - policy_retrieval
            - risk_analysis
            - compliance_analysis
            - recommendation
            """
        )

        return response
```

Notice something important.

The Planner doesn't execute anything.

It produces:

```text
ExecutionPlan
```

---

# 9. Don't allow the LLM to freely invent tasks

This is a major production concern.

A naïve implementation says:

```python
plan = llm.generate("Decide what agents to call")
```

and then executes whatever the LLM returns.

That's dangerous.

Instead define an allowlist:

```python
ALLOWED_TASKS = {
    "research",
    "policy_retrieval",
    "risk_analysis",
    "compliance_analysis",
    "recommendation"
}
```

Validate:

```python
def validate_plan(plan: ExecutionPlan):

    for task in plan.tasks:

        if task.type not in ALLOWED_TASKS:
            raise InvalidPlanError(
                f"Unsupported task: {task.type}"
            )
```

At Staff level, you want:

```text
LLM proposes
      ↓
Policy validates
      ↓
Engine executes
```

not:

```text
LLM decides everything
      ↓
System blindly executes
```

---

# 10. Supervisor

The Supervisor receives:

```text
Current state
+
Execution plan
+
Available agents
```

and decides the next action.

For example:

```python
class Supervisor:

    async def decide(
        self,
        state: WorkflowState
    ) -> str:

        if state["research_result"] is None:
            return "research"

        if state["risk_result"] is None:
            return "risk_analysis"

        if state["compliance_result"] is None:
            return "compliance_analysis"

        if state["evaluator_result"] is None:
            return "evaluate"

        if state["evaluator_result"]["status"] == "fail":
            return "rework"

        return "finalize"
```

This version is deterministic.

That's actually desirable for many enterprise workflows.

---

# 11. Should the Supervisor always be an LLM?

**No.**

This is one of the most important Staff Engineer decisions.

You can implement the Supervisor in three ways.

### Option 1 — Deterministic Supervisor

```python
if research_missing:
    return "research"

if risk_missing:
    return "risk"

return "evaluate"
```

Best when:

* workflow is predictable
* compliance requirements are strict
* deterministic behavior is important

---

### Option 2 — LLM Supervisor

The LLM decides:

```text
research
risk
compliance
retry
human_review
finalize
```

Useful when:

* workflow is dynamic
* tasks vary substantially
* the next step depends on semantic reasoning

But you need strict validation.

---

### Option 3 — Hybrid Supervisor

This is often the strongest production design.

```text
              Supervisor
                  │
         ┌────────┴─────────┐
         │                  │
 Deterministic rules     LLM reasoning
         │                  │
         └────────┬─────────┘
                  ▼
            Policy Validator
                  │
                  ▼
             Next Action
```

For example:

```python
class Supervisor:

    async def decide(self, state):

        if state["retry_count"] >= 3:
            return "human_review"

        if state["requires_human_approval"]:
            return "human_review"

        if not state["research_result"]:
            return "research"

        return await self.llm_decide(state)
```

This is much safer.

---

# 12. Executor design

Now suppose Supervisor says:

```text
risk_analysis
```

The executor registry maps that task to an actual implementation.

```python
class ExecutorRegistry:

    def __init__(self):
        self.executors = {}

    def register(self, name, executor):
        self.executors[name] = executor

    def get(self, name):
        return self.executors[name]
```

Register:

```python
registry.register(
    "risk_analysis",
    RiskAnalysisAgent()
)

registry.register(
    "research",
    ResearchAgent()
)

registry.register(
    "compliance_analysis",
    ComplianceAgent()
)
```

Then:

```python
executor = registry.get(next_action)

result = await executor.execute(state)
```

This creates a clean abstraction.

---

# 13. Executor contract

Every agent should ideally follow a common contract.

```python
from typing import Protocol


class AgentExecutor(Protocol):

    async def execute(
        self,
        state: WorkflowState
    ) -> dict:
        ...
```

Then:

```python
class RiskAgent:

    async def execute(
        self,
        state: WorkflowState
    ) -> dict:

        documents = await self.retrieve_policy(
            state["user_request"]
        )

        assessment = await self.analyze_risk(
            documents
        )

        return assessment
```

And:

```python
class ResearchAgent:

    async def execute(
        self,
        state: WorkflowState
    ) -> dict:

        data = await self.search_market_data(
            state["user_request"]
        )

        return {
            "sources": data.sources,
            "summary": data.summary
        }
```

The Supervisor doesn't need to know how these agents work internally.

---

# 14. LangGraph implementation

This architecture maps very naturally to LangGraph.

Conceptually:

```text
START
  ↓
Planner
  ↓
Supervisor
  ↓
┌───────────────────────────┐
│                           │
Research / Risk / Compliance │
│                           │
└──────────────┬────────────┘
               ↓
           Evaluator
               │
         ┌─────┴─────┐
         ↓           ↓
       PASS        REWORK
         │           │
         ↓           └──→ Supervisor
      Finalize
         ↓
        END
```

A simplified implementation:

```python
from langgraph.graph import StateGraph, END


workflow = StateGraph(WorkflowState)

workflow.add_node(
    "planner",
    planner_node
)

workflow.add_node(
    "supervisor",
    supervisor_node
)

workflow.add_node(
    "research",
    research_node
)

workflow.add_node(
    "risk",
    risk_node
)

workflow.add_node(
    "compliance",
    compliance_node
)

workflow.add_node(
    "evaluator",
    evaluator_node
)

workflow.add_node(
    "finalize",
    finalize_node
)
```

---

# 15. Conditional routing

The Supervisor determines where to go.

```python
def route_from_supervisor(
    state: WorkflowState
) -> str:

    return state["next_action"]
```

Then:

```python
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research": "research",
        "risk": "risk",
        "compliance": "compliance",
        "evaluate": "evaluator",
        "finalize": "finalize"
    }
)
```

This is an important enterprise pattern.

The LLM isn't directly controlling LangGraph.

Instead:

```text
LLM decision
      ↓
Structured output
      ↓
Validation
      ↓
Allowed route
      ↓
LangGraph
```

---

# 16. Evaluator

A serious production architecture should have an evaluator.

Why?

Because:

```text
Agent completed successfully
```

does **not** mean:

```text
Agent produced a correct result
```

For example, the Risk Agent might return:

```json
{
  "risk": "low"
}
```

but the supporting evidence might actually indicate high risk.

The evaluator checks:

```text
Completeness
Correctness
Evidence
Citations
Policy compliance
Schema validity
Confidence
```

---

# 17. Evaluator → Rework loop

This is where the architecture becomes much more powerful.

```text
Risk Agent
    ↓
Evaluator
    ↓
FAIL
    ↓
Supervisor
    ↓
Risk Agent
```

But don't create unlimited loops.

Use:

```python
MAX_RETRIES = 3
```

and:

```python
if state["retry_count"] >= MAX_RETRIES:
    return "human_review"
```

So:

```text
Attempt 1
   ↓
Evaluator FAIL
   ↓
Attempt 2
   ↓
Evaluator FAIL
   ↓
Attempt 3
   ↓
Evaluator FAIL
   ↓
Human Review
```

This prevents agent loops from becoming expensive or infinite.

---

# 18. HITL

Now add human approval.

Suppose the system determines:

```text
High financial risk
```

The Supervisor should be able to route to:

```text
HUMAN_REVIEW
```

Production flow:

```text
Supervisor
    ↓
HITL checkpoint
    ↓
Persist state
    ↓
Workflow paused
    ↓
Human reviews
    ↓
Approve / Reject / Request changes
    ↓
Resume workflow
```

This is different from simply returning:

```json
{
  "status": "pending"
}
```

A true production HITL system needs **durable workflow state**.

For example:

```text
PostgreSQL
───────────────
workflow_id
status
current_node
state
plan
retry_count
human_decision
created_at
updated_at
```

Then later:

```text
POST /workflows/{workflow_id}/resume
```

can resume the workflow.

---

# 19. Planner + Supervisor + Executor + Evaluator

Now let's put everything together.

```text
                         USER
                           │
                           ▼
                     FastAPI API
                           │
                           ▼
                    Application Service
                           │
                           ▼
                       PLANNER
                           │
                           ▼
                    Execution Plan
                           │
                           ▼
                      SUPERVISOR
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Research        Risk       Compliance
          Executor       Executor     Executor
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                       EVALUATOR
                           │
                  ┌────────┴─────────┐
                  ▼                  ▼
                PASS               FAIL
                  │                  │
                  ▼                  ▼
              FINALIZE            REWORK
                  │                  │
                  │                  └───────┐
                  ▼                          │
                RESULT                       │
                                             ▼
                                        SUPERVISOR
```

---

# 20. Where RAG fits

RAG is **not** the Supervisor.

RAG is normally an executor capability.

For example:

```text
Supervisor
     ↓
Compliance Agent
     ↓
RAG Service
     ↓
Retriever Strategy
     ↓
VectorStore Port
     ↓
pgvector Adapter
     ↓
PostgreSQL
```

And:

```text
Risk Agent
     ↓
RAG
     ↓
Policy documents
     ↓
LLM
     ↓
Risk assessment
```

---

# 21. Where Strategy Pattern fits

This connects directly to what we discussed earlier.

Suppose your Research Agent needs retrieval.

You might have:

```python
class RetrievalStrategy(Protocol):

    async def retrieve(
        self,
        query: str
    ) -> list:
        ...
```

Implementations:

```text
VectorRetrievalStrategy
BM25RetrievalStrategy
HybridRetrievalStrategy
```

The agent chooses:

```text
Research Agent
      ↓
Retrieval Strategy
      ↓
Hybrid Retrieval
      ↓
VectorStore Adapter
      ↓
pgvector
```

So:

> **Supervisor chooses the task. Strategy chooses an algorithm. Adapter connects that algorithm to an external technology.**

This distinction is very important.

---

# 22. Where Adapter fits

For example:

```text
Risk Agent
     ↓
LLMProvider
     ↓
LLM Router
     ↓
OpenAI Adapter
     ↓
OpenAI
```

or:

```text
Compliance Agent
       ↓
VectorStore Port
       ↓
PgVector Adapter
       ↓
PostgreSQL
```

Therefore your architecture can combine:

```text
Planner
   ↓
Supervisor
   ↓
Executor
   ↓
Strategy
   ↓
Port
   ↓
Adapter
   ↓
Infrastructure
```

---

# 23. Production package structure

For the modular architecture we discussed, I would structure it roughly like this:

```text
financial-ai-platform/

├── packages/
│
├── ai-agents/
│   └── src/ai_agents/
│       ├── contracts/
│       │   ├── agent.py
│       │   ├── planner.py
│       │   ├── supervisor.py
│       │   └── evaluator.py
│       │
│       ├── planner/
│       │   ├── planner.py
│       │   └── validators.py
│       │
│       ├── supervisor/
│       │   ├── supervisor.py
│       │   ├── router.py
│       │   └── policy.py
│       │
│       ├── runtime/
│       │   ├── executor.py
│       │   └── registry.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   └── rework.py
│       │
│       └── workflow/
│           ├── graph.py
│           ├── state.py
│           └── checkpoint.py
│
├── ai-llm/
│   └── src/ai_llm/
│       ├── ports.py
│       ├── router.py
│       └── adapters/
│           ├── openai.py
│           ├── anthropic.py
│           └── bedrock.py
│
├── ai-rag/
│   └── src/ai_rag/
│       ├── strategies/
│       │   ├── vector.py
│       │   ├── bm25.py
│       │   └── hybrid.py
│       └── adapters/
│           ├── pgvector.py
│           └── qdrant.py
│
└── applications/
    │
    ├── financial-compliance/
    │   └── src/financial_compliance/
    │       ├── agents/
    │       │   ├── research.py
    │       │   ├── risk.py
    │       │   └── compliance.py
    │       │
    │       ├── workflows/
    │       │   └── compliance_review.py
    │       │
    │       └── services/
    │           └── review_service.py
    │
    └── investment-intelligence/
        └── src/investment_intelligence/
            ├── agents/
            ├── workflows/
            └── services/
```

This is a much cleaner Staff-level separation.

---

# 24. Real production request

Let's walk through an actual request.

User sends:

```http
POST /v1/investment-analysis
```

```json
{
  "security": "ABC Corp",
  "question": "Should we recommend this investment?"
}
```

### Step 1 — FastAPI

```text
HTTP request
     ↓
Authentication
     ↓
RBAC
     ↓
Application Service
```

---

### Step 2 — Create workflow

```python
workflow_id = uuid4()
```

Persist:

```text
workflow_id = 9a31...
status = RUNNING
```

---

### Step 3 — Planner

Planner determines:

```text
1. Research ABC
2. Retrieve investment policy
3. Analyze risk
4. Analyze fees
5. Generate recommendation
```

---

### Step 4 — Supervisor

Supervisor sees:

```text
research_result = None
```

Therefore:

```text
→ research
```

---

### Step 5 — Research Agent

Research Agent:

```text
Web/API tools
      ↓
Market data
      ↓
Normalize
      ↓
Research result
```

Output:

```json
{
  "company": "ABC Corp",
  "market_cap": "...",
  "sources": [...]
}
```

State is persisted.

---

### Step 6 — Supervisor

Now:

```text
research_result != None
policy_result == None
```

Supervisor:

```text
→ policy_retrieval
```

---

### Step 7 — Risk Agent

Once required dependencies exist:

```text
Risk Agent
    ↓
RAG
    ↓
Policy documents
    ↓
Market data
    ↓
LLM
```

Produces:

```json
{
  "risk_level": "HIGH",
  "factors": [
    "High volatility",
    "Concentration risk"
  ],
  "citations": [...]
}
```

---

### Step 8 — Fee Agent

Produces:

```json
{
  "annual_fee": 1.25,
  "estimated_impact": 0.87
}
```

---

# 25. Evaluator

Now evaluator receives:

```text
Research
+
Policy
+
Risk
+
Fee
```

It checks:

```text
Are all required sections present?
Are citations valid?
Are risk claims supported?
Are calculations valid?
Does output match schema?
Does policy permit recommendation?
```

Suppose:

```json
{
  "status": "FAIL",
  "reason": "Risk assessment lacks evidence for concentration-risk claim."
}
```

---

# 26. Rework

Supervisor sees:

```text
evaluator.status = FAIL
```

and:

```text
retry_count = 1
```

Routes:

```text
→ risk_analysis
```

Risk Agent performs another retrieval.

Perhaps it finds:

```text
Policy document section 4.3
```

and produces stronger evidence.

---

# 27. Evaluator passes

```json
{
  "status": "PASS",
  "score": 0.94
}
```

Supervisor:

```text
→ finalize
```

Finalizer generates:

```json
{
  "recommendation": "CONDITIONAL_APPROVAL",
  "risk": "HIGH",
  "fees": {...},
  "evidence": [...],
  "confidence": 0.94
}
```

---

# 28. Human approval

Because:

```text
risk == HIGH
```

policy says:

```text
HIGH_RISK_RECOMMENDATION → HUMAN_APPROVAL
```

Therefore:

```text
Finalizer
    ↓
HITL
    ↓
PostgreSQL checkpoint
    ↓
PAUSED
```

The API might return:

```json
{
  "workflow_id": "9a31...",
  "status": "PENDING_HUMAN_APPROVAL"
}
```

Human approves.

```http
POST /v1/workflows/9a31/resume
```

```json
{
  "decision": "APPROVED"
}
```

Workflow resumes:

```text
checkpoint
    ↓
Supervisor
    ↓
Finalize
    ↓
Audit record
    ↓
Response
```

---

# 29. PostgreSQL, Redis and Celery

A production implementation needs clear responsibility.

### PostgreSQL

Use for:

```text
Users
Workflow state
Plans
Agent results
Audit records
HITL decisions
Policy versions
Evaluation results
Vector data
```

PostgreSQL is your durable source of truth.

---

### Redis

Use for:

```text
Caching
Rate limiting
Distributed locks
Short-lived state
Celery broker
Idempotency support
```

Don't make Redis your only durable workflow state.

---

### Celery

Use when workflows are long-running.

For example:

```text
API
 ↓
Create workflow
 ↓
Celery task
 ↓
LangGraph execution
 ↓
Agents
 ↓
Checkpoint
```

This prevents a 5-minute AI workflow from tying up an HTTP request.

---

# 30. Production reliability

This architecture becomes enterprise-grade when you add:

### Idempotency

```text
same request
     ↓
same idempotency key
     ↓
don't execute twice
```

### Retry policies

```text
LLM timeout → retry
429 → exponential backoff
5xx → retry
invalid output → rework
business violation → human
```

### Maximum steps

```python
MAX_STEPS = 20
```

Never allow:

```text
Agent → Supervisor → Agent → Supervisor → ...
```

forever.

---

# 31. Security

The Supervisor must not bypass authorization.

For example:

```text
User
 ↓
RBAC
 ↓
Supervisor
 ↓
Tool authorization
 ↓
Tool
```

Every tool should have permissions.

```python
TOOL_PERMISSIONS = {
    "market_data": "market.read",
    "policy_search": "policy.read",
    "portfolio_update": "portfolio.write"
}
```

If an agent tries:

```text
portfolio.write
```

without permission:

```text
DENY
```

The LLM should never be considered a security boundary.

---

# 32. Observability

For every workflow:

```text
trace_id
workflow_id
tenant_id
user_id
agent
node
model
latency
tokens
cost
tool
status
retry_count
```

Example:

```text
workflow_id=abc123

Planner
 └─ 1.2 sec
    └─ 1,200 tokens

Supervisor
 └─ 0.4 sec

Research Agent
 ├─ web_search: 2.1 sec
 └─ GPT: 1.8 sec

Risk Agent
 ├─ pgvector: 120 ms
 └─ GPT: 2.3 sec

Evaluator
 └─ FAIL

Risk Agent
 └─ REWORK

Evaluator
 └─ PASS
```

This makes production debugging possible.

---

# 33. Don't over-agentize

This is another Staff-level principle.

You don't need:

```text
Planner Agent
Supervisor Agent
Research Agent
Search Agent
Retriever Agent
Chunk Agent
Embedding Agent
Risk Agent
Fee Agent
Compliance Agent
Writer Agent
Evaluator Agent
Critic Agent
Reviewer Agent
```

just because you can.

Every agent introduces:

* latency
* token cost
* failure modes
* state complexity
* observability complexity
* security surface

A good Staff Engineer asks:

> **“Does this need autonomous reasoning, or can deterministic code do it?”**

For example:

```text
Calculate fee
```

should probably be deterministic Python.

Not:

```text
FeeCalculationLLMAgent
```

---

# 34. When to use Planner–Supervisor–Executor

Use it when you have:

* multiple specialized capabilities
* dynamic workflows
* conditional routing
* parallel tasks
* retries/rework
* human approval
* long-running workflows
* heterogeneous tools
* complex dependencies

Don't use it for:

```text
Simple RAG question
```

where:

```text
query
 ↓
retrieve
 ↓
LLM
 ↓
answer
```

is sufficient.

---

# 35. Planner–Supervisor–Executor vs simpler architectures

### Simple RAG

```text
API
 ↓
Retriever
 ↓
LLM
 ↓
Answer
```

Use for straightforward Q&A.

### Sequential agent workflow

```text
Research
 ↓
Risk
 ↓
Writer
```

Use when flow is fixed.

### Planner–Executor

```text
Planner
 ↓
Tasks
 ↓
Executors
```

Use when task decomposition is dynamic.

### Supervisor architecture

```text
Supervisor
 ↓
Agent A / B / C
```

Use when routing is dynamic.

### Planner + Supervisor + Executor

```text
Planner
   ↓
Plan
   ↓
Supervisor
   ↓
Executors
   ↓
Evaluator
   ↓
Rework
```

Use when you have a genuinely complex enterprise workflow.

---

# 36. The Staff Engineer mental model

The easiest way to remember the architecture is:

```text
                 PLANNER
                    │
             "What needs doing?"
                    │
                    ▼
             EXECUTION PLAN
                    │
                    ▼
               SUPERVISOR
                    │
             "What's next?"
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    EXECUTOR      EXECUTOR     EXECUTOR
       │            │            │
       ▼            ▼            ▼
    Research       Risk       Compliance
       │            │            │
       └────────────┼────────────┘
                    ▼
                EVALUATOR
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        PASS                 FAIL
          │                   │
          ▼                   ▼
       FINALIZE             REWORK
          │                   │
          │                   └──→ Supervisor
          ▼
        HITL if needed
          │
          ▼
        RESULT
```

And underneath that:

```text
                  APPLICATION
                       │
                  ┌────┴────┐
                  │         │
               Strategy   Ports
                  │         │
                  │      Adapters
                  │         │
                  ▼         ▼
              Algorithms Infrastructure
```

So the patterns have **different jobs**:

| Pattern                  | Main question                                |
| ------------------------ | -------------------------------------------- |
| **Planner**              | What work needs to happen?                   |
| **Supervisor**           | What should happen next?                     |
| **Executor**             | How do I perform this task?                  |
| **Evaluator**            | Is the result good enough?                   |
| **Strategy**             | Which algorithm/behavior should I use?       |
| **Adapter**              | How do I integrate this external system?     |
| **Repository**           | How do I persist/retrieve domain data?       |
| **Factory**              | Which implementation should I construct?     |
| **Dependency Injection** | How do I provide that implementation?        |
| **HITL**                 | When does a human need to make the decision? |

That separation is what makes a multi-agent system **architecturally maintainable rather than just a collection of LLM calls**.

For your two finance applications, the strongest architecture would therefore be **shared Planner/Supervisor/Executor runtime in `ai-agents`, shared RAG/LLM/guardrail infrastructure in packages, and business-specific Research/Risk/Compliance/Market agents inside each application**. This lets both applications reuse the orchestration engine without forcing one application's business logic into the other.
