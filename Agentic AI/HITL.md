Yes. For a **production-level AI system**, Human-in-the-Loop (HITL) is not simply:

```python
input("Approve?")
```

It is a **durable workflow** where the AI can pause, persist its state, expose an approval task to a human, and later resume from exactly where it stopped.

For your **FastAPI + LangGraph + RAG + tools + PostgreSQL + Redis enterprise AI project**, a good architecture is:

```text
User
 │
 ▼
FastAPI
 │
 ▼
LangGraph Workflow
 │
 ├── Input Guardrails
 │
 ├── Planner
 │
 ├── RAG / Research
 │
 ├── Tool Decision
 │
 ▼
Risk Check
 │
 ├── LOW RISK ───────────────► Execute
 │
 └── HIGH RISK
          │
          ▼
   Create Approval Task
          │
          ▼
     PostgreSQL
          │
          ▼
     PAUSED GRAPH
          │
          ▼
      Human UI
          │
       Approve / Reject
          │
          ▼
     Resume Graph
          │
          ▼
      Execute Tool
          │
          ▼
     Output Guardrails
          │
          ▼
       Response
```

The critical idea is:

> **The human approval must be persisted outside the LLM and the workflow must be resumable.**

---

# 1. Real Production Example

Let's use a financial AI assistant.

A user says:

```text
Transfer ₹50,000 from my account to account XYZ.
```

The AI can understand the request, but it should **not autonomously execute the transfer**.

Instead:

```text
User
 ↓
AI
 ↓
Understands transfer request
 ↓
Validates parameters
 ↓
Checks permissions
 ↓
Risk engine
 ↓
HIGH RISK
 ↓
Create HITL approval
 ↓
Pause workflow
 ↓
Human approves
 ↓
Resume workflow
 ↓
Transfer API
 ↓
Verify transaction
 ↓
Response
```

---

# 2. Project Structure

A production-oriented implementation:

```text
enterprise_ai/
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py
│   │       └── approvals.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── exceptions.py
│   │   └── security.py
│   │
│   ├── schemas/
│   │   ├── chat.py
│   │   ├── approval.py
│   │   └── workflow.py
│   │
│   ├── models/
│   │   └── approval.py
│   │
│   ├── workflows/
│   │   ├── state.py
│   │   ├── graph.py
│   │   └── nodes/
│   │       ├── planner.py
│   │       ├── risk_check.py
│   │       ├── approval.py
│   │       ├── execute.py
│   │       └── response.py
│   │
│   ├── services/
│   │   ├── chat_service.py
│   │   ├── approval_service.py
│   │   └── transaction_service.py
│   │
│   ├── tools/
│   │   └── transaction_tool.py
│   │
│   └── repositories/
│       └── approval_repository.py
│
├── tests/
│   ├── test_approval.py
│   └── test_workflow.py
│
├── requirements.txt
└── .env
```

---

# 3. Dependencies

```text
fastapi
uvicorn
langgraph
langchain
pydantic
pydantic-settings
sqlalchemy
asyncpg
psycopg
```

For example:

```text
requirements.txt
```

```text
fastapi
uvicorn[standard]
langgraph
langchain
pydantic
pydantic-settings
sqlalchemy
asyncpg
```

---

# 4. Configuration

## `app/core/config.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    database_url: str = (
        "postgresql+asyncpg://"
        "postgres:postgres@localhost:5432/ai_db"
    )

    environment: str = "development"

    approval_timeout_minutes: int = 30

    max_transfer_amount: float = 100000

    class Config:
        env_file = ".env"


settings = Settings()
```

---

# 5. Database

## `app/core/database.py`

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)

from app.core.config import settings


engine = create_async_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)


SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db():

    async with SessionLocal() as session:

        yield session
```

In production:

```text
FastAPI
   ↓
DB connection pool
   ↓
PostgreSQL
```

Don't create a new database connection for every request.

---

# 6. Approval Database Model

The approval needs to survive:

* application restart
* pod restart
* workflow crash
* human taking 10 minutes to respond
* network failure

Therefore we persist it.

## `app/models/approval.py`

```python
from datetime import datetime

from sqlalchemy import (
    String,
    DateTime,
    Text,
    Float
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column
)


class Base(DeclarativeBase):
    pass


class Approval(Base):

    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True
    )

    thread_id: Mapped[str] = mapped_column(
        String(200),
        index=True
    )

    user_id: Mapped[str] = mapped_column(
        String(100),
        index=True
    )

    action: Mapped[str] = mapped_column(
        String(100)
    )

    amount: Mapped[float] = mapped_column(
        Float
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
        index=True
    )

    reason: Mapped[str] = mapped_column(
        Text
    )

    reviewer_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )
```

A row might look like:

```text
id              approval-123
thread_id       thread-789
user_id         user-001
action          transfer_money
amount          50000
status          pending
reason          High risk financial transaction
reviewer_id     NULL
```

---

# 7. Approval Schema

## `app/schemas/approval.py`

```python
from datetime import datetime

from pydantic import BaseModel


class ApprovalRequest(BaseModel):

    decision: str

    comment: str | None = None


class ApprovalResponse(BaseModel):

    id: str

    status: str

    action: str

    amount: float

    created_at: datetime
```

---

# 8. Workflow State

This is extremely important for LangGraph.

## `app/workflows/state.py`

```python
from typing import TypedDict


class WorkflowState(TypedDict, total=False):

    thread_id: str

    user_id: str

    message: str

    action: str

    amount: float

    destination_account: str

    risk_level: str

    approval_id: str

    approval_status: str

    transaction_id: str

    response: str
```

Think of this as:

```text
WorkflowState = memory of this workflow execution
```

Example:

```python
{
    "thread_id": "thread-123",
    "user_id": "user-456",
    "action": "transfer_money",
    "amount": 50000,
    "risk_level": "high",
    "approval_id": "approval-789"
}
```

---

# 9. Planner Node

The planner decides what the user wants.

## `app/workflows/nodes/planner.py`

```python
from app.workflows.state import WorkflowState


async def planner(
    state: WorkflowState
) -> WorkflowState:

    message = state["message"]

    # In a real application,
    # this would be an LLM structured-output call.

    if "transfer" in message.lower():

        return {
            **state,

            "action": "transfer_money",

            "amount": 50000,

            "destination_account": "ACC-XYZ"
        }

    return {
        **state,
        "action": "information_request"
    }
```

Production version:

```text
LLM
 ↓
Structured output
 ↓
Pydantic validation
 ↓
WorkflowState
```

Never trust raw LLM output.

---

# 10. Risk Engine

The AI should not determine whether approval is necessary.

Use deterministic business logic.

## `app/workflows/nodes/risk_check.py`

```python
from app.workflows.state import WorkflowState


async def risk_check(
    state: WorkflowState
) -> WorkflowState:

    action = state["action"]

    amount = state.get(
        "amount",
        0
    )

    if action == "transfer_money":

        if amount >= 10000:

            return {
                **state,
                "risk_level": "high"
            }

        return {
            **state,
            "risk_level": "medium"
        }

    return {
        **state,
        "risk_level": "low"
    }
```

This is much safer than asking:

```text
LLM, should I require approval?
```

The LLM can recommend an action.

The application decides whether that action requires authorization.

---

# 11. Approval Node

Now comes HITL.

## `app/workflows/nodes/approval.py`

```python
import uuid

from app.workflows.state import WorkflowState


async def create_approval(
    state: WorkflowState
) -> WorkflowState:

    approval_id = str(
        uuid.uuid4()
    )

    # In production:
    #
    # await approval_repository.create(...)

    print(
        f"Creating approval: {approval_id}"
    )

    return {
        **state,

        "approval_id": approval_id,

        "approval_status": "pending"
    }
```

But there's a critical issue.

Creating an approval isn't enough.

The workflow must **pause**.

---

# 12. LangGraph Interrupt

This is where LangGraph becomes particularly useful.

## `app/workflows/nodes/approval.py`

```python
from langgraph.types import interrupt

from app.workflows.state import WorkflowState


async def approval_node(
    state: WorkflowState
) -> WorkflowState:

    approval_request = {
        "action": state["action"],
        "amount": state["amount"],
        "destination": state[
            "destination_account"
        ],
        "reason": (
            "Financial transaction requires "
            "human approval."
        )
    }

    decision = interrupt(
        approval_request
    )

    return {
        **state,

        "approval_status": decision[
            "decision"
        ]
    }
```

This is the key concept:

```python
interrupt(...)
```

The graph pauses here.

---

# 13. What `interrupt()` Means

Before:

```text
planner
   ↓
risk_check
   ↓
approval_node
   ↓
execute
```

When `approval_node` reaches:

```python
interrupt(...)
```

the graph becomes:

```text
planner
   ↓
risk_check
   ↓
approval_node
   ↓
████████████████
     PAUSED
████████████████
```

It doesn't continue to:

```text
execute
```

until the human responds.

---

# 14. Execute Node

## `app/workflows/nodes/execute.py`

```python
from app.workflows.state import WorkflowState


async def execute_transaction(
    state: WorkflowState
) -> WorkflowState:

    if state["approval_status"] != "approved":

        return {
            **state,

            "response": (
                "Transaction was not approved."
            )
        }

    # In production:
    #
    # transaction_id = await transaction_service.transfer(...)
    #

    transaction_id = "TXN-123456"

    return {
        **state,

        "transaction_id": transaction_id
    }
```

Notice:

```python
if state["approval_status"] != "approved":
```

Even if someone accidentally reaches this node, the transaction won't execute.

That's **defense in depth**.

---

# 15. Response Node

## `app/workflows/nodes/response.py`

```python
from app.workflows.state import WorkflowState


async def response_node(
    state: WorkflowState
) -> WorkflowState:

    if state.get("transaction_id"):

        return {
            **state,

            "response": (
                f"Transaction completed successfully. "
                f"Transaction ID: "
                f"{state['transaction_id']}"
            )
        }

    return state
```

---

# 16. Build LangGraph

## `app/workflows/graph.py`

```python
from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.checkpoint.memory import (
    MemorySaver
)

from app.workflows.state import WorkflowState

from app.workflows.nodes.planner import (
    planner
)

from app.workflows.nodes.risk_check import (
    risk_check
)

from app.workflows.nodes.approval import (
    approval_node
)

from app.workflows.nodes.execute import (
    execute_transaction
)

from app.workflows.nodes.response import (
    response_node
)


def build_graph():

    builder = StateGraph(
        WorkflowState
    )

    builder.add_node(
        "planner",
        planner
    )

    builder.add_node(
        "risk_check",
        risk_check
    )

    builder.add_node(
        "approval",
        approval_node
    )

    builder.add_node(
        "execute",
        execute_transaction
    )

    builder.add_node(
        "response",
        response_node
    )

    builder.add_edge(
        START,
        "planner"
    )

    builder.add_edge(
        "planner",
        "risk_check"
    )

    builder.add_edge(
        "risk_check",
        "approval"
    )

    builder.add_edge(
        "approval",
        "execute"
    )

    builder.add_edge(
        "execute",
        "response"
    )

    builder.add_edge(
        "response",
        END
    )

    checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


graph = build_graph()
```

For a real production deployment, don't use an in-memory checkpointer as the durable source of workflow state. Use a persistent checkpointer/database appropriate to your LangGraph deployment.

---

# 17. Thread ID

This is critical.

Every workflow execution needs a unique ID.

Example:

```python
config = {
    "configurable": {
        "thread_id": "thread-123"
    }
}
```

Think of:

```text
thread_id
```

as:

```text
workflow execution ID
```

For example:

```text
thread-123
```

contains:

```text
User request
       ↓
Planner state
       ↓
Risk state
       ↓
Approval state
       ↓
Human decision
       ↓
Execution state
```

---

# 18. Starting the Workflow

## `app/services/chat_service.py`

```python
import uuid

from app.workflows.graph import graph


class ChatService:

    async def start(
        self,
        user_id: str,
        message: str
    ):

        thread_id = str(
            uuid.uuid4()
        )

        initial_state = {

            "thread_id": thread_id,

            "user_id": user_id,

            "message": message
        }

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = await graph.ainvoke(
            initial_state,
            config=config
        )

        return {
            "thread_id": thread_id,
            "result": result
        }
```

If the graph hits:

```python
interrupt(...)
```

the execution pauses.

---

# 19. FastAPI Chat Endpoint

## `app/api/routes/chat.py`

```python
from fastapi import (
    APIRouter,
    Depends
)

from app.services.chat_service import (
    ChatService
)

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


chat_service = ChatService()


@router.post("")
async def chat(
    message: str
):

    result = await chat_service.start(
        user_id="user-123",
        message=message
    )

    return result
```

The response could tell the frontend:

```json
{
    "thread_id": "thread-123",
    "status": "waiting_for_approval"
}
```

---

# 20. Human Approval Endpoint

This is the other half of HITL.

## `app/api/routes/approvals.py`

```python
from fastapi import APIRouter

from langgraph.types import Command

from app.workflows.graph import graph


router = APIRouter(
    prefix="/approvals",
    tags=["approvals"]
)


@router.post(
    "/{thread_id}"
)
async def resolve_approval(
    thread_id: str,
    decision: str
):

    if decision not in {
        "approved",
        "rejected"
    }:

        return {
            "error": "Invalid decision"
        }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = await graph.ainvoke(
        Command(
            resume={
                "decision": decision
            }
        ),
        config=config
    )

    return {
        "thread_id": thread_id,
        "result": result
    }
```

This resumes the paused graph.

---

# 21. Full HITL Flow

Suppose:

```text
POST /chat
```

with:

```text
Transfer ₹50,000 to account XYZ.
```

### Step 1

```text
FastAPI
 ↓
LangGraph
```

### Step 2

Planner:

```text
action = transfer_money
amount = 50000
```

### Step 3

Risk:

```text
amount >= 10000
```

therefore:

```text
risk = high
```

### Step 4

Approval:

```python
interrupt(...)
```

Graph pauses.

---

# 22. Frontend Gets Approval Request

Your UI can display:

```text
┌──────────────────────────────────────┐
│       Transaction Approval           │
├──────────────────────────────────────┤
│                                      │
│ Action: Transfer Money               │
│ Amount: ₹50,000                      │
│ Destination: ACC-XYZ                 │
│                                      │
│ Reason: High-risk transaction        │
│                                      │
│ [ Reject ]        [ Approve ]        │
│                                      │
└──────────────────────────────────────┘
```

The human clicks:

```text
Approve
```

Frontend calls:

```text
POST /approvals/thread-123
```

with:

```json
{
    "decision": "approved"
}
```

---

# 23. Graph Resumes

The graph resumes from:

```text
approval
   │
   ▼
execute
   │
   ▼
response
```

It does **not** restart from:

```text
planner
```

That's one of the major benefits of durable workflow orchestration.

---

# 24. Production Approval Service

In reality, I would separate approval logic from the graph.

## `app/services/approval_service.py`

```python
from datetime import datetime

from sqlalchemy import select

from app.models.approval import Approval


class ApprovalService:

    def __init__(self, session):

        self.session = session

    async def create(
        self,
        approval_id: str,
        thread_id: str,
        user_id: str,
        action: str,
        amount: float,
        reason: str
    ):

        approval = Approval(

            id=approval_id,

            thread_id=thread_id,

            user_id=user_id,

            action=action,

            amount=amount,

            status="pending",

            reason=reason
        )

        self.session.add(
            approval
        )

        await self.session.commit()

        return approval

    async def get(
        self,
        approval_id: str
    ):

        result = await self.session.execute(

            select(Approval).where(
                Approval.id == approval_id
            )
        )

        return result.scalar_one_or_none()

    async def approve(
        self,
        approval_id: str,
        reviewer_id: str
    ):

        approval = await self.get(
            approval_id
        )

        if approval is None:
            raise ValueError(
                "Approval not found"
            )

        if approval.status != "pending":
            raise ValueError(
                "Approval already resolved"
            )

        approval.status = "approved"

        approval.reviewer_id = reviewer_id

        approval.reviewed_at = datetime.utcnow()

        await self.session.commit()

        return approval

    async def reject(
        self,
        approval_id: str,
        reviewer_id: str
    ):

        approval = await self.get(
            approval_id
        )

        if approval is None:
            raise ValueError(
                "Approval not found"
            )

        if approval.status != "pending":
            raise ValueError(
                "Approval already resolved"
            )

        approval.status = "rejected"

        approval.reviewer_id = reviewer_id

        approval.reviewed_at = datetime.utcnow()

        await self.session.commit()

        return approval
```

---

# 25. Why PostgreSQL?

You don't want:

```python
approval = {
    "status": "pending"
}
```

living only in Python memory.

Imagine:

```text
Pod 1
 │
 ├── Workflow paused
 │
 └── Human approval pending

Pod crashes
```

If approval state was only in memory:

```text
Approval LOST
```

With PostgreSQL:

```text
Pod 1
 │
 ├── Workflow
 │
 ▼
PostgreSQL
 │
 │
 Pod crashes
 │
 ▼
Pod 2
 │
 ▼
Resume workflow
```

That's why persistence matters.

---

# 26. Production Tool

Let's build the actual transaction tool.

## `app/tools/transaction_tool.py`

```python
class TransactionTool:

    async def transfer(
        self,
        account_id: str,
        destination_account: str,
        amount: float
    ):

        if amount <= 0:
            raise ValueError(
                "Amount must be positive"
            )

        if amount > 100000:
            raise ValueError(
                "Transfer limit exceeded"
            )

        # Actual banking API call here.

        return {
            "transaction_id": "TXN-987654",
            "status": "completed"
        }
```

---

# 27. Never Let the LLM Directly Execute This

Bad architecture:

```text
LLM
 │
 ▼
transfer()
```

Good architecture:

```text
LLM
 │
 ▼
Tool Request
 │
 ▼
Schema Validation
 │
 ▼
Authentication
 │
 ▼
Authorization
 │
 ▼
Risk Engine
 │
 ▼
HITL
 │
 ▼
TransactionTool
```

This is a major Staff-level design point.

---

# 28. Add Authorization

Before creating approval:

```python
async def authorize_transfer(
    user_id: str,
    account_id: str
):

    # Query account ownership / permissions.

    if user_id != account_id:

        raise PermissionError(
            "User does not own this account"
        )
```

Never assume that because the LLM says:

```text
account_id = ABC
```

the user is allowed to operate on ABC.

---

# 29. Add Approval Expiration

Production approval shouldn't remain pending forever.

Example:

```text
Created:
10:00 AM

Expires:
10:30 AM
```

Add:

```python
expires_at
```

to your database.

Then:

```python
if datetime.utcnow() > approval.expires_at:

    approval.status = "expired"
```

And don't resume the transaction.

---

# 30. Prevent Double Approval

Imagine:

```text
Reviewer A → Approve
Reviewer B → Approve
```

You must make the operation idempotent.

The database transition should effectively be:

```text
pending
   │
   ├── approve → approved
   │
   └── reject  → rejected
```

Once:

```text
approved
```

you cannot transition again.

That's why:

```python
if approval.status != "pending":
    raise ValueError("Already resolved")
```

is important.

For high-concurrency production systems, enforce the state transition atomically at the database level as well.

---

# 31. Multi-Level Approval

For very high-risk actions:

```text
₹50,000
    ↓
1 approval

₹5,00,000
    ↓
2 approvals

₹50,00,000
    ↓
Finance + Compliance
```

Architecture:

```text
                Risk Engine
                     │
          ┌──────────┼───────────┐
          │          │           │
       Low Risk   Medium Risk   High Risk
          │          │           │
          ▼          ▼           ▼
       Execute    Manager      Compliance
                     │             │
                     ▼             ▼
                  Execute       Finance
```

This can be implemented as workflow branches.

---

# 32. Example Conditional Graph

Conceptually:

```python
def route_after_risk(
    state
):

    if state["risk_level"] == "low":

        return "execute"

    if state["risk_level"] == "medium":

        return "manager_approval"

    return "compliance_approval"
```

Then:

```python
builder.add_conditional_edges(
    "risk_check",
    route_after_risk,
    {
        "execute": "execute",
        "manager_approval": "manager_approval",
        "compliance_approval": "compliance_approval"
    }
)
```

This is a very common agentic workflow pattern.

---

# 33. HITL Doesn't Mean Only Approve/Reject

You can support:

```text
APPROVE
REJECT
EDIT
REQUEST_MORE_INFORMATION
ESCALATE
```

For example:

```json
{
    "decision": "edit",
    "amount": 25000
}
```

Then:

```text
AI proposed ₹50,000
       ↓
Human edits
       ↓
₹25,000
       ↓
Revalidate
       ↓
Execute
```

**Important:** if the human edits an important parameter, run the authorization and risk checks again.

Don't blindly resume execution.

---

# 34. HITL + Guardrails

This is where your previous guardrails architecture connects.

```text
                  User
                   │
                   ▼
              Input Guard
                   │
                   ▼
                Agent
                   │
                   ▼
               Tool Call
                   │
                   ▼
            Tool Guardrails
                   │
                   ▼
              Risk Engine
                   │
             high risk?
              /       \
            NO         YES
            │           │
            ▼           ▼
         Execute       HITL
                         │
                         ▼
                      Human
                      /   \
                 Reject   Approve
                    │        │
                    ▼        ▼
                   END     Revalidate
                              │
                              ▼
                           Execute
                              │
                              ▼
                       Output Guardrails
                              │
                              ▼
                           Response
```

This is a **production-grade pattern**.

---

# 35. What About Human Approval UI?

Typically you'd have:

```text
React / Next.js
        │
        ▼
GET /approvals
        │
        ▼
FastAPI
        │
        ▼
PostgreSQL
```

The UI might show:

```text
Pending Approvals

------------------------------------------------
Request       Amount       Risk       Action
------------------------------------------------
Transfer      ₹50,000      HIGH       Review
Transfer      ₹25,000      HIGH       Review
Account       Delete       CRITICAL    Review
------------------------------------------------
```

Clicking Review:

```text
GET /approvals/{approval_id}
```

Then:

```text
┌─────────────────────────────┐
│ Transaction Review          │
├─────────────────────────────┤
│ User: user-123              │
│ Amount: ₹50,000             │
│ Destination: ACC-XYZ        │
│ Risk: HIGH                  │
│                             │
│ [Reject]       [Approve]    │
└─────────────────────────────┘
```

---

# 36. Don't Trust the Frontend

This is critical.

Never do:

```python
@app.post("/approve")
async def approve():

    execute_transaction()
```

because anyone who can call that API might bypass the UI.

Instead:

```text
Frontend
   ↓
JWT
   ↓
FastAPI
   ↓
Reviewer authentication
   ↓
Reviewer authorization
   ↓
Approval ownership / scope check
   ↓
Atomic approval transition
   ↓
Resume workflow
```

The frontend is just a UI.

Security belongs on the backend.

---

# 37. Reviewer Authorization

Example:

```python
async def can_review(
    reviewer,
    approval
):

    if reviewer.role not in {
        "manager",
        "compliance",
        "admin"
    }:

        return False

    return True
```

But production systems should make this more granular.

For example:

```text
customer support
    ↓
cannot approve transfers

manager
    ↓
can approve <= ₹1 lakh

compliance
    ↓
can approve high-risk transactions

admin
    ↓
can review operational actions
```

---

# 38. Audit Log

Every HITL decision should be auditable.

Example:

```json
{
    "event": "approval_decision",
    "approval_id": "approval-123",
    "thread_id": "thread-456",
    "reviewer_id": "employee-789",
    "decision": "approved",
    "timestamp": "2026-09-20T11:00:00Z"
}
```

For enterprise systems:

```text
User request
      ↓
AI decision
      ↓
Risk decision
      ↓
Approval created
      ↓
Reviewer
      ↓
Approval/rejection
      ↓
Tool execution
      ↓
Final response
```

should be traceable.

---

# 39. Idempotency

This is another production-level requirement.

Suppose the human clicks:

```text
Approve
```

and the network times out.

They click again.

You don't want:

```text
₹50,000 transferred
₹50,000 transferred again
```

Therefore use an idempotency key:

```text
approval_id
```

or:

```text
transaction_request_id
```

Example:

```python
async def execute_transaction(
    approval_id: str
):

    existing = await find_transaction(
        approval_id
    )

    if existing:

        return existing

    return await create_transaction(
        approval_id
    )
```

The database should enforce uniqueness.

---

# 40. What Happens if the Pod Crashes?

This is a classic interview question.

### Before approval

```text
Pod
 ↓
LangGraph
 ↓
Approval
 ↓
PostgreSQL
 ↓
PAUSED
```

Pod crashes.

Human hasn't approved yet.

New pod starts:

```text
Pod 2
 ↓
PostgreSQL
 ↓
Find pending workflow
```

Human approves:

```text
Approval
 ↓
Resume workflow
 ↓
Execute
```

The workflow doesn't depend on the lifetime of a single HTTP request or pod.

---

# 41. What Happens if the LLM Crashes?

Suppose:

```text
Planner
 ↓
LLM
 ↓
timeout
```

Use retry policies around transient LLM calls.

But don't retry dangerous tool execution blindly.

For example:

```text
LLM call
→ safe to retry depending on provider semantics

GET account balance
→ generally retryable

TRANSFER MONEY
→ must be idempotent before retry
```

That's another reason tool execution must be isolated from the model.

---

# 42. A Better Production Architecture

For your enterprise project, I'd ultimately structure it like this:

```text
                         ┌───────────────┐
                         │ React / Web UI│
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    FastAPI    │
                         └───────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
               Chat API                 Approval API
                    │                         │
                    ▼                         ▼
              Auth + RBAC              Auth + RBAC
                    │                         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                          LangGraph
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
               ▼                 ▼                 ▼
            Planner             RAG             Tools
               │                 │                 │
               │                 ▼                 │
               │            Qdrant                 │
               │                                   │
               └────────────────┬──────────────────┘
                                │
                                ▼
                           Risk Engine
                                │
                       ┌────────┴────────┐
                       │                 │
                     LOW               HIGH
                       │                 │
                       ▼                 ▼
                    Execute             HITL
                                         │
                                         ▼
                                    PostgreSQL
                                         │
                                         ▼
                                      Human
                                         │
                              ┌──────────┴─────────┐
                              │                    │
                           Reject                Approve
                              │                    │
                              ▼                    ▼
                             END              Revalidate
                                                   │
                                                   ▼
                                                Execute
                                                   │
                                                   ▼
                                          Output Guardrails
                                                   │
                                                   ▼
                                                Response
```

---

# 43. Redis's Role

PostgreSQL and Redis have different jobs.

You can use Redis for:

```text
rate limiting
short-lived UI state
notifications
caching
pub/sub
WebSocket events
```

For example:

```text
Approval created
      ↓
PostgreSQL
      ↓
Redis Pub/Sub
      ↓
Frontend WebSocket
      ↓
"New approval required"
```

PostgreSQL remains the durable source of truth for the approval record.

---

# 44. Notification Flow

Production system:

```text
LangGraph
   │
   ▼
Create Approval
   │
   ├──────────────► PostgreSQL
   │
   └──────────────► Event
                         │
                         ▼
                    Redis / Queue
                         │
                  ┌──────┴──────┐
                  ▼             ▼
              WebSocket       Email
                  │             │
                  ▼             ▼
               Reviewer      Reviewer
```

This avoids keeping an HTTP request open while waiting for a human.

---

# 45. The Most Important Design Decision

**Do not do this:**

```python
@app.post("/chat")
async def chat():

    result = await agent()

    approval = input(
        "Approve? "
    )

    return result
```

This is a demo, not production HITL.

Instead:

```text
HTTP request
     ↓
Start workflow
     ↓
Persist state
     ↓
Return "waiting_for_approval"
```

Then:

```text
Human acts later
     ↓
Approval API
     ↓
Persist decision
     ↓
Resume workflow
```

This is the production pattern.

---

# 46. HITL State Machine

You should think of approval as a state machine:

```text
                  ┌───────────────┐
                  │    PENDING    │
                  └───────┬───────┘
                          │
             ┌────────────┼─────────────┐
             │            │             │
             ▼            ▼             ▼
          APPROVED     REJECTED       EXPIRED
             │            │             │
             ▼            ▼             ▼
          EXECUTE         END            END
             │
             ▼
          COMPLETED
```

This makes your system deterministic.

---

# 47. Testing HITL

## `tests/test_approval.py`

```python
import pytest


@pytest.mark.asyncio
async def test_approval_required():

    state = {
        "action": "transfer_money",
        "amount": 50000
    }

    assert state["amount"] >= 10000
```

Test approval state:

```python
@pytest.mark.asyncio
async def test_rejected_transaction():

    state = {
        "approval_status": "rejected"
    }

    assert (
        state["approval_status"]
        != "approved"
    )
```

Test idempotency:

```python
@pytest.mark.asyncio
async def test_already_approved():

    status = "approved"

    with pytest.raises(ValueError):

        if status != "pending":
            raise ValueError(
                "Approval already resolved"
            )
```

---

# 48. HITL + LangGraph Interview Answer

If an interviewer asks:

> **"How would you implement Human-in-the-Loop in an enterprise agentic AI system?"**

A strong answer would be:

> "I model HITL as a durable workflow interruption rather than blocking an HTTP request. LangGraph manages the workflow state and pauses at an approval node when a deterministic risk or authorization policy identifies a high-impact action. I persist the approval request, thread ID, action details, expiration and reviewer information in PostgreSQL. The approval API authenticates and authorizes the reviewer, performs an atomic state transition such as pending → approved or rejected, records an audit event, and resumes the workflow using the persisted thread state. Before executing the tool, I revalidate authorization, parameters and risk because the approval itself isn't a security boundary. Tool execution is idempotent to prevent duplicate side effects, and high-risk operations fail closed if the approval or policy services are unavailable."

That's the level of explanation I'd use for a **Senior/Staff AI Engineer interview**.

---

# 49. The Complete Mental Model

Remember this:

```text
                 LLM
                  │
                  │ proposes
                  ▼
               ACTION
                  │
                  ▼
          ┌───────────────┐
          │ Policy Engine │
          └───────┬───────┘
                  │
           Is approval needed?
              /          \
            NO            YES
            │              │
            ▼              ▼
         Execute          HITL
                            │
                            ▼
                     Persist Approval
                            │
                            ▼
                          Human
                         /     \
                    Reject     Approve
                       │          │
                       ▼          ▼
                      END      Revalidate
                                  │
                                  ▼
                               Execute
                                  │
                                  ▼
                           Output Guardrails
                                  │
                                  ▼
                              Response
```

The fundamental principle is:

> **The AI proposes; deterministic policy decides; the human approves high-impact actions; the backend executes.**

For your enterprise **LangGraph + RAG + FastAPI + PostgreSQL + Redis + multi-agent financial copilot**, this is the pattern I'd recommend rather than putting a generic `human approval` node into the graph and considering the problem solved.
