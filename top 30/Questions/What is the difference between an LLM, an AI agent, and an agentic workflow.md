# Difference between an LLM, an AI Agent, and an Agentic Workflow

These three concepts are closely related, but they represent three different levels of intelligence and system design.

- LLM: The model that understands and generates language.
- AI Agent: A system that uses an LLM to decide what action to take and can use tools to accomplish a goal.
- Agentic Workflow: An orchestrated process in which one or more AI agents, tools, and deterministic components work together to complete a multi-step task.

Let's understand this from a Senior/Staff AI Engineer perspective, including architecture, execution flow, code, and production use cases.

## 1. LLM — Large Language Model

An LLM is a neural network trained on large amounts of text and other data to understand context and generate tokens.

Examples include:

- GPT
- Claude
- Llama
- Gemini

An LLM receives input, processes it through its Transformer architecture, and generates an output.

### Architecture

User Prompt

"Explain portfolio diversification"

Tokenizer

Text → Token IDs

Transformer Model

Embedding → Attention → Feed Forward → Output logits

Token Generation

Temperature, Top-K, Top-P, sampling

Generated Response

"Diversification means spreading investments..."

### What can an LLM do?

- Answer questions.
- Summarize documents.
- Generate code.
- Translate languages.
- Extract structured information.
- Classify text.
- Generate JSON.
- Explain technical concepts.

### What can't a standalone LLM reliably do?

A basic LLM call does not inherently:

- Execute a bank transaction.
- Query your production database.
- Send an email.
- Monitor an external system continuously.
- Independently complete a multi-step business process.

It can generate instructions for these actions, but an application must provide tools and execution capabilities.

### Example: Simple LLM application

```
from openai import AsyncOpenAIclient = AsyncOpenAI()async def ask_llm(question: str) -> str:    response = await client.responses.create(        model="YOUR_MODEL",        input=question,    )    return response.output_text
```

Execution:

```
User
  |
  v
FastAPI
  |
  v
LLM API
  |
  v
Generated response
  |
  v
User
```

This is a simple LLM-powered application. There is no autonomous decision-making loop or tool orchestration here.

## 2. AI Agent

An AI agent is an application that uses an LLM as its reasoning engine and gives it the ability to interact with tools, observe results, and decide what to do next.

The important difference:

An LLM generates an answer. An agent can take actions toward achieving a goal.

For example, suppose a user asks:

> Analyze my investment portfolio, identify risky stocks, and prepare a risk report.

A basic LLM might explain how to analyze a portfolio.

An AI agent could:

1. Understand the user's goal.
2. Decide it needs portfolio data.
3. Call a portfolio API.
4. Analyze the retrieved data.
5. Decide it needs market information.
6. Call a market-data tool.
7. Evaluate the portfolio risk.
8. Generate a report.

### AI agent architecture

User Goal

Analyze my portfolio risk

LLM — Reasoning and Decision Making

What information or action is needed next?

Portfolio API

Retrieve holdings

Market API

Retrieve prices

Risk Calculator

Calculate exposure

Report Tool

Prepare report

Tool results return to the agent

LLM evaluates the results

Continue, use another tool, or finish

Final Risk Report

Notice something important: the agent doesn't necessarily know in advance every tool it will need. It can choose the next action based on the previous tool result.

### The agent execution loop

A common agent pattern is:

```
1. Receive user goal
       |
       v
2. LLM reasons about the goal
       |
       v
3. Select a tool
       |
       v
4. Execute the tool
       |
       v
5. Observe tool result
       |
       v
6. LLM reasons again
       |
       +------> Need another action?
       |                |
       |              Yes
       |                |
       |          Select next tool
       |
       No
       |
       v
7. Return final response
```

This is often called the Reason → Act → Observe loop.

### Example: Implementing a simple tool-using agent

Here is an illustrative implementation using Python and OpenAI tool calling. The application executes the tool; the LLM only requests it.

```
import jsonfrom openai import AsyncOpenAIclient = AsyncOpenAI()# -------------------------------# 1. Business tools# -------------------------------async def get_portfolio(user_id: str):    # Replace with a real authenticated service call.    return {        "user_id": user_id,        "holdings": [            {"symbol": "AAPL", "value": 50000},            {"symbol": "MSFT", "value": 30000},            {"symbol": "TSLA", "value": 20000},        ]    }async def calculate_exposure(holdings: list):    total = sum(item["value"] for item in holdings)    return {        item["symbol"]: round(            item["value"] / total * 100, 2        )        for item in holdings    }# -------------------------------# 2. Tool definitions# -------------------------------
```

The execution is:

```
User request
    |
    v
LLM decides to call get_portfolio
    |
    v
Python executes get_portfolio()
    |
    v
Portfolio data returned to LLM
    |
    v
LLM decides to call calculate_exposure
    |
    v
Python executes calculate_exposure()
    |
    v
Exposure returned to LLM
    |
    v
LLM generates final response
```

This is a simplified teaching example, not a complete production agent. Production code also needs authenticated user context, tool-level authorization, input validation, timeouts, retries, idempotency, audit logs, tracing, token budgets, and durable execution state. In particular, never trust an LLM-generated `user_id` as authorization: derive identity from the authenticated request.

### Important components of an AI agent

| Component  | Responsibility                       |
| ---------- | ------------------------------------ |
| LLM        | Reasoning and decision-making        |
| Tools      | Interact with external systems       |
| Memory     | Maintain relevant context            |
| State      | Track current execution              |
| Planner    | Decide the next steps                |
| Executor   | Execute selected actions             |
| Guardrails | Enforce safety and business rules    |
| Evaluator  | Assess whether the goal was achieved |

Not every agent needs every component. For example, a simple tool-calling agent might not need a separate planner or long-term memory.

## 3. Agentic Workflow

An agentic workflow is a larger system that coordinates multiple steps, potentially multiple agents, deterministic services, and human approvals to achieve a business objective.

This is where things become particularly important for enterprise AI engineering.

Consider a financial compliance platform.

A user uploads a financial marketing document and asks:

> Check whether this document violates financial compliance policies and prepare a compliance report.

You may need several different responsibilities:

1. Document ingestion.
2. Document parsing.
3. Policy retrieval using RAG.
4. Compliance analysis.
5. Risk classification.
6. Independent evaluation.
7. Human approval.
8. Final report generation.
9. Audit logging.

One LLM call cannot reliably own this entire process.

### Agentic workflow architecture

User uploads financial document

Document Ingestion Service

Parse → Validate → Chunk → Embed

Supervisor / Orchestrator

Maintains state and coordinates execution

Retrieval Agent

Find applicable policies

Compliance Agent

Analyze claims

Risk Agent

Classify violations

Evaluator

Check quality and evidence

Decision / Conditional Routing

Pass? Retry? Escalate?

Human-in-the-Loop Approval

Required for designated high-risk decisions

Final Compliance Report + Audit Trail

The main difference from a standalone agent is that an agentic workflow has an explicit execution architecture. It defines:

- Which component executes first.
- Which components can execute in parallel.
- Which decisions control routing.
- Where retries happen.
- Where human approval is required.
- How execution resumes after a failure.
- How the final output is validated.

An agentic workflow does not necessarily require every node to be an LLM agent. In fact, many production nodes should be deterministic Python services.

For example:

| Workflow step        | Implementation           |
| -------------------- | ------------------------ |
| PDF parsing          | Deterministic parser     |
| Chunking             | Python service           |
| Embeddings           | Embedding model          |
| Policy retrieval     | Vector search + reranker |
| Compliance reasoning | LLM-powered agent        |
| Risk scoring         | Rules + model            |
| Schema validation    | Pydantic                 |
| Approval             | Human                    |
| Audit logging        | PostgreSQL               |

This combination is what makes enterprise systems reliable.

## 4. Direct comparison

| Dimension       | LLM                          | AI Agent                       | Agentic Workflow                         |
| --------------- | ---------------------------- | ------------------------------ | ---------------------------------------- |
| Core purpose    | Generate responses           | Achieve a goal through actions | Coordinate a complex task                |
| Reasoning       | Model inference              | LLM-driven decisions           | Distributed across agents and components |
| Tool usage      | Not inherent                 | Yes                            | Yes                                      |
| Multiple steps  | Can generate multi-step text | Executes iterative actions     | Orchestrates explicit steps              |
| Memory/state    | Context window               | Optional memory/state          | Managed workflow state                   |
| Multiple agents | No inherent concept          | Possible                       | Common                                   |
| Human approval  | Application-level            | Possible                       | Can be a formal workflow step            |
| Recovery        | Retry the request            | Resume if state is persisted   | Checkpoint and resume workflow           |
| Example         | Answer a question            | Retrieve portfolio data        | Complete compliance review               |

## 5. Same problem solved in three different ways

Let's use one example:&#x20;

"Find the top five risky investments in my portfolio and prepare a report."

Level 1 — LLM

Input: Analyze my portfolio risk.

Execution: LLM generates a textual explanation based on the prompt and available context.

Output: A general explanation of portfolio risk, not necessarily based on real holdings.

Level 2 — AI Agent

Input: Analyze my portfolio risk.

Execution: Agent decides to call portfolio API → retrieves holdings → calls market API → calculates exposure → generates report.

Output: A report based on retrieved portfolio data.

Level 3 — Agentic Workflow

Input: Analyze my portfolio risk.

Execution: Workflow validates the request → retrieves portfolio → runs risk analysis → performs independent evaluation → routes high-risk cases for approval → generates report → stores audit records.

Output: A controlled, traceable business process with validation and recovery.

## 6. How do we implement an agentic workflow in production?

For enterprise systems, a framework such as LangGraph can help implement the workflow.

Its important capabilities include:

- State management.
- Conditional routing.
- Parallel execution.
- Cycles and evaluator loops.
- Checkpointing.
- Human-in-the-loop interruptions.
- Workflow resumption.

A simplified LangGraph design:

```
from typing import TypedDictfrom langgraph.graph import StateGraph, START, ENDclass ComplianceState(TypedDict):    document: str    policies: list    findings: list    risk_level: str    approved: bool    report: str# 1. Retrieve relevant policiesasync def retrieve_policies(state):    policies = await policy_retriever.search(        state["document"]    )    return {"policies": policies}# 2. Analyze complianceasync def analyze_compliance(state):    findings = await compliance_agent.analyze(        document=state["document"],        policies=state["policies"]    )    return {"findings": findings}# 3. Classify riskasync def classify_risk(state):    risk = await risk_service.classify(        state["findings"]
```

This illustrates graph orchestration, but the human review node above is not a real approval pause. A production implementation needs a durable checkpointer, an interrupt before approval, an authenticated approval API, and a resume operation. Approval must also be verified against the correct document, reviewer permissions, and workflow version.

### How the production architecture fits together

FastAPI — API Layer

Application Service

Authentication, validation, idempotency

LangGraph Orchestration

Agents, tools, routing, checkpoints, HITL

PostgreSQL

Business data, audit, workflow metadata

Qdrant

Vector retrieval

Redis

Cache, rate limits

Celery / Workers

Background execution

Observability

OpenTelemetry, logs, metrics, traces

For long-running workflows, the API should normally create a workflow execution and return its ID rather than waiting for the entire process. Workers execute the workflow, and clients can retrieve status or submit an approval. Durable checkpoints and idempotent external actions are essential for recovery.

## 7. When should you use each?

| Requirement                                | Recommended approach |
| ------------------------------------------ | -------------------- |
| Chatbot answering general questions        | LLM                  |
| Document summarization                     | LLM                  |
| Structured information extraction          | LLM + validation     |
| Search database and answer questions       | LLM + RAG            |
| Decide which API to call                   | AI Agent             |
| Research across multiple external sources  | AI Agent             |
| Execute a sequence of business operations  | Agentic Workflow     |
| Financial compliance review with approvals | Agentic Workflow     |
| Multi-agent financial advisor copilot      | Agentic Workflow     |

## 8. Senior AI Engineer interview questions

These are important questions to prepare:

1. Can an AI agent exist without an LLM?
2. Is every tool-calling LLM an AI agent?
3. What is the difference between a workflow and an agentic workflow?
4. When would you prefer deterministic workflows over autonomous agents?
5. How do you prevent an agent from executing an infinite loop?
6. How do you persist agent state?
7. How do you resume an agent after a worker crashes?
8. How do you implement human-in-the-loop approval?
9. How do you prevent multiple agents from executing conflicting actions?
10. How do you evaluate whether an agent has successfully completed its task?

## 9. My recommendation for your Senior/Staff AI Engineer preparation

Understand these concepts as three layers:

Layer 1: Intelligence

LLMs, Transformers, tokens, prompting, sampling, structured outputs

Layer 2: Action

Agents, tool calling, planning, memory, reasoning, execution loops

Layer 3: Orchestration

LangGraph, multi-agent systems, state management, retries, HITL, checkpointing, observability

For your enterprise financial advisor copilot and compliance platform, the strongest architecture is generally LLM-powered agents inside a controlled agentic workflow, not one autonomous agent responsible for everything.

The Staff Engineer distinction is this: you don't just know how to make an LLM call a tool. You know how to build a system that executes safely, handles failures, resumes work, controls costs, and produces auditable results.