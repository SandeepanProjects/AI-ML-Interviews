# Defining agent contracts and limiting tool access in LangGraph

At senior/staff AI engineer level, I treat agent contracts as enforceable interfaces and tool access as a security boundary.

In a production multi-agent system, an agent should not be allowed to decide for itself which data it can access, which tools it can invoke, or which business actions it can authorize.

Consider your AI-powered financial compliance platform:

- A planner decides which checks are required.
- A policy retrieval agent searches authorized compliance documents.
- A performance agent evaluates return-related claims.
- A risk agent evaluates risk disclosures.
- An evaluator checks whether the findings are adequately supported.
- A supervisor coordinates the workflow.
- A human reviewer approves or rejects the final result when required.

If the risk agent can invoke an unrestricted database query, read another tenant's documents, or publish an approved compliance report, the architecture has a security problem—even if the LLM follows its instructions correctly 99% of the time.

The design goal is:

1. Define exactly what each agent receives.
2. Define exactly what it must return.
3. Define which tools it may invoke.
4. Validate inputs, outputs, and tool arguments.
5. Enforce authorization in trusted code—not in the prompt.
6. Keep sensitive side effects and final approval under explicit control.

## 1. What is an agent contract?

An agent contract specifies the agent's permitted behavior at the interface level.

Input contract

What data the agent receives, required fields, size limits, tenant scope, and evidence it is allowed to use.

Output contract

The required schema, allowed values, evidence references, error states, and rules for representing uncertainty.

Capability contract

The exact tools the agent may call, their allowed arguments, call limits, and authorization requirements.

Invariant contract

Rules that must never be violated: no cross-tenant access, no fabricated policy citations, no unauthorized approval, and no unapproved external side effects.

A system prompt is useful for communicating the contract to the model, but it is not sufficient to enforce it. The application must validate the contract at runtime.

## 2. Define contracts with Pydantic

First, define what the performance agent is allowed to receive and return.

```
from typing import Literalfrom pydantic import BaseModel, ConfigDict, Fieldclass PerformanceTask(BaseModel):    model_config = ConfigDict(extra="forbid", frozen=True)    task_id: str    claim_text: str = Field(min_length=1, max_length=3000)    evidence_ids: tuple[str, ...] = Field(max_length=30)class PerformanceFinding(BaseModel):    model_config = ConfigDict(extra="forbid", frozen=True)    task_id: str    status: Literal["supported", "unsupported", "uncertain"]    severity: Literal["low", "medium", "high", "critical"]    rationale: str = Field(min_length=1, max_length=3000)    evidence_ids: tuple[str, ...] = Field(max_length=20)
```

These models establish a structural contract:

- The agent receives a specific task rather than the entire application state.
- It cannot introduce arbitrary fields such as `tenant_id`, `is_admin`, or `approved`.
- It must express uncertainty explicitly.
- Its findings must identify evidence that can be validated downstream.

`frozen=True` prevents ordinary field reassignment, while `extra="forbid"` rejects unexpected fields during validation. Neither setting proves the content is factually correct or makes every nested Python object immutable.

### Validate the semantic contract too

After receiving a finding, verify that its evidence IDs belong to the evidence supplied for that task.

```
def validate_finding(    task: PerformanceTask,    finding: PerformanceFinding,) -> PerformanceFinding:    if finding.task_id != task.task_id:        raise ValueError("Finding belongs to another task")    allowed_ids = set(task.evidence_ids)    if not set(finding.evidence_ids).issubset(allowed_ids):        raise ValueError("Finding references unauthorized evidence")    return finding
```

This is a critical distinction: a valid schema is not proof of a valid conclusion. Your evaluator still needs to verify policy applicability, citation accuracy, and whether the evidence supports the claim.

## 3. Limit tools by giving each agent a separate capability set

The most straightforward control is to construct a different tool set for each specialist.

Performance agent

Search performance policies

Read authorized evidence

Risk agent

Search risk policies

Read authorized evidence

Supervisor

Dispatch specialist tasks

Read validated findings

Human approval service

Approve or reject review

This capability is not exposed to the specialist LLMs.

In LangGraph, `ToolNode` executes the tools you register with it, while `tools_condition` can route a model response to the tool-execution node when tool calls are present.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://reference.langchain.com\&sz=32)

LangChain Reference

+1



The important rule is: do not give every agent the same unrestricted tool registry and expect its prompt to enforce access control.

## 4. Implement scoped tools in Python

Let's build a policy-search tool. The LLM can provide a search query, but it cannot choose the tenant or invent its own authorization scope.

The example below uses an application service called `policy_service`. You would implement that service against your PostgreSQL/pgvector repository, including document-level ACL enforcement.

```
from typing import Callablefrom pydantic import BaseModel, ConfigDict, Fieldfrom langchain_core.tools import toolclass PolicySearchArgs(BaseModel):    model_config = ConfigDict(extra="forbid")    query: str = Field(min_length=3, max_length=500)    limit: int = Field(default=5, ge=1, le=10)def build_policy_search_tool(    *,    principal_id: str,    tenant_id: str,    policy_service,):    # These values come from trusted server-side authentication    # and authorization, never from the LLM's arguments.    @tool(args_schema=PolicySearchArgs)    def search_authorized_policies(query: str, limit: int = 5) -> list[dict]:        """Search policies this authenticated workflow is permitted to access."""        return policy_service.search(            principal_id=principal_id,            tenant_id=tenant_id,            query=query,            limit=limit,        )    return search_authorized_policies
```

Here, `principal_id` and `tenant_id` are captured by the tool implementation. They are not exposed as model-controlled arguments.

The service must still enforce authorization at the database/repository layer. A closure is not a substitute for database ACLs, row-level security, or service authentication.

### Give each specialist its own tools

```
def build_agent_tools(    *,    agent_role: str,    principal_id: str,    tenant_id: str,    policy_service,):    policy_search = build_policy_search_tool(        principal_id=principal_id,        tenant_id=tenant_id,        policy_service=policy_service,    )    if agent_role == "performance":        return [policy_search]    if agent_role == "risk":        return [policy_search]    if agent_role == "supervisor":        # Supervisor receives no arbitrary database tool.        return []    raise ValueError(f"Unknown agent role: {agent_role}")
```

This is an allowlist. The performance agent cannot call a fee-update tool that was never supplied to it.

For a real application, I would use separate, purpose-specific tools such as `search_performance_policies` and `search_risk_policies`, each with server-enforced filters. This improves auditability and makes it harder for an agent to search unrelated data accidentally.

## 5. Connect the allowlisted tools to LangGraph

For a custom graph, bind only the allowed tools to the model and register that same tool set with its `ToolNode`.

```
from typing import Annotatedfrom typing_extensions import TypedDictfrom langchain_core.messages import AnyMessagefrom langgraph.graph import StateGraph, START, ENDfrom langgraph.graph.message import add_messagesfrom langgraph.prebuilt import ToolNode, tools_conditionclass AgentState(TypedDict):    messages: Annotated[list[AnyMessage], add_messages]def build_specialist_graph(model, tools):    # The model sees only these tools.    model_with_tools = model.bind_tools(tools)    def call_model(state: AgentState):        response = model_with_tools.invoke(state["messages"])        return {"messages": [response]}    builder = StateGraph(AgentState)    builder.add_node("agent", call_model)    if tools:        builder.add_node("tools", ToolNode(tools))        builder.add_edge(START, "agent")        builder.add_conditional_edges(            "agent",            tools_condition,            {"tools": "tools", END: END},        )        builder.add_edge("tools", "agent")    else:        builder.add_edge(START, "agent")
```

Example setup:

```
performance_tools = build_agent_tools(    agent_role="performance",    principal_id=authenticated_user.id,    tenant_id=authenticated_user.tenant_id,    policy_service=policy_service,)performance_graph = build_specialist_graph(    model=chat_model,    tools=performance_tools,)
```

`authenticated_user` here represents a server-verified identity object, not fields copied from the request body.

This graph demonstrates capability separation, but it is not a complete authorization framework. In a production system, the authenticated identity should be bound to the workflow invocation by trusted application code, and tool execution must revalidate access to each requested resource.

## 6. Add tool-level authorization, not just tool allowlists

There is a subtle but important security distinction:

- Tool allowlist: the risk agent has access to `search_risk_policies`.
- Argument validation: the search query and result limit must meet defined constraints.
- Resource authorization: the service verifies that the authenticated principal may access the specific documents returned.
- Action authorization: the service verifies that the requested operation is permitted for that principal and workflow.

All four matter. A tool can be on an agent's allowlist and still be insecure if it trusts a model-supplied document ID without checking access.

For a financial application, I would enforce the following:

```
def get_authorized_document(    *,    principal_id: str,    tenant_id: str,    document_id: str,    repository,):    document = repository.get_document(        document_id=document_id,        tenant_id=tenant_id,    )    if document is None:        # Avoid revealing whether another tenant's document exists.        raise PermissionError("Document is unavailable")    if not repository.can_read_document(        principal_id=principal_id,        document_id=document_id,    ):        raise PermissionError("Document is unavailable")    return document
```

The repository methods here are application-specific. In a real implementation, enforce tenant scope and ACLs in the database query itself wherever possible, and consider PostgreSQL row-level security as an additional layer.

Do not trust an LLM-generated `principal_id`, `tenant_id`, role, or permission list. Derive identity from verified authentication and authorization data. This aligns with established agent security guidance distinguishing authentication from authorization.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.langchain.com\&sz=32)

langchain.com

+1



## 7. Put limits on tool calls and expensive operations

Even an authorized tool can be abused through excessive repetition.

For agents built with LangChain's `create_agent`, `ToolCallLimitMiddleware` provides global and tool-specific call limits, including limits across a thread when a checkpointer is configured.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



For example:

```
from langchain.agents import create_agentfrom langchain.agents.middleware import ToolCallLimitMiddlewareagent = create_agent(    model=chat_model,    tools=performance_tools,    middleware=[        ToolCallLimitMiddleware(            run_limit=8,        ),        ToolCallLimitMiddleware(            tool_name="search_authorized_policies",            run_limit=4,        ),    ],)
```

Check the middleware's behavior and API against the installed LangChain version before adopting this exact snippet. This example applies to `create_agent`, not automatically to every custom `StateGraph` you build.

For custom LangGraph workflows, enforce budgets in the tool executor or a common wrapper, and use graph recursion limits as an additional safety net. A model-call limit, tool-call limit, and graph superstep limit are different controls.

## 8. Require human approval for sensitive actions

Read-only retrieval and report drafting can often run autonomously. Actions such as publishing a compliance decision, changing a policy, deleting a document, or sending an external communication may need human approval.

A safer workflow looks like this:

Agent proposes action

Validate schema and permissions

Check action risk

Allowed read-only action

Approval required

Approval must be checked and enforced by trusted application code before executing the side effect.

LangGraph supports interrupt-based human-in-the-loop workflows. When using an interrupt, persist the workflow using a checkpointer and revalidate the action and permissions when the graph resumes.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



Do not assume that a tool is safe merely because its metadata labels it as read-only or destructive. Treat tool metadata as useful classification, not as a substitute for server-side authorization.

## 9. Production checklist

For a staff-level implementation, I would verify these controls before deployment:

Agent security review

0/10

Each agent has a documented input and output contract.

Unexpected output fields and invalid enum values are rejected.

Each agent receives only the tools it needs.

Tool arguments are validated before execution.

Tenant and document ACLs are enforced server-side.

Tool calls, retries, latency, and token costs are bounded.

Destructive actions require the appropriate approval.

Every tool invocation records actor, tool, resource, decision, and outcome.

Retries and checkpoint resumes cannot duplicate unsafe side effects.

Negative tests verify cross-tenant access and unauthorized tool calls are rejected.

## 10. Interview-ready answer

> “I define agent contracts using typed input and output schemas, explicit invariants, and capability boundaries. For tool security, I use per-agent allowlists and expose only the tools required for that role. Each tool validates its arguments and enforces authorization in the underlying service, using trusted identity rather than model-supplied tenant or user IDs. I add call limits, timeouts, audit logging, and human approval for sensitive actions. Finally, I test both invalid model outputs and adversarial tool requests. The key principle is that prompts guide agent behavior, but trusted code enforces permissions.”

The staff-level design principle is simple: an agent may propose an action, but only the trusted application layer decides whether that action is permitted.