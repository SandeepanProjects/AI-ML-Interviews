# Explain Planner-Executor, Supervisor, and ReAct architectures

This is a very important Senior AI Engineer interview question, especially when working with LangGraph, multi-agent systems, and enterprise agentic AI applications.

All three architectures describe different ways of organizing an AI agent's reasoning, task execution, and tool usage.

The key difference is:

- Planner-Executor: One component plans the work, and another executes the plan.
- Supervisor: A central agent manages and coordinates multiple specialist agents.
- ReAct: An agent repeatedly reasons, takes an action, observes the result, and decides what to do next.

Let's understand them from a production engineering perspective.

# 1. Planner-Executor architecture

## What is Planner-Executor?

Planner-Executor is an agentic architecture in which we separate task planning from task execution.

Instead of asking one LLM to solve a complex problem in one go, we divide the work into two responsibilities.

1. Planner: Understands the user request and creates a sequence of tasks.
2. Executor: Executes those tasks using available tools, APIs, databases, or other agents.

[How Enterprise AI Agents Work - Cognitive Architecture Explained | Sema4.ai](https://images.openai.com/static-rsc-4/Hu-ST6CK9l6UiTdmhFfikm4ou20AY2nMiUzRkO6VEHJ8Yz4jum5DUC_bcpJ7H77JAbmUuq8OTc5lXLSMwzEOZbg9o_pcTn8Y6kyStTfTGIATV3LO_6N9ZZlWlIAHzamhMrrkGSR3gVRDNSZfolaRyl3DtTrGNZdDnCwmEFGrE40?purpose=inline)

[medium.com](https://medium.com/sema4-ai/whats-inside-an-enterprise-ai-agent-1394fad33c1e?utm_source=chatgpt.com)

### Example: Financial Advisor Copilot

Suppose a user asks:

"What is the best investment strategy for me considering my portfolio, market conditions, and risk profile?"

The system needs to perform several operations.

Step 1: Planner

The planner analyzes the request and generates a plan.

```
plan = [    {        "task_id": 1,        "task": "Analyze the user's portfolio"    },    {        "task_id": 2,        "task": "Analyze current market conditions"    },    {        "task_id": 3,        "task": "Evaluate the user's risk profile"    },    {        "task_id": 4,        "task": "Generate investment recommendations"    }]
```

Step 2: Executor

The executor processes the tasks.

| Task               | Executor responsibility                          |
| ------------------ | ------------------------------------------------ |
| Portfolio analysis | Query portfolio database                         |
| Market analysis    | Call market data API                             |
| Risk analysis      | Execute risk assessment                          |
| Recommendations    | Generate recommendations using collected results |

Step 3: Aggregation

The system collects all the results and produces a final response.

### Architecture flow

User Request

Planner Agent

Breaks the problem into tasks

Execution Plan

Task 1 → Task 2 → Task 3 → Task 4

Executor

Runs tasks using tools and services

Results Aggregator

Final Response

## How do we implement Planner-Executor using LangGraph?

In LangGraph, we can represent the planner and executor as separate nodes.

- `planner_node`: Generates the task plan.
- `executor_node`: Executes individual tasks.
- `aggregator_node`: Combines execution results.

Here is a simplified but functional implementation.

### Project structure

```
planner_executor/
│
├── main.py
├── state.py
├── nodes.py
├── graph.py
└── requirements.txt
```

### state.py

The state maintains information throughout the workflow.

```
from typing import TypedDict, Anyclass AgentState(TypedDict):    user_query: str    plan: list[dict[str, Any]]    current_task: int    results: list[dict[str, Any]]    final_answer: str
```

### nodes.py

```
from langchain_openai import ChatOpenAIfrom langchain_core.prompts import ChatPromptTemplatefrom state import AgentStatellm = ChatOpenAI(    model="gpt-4o-mini",    temperature=0)async def planner_node(state: AgentState):    prompt = ChatPromptTemplate.from_messages([        (            "system",            """            Break the user request into independent,            actionable tasks.            Return a numbered execution plan.            """        ),        ("human", "{query}")    ])    chain = prompt | llm    response = await chain.ainvoke({        "query": state["user_query"]    })    # In production, parse a structured Pydantic    # response instead of parsing free-form text.    tasks = [        {"task": line.strip()}
```

### graph.py

```
from langgraph.graph import (    StateGraph,    START,    END)from state import AgentStatefrom nodes import (    planner_node,    executor_node,    aggregator_node)def should_continue(state: AgentState):    if state["current_task"] < len(state["plan"]):        return "execute"    return "aggregate"builder = StateGraph(AgentState)builder.add_node("planner", planner_node)builder.add_node("executor", executor_node)builder.add_node("aggregator", aggregator_node)builder.add_edge(START, "planner")builder.add_edge("planner", "executor")builder.add_conditional_edges(    "executor",    should_continue,
```

### main.py

```
import asynciofrom graph import graphasync def main():    result = await graph.ainvoke({        "user_query": (            "Analyze my investment portfolio "            "and suggest an investment strategy."        )    })    print(result["final_answer"])if __name__ == "__main__":    asyncio.run(main())
```

This demonstrates the basic Planner-Executor pattern. The executor above is an LLM-based placeholder, not a production financial tool implementation.

Production considerations:

- Validate plans using Pydantic schemas.
- Add task dependencies and execution limits.
- Run independent tasks concurrently.
- Persist execution state with a checkpoint backend.
- Make external operations idempotent.
- Add retry policies and timeouts.
- Validate results before aggregation.

For example, portfolio analysis and market analysis can run in parallel, while recommendations must wait for both.

# 2. Supervisor architecture

## What is Supervisor architecture?

Supervisor architecture is a multi-agent design pattern where one central agent coordinates multiple specialized agents.

The supervisor does not necessarily perform all the work itself.

Instead, it:

1. Understands the user's request.
2. Identifies which specialist agents are required.
3. Delegates tasks to those agents.
4. Collects their results.
5. Evaluates whether the results are sufficient.
6. Decides whether to call another agent or generate the final response.

[LangGraph’s create_supervisor: The Multi-Agent Coordinator You Need to Understand](https://images.openai.com/static-rsc-4/S95o2vVu93Rqc6HWx0nRKsUpP_CmHD3_1XhbbPWBFXZADHPr9_MZtdDsKxzUkQSG0cAgl0d0UgVaC88HC-R-YJ90uRdKf8lW5U_NK39dzr_nSa6ijC2EY1DrBj_Si0BllTtn9HUDcuKeqreFYPl0MOIgw8qQ8Eyn4A_SN20myIQ?purpose=inline)

[linkedin.com](https://www.linkedin.com/pulse/langgraphs-createsupervisor-multi-agent-coordinator-you-kumar-v-eu4zc?utm_source=chatgpt.com)

### Example: Enterprise Financial Advisor Copilot

Suppose a user asks:

"Analyze my investment portfolio and tell me whether I should rebalance it."

We have four specialist agents:

| Agent                 | Responsibility                        |
| --------------------- | ------------------------------------- |
| Portfolio Agent       | Analyzes current portfolio allocation |
| Market Research Agent | Analyzes market conditions            |
| Risk Agent            | Evaluates portfolio risk              |
| Recommendation Agent  | Generates recommendations             |

The Supervisor coordinates all four.

### Execution flow

User Request

Supervisor Agent

Decides which specialist to invoke

Portfolio Agent

Allocation

Market Agent

Market data

Risk Agent

Risk analysis

Recommendation Agent

Recommendations

Results returned to Supervisor

Final Response

Notice that the Supervisor controls the workflow.

For example:

- If the request is only about portfolio allocation, it may call the Portfolio Agent.
- If the user asks about investment risk, it may call the Risk Agent.
- If the user asks for a complete portfolio review, it may call multiple agents.
- If an agent returns incomplete results, it may invoke that agent again or request another specialist's analysis.

## How to implement Supervisor using LangGraph

Let's build a simplified implementation.

### Project structure

```
supervisor_architecture/
│
├── main.py
├── state.py
├── agents.py
├── supervisor.py
└── graph.py
```

### state.py

```
from typing import TypedDictclass AgentState(TypedDict):    user_query: str    next_agent: str    portfolio_result: str    market_result: str    risk_result: str    final_answer: str
```

### agents.py

```
from langchain_openai import ChatOpenAIllm = ChatOpenAI(    model="gpt-4o-mini",    temperature=0)async def portfolio_agent(state):    response = await llm.ainvoke(        f"""        Analyze the portfolio.        User request:        {state['user_query']}        """    )    return {        "portfolio_result": response.content    }async def market_agent(state):    response = await llm.ainvoke(        f"""        Analyze market conditions.        User request:        {state['user_query']}        """    )    return {
```

These are illustrative specialist agents. In a real financial application, each agent would use authorized data sources and validated domain services rather than relying on an LLM to invent portfolio or market data.

### supervisor.py

The Supervisor decides which agent should execute next.

```
from typing import Literalfrom pydantic import BaseModelfrom langchain_openai import ChatOpenAIfrom state import AgentStateclass RoutingDecision(BaseModel):    next_agent: Literal[        "portfolio",        "market",        "risk",        "finish"    ]llm = ChatOpenAI(    model="gpt-4o-mini",    temperature=0)router = llm.with_structured_output(    RoutingDecision)async def supervisor_node(state: AgentState):    prompt = f"""    You are the supervisor of a financial    advisor multi-agent system.    User query:    {state['user_query']}
```

### graph.py

```
from langgraph.graph import (    StateGraph,    START,    END)from state import AgentStatefrom agents import (    portfolio_agent,    market_agent,    risk_agent)from supervisor import supervisor_nodedef route_agent(state):    return state["next_agent"]builder = StateGraph(AgentState)builder.add_node("supervisor", supervisor_node)builder.add_node("portfolio", portfolio_agent)builder.add_node("market", market_agent)builder.add_node("risk", risk_agent)builder.add_edge(START, "supervisor")builder.add_conditional_edges(    "supervisor",
```

The execution cycle is:

```
START
  |
  v
SUPERVISOR
  |
  +----> PORTFOLIO AGENT
  |              |
  |              v
  |          SUPERVISOR
  |
  +----> MARKET AGENT
  |              |
  |              v
  |          SUPERVISOR
  |
  +----> RISK AGENT
  |              |
  |              v
  |          SUPERVISOR
  |
  +----> FINISH
```

For this example, the Supervisor invokes one specialist at a time. A production implementation can use parallel execution for independent specialists through LangGraph's fan-out and fan-in patterns.

Production considerations:

- Add a maximum number of supervisor iterations.
- Track completed agents to prevent duplicate calls.
- Validate agent outputs against schemas.
- Add retries and timeouts.
- Enforce permissions at the tool and service layers.
- Persist state with durable checkpointing.
- Use deterministic aggregation for financial calculations.
- Add HITL before high-impact operations.

One important point: the Supervisor should not have unrestricted access to every tool. Each specialist should receive only the permissions it needs.

# 3. ReAct architecture

## What is ReAct?

ReAct stands for Reason + Act.

It is an agent architecture where an LLM alternates between reasoning about a problem, taking an action, observing the result, and deciding what to do next.

Unlike Planner-Executor, ReAct does not necessarily generate the complete execution plan upfront.

It determines its next action based on the observation from the previous action.

[ReAct vs. naive prompt chaining on Llama Stack | Red Hat Developer](https://images.openai.com/static-rsc-4/igSW6pQMPa45IRtNm8oT3JycvPSAitUmk1Ywi9yoKSm-6rhmvxDeRDjR_FWNjZOXIpdtSuhFki-2ZBAx78ox5OvRiPqkQb_bcyrgjDXzckzm4bO9Twp-pjVW7iLpTUcecMuhmGXW9k_iZIuO-z5iNgABflbVr1JUL3wvKnmyZi0?purpose=inline)

[developers.redhat.com](https://developers.redhat.com/articles/2025/07/22/react-vs-naive-prompt-chaining-llama-stack?utm_source=chatgpt.com)

The fundamental loop is:

Reason

What should I do next?

Act

Call a tool or API

Observe

Read the tool result

Repeat until the task is complete

### Example: Financial research agent

User asks:

"Find the latest stock price of Company ABC and calculate its percentage change from yesterday."

The agent might execute the following sequence:

Iteration 1: Reason

I need the latest stock price.

Iteration 2: Act

Call the stock market API.

```
latest_price = get_latest_stock_price(    symbol="ABC")
```

Iteration 3: Observe

```
{
    "symbol": "ABC",
    "latest_price": 150,
    "previous_close": 145
}
```

Iteration 4: Reason

I now have both prices. I need to calculate the percentage change.

Iteration 5: Act

```
percentage_change = (    (150 - 145) / 145) * 100
```

Iteration 6: Observe

```
Percentage change = 3.45%
```

Iteration 7: Final response

"Company ABC is trading at ₹150, compared with the previous close of ₹145. This represents an increase of approximately 3.45%."

The important feature is that the next action is determined using the latest observation.

## How to implement ReAct using LangGraph

LangGraph provides an agent loop that can be implemented using an LLM node, tool node, and conditional routing.

### Project structure

```
react_agent/
│
├── main.py
├── tools.py
└── agent.py
```

### tools.py

```
from langchain_core.tools import tool@tooldef calculate_percentage_change(    current_price: float,    previous_price: float) -> float:    """Calculate percentage change between prices."""    if previous_price <= 0:        raise ValueError("Previous price must be positive")    return (        (current_price - previous_price)        / previous_price    ) * 100@tooldef get_stock_price(symbol: str) -> dict:    """    Get current and previous closing prices.    Demonstration implementation.    Replace with an authenticated market data API.    """    return {        "symbol": symbol,        "current_price": 150,        "previous_close": 145    }
```

### agent.py

```
from langchain_openai import ChatOpenAIfrom langgraph.prebuilt import create_react_agentfrom tools import (    get_stock_price,    calculate_percentage_change)llm = ChatOpenAI(    model="gpt-4o-mini",    temperature=0)tools = [    get_stock_price,    calculate_percentage_change]agent = create_react_agent(    model=llm,    tools=tools)
```

### main.py

```
import asynciofrom agent import agentasync def main():    result = await agent.ainvoke({        "messages": [            {                "role": "user",                "content": (                    "Find the latest stock price of ABC "                    "and calculate its percentage change."                )            }        ]    })    print(result["messages"][-1].content)if __name__ == "__main__":    asyncio.run(main())
```

This is a minimal ReAct implementation using LangGraph's prebuilt agent. The stock-price tool contains demonstration data, not live market data.

Production considerations:

- Restrict available tools.
- Validate all tool arguments.
- Set maximum iterations.
- Apply per-tool timeouts.
- Handle API failures and rate limits.
- Add idempotency for side-effecting tools.
- Log tool calls and observations.
- Require human approval for financial transactions or other high-impact actions.

# 4. Comparison: Planner-Executor vs Supervisor vs ReAct

This is one of the most important tables to remember for interviews.

| Feature         | Planner-Executor      | Supervisor                     | ReAct                       |
| --------------- | --------------------- | ------------------------------ | --------------------------- |
| Main purpose    | Break task into steps | Coordinate specialist agents   | Reason and act iteratively  |
| Planning        | Usually upfront       | Supervisor decides dynamically | Step-by-step                |
| Execution       | Executor              | Multiple specialist agents     | Agent invokes tools         |
| Control         | Execution plan        | Central supervisor             | Agent reasoning/action loop |
| Multiple agents | Optional              | Core design pattern            | Not required                |
| Feedback        | Execution results     | Specialist results             | Tool observations           |
| Best suited for | Complex workflows     | Enterprise multi-agent systems | Dynamic tool-based tasks    |

## 5. When should you use which architecture?

Let's consider three real-world scenarios.

Scenario 1: Document processing

You need to extract information from 100 documents, classify them, validate them, and generate a report.

Use Planner-Executor

Why? The task can be divided into clear execution steps, and independent tasks can run concurrently.

Scenario 2: Financial Advisor Copilot

You need portfolio analysis, market research, risk evaluation, and recommendations.

Use Supervisor

Why? Different specialist agents have different responsibilities, and one component must coordinate them.

Scenario 3: Research assistant

A user asks for the latest company information. The agent must search, inspect results, follow links, and decide whether more research is required.

Use ReAct

Why? The next action depends on the observation from the previous action.

# 6. Can we combine all three architectures?

Absolutely. In fact, this is a common approach for sophisticated enterprise AI systems.

For your Enterprise Multi-Agent Financial Advisor Copilot, I would consider combining them.

[Beyond Simple Chatbots: The AI Agent Patterns That Power Real Business Workflows(Part 1) - FlowGenX AI](https://images.openai.com/static-rsc-4/mLHgi_kjQNTtf6Dxa7FigKDvSs67GofsUahKkwtVUH8wzStH8u0780dmu17KXoc_eNYX62SIAcaYUkxvZ6Z93nQ6z3ttFVabuXUKAfbBS4CGTP6bd6R8SKwJax_YODNclU0tfJJzAcwoady5WctBeQtLO85kQjr1WnlEva4MlSk?purpose=inline)

[blog.flowgenx.ai](https://blog.flowgenx.ai/beyond-simple-chatbots-the-ai-agent-patterns-that-power-real-business-workflows-part-1/?utm_source=chatgpt.com)

### Hybrid architecture

```
                 USER REQUEST
                      |
                      v
                PLANNER AGENT
                      |
              Creates execution plan
                      |
                      v
              SUPERVISOR AGENT
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
      PORTFOLIO     RESEARCH      RISK
       AGENT         AGENT       AGENT
          |           |           |
          v           v           v
        ReAct       ReAct       ReAct
        Loop        Loop        Loop
          |           |           |
          +-----------+-----------+
                      |
                      v
               RESULT EVALUATOR
                      |
               Is result sufficient?
                  /       \
                No         Yes
                |           |
                v           v
            Rework       AGGREGATOR
                            |
                            v
                       HITL APPROVAL
                            |
                            v
                       FINAL RESPONSE
```

Here is how the responsibilities are divided:

| Component    | Responsibility                        |
| ------------ | ------------------------------------- |
| Planner      | Creates the initial execution plan    |
| Supervisor   | Coordinates specialist agents         |
| ReAct agents | Perform dynamic tool-based operations |
| Evaluator    | Checks completeness and quality       |
| HITL         | Obtains human approval when required  |
| Aggregator   | Combines validated results            |

This is more flexible than using one architecture for everything.

However, combining all three also increases complexity, latency, cost, and failure modes. I would only introduce each pattern when its responsibilities justify it.

# 7. Important senior-level interview follow-up questions

Prepare these as well:

1. How do you prevent infinite loops in Supervisor architecture?
2. How do you maintain shared state across multiple agents?
3. How do you execute independent agents in parallel?
4. How do you handle agent failures and retries?
5. How do you implement checkpointing and workflow recovery?
6. How do you prevent two agents from executing the same task?
7. How do you control tool permissions and prevent unauthorized execution?
8. How do you evaluate the output of individual agents?
9. When should you avoid using multi-agent architecture?
10. How do you make a multi-agent workflow observable?

## 8. My recommended interview answer

If an interviewer asks you to explain these three architectures, you can answer:

> Planner-Executor, Supervisor, and ReAct are three different agent orchestration patterns.
>
> In Planner-Executor, we separate planning and execution. The planner breaks a complex request into smaller tasks, and the executor performs those tasks. I would use this when the workflow has multiple well-defined steps.
>
> In Supervisor architecture, we have a central orchestrator that delegates tasks to specialized agents, collects their results, and decides what should happen next. I would use this for enterprise multi-agent applications where different agents have different responsibilities.
>
> In ReAct architecture, the agent follows a reasoning, action, and observation loop. It invokes tools, examines the results, and dynamically determines its next action. I would use this for research assistants and dynamic tool-based workflows.
>
> In production, I may combine these patterns. For example, I can use Planner-Executor for task decomposition, Supervisor for specialist coordination, and ReAct inside individual agents for dynamic tool execution.
>
> I would implement the orchestration using LangGraph, with typed state, conditional routing, bounded execution, retries, durable checkpoints, observability, and human approval for sensitive operations.

One final Staff Engineer insight: Don't use multi-agent architecture just because you can. If a deterministic workflow or a single LLM with a few tools can solve the problem reliably, that is often the better production design. Multi-agent systems should solve a real separation-of-responsibilities or orchestration problem, not introduce unnecessary complexity.