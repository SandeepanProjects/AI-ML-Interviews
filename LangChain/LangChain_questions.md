For **Senior AI Engineer (5–10+ years)** interviews, LangChain and LangGraph questions are usually not about APIs alone. Interviewers expect you to know **when to use them, how they work internally, how to build production systems, and their limitations**.

Below is a comprehensive list of commonly asked questions.

---

# Part 1: LangChain Fundamentals

## 1. What is LangChain?

Expected answer

* Framework for building LLM-powered applications
* Provides reusable abstractions
* Connects LLMs with external systems
* Supports RAG
* Supports agents
* Supports memory
* Supports tools
* Supports structured outputs

Example

```
User
   |
Prompt
   |
LLM
   |
Parser
   |
Application
```

---

## 2. Why use LangChain instead of calling OpenAI directly?

Interviewers expect

Direct API

```
User
   |
OpenAI API
   |
Result
```

LangChain

```
User
   |
Prompt Template
   |
LLM
   |
Output Parser
   |
Retriever
   |
Memory
   |
Result
```

Advantages

* reusable components
* tool calling
* chains
* retrieval
* memory
* callbacks
* observability

---

## 3. Explain LangChain architecture.

Typical flow

```
User

↓

PromptTemplate

↓

LLM

↓

OutputParser

↓

Application
```

Production

```
User

↓

Prompt

↓

Retriever

↓

Vector DB

↓

LLM

↓

Output Parser

↓

Business Logic
```

---

## 4. What are Chains?

A chain connects multiple components.

Example

```
Prompt

↓

LLM

↓

Parser
```

Instead of

```
Prompt

↓

LLM
```

---

## 5. What is LCEL?

LangChain Expression Language

Example

```python
chain = prompt | llm | parser
```

Instead of

```python
output = parser.parse(
    llm.invoke(
        prompt.invoke(data)
    )
)
```

Advantages

* readable
* composable
* streaming
* async
* reusable

---

## 6. Explain Runnable interface.

Questions

* invoke()
* batch()
* stream()
* ainvoke()
* astream()

Example

```python
result = chain.invoke({
    "question": "What is AI?"
})
```

Batch

```python
results = chain.batch([
    {"question":"AI"},
    {"question":"ML"}
])
```

---

## 7. PromptTemplate vs ChatPromptTemplate

PromptTemplate

```
Translate to French

{sentence}
```

ChatPromptTemplate

```
System:
You are helpful.

User:
Translate hello.
```

---

## 8. Explain OutputParser.

Example

LLM output

```
Name: John
Age: 22
```

Need

```python
{
 "name":"John",
 "age":22
}
```

Parser converts

```
Text

↓

JSON

↓

Pydantic
```

---

## 9. Explain Structured Output.

Modern interview question.

Example

```python
class Person(BaseModel):
    name: str
    age: int
```

LLM

↓

Validated object

---

## 10. Explain Tool Calling.

Example

```
User

↓

Agent

↓

Weather Tool

↓

LLM
```

---

# Part 2: Retrieval

## 11. How does RetrievalQA work?

Pipeline

```
Question

↓

Embedding

↓

Vector Search

↓

Top K

↓

Prompt

↓

LLM
```

---

## 12. Difference between Retriever and Vector Store.

Retriever

```
retrieve(query)
```

Vector Store

```
stores vectors
```

Retriever is abstraction.

---

## 13. Similarity Search

Questions

* cosine similarity
* top-k
* metadata filtering

---

## 14. MultiQueryRetriever

Instead of

```
What is AI?
```

Generate

```
Explain AI

Artificial Intelligence

Define AI
```

Improves recall.

---

## 15. Contextual Compression Retriever

Retrieves

20 chunks

↓

compress

↓

5 chunks

↓

LLM

---

## 16. Parent Document Retriever

Retrieve

Small chunk

↓

Return

Entire parent document

---

## 17. Ensemble Retriever

Combine

BM25

*

Dense search

*

Reranker

---

# Part 3: Memory

## 18. Types of Memory

Questions

ConversationBufferMemory

ConversationSummaryMemory

EntityMemory

TokenBufferMemory

VectorStoreMemory

---

## 19. Why is memory difficult?

Problems

* token limits
* hallucinations
* cost
* stale memory

---

## 20. Long-term memory implementation

```
Conversation

↓

Embedding

↓

Vector DB

↓

Relevant history

↓

Prompt
```

---

# Part 4: Agents

## 21. What is an Agent?

Instead of

```
Prompt

↓

LLM
```

Agent

```
Prompt

↓

Reason

↓

Choose Tool

↓

Execute

↓

Observe

↓

Repeat
```

---

## 22. Explain ReAct.

Thought

↓

Action

↓

Observation

↓

Repeat

---

## 23. AgentExecutor

Responsibilities

* execute agent
* call tools
* stop loops
* collect responses

---

## 24. How do tools work?

Tool

```python
@tool
def weather(city):
    ...
```

Agent

↓

Calls tool

↓

Gets output

↓

Continues reasoning

---

## 25. How do you prevent infinite loops?

Expected answers

* max iterations
* timeout
* recursion limits
* stop conditions

---

# Part 5: LangGraph

This is becoming more common than LangChain alone.

---

## 26. What is LangGraph?

Framework for building

State Machines

instead of

Linear Chains

---

## 27. Why LangGraph?

Instead of

```
A

↓

B

↓

C
```

You can build

```
A

↙    ↘

B      C

 ↘    ↙

D
```

---

## 28. Explain StateGraph.

Everything shares one state.

```
State

↓

Node A

↓

Updated State

↓

Node B
```

---

## 29. What is State?

Example

```python
class AgentState(TypedDict):
    messages: list
    documents: list
    answer: str
```

Every node modifies it.

---

## 30. Explain Nodes.

Node

```
Input State

↓

Logic

↓

Updated State
```

---

## 31. Explain Edges.

Edges determine

```
Node A

↓

Node B
```

or

```
if score < 0.8

↓

Retrieve Again
```

---

## 32. Conditional Edges

Example

```
Enough context?

Yes

↓

Answer

No

↓

Retrieve Again
```

---

## 33. Cycles in LangGraph

One major feature.

```
Retrieve

↓

Grade

↓

Bad?

↓

Retrieve Again
```

Impossible in normal chains.

---

## 34. Explain Checkpointing.

State saved after every node.

Useful for

* resume execution
* failures
* human approval

---

## 35. Human-in-the-loop

```
LLM

↓

Human Approval

↓

Continue
```

---

## 36. Multi-agent architecture

```
Planner

↓

Researcher

↓

Coder

↓

Reviewer

↓

Final
```

---

## 37. Explain Supervisor Pattern.

```
Supervisor

↓

Agent A

↓

Agent B

↓

Agent C
```

Supervisor decides routing.

---

## 38. Explain Planner-Executor Pattern.

Planner

↓

Task list

↓

Executor

↓

Results

↓

Planner

---

## 39. Explain Reflection Pattern.

LLM

↓

Answer

↓

Critique

↓

Improve

↓

Final

---

## 40. Explain Retry Pattern.

```
Call API

↓

Failure

↓

Retry

↓

Fallback

↓

Success
```

---

# Part 6: Production Questions

## 41. How do you monitor LangChain?

Expected

* LangSmith
* OpenTelemetry
* Prometheus
* Grafana
* Phoenix
* MLflow

---

## 42. How do you reduce latency?

* cache prompts
* cache embeddings
* async tools
* parallel nodes
* streaming
* reranking only when necessary

---

## 43. How do you reduce cost?

* prompt optimization
* semantic cache
* smaller models
* batching
* compress context

---

## 44. How do you debug an agent?

Expected

* trace execution
* inspect prompts
* inspect tool calls
* inspect state
* inspect retrieved documents

---

## 45. LangChain vs LangGraph

| LangChain            | LangGraph                    |
| -------------------- | ---------------------------- |
| Linear workflows     | Graph workflows              |
| Simple pipelines     | Complex orchestration        |
| Few steps            | Hundreds of steps            |
| Stateless by default | Stateful execution           |
| No cycles            | Supports loops               |
| Limited branching    | Rich conditional routing     |
| Good for RAG         | Good for multi-agent systems |

---

# Part 7: Coding Questions

You may be asked to implement:

1. Build a RAG pipeline with LangChain.
2. Create a custom tool and integrate it with an agent.
3. Implement a conversation memory system.
4. Build a router chain that selects among multiple LLMs.
5. Parse structured JSON output using Pydantic.
6. Create an LCEL pipeline with prompts, retrieval, and parsing.
7. Build a LangGraph workflow with conditional branching.
8. Implement a planner–executor multi-agent architecture.
9. Add human approval checkpoints to a LangGraph.
10. Add retry logic and fallback models for failed tool calls.
11. Build a retrieval grader that loops until sufficient context is found.
12. Stream tokens from an LLM through a LangChain pipeline.
13. Add observability and tracing to a LangChain application.
14. Design a scalable multi-tenant AI agent architecture.
15. Build an end-to-end agent using LangGraph, FastAPI, Redis, PostgreSQL, and a vector database.

For senior AI engineering roles, interviewers increasingly focus on **design decisions** rather than API memorization. Be prepared to explain **why** you chose LangChain versus LangGraph, how you manage state, handle failures and retries, prevent infinite loops, optimize latency and cost, secure tool usage, and monitor agents in production. These architectural discussions often carry more weight than writing a few lines of framework-specific code.
