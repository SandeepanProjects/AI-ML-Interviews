# How do you investigate a production AI incident where the system suddenly starts returning incorrect answers?

This is a very important Senior AI Engineer / Staff AI Engineer interview question.

In production, an AI application can suddenly start returning incorrect answers even when the application code has not changed.

For example, imagine you have deployed an enterprise financial advisor copilot using:

- FastAPI
- LangGraph
- OpenAI LLM
- PostgreSQL
- Qdrant vector database
- Redis
- RAG pipeline
- Multiple AI agents

A user asks:

> What is the current expense ratio of Fund X?

The correct answer in the document is 0.45%, but suddenly the AI starts returning 0.75%.

As a Staff AI Engineer, I would not immediately change the prompt or switch the LLM.

I would follow a systematic incident investigation process to identify which layer is responsible.

## 1. First understand the possible failure points

An AI application has multiple layers, and incorrect answers can originate from any of them.

User Query

FastAPI / API Gateway

Authentication, tenant context, request validation

Query Processing

Normalization, rewriting, metadata filters

Retrieval Layer

Qdrant, BM25, hybrid search, reranking

Context Construction

Chunk selection, token limits, prompt assembly

LLM / Agent Workflow

Model generation, tools, LangGraph execution

Response Validation and Final Answer

My first objective is to determine which layer introduced the incorrect information.

There are six major areas I investigate:

| Layer               | Possible problem                       |
| ------------------- | -------------------------------------- |
| Data ingestion      | Stale or incorrectly parsed documents  |
| Retrieval           | Wrong chunks retrieved                 |
| Reranking           | Relevant chunks ranked too low         |
| Prompt construction | Missing or truncated context           |
| LLM                 | Hallucination or model behavior change |
| Agent workflow      | Wrong tool, state, or execution path   |

## 2. Step 1 — Detect and classify the incident

First, I establish the scope of the incident.

I investigate:

1. Did incorrect answers start suddenly or gradually?
2. Are all users affected or only specific tenants?
3. Are all questions affected or only financial questions?
4. Is the issue isolated to one model or one workflow?
5. Did a recent deployment, prompt update, model update, or document ingestion occur?

I compare production metrics against the previous baseline.

For example:

| Metric              | Previous baseline | Current |
| ------------------- | ----------------- | ------- |
| Answer faithfulness | 0.94              | 0.61    |
| Answer relevancy    | 0.91              | 0.73    |
| Retrieval precision | 0.90              | 0.55    |
| API error rate      | 1%                | 1.2%    |
| Average latency     | 1.8 sec           | 2.1 sec |

Notice something important.

The API is still working. Latency is normal. There are no major infrastructure errors.

However, retrieval precision and answer faithfulness have dropped.

This suggests that the problem could be in retrieval or context construction rather than infrastructure.

These metrics are illustrative, not universal production thresholds.

## 3. Step 2 — Check recent changes

I investigate what changed immediately before the incident.

I check:

- Git deployment history.
- LLM model version.
- System prompt version.
- Embedding model version.
- Chunking configuration.
- Qdrant collection changes.
- Document ingestion jobs.
- Reranker version.
- LangGraph workflow changes.
- Redis cache behavior.

For example:

Suppose the team recently changed the embedding model from:

`text-embedding-3-small`

to another embedding model.

But the existing Qdrant vectors were generated using the previous embedding model.

Now new queries use a different embedding space.

This can cause retrieval quality to deteriorate significantly.

Important: Never mix vectors from incompatible embedding spaces in the same retrieval operation. Re-embed the corpus into a separate collection and validate it before switching traffic.

If a recent change correlates with the incident, I consider rolling it back or routing traffic to the last known-good version.

## 4. Step 3 — Investigate the RAG retrieval pipeline

This is one of the most important steps.

I take an incorrect production request and replay it against the retrieval pipeline.

Suppose the user asks:

```
What is the expense ratio of Fund X?
```

Expected document:

```
Fund X
Expense Ratio: 0.45%
Effective Date: October 2026
```

But the system retrieves:

```
Fund X
Expense Ratio: 0.75%
Effective Date: March 2025
```

The LLM may actually be doing exactly what it was instructed to do: answer using the retrieved context.

The real problem is stale retrieval.

I investigate four things.

A. Check the retrieved documents

I log the top-k retrieved chunks, their document IDs, versions, scores, and metadata.

```
async def debug_retrieval(query: str, retriever):    results = await retriever.retrieve(        query=query,        top_k=5,    )    for rank, result in enumerate(results, start=1):        print({            "rank": rank,            "document_id": result.document_id,            "chunk_id": result.chunk_id,            "score": result.score,            "document_version": result.metadata.get(                "document_version"            ),            "effective_date": result.metadata.get(                "effective_date"            ),            "content": result.content[:500],        })    return results
```

In production, I would send these records to structured observability logs rather than use `print`.

B. Check metadata filtering

Sometimes retrieval is correct semantically but wrong because metadata filters are missing or incorrect.

For example:

```
query_filter = {    "tenant_id": "tenant_123",    "document_status": "active",    "document_type": "fund_prospectus",}
```

I verify that:

- The correct tenant is selected.
- Only active documents are retrieved.
- Document version is correct.
- Effective dates are respected.
- Deleted documents are excluded.

C. Check hybrid search

If semantic search misses exact financial terminology, I compare vector search with keyword search.

For example:

```
Expense Ratio: 0.45%
```

A keyword search can find the exact phrase, while semantic search may retrieve a related but outdated document.

I compare:

- Vector retrieval results.
- BM25 retrieval results.
- Hybrid retrieval results.
- Reranked results.

D. Check reranking

A reranker can sometimes push the correct chunk down the ranking.

I inspect the initial retrieval ranking and the final reranked ranking separately.

This helps identify whether the problem is in the retriever or reranker.

## 5. Step 4 — Investigate prompt construction

Suppose Qdrant retrieves the correct document.

But the final answer is still incorrect.

Now I inspect the exact prompt sent to the LLM.

For example, the application may have retrieved the correct chunk but removed it because of token limitations.

Or the prompt may contain conflicting information:

```
Context 1:
Fund X expense ratio is 0.45%.

Context 2:
Fund X expense ratio was previously 0.75%.
```

The model might select the outdated value.

I inspect:

1. System prompt.
2. User prompt.
3. Retrieved context.
4. Context ordering.
5. Token count.
6. Truncation behavior.
7. Prompt template version.

Example:

```
def build_rag_prompt(question: str, documents: list) -> str:    context = "\n\n".join(        f"""        Document ID: {doc.document_id}        Version: {doc.metadata['version']}        Effective Date: {doc.metadata['effective_date']}        Content:        {doc.content}        """        for doc in documents    )    return f"""    You are an enterprise financial assistant.    Rules:    1. Answer only using the supplied context.    2. Prefer the latest valid effective document.    3. Never invent financial values.    4. If conflicting information exists, explain the conflict.    5. If the answer is unavailable, say so.    6. Cite the document supporting your answer.    Context:    {context}    User Question:    {question}    Answer:    """
```

This is a simplified example. In production, I would also enforce document validity and version selection before prompt construction instead of relying on the LLM to resolve every conflict.

## 6. Step 5 — Investigate the LLM itself

Suppose:

- Correct documents are retrieved.
- Prompt contains the correct context.
- Token truncation is not happening.
- The answer is still wrong.

Now I investigate model behavior.

I check:

| Parameter         | What I investigate                         |
| ----------------- | ------------------------------------------ |
| Model             | Was the provider or model version changed? |
| Temperature       | Did sampling configuration change?         |
| Max tokens        | Is the response being cut off?             |
| System prompt     | Was the instruction modified?              |
| Structured output | Is the schema being violated?              |
| Provider          | Is a fallback model being used?            |

One common production issue is silent model fallback.

For example:

```
Primary model: GPT model A
Fallback model: GPT model B
```

If model A experiences an outage, the application may automatically route requests to model B.

Model B might have different reasoning behavior or instruction-following characteristics.

Therefore, I log the actual model used for every request, not just the configured primary model.

I also replay the same prompt against the previous known-good model configuration.

If the previous model produces the correct response while the current model does not, I investigate model regression.

## 7. Step 6 — Investigate the agent workflow

This becomes particularly important in a LangGraph multi-agent application.

Imagine the workflow:

```
User Query
    |
    v
Planner Agent
    |
    v
Research Agent
    |
    v
Retrieval Agent
    |
    v
Financial Analysis Agent
    |
    v
Response Agent
```

The problem might not be the LLM or RAG system.

It could be an incorrect state transition.

For example:

- Planner sends the wrong question to the research agent.
- Research agent retrieves outdated information.
- Analysis agent overwrites a correct value.
- A conditional edge routes execution to the wrong node.
- An old checkpoint resumes with stale state.
- A tool returns an incorrect result.

I log every node execution.

Example:

```
from datetime import datetime, timezoneimport logginglogger = logging.getLogger(__name__)def log_agent_execution(    request_id: str,    node_name: str,    input_state: dict,    output_state: dict,):    logger.info(        "agent_execution",        extra={            "request_id": request_id,            "node_name": node_name,            "timestamp": datetime.now(                timezone.utc            ).isoformat(),            "input_keys": list(input_state.keys()),            "output_keys": list(output_state.keys()),        },    )
```

For production, I would avoid logging raw sensitive financial documents or personal information. I would use redaction, access controls, and secure trace storage.

I also capture:

- Graph execution ID.
- Checkpoint ID.
- Node name.
- Tool name.
- Tool execution result status.
- Retry count.
- State version.
- Model version.

This allows me to identify exactly where the incorrect information entered the workflow.

## 8. Step 7 — Use distributed tracing to identify the failure

For production AI applications, I recommend OpenTelemetry.

Every request should have a correlation ID and trace ID.

Example trace:

Request trace: `req-98231`

1. FastAPI request

   120 ms

   Success
2. Query embedding

   90 ms

   Success
3. Qdrant retrieval

   80 ms

   Success

   Wrong document version retrieved
4. Reranking

   110 ms

   Success
5. LLM generation

   900 ms

   Success
6. Response validation

   30 ms

   Incorrect answer

Total latency

# 1.33 sec

Illustrative trace showing a successful execution that produces an incorrect answer.

Notice that every component returned successfully.

This is why normal API monitoring is not enough for AI applications.

We need to monitor both technical health and answer quality.

A useful production observability stack:

| Tool             | Responsibility                |
| ---------------- | ----------------------------- |
| OpenTelemetry    | Distributed traces            |
| Prometheus       | Infrastructure metrics        |
| Grafana          | Dashboards and alerts         |
| LangSmith        | LangGraph / LLM tracing       |
| MLflow           | Model and experiment tracking |
| RAGAS            | RAG quality evaluation        |
| ELK / OpenSearch | Centralized logs              |

## 9. Step 8 — Run an evaluation to identify quality regression

Once I identify the suspicious layer, I run a controlled evaluation.

I maintain a golden evaluation dataset containing previously validated questions and answers.

Example:

```
golden_dataset = [    {        "question": "What is Fund X expense ratio?",        "expected_answer": "0.45%",        "expected_document": "fund_x_prospectus_2026",    },    {        "question": "What is Fund Y expense ratio?",        "expected_answer": "0.60%",        "expected_document": "fund_y_prospectus_2026",    },]
```

I replay the dataset against:

1. Previous production version.
2. Current production version.
3. Previous embedding configuration.
4. Current embedding configuration.
5. Previous LLM model.
6. Current LLM model.

Then I compare the results.

### RAG evaluation metrics

| Metric             | What it tells me                              |
| ------------------ | --------------------------------------------- |
| Context Precision  | Are retrieved chunks relevant?                |
| Context Recall     | Did retrieval find the necessary information? |
| Faithfulness       | Is the answer supported by retrieved context? |
| Answer Relevancy   | Does the answer address the question?         |
| Answer Correctness | Does the answer match the ground truth?       |

For example, suppose:

Illustrative RAG quality regression

Comparison between the previous and current production versions.

Previous version

Current version

00.250.50.751FaithfulnessContext precisionContext recallAnswer relevancy

Here, context recall is relatively stable, but context precision has fallen significantly.

My investigation would focus on irrelevant or outdated chunks entering the context.

I would not immediately fine-tune the LLM because the evidence points toward retrieval quality.

## 10. Step 9 — Apply immediate mitigation

Once the issue is identified, I prioritize restoring correct answers over investigating every technical detail.

Depending on the root cause, I take the following actions:

| Root cause                 | Immediate mitigation                                       |
| -------------------------- | ---------------------------------------------------------- |
| Bad deployment             | Roll back to previous version                              |
| Incorrect embedding model  | Restore compatible model and vectors                       |
| Stale documents            | Re-ingest and validate documents                           |
| Bad reranker               | Disable or roll back reranker                              |
| Prompt regression          | Restore previous prompt                                    |
| LLM regression             | Route to known-good model                                  |
| Redis stale cache          | Invalidate affected cache entries                          |
| Incorrect agent state      | Stop affected workflow and recover from a valid checkpoint |
| Incorrect metadata filters | Fix tenant/version filters                                 |

For financial applications, I would introduce an additional safety mechanism.

If the system cannot establish reliable evidence, it should abstain instead of inventing a financial value.

For example:

```
def validate_answer(    answer: str,    retrieved_context: str,    confidence: float,) -> dict:    if confidence < 0.80:        return {            "status": "REVIEW_REQUIRED",            "answer": (                "I cannot confidently verify "                "this information from the available documents."            ),        }    if not retrieved_context.strip():        return {            "status": "INSUFFICIENT_CONTEXT",            "answer": (                "I could not find supporting "                "information in the available documents."            ),        }    return {        "status": "SUCCESS",        "answer": answer,    }
```

Important: This is illustrative application logic, not a reliable confidence estimator. A raw LLM-generated confidence score should not be trusted as proof of correctness.

In a real financial application, I would validate the answer against source evidence, effective dates, structured financial data, and deterministic business rules.

## 11. Step 10 — Prevent the incident from happening again

After recovery, I focus on prevention.

I introduce the following controls:

A. Version everything

Maintain versions for:

- Prompts.
- Embedding models.
- Chunking configurations.
- Retrieval configurations.
- Rerankers.
- LLM models.
- Agent workflows.

B. Introduce automated regression testing

Every prompt, model, embedding, or retrieval change must pass the golden evaluation dataset before production deployment.

C. Introduce canary deployments

Instead of deploying a new model to 100% of traffic:

- Send 5% of traffic to the new version.
- Compare quality metrics.
- Check latency and cost.
- Increase traffic gradually.
- Automatically roll back when quality deteriorates.

D. Introduce document freshness monitoring

Track:

- Document ingestion timestamp.
- Document version.
- Embedding generation timestamp.
- Index synchronization status.
- Document deletion status.

E. Introduce answer-level monitoring

Track the percentage of answers that:

- Have supporting citations.
- Pass schema validation.
- Pass deterministic business rules.
- Fail groundedness evaluation.
- Require human review.

## 12. Production incident investigation code structure

As a Staff AI Engineer, I would separate incident diagnosis from the business logic.

A practical project structure:

```
app/
│
├── api/
│   └── routes/
│       └── chat.py
│
├── rag/
│   ├── retriever.py
│   ├── reranker.py
│   ├── prompt_builder.py
│   └── answer_generator.py
│
├── agents/
│   └── workflow.py
│
├── observability/
│   ├── tracing.py
│   ├── metrics.py
│   └── logging.py
│
├── evaluation/
│   ├── golden_dataset.py
│   ├── ragas_evaluator.py
│   └── regression_test.py
│
├── incident/
│   ├── incident_detector.py
│   ├── diagnostic_service.py
│   └── recovery_service.py
│
└── tests/
    ├── test_retrieval.py
    ├── test_generation.py
    └── test_end_to_end.py
```

The important architectural decision is that every layer exposes enough information to investigate its own behavior.

For example:

```
from dataclasses import dataclassfrom typing import Any@dataclassclass AIRequestTrace:    request_id: str    tenant_id: str    prompt_version: str    model_name: str    embedding_model: str    retrieved_document_ids: list[str]    retrieved_scores: list[float]    retrieved_versions: list[str]    workflow_id: str | None    checkpoint_id: str | None    response_status: strclass IncidentDiagnosticService:    def __init__(        self,        trace_repository,        evaluator,        alert_service,    ):        self.trace_repository = trace_repository        self.evaluator = evaluator        self.alert_service = alert_service    async def investigate(        self,        request_id: str,    ) -> dict[str, Any]:
```

This is a simplified diagnostic service. The evaluator needs to be implemented using actual evaluation logic, and the alert service needs integration with your monitoring platform.

The important principle is that the diagnostic service gathers evidence; it should not automatically declare a root cause from one metric.

## 13. How I would handle the incident in the first 30 minutes

This is how I would organize the actual production response.

| Time      | Action                               | Objective                                  |
| --------- | ------------------------------------ | ------------------------------------------ |
| 0–5 min   | Detect and classify                  | Establish impact and severity              |
| 5–10 min  | Inspect deployment and model changes | Identify recent regressions                |
| 10–15 min | Replay affected requests             | Identify incorrect retrieval or generation |
| 15–20 min | Inspect traces and prompts           | Locate the failure layer                   |
| 20–25 min | Rollback or apply mitigation         | Restore answer quality                     |
| 25–30 min | Run regression evaluation            | Verify recovery                            |

For a high-impact financial incident, I would immediately disable affected answer paths or switch to a verified fallback if correctness cannot be guaranteed. I would not wait 30 minutes to mitigate.

## 14. Senior AI Engineer interview answer

If the interviewer asks this question, you can give the following answer:

"I approach production AI incidents systematically rather than immediately modifying prompts or changing models.

First, I identify the scope of the incident. I check whether all users are affected, whether the issue is tenant-specific, and whether incorrect answers started after a recent deployment, model update, prompt change, or document ingestion.

Second, I inspect distributed traces using OpenTelemetry. I trace the request across FastAPI, query processing, embedding generation, vector retrieval, reranking, prompt construction, LLM generation, and response validation.

Third, I isolate the failure layer.

If retrieval is returning outdated documents, I investigate document versioning, metadata filters, embedding compatibility, hybrid search, and reranking.

If retrieval is correct but the answer is wrong, I inspect the exact prompt, context truncation, model version, temperature, and fallback behavior.

If I am dealing with a multi-agent LangGraph system, I inspect node executions, state transitions, tool calls, retries, and checkpoint recovery.

Fourth, I replay affected requests against a golden evaluation dataset. I use RAGAS metrics such as faithfulness, context precision, context recall, and answer relevancy to identify quality regressions.

Fifth, I mitigate the incident by rolling back the problematic deployment, restoring a known-good model or prompt, correcting document indexes, or invalidating stale caches.

Finally, I introduce regression tests, canary deployments, versioned prompts and embeddings, document freshness monitoring, and answer-quality alerts.

My main objective is to identify the root cause, restore trustworthy responses, and prevent recurrence.

I also ensure that the system abstains rather than generating unsupported answers, particularly in financial or other high-impact applications."

## 15. Important follow-up questions interviewers may ask

These are particularly relevant for Senior/Staff AI Engineer interviews.

1. How do you differentiate between an LLM hallucination and a retrieval failure?
2. What if RAGAS metrics suddenly drop in production?
3. How do you detect that an embedding model update has degraded retrieval quality?
4. How do you roll back a model without downtime?
5. How do you investigate incorrect answers in a multi-agent LangGraph workflow?
6. How do you implement automated AI quality monitoring?
7. How do you prevent stale documents from being retrieved?
8. What if the LLM returns incorrect answers even when the retrieved context is correct?

My recommendation: Pay particular attention to questions 1, 2, 4, and 8. They test whether you understand the difference between infrastructure reliability, retrieval quality, model behavior, and end-to-end AI correctness.