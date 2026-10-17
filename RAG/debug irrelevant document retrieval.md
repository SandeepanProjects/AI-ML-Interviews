# How to debug irrelevant document retrieval in a production RAG system

As a senior or staff AI engineer, I would treat this as a retrieval-quality incident, not immediately as an LLM or prompt-engineering problem.

If a RAG system retrieves irrelevant documents, the model may still produce a fluent answer, but it will be grounded in the wrong evidence. The objective is to identify the earliest stage where relevance breaks down, prove the root cause with measurements, and fix that stage without degrading other queries.

## 1. Understand the retrieval pipeline first

A typical enterprise RAG pipeline looks like this:

1\. User query

“What are the early withdrawal penalties for this account?”

2\. Query processing

Normalization, query rewriting, embedding

3\. Candidate retrieval

Vector search, keyword search, metadata and ACL filters

4\. Reranking and selection

Relevance scoring, deduplication, top-k selection

5\. Context assembly

Token budget, document ordering, truncation

6\. LLM generation

Answer grounded in the retrieved evidence

The critical distinction is between candidate recall and final ranking:

- Candidate recall: Did the system retrieve the document that contains the answer?
- Ranking quality: Was that document ranked highly enough to survive reranking and top-k selection?
- Context quality: Did the correct evidence actually reach the LLM?

If the correct document never enters the candidate set, changing the final prompt or model will not fix the root cause.

## 2. Start with a reproducible failing query

Do not start by changing `top_k`, the embedding model, or the prompt. First preserve a failing example and inspect every retrieval stage.

For each failed query, record:

| Field         | What to capture                                 |
| ------------- | ----------------------------------------------- |
| Query         | Original user query, appropriately redacted     |
| Query version | Normalization and rewriting version             |
| Corpus        | Index version, tenant, document scope           |
| Embedding     | Model name and version, dimensions              |
| Retrieval     | Search type, filters, candidate IDs and scores  |
| Reranking     | Model version, scores and final order           |
| Context       | Final chunk IDs sent to the LLM                 |
| Outcome       | Expected relevant document and observed failure |

For example, suppose the user asks about early withdrawal penalties. The relevant policy is indexed, but retrieval returns a general account overview and an unrelated fee schedule.

The debugging questions are:

1. Is the relevant policy actually in the active index?
2. Does the original query retrieve it?
3. Does keyword search find it?
4. Does vector search find it?
5. Do metadata filters exclude it?
6. Does reranking demote it?
7. Does context assembly drop it?

Always inspect document IDs and text, not just similarity scores. A score of `0.82` is not universally meaningful; score scales differ across embedding models and vector databases.

## 3. Debug in the right order

I would follow this sequence in production. It minimizes random changes and makes the root cause measurable.

Step 1 — Verify the document exists in the active index

Check ingestion status, document version, chunk count, embedding status, tenant ID, access permissions and index alias. A source file existing in object storage does not prove its chunks exist in the searchable index.

Step 2 — Test vector search and keyword search independently

Run each against the same corpus and equivalent security filters. If keyword search finds the document but vector search does not, investigate semantic representation, embeddings and vector ranking. If neither finds it, investigate indexing, filters, chunking and query terms.

Step 3 — Validate filters and authorization

Inspect tenant, document type, date, product, jurisdiction and ACL filters. A correct document can be silently excluded by a filter built from an incorrect default or mismatched metadata type.

Step 4 — Check chunking and embeddings

Verify that chunks contain the relevant text, tables and headings; that query and document embeddings use compatible models and dimensions; and that indexing and query-time preprocessing are consistent.

Step 5 — Inspect reranking and context assembly

Compare the initial candidates with the final selected chunks. If the right evidence is found but discarded, fix reranking, selection, deduplication or token-budget logic—not the vector index.

## 4. Implement a retrieval diagnostic harness

A production retrieval service should be testable independently of the LLM. Here is a reusable pattern for recording candidates and measuring whether known relevant documents are being retrieved.

### A. Define typed retrieval results

```
from dataclasses import dataclassfrom typing import Any@dataclass(frozen=True)class RetrievedChunk:    chunk_id: str    document_id: str    text: str    score: float    metadata: dict[str, Any]@dataclass(frozen=True)class RetrievalDiagnostics:    query: str    candidate_count: int    candidate_document_ids: list[str]    candidate_chunk_ids: list[str]    candidate_scores: list[float]
```

The retrieval adapter should return these results from your actual vector database. Keep the adapter separate from business logic so you can test Qdrant, pgvector or another backend without changing the diagnostic logic.

### B. Record candidates before and after reranking

```
import logginglogger = logging.getLogger(__name__)def log_retrieval_stage(    *,    trace_id: str,    stage: str,    query_version: str,    chunks: list[RetrievedChunk],) -> None:    logger.info(        "rag_retrieval_stage",        extra={            "trace_id": trace_id,            "stage": stage,            "query_version": query_version,            "candidate_count": len(chunks),            "candidates": [                {                    "chunk_id": c.chunk_id,                    "document_id": c.document_id,                    "score": round(c.score, 5),                }                for c in chunks            ],        },    )
```

Usage:

```
initial_candidates = await retriever.search(    query=query,    filters=authorized_filters,    top_k=50,)log_retrieval_stage(    trace_id=trace_id,    stage="initial_retrieval",    query_version="v3",    chunks=initial_candidates,)reranked_candidates = await reranker.rerank(    query=query,    candidates=initial_candidates,)log_retrieval_stage(    trace_id=trace_id,    stage="reranking",    query_version="v3",    chunks=reranked_candidates,)final_chunks = reranked_candidates[:8]
```

The interfaces are illustrative; adapt them to your actual retriever and reranker APIs.

Security note: In financial or enterprise RAG, do not indiscriminately log raw queries, document contents, customer identifiers or authorization claims. Log stable IDs, versions, scores and redacted metadata by default. If text samples are necessary for an incident, use a controlled, access-restricted diagnostic workflow.

### C. Measure Recall\@k

For known test queries, define the relevant document or chunk IDs before running retrieval. Then measure how often those items appear in the first \\(k\\) results.

\\[ \operatorname{Recall\@k} = \frac{\text{relevant items retrieved in top-k}} {\text{all known relevant items}} \\]

```
def recall_at_k(    retrieved_ids: list[str],    relevant_ids: set[str],    k: int,) -> float:    if not relevant_ids:        raise ValueError("Relevant IDs must not be empty")    retrieved = set(retrieved_ids[:k])    return len(retrieved & relevant_ids) / len(relevant_ids)relevant = {"policy-chunk-42", "policy-chunk-43"}retrieved = [    "overview-chunk-2",    "fees-chunk-9",    "policy-chunk-42",    "faq-chunk-7",]print(recall_at_k(retrieved, relevant, k=4))# 0.5
```

Here, only one of two relevant chunks was retrieved, so Recall\@4 is `0.5`.

Use both chunk-level and document-level evaluation when documents are split into multiple chunks. The expected answer may require only one chunk, or several chunks containing separate conditions and exceptions.

## 5. Root cause: the relevant document is not in the initial candidates

This is a candidate retrieval failure. Focus on ingestion, query representation, filters and search configuration.

### Cause A: Chunking destroyed the meaning

Suppose the source policy says:

> Retirement Account — Early Withdrawal. Withdrawals before age 59½ may incur an additional tax unless an exception applies. Refer to the exceptions section for qualifying circumstances.

A naive chunker might split it into:

- Chunk 1: “Retirement Account — Early Withdrawal. Withdrawals before age 59½…”
- Chunk 2: “unless an exception applies. Refer to the exceptions section…”

The second chunk has lost important context. The first chunk may omit the exception. Neither alone fully answers the question.

Senior-level fix:

- Use structure-aware splitting for PDFs, HTML, Markdown and tables.
- Preserve section headings and document identifiers in chunk metadata or the embedded text.
- Use overlap where it helps preserve continuity.
- Keep parent-document or neighboring-chunk references for retrieving surrounding context.
- Test actual chunks against real questions rather than choosing chunk size by convention.

Example:

```
def build_chunk_text(    *,    document_title: str,    section_path: str,    chunk_body: str,) -> str:    return (        f"Document: {document_title}\n"        f"Section: {section_path}\n\n"        f"{chunk_body.strip()}"    )
```

Embed the resulting text if your embedding strategy is designed for this representation. Preserve the original text separately for citations and display.

Do not blindly prepend huge amounts of metadata to every chunk: irrelevant metadata can dilute semantic similarity.

### Cause B: Query and document embeddings are inconsistent

Common mistakes include:

- Documents embedded with model A, queries embedded with model B.
- A new embedding model deployed without rebuilding the index.
- Different normalization or preprocessing between ingestion and retrieval.
- Incorrect vector dimensions or the wrong named vector being searched.
- Using a model that is poorly suited to the query language or domain.

Important: Changing the embedding model generally requires re-embedding the corpus into a new, versioned index. Do not mix unrelated embedding spaces in the same search path.

A deployment manifest can help:

```
from dataclasses import dataclass@dataclass(frozen=True)class EmbeddingManifest:    model_name: str    model_version: str    dimensions: int    preprocessing_version: str    corpus_version: strEXPECTED = EmbeddingManifest(    model_name="your-embedding-model",    model_version="v2",    dimensions=1536,  # Example only; use the model's real dimensions.    preprocessing_version="v3",    corpus_version="policies-2026-10",)
```

Validate the manifest at ingestion and deployment time. The actual dimensions must match the model and vector-store schema.

### Cause C: Metadata filters are too restrictive

Imagine the relevant policy has this metadata:

```
{    "tenant_id": "tenant-17",    "document_type": "policy",    "jurisdiction": "US",    "status": "active"}
```

But the request applies:

```
filters = {    "tenant_id": "tenant-17",    "document_type": "faq",    "jurisdiction": "US",}
```

The vector search may be working perfectly. The policy is excluded because the filter asks for FAQs.

For every failed query, compare:

1. The actual metadata stored on the relevant chunk.
2. The filters produced by the application.
3. The result count with those filters.
4. The result count under a controlled, authorized diagnostic scope.

Do not remove tenant or ACL restrictions in production to improve recall. A relevant document that the user is not authorized to access must remain inaccessible.

### Cause D: Vector search alone is insufficient

Semantic search is useful for paraphrases, but can struggle with exact identifiers, product codes, section numbers, acronyms and rare financial terminology.

For example:

- “early withdrawal penalty”
- “additional tax under section 72(t)”
- “account code IRA-401”

These phrases may refer to related concepts without having the same wording.

A strong baseline is hybrid retrieval:

- Dense vector search for semantic similarity.
- BM25 or another lexical search method for exact terminology.
- Rank fusion to combine the candidate lists.
- Reranking for query-document relevance.

Conceptually:

\\[ S\_{\text{hybrid}} = \alpha S\_{\text{dense}} + (1-\alpha)S\_{\text{lexical}} \\]

This weighted-score approach is appropriate only when the scores are calibrated or normalized appropriately. A rank-fusion method such as Reciprocal Rank Fusion (RRF) is often a safer starting point when dense and lexical scores are not directly comparable.

For RRF:

\\[ \operatorname{RRF}(d)= \sum\_{r \in R} \frac{1}{c+\operatorname{rank}\_{r}(d)} \\]

Here, \\(R\\) is the set of retrieval rankings, and \\(c\\) is a smoothing constant.

The production workflow becomes:

```
User query
   ├── Dense retrieval ─── top 50 candidates
   └── BM25 retrieval ──── top 50 candidates
                ↓
          Deduplicate candidates
                ↓
          Rank fusion (RRF)
                ↓
          Select top 50
                ↓
          Cross-encoder reranker
                ↓
          Final top 5–10 chunks
```

Tune these numbers using evaluation data and latency budgets, rather than assuming they are universal defaults.

## 6. Root cause: the correct document is retrieved but ranked too low

This is a ranking failure, not necessarily an embedding failure.

Suppose initial retrieval returns:

| Rank | Candidate                | Initial score | Relevant? |
| ---- | ------------------------ | ------------- | --------- |
| 1    | General account overview | 0.84          | No        |
| 2    | Account fee schedule     | 0.81          | No        |
| 3    | Early withdrawal policy  | 0.78          | Yes       |
| 4    | Tax exceptions policy    | 0.75          | Yes       |

If your system takes only the top two results, it drops the evidence needed to answer correctly.

### Fix 1: Retrieve more candidates before reranking

Increase the initial candidate pool if Recall\@k is poor, then use a stronger reranker to narrow it down. Measure the additional latency and compute cost.

Do not simply increase the final context size. That can increase noise and distract the model.

### Fix 2: Use a query-aware reranker

A cross-encoder or other query-document reranker evaluates the query and each candidate together, rather than relying only on the independently computed vector similarity.

The interface might look like this:

```
async def retrieve_and_rerank(    query: str,    retriever,    reranker,    *,    candidate_k: int = 50,    final_k: int = 8,):    candidates = await retriever.search(        query=query,        top_k=candidate_k,    )    if not candidates:        return []    ranked = await reranker.rerank(        query=query,        candidates=candidates,    )    return ranked[:final_k]
```

This is a framework-independent example, not a drop-in implementation for a specific vector database. In a real service, pass server-derived authorization filters into retrieval and enforce them before reranking as well.

### Fix 3: Evaluate the ranking stage separately

Use metrics appropriate to your failure:

- Recall\@k: Did the relevant evidence appear in the candidates?
- MRR: How high was the first relevant result?
- nDCG\@k: Were the most relevant results ranked near the top, accounting for graded relevance?
- Context precision: How much of the final context is actually relevant?
- Context recall: How much of the evidence needed to answer was retrieved?

If candidate Recall\@50 is excellent but final Recall\@8 drops sharply, investigate reranking, deduplication and final selection first.

## 7. Root cause: retrieval finds the right document, but context assembly removes it

This is a frequently overlooked production issue.

Your retrieval logs may show the correct chunk, but the model never sees it because:

- A token budget is exceeded.
- The chunk is dropped during deduplication.
- A parent-child expansion step replaces the relevant chunk.
- Chunks are sorted incorrectly.
- The context builder takes a fixed number of characters instead of accounting for tokens.
- A prompt template accidentally omits the selected context.

Trace the exact chunk IDs from initial retrieval to the final LLM request:

```
def inspect_context_selection(    candidates: list[RetrievedChunk],    final_context_ids: list[str],) -> dict:    candidate_ids = {c.chunk_id for c in candidates}    selected_ids = set(final_context_ids)    return {        "candidate_count": len(candidate_ids),        "final_context_count": len(selected_ids),        "dropped_candidate_ids": sorted(candidate_ids - selected_ids),        "selected_ids_not_in_candidates": sorted(            selected_ids - candidate_ids        ),    }
```

In production, also record the reason each candidate was dropped: token budget, duplicate, authorization, relevance threshold or another explicit rule.

A useful diagnostic question is:

> Was the relevant evidence retrieved, reranked, selected, and included in the final model input?

If you cannot answer that from trace data, improve observability before making another retrieval change.

## 8. Root cause: the user's question is ambiguous or poorly represented

Sometimes the query is too vague or uses different terminology from the corpus.

For example:

> “What happens if I take the money out early?”

A financial policy might use “premature distribution,” “early redemption,” or “early withdrawal.”

Query rewriting can help, but it must be controlled and observable.

```
from pydantic import BaseModel, Fieldclass QueryPlan(BaseModel):    original_query: str    rewritten_queries: list[str] = Field(        min_length=1,        max_length=4,    )async def prepare_queries(    original_query: str,    query_rewriter,) -> QueryPlan:    result = await query_rewriter.rewrite(original_query)    # A production implementation should also validate relevance,    # enforce timeouts, and fall back to the original query on failure.    return QueryPlan(        original_query=original_query,        rewritten_queries=result,    )
```

For example, a rewritten query set could include:

- “early withdrawal penalty”
- “premature distribution additional tax”
- “exceptions to early withdrawal penalties”

Run retrieval for these alternatives, fuse the results, and rerank the combined candidates.

Guardrail: Always retain the original query. A rewrite can change the user's intent, introduce unsupported assumptions, or accidentally omit an important constraint. Compare retrieval performance with and without rewriting on a fixed evaluation dataset before enabling it broadly.

## 9. Build an offline retrieval evaluation suite

A senior engineer should not rely on manually testing a few queries. Create a golden dataset with known relevant evidence and run it whenever the index, embedding model, chunking strategy, filters, reranker or query-rewriting logic changes.

Example:

```
golden_queries = [    {        "query": "What are the early withdrawal penalties?",        "relevant_chunk_ids": {            "withdrawal-policy-12",            "withdrawal-policy-13",        },        "category": "policy_lookup",    },    {        "query": "Which exceptions apply to the additional tax?",        "relevant_chunk_ids": {            "tax-exceptions-4",        },        "category": "exception_lookup",    },]
```

A basic evaluator:

```
def evaluate_retrieval(    golden_queries: list[dict],    retrieve,    *,    k: int = 10,) -> dict:    scores = []    for item in golden_queries:        chunks = retrieve(item["query"])        retrieved_ids = [chunk.chunk_id for chunk in chunks]        score = recall_at_k(            retrieved_ids,            item["relevant_chunk_ids"],            k,        )        scores.append(score)    return {        "query_count": len(scores),        "mean_recall_at_k": (            sum(scores) / len(scores) if scores else 0.0        ),    }
```

This example assumes a synchronous retrieval callable. If your service uses async retrieval, make the evaluator async and await the retriever.

For more mature evaluation, measure results by query category and use graded relevance labels for ranking metrics. Also distinguish retrieval correctness from answer correctness.

A useful test matrix includes:

| Test category                     | What it detects                 |
| --------------------------------- | ------------------------------- |
| Exact terminology and identifiers | Weak lexical matching           |
| Paraphrases and synonyms          | Weak semantic retrieval         |
| Multi-condition questions         | Missing evidence chunks         |
| Tables and PDF-derived content    | Parsing and chunking failures   |
| New or updated policies           | Index freshness issues          |
| Multi-tenant queries              | Metadata and authorization bugs |
| No-answer questions               | Poor abstention behavior        |
| Adversarial or ambiguous queries  | Unstable rewriting and ranking  |

Set release gates from a baseline and business risk. For example, a regulated policy-review system may require very high recall for critical policy evidence, while tolerating more latency than a consumer chatbot. The actual threshold should be agreed with product, compliance and risk stakeholders—not chosen arbitrarily.

## 10. Debug it in production using traces and metrics

For an enterprise RAG application using FastAPI, LangGraph, PostgreSQL/pgvector or Qdrant, and Redis, I would create one trace for each request and spans for each retrieval stage.

```
rag.request
 ├── query.normalize
 ├── query.rewrite
 ├── query.embed
 ├── retrieval.dense
 ├── retrieval.bm25
 ├── retrieval.fusion
 ├── retrieval.rerank
 ├── retrieval.context_select
 ├── llm.generate
 └── answer.validate
```

For each retrieval span, capture:

- Query and index version identifiers, with sensitive values redacted.
- Embedding model version and vector dimension.
- Active filter names and candidate counts.
- Retrieval latency and reranker latency.
- Candidate IDs and ranking scores, subject to data-access controls.
- Final context IDs and dropped-candidate reasons.
- Whether the answer passed citation and grounding validation.

### Metrics I would put on a dashboard

| Metric                               | Why it matters                                |
| ------------------------------------ | --------------------------------------------- |
| Recall\@10 on golden queries         | Detects missing evidence                      |
| MRR / nDCG\@10                       | Detects ranking degradation                   |
| Empty-result rate                    | Detects filters, indexing or query issues     |
| Candidate-to-final retention rate    | Detects excessive candidate dropping          |
| Retrieval p50 / p95 latency          | Detects performance regressions               |
| Stale-document rate                  | Detects ingestion and index-refresh problems  |
| Citation correctness                 | Detects incorrect evidence attribution        |
| Retrieval quality by tenant/category | Detects localized failures hidden by averages |

Avoid relying on a single aggregate metric. A system can have good overall recall while failing badly on one important policy category.

### A practical incident decision table

| Observation                                            | Likely area to investigate                      |
| ------------------------------------------------------ | ----------------------------------------------- |
| Relevant document absent from index                    | Ingestion or indexing                           |
| Present in index but absent from dense results         | Embeddings, query semantics or vector search    |
| Found by BM25 but not dense retrieval                  | Exact terms, identifiers, embedding suitability |
| Found without filters but not with filters             | Filter construction, metadata or ACLs           |
| Present in initial results but missing after reranking | Reranker or candidate selection                 |
| Present after reranking but missing from prompt        | Context assembly or token budget                |
| Correct context reaches model, answer is still wrong   | Generation, prompt, grounding or validation     |

Run diagnostic comparisons under the same corpus version, query, tenant scope and authorization rules. Otherwise, you may mistake differences in the test conditions for differences in retrieval quality.

## 11. Common mistakes I would avoid as a staff engineer

Mistake 1: Increasing `top_k` without measuring recall.

More candidates may recover the missing document, but they can also introduce noise, increase latency and raise token costs. Measure candidate recall first.

Mistake 2: Changing the embedding model without reindexing.

If document vectors and query vectors are incompatible, retrieval can become unreliable. Version the embedding configuration and build a new index before switching traffic.

Mistake 3: Using similarity thresholds without calibration.

A threshold that works for one embedding model or corpus may reject valid documents in another. Calibrate it on labeled queries, and measure both false exclusions and irrelevant matches.

Mistake 4: Assuming hybrid search automatically solves relevance.

Hybrid retrieval improves coverage for many workloads, but it still needs suitable lexical analyzers, metadata filters, fusion, ranking and evaluation.

Mistake 5: Letting an LLM decide whether a document is relevant without testing the decision.

LLM-based query rewriting and relevance grading can help, but they introduce latency, cost and nondeterminism. Evaluate them against labeled data and provide safe fallbacks.

Mistake 6: Optimizing answer quality while ignoring retrieval quality.

A more capable LLM can sometimes compensate for weak evidence, but it cannot reliably cite a policy it never receives. Measure retrieval and generation separately.

## 12. My recommended remediation workflow

1. Reproduce the failure. Save a sanitized query, index version, expected relevant chunks and the observed candidate list.
2. Verify index integrity. Confirm the source document was ingested, chunked and embedded successfully, and is available in the active index.
3. Run controlled retrieval experiments. Compare dense search, lexical search and hybrid search with identical authorized filters.
4. Inspect candidate recall. If the relevant chunk is absent, fix ingestion, filters, chunking, embeddings or candidate search.
5. Inspect ranking and context assembly. If the chunk exists early but disappears later, fix reranking, deduplication or selection.
6. Evaluate on a golden dataset. Compare the proposed change with the current baseline, including latency, relevance and cost.
7. Deploy with monitoring and rollback. Version the index and retrieval configuration, canary the change, and monitor for regressions.

## 13. Interview-ready answer

You can explain your approach this way:

> “When a RAG system starts retrieving irrelevant documents, I debug the retrieval pipeline stage by stage instead of immediately changing the prompt or LLM. First, I reproduce the issue using a known query and inspect the active corpus, document versions, chunking, embedding configuration and authorization filters.
>
> I then measure candidate Recall\@k and compare dense retrieval with lexical retrieval to establish whether the problem is candidate generation or ranking. If the relevant document is missing from the initial candidates, I investigate ingestion, chunk boundaries, embedding consistency, query rewriting and metadata filters. If it appears in the initial results but is lost later, I inspect reranking, deduplication and context assembly.
>
> I instrument each stage with distributed traces, candidate IDs, ranking scores, latency and index versions. I maintain a golden evaluation dataset and track Recall\@k, MRR, nDCG and context quality across releases. Finally, I validate fixes through offline evaluation and canary deployment, while preserving tenant isolation, document permissions and sensitive-data controls.
>
> My objective is to identify the earliest stage where relevance degrades, fix the root cause, and demonstrate with measurements that retrieval quality improved without unacceptable latency, cost or security regressions.”

The key principle is: prove where relevant evidence is lost before changing the retrieval system. That is the difference between random RAG tuning and systematic production debugging.