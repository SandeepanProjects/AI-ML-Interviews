# 1. What is reranking in RAG?

Reranking is the process of taking documents retrieved by an initial search system and reordering them based on how relevant they are to the user's actual query.

In a RAG system, vector search retrieves documents based on embedding similarity. However, a document with a high semantic similarity score is not necessarily the most useful document for answering the question.

A reranker performs a more detailed relevance evaluation between the query and each retrieved document.

Think of it this way:

- Vector retrieval finds potentially relevant documents.
- Reranking identifies which of those documents are actually most relevant.
- The LLM receives the best-ranked documents to generate its answer.

Vector search is candidate retrieval. Reranking is candidate refinement.

# 2. Why do we need a reranker after vector retrieval?

Let's understand this with an enterprise financial compliance example.

Suppose your knowledge base contains these documents:

| Document | Content                                                                                   |
| -------- | ----------------------------------------------------------------------------------------- |
| Doc A    | Investment recommendations must be approved by the compliance department.                 |
| Doc B    | Investment portfolios should be reviewed periodically to manage risk.                     |
| Doc C    | Compliance approval is mandatory before sharing financial recommendations with customers. |
| Doc D    | Investment products carry market risks and may result in financial losses.                |

User query:

"What approval is required before sharing investment recommendations with customers?"

Your vector database searches using the query embedding.

It might return:

| Document | Vector similarity score |
| -------- | ----------------------- |
| Doc A    | 0.89                    |
| Doc B    | 0.86                    |
| Doc D    | 0.84                    |
| Doc C    | 0.82                    |

Notice something interesting.

Doc C is probably the most directly relevant answer, but vector search ranked it fourth.

Why?

Because embedding similarity measures semantic similarity in vector space. It does not guarantee that the document contains the exact evidence needed to answer the question.

A reranker can evaluate the query-document relationship more deeply.

After reranking:

| Document | Reranker score | New rank |
| -------- | -------------- | -------- |
| Doc C    | 0.98           | 1        |
| Doc A    | 0.95           | 2        |
| Doc B    | 0.42           | 3        |
| Doc D    | 0.21           | 4        |

Improved ranking

Now the most useful documents are at the top.

[Kudos AI | Blog | Retrieval-Augmented Generation (RAG): From First Principles to Production Systems](https://images.openai.com/static-rsc-4/CbsTsJqQSfePn60qudsHsQv5xrqNrVam_eBwVgtQ-3swZGqzcyy_mwT4iOcqso-J466HzpjRqXBhA0NGEnOD7SNBXdZJfolfrLuOJg3pVCC6y43tYxvSK1O1UyfZ5FSMjBQt86rgStW38g3PumfC2wbQNM53cSxUmRDj9uDUPKo?purpose=inline)

[kudosai.com](https://www.kudosai.com/Blog/Retrieval-Augmented-Generation?utm_source=chatgpt.com)

This is the main reason we use reranking: to improve the relevance of the context passed to the LLM.

# 3. How does a reranker actually work?

To understand reranking, you need to understand the difference between a bi-encoder and a cross-encoder.

These are two common model architectures used in information retrieval.

## A. Bi-encoder: Used for vector retrieval

A bi-encoder independently converts the query and document into embeddings.

For example:

Query

What approval is needed before sharing investment advice?

Document

Investment recommendations require compliance approval.

Embedding model

Query vector

[0.21, 0.45, ...]

Document vector

[0.19, 0.43, ...]

Cosine similarity

Similarity score: 0.89

The important point is that the query and document are encoded independently.

This makes vector retrieval efficient because you can generate document embeddings once, store them, and compare a new query against millions of stored vectors.

However, the model doesn't perform a detailed joint analysis of the query and document at retrieval time.

## B. Cross-encoder: Used for reranking

A cross-encoder takes the query and document together as input.

For example:

```
Query:
What approval is needed before sharing investment advice?

Document:
Investment recommendations require compliance approval.
```

The model processes both together and produces a relevance score.

Query + Document

Combined input pair

Cross-encoder model

Jointly evaluates query-document relevance

Relevance score: 0.98

Illustrative score

The cross-encoder can consider how individual words and phrases relate to each other across the query and document.

For example, it can distinguish between:

- A document explaining how compliance approval is obtained.
- A document simply mentioning compliance approval.
- A document discussing investment risk without addressing approval.

This is why cross-encoder reranking can improve relevance.

Important difference:

| Feature             | Bi-encoder                    | Cross-encoder                                   |
| ------------------- | ----------------------------- | ----------------------------------------------- |
| Input               | Query and document separately | Query and document together                     |
| Output              | Embeddings                    | Relevance score                                 |
| Computation         | Efficient                     | More expensive                                  |
| Document embeddings | Can be precomputed            | Not independently precomputed for query scoring |
| Best use            | Search millions of chunks     | Rerank a smaller candidate set                  |
| Typical role        | Initial retrieval             | Final ranking                                   |

The cross-encoder is usually more computationally expensive because it evaluates each query-document pair separately.

That is why we don't normally run it against every document in the database.

# 4. Complete RAG pipeline with reranking

A typical production pipeline looks like this:

User query

Query embedding

Vector retrieval — Qdrant

Retrieve top 50 candidates

Cross-encoder reranker

Score and reorder 50 candidates

Select top 5–10 chunks

Apply relevance threshold and context budget

Context builder

LLM generates grounded answer with citations

One important distinction: if you're already using hybrid retrieval, the pipeline becomes:

```
                  User Query
                      |
             Query Processing
                      |
           +----------+----------+
           |                     |
       BM25 Search          Vector Search
           |                     |
           +----------+----------+
                      |
                 RRF Fusion
                      |
              Candidate Chunks
                      |
              Cross-Encoder
                 Reranker
                      |
               Top-K Chunks
                      |
                Context Builder
                      |
                     LLM
                      |
               Final Answer
```

RRF and reranking solve different problems:

- RRF combines results from multiple retrieval methods.
- Reranking evaluates and reorders the resulting candidates.

Neither replaces the other.

# 5. Implement a reranker using Python

Let's implement a cross-encoder reranker using the `sentence-transformers` library.

We'll use a BGE reranker model as an example.

Install:

```
pip install sentence-transformers torch
```

Model:

`BAAI/bge-reranker-base`

### Step 1: Create the reranker

```
from sentence_transformers import CrossEncoderclass CrossEncoderReranker:    def __init__(        self,        model_name: str = "BAAI/bge-reranker-base",        max_length: int = 512,    ):        self.model = CrossEncoder(            model_name,            max_length=max_length,        )    def rerank(        self,        query: str,        documents: list[dict],        top_k: int = 5,    ) -> list[dict]:        if not documents:            return []        pairs = [            (query, document["content"])            for document in documents        ]        scores = self.model.predict(            pairs,            batch_size=16,            show_progress_bar=False,        )
```

### Step 2: Pass vector search results

Imagine Qdrant has already returned these chunks:

```
retrieved_documents = [    {        "chunk_id": "chunk_001",        "content": (            "Investment portfolios should be reviewed "            "periodically to manage risk."        ),    },    {        "chunk_id": "chunk_002",        "content": (            "Compliance approval is mandatory before "            "sharing financial recommendations with customers."        ),    },    {        "chunk_id": "chunk_003",        "content": (            "Investment products carry market risks."        ),    },    {        "chunk_id": "chunk_004",        "content": (            "Employees must obtain approval from the "            "compliance department before sharing "            "investment recommendations."        ),    },]
```

### Step 3: Rerank

```
reranker = CrossEncoderReranker()query = (    "What approval is required before sharing "    "investment recommendations with customers?")ranked_results = reranker.rerank(    query=query,    documents=retrieved_documents,    top_k=2,)for result in ranked_results:    print(        result["chunk_id"],        result["rerank_score"],        result["content"],    )
```

The reranker will score the query-document pairs and return the highest-scoring candidates.

Scores are model-specific relevance scores, not calibrated probabilities. Do not assume a score of 0.98 means a 98% probability of correctness. Validate the model and thresholds on your own data.

# 6. How would you integrate reranking with Qdrant?

In your FastAPI + Qdrant RAG architecture, I would keep reranking as a separate service.

Suggested folder structure:

```
app/
├── retrieval/
│   ├── vector_search.py
│   ├── keyword_search.py
│   ├── fusion.py
│   ├── reranker.py
│   └── retrieval_service.py
│
├── generation/
│   ├── context_builder.py
│   └── answer_generator.py
│
└── api/
    └── routes/
        └── query.py
```

The retrieval service orchestrates the components:

```
class RetrievalService:    def __init__(        self,        vector_search,        keyword_search,        fusion,        reranker,    ):        self.vector_search = vector_search        self.keyword_search = keyword_search        self.fusion = fusion        self.reranker = reranker    async def retrieve(        self,        query: str,        tenant_id: str,        allowed_groups: list[str],        top_k: int = 5,    ):        # 1. Execute both retrieval strategies.        import asyncio        vector_results, keyword_results = await asyncio.gather(            self.vector_search.search(                query=query,                tenant_id=tenant_id,                allowed_groups=allowed_groups,                limit=30,            ),            self.keyword_search.search(                query=query,                tenant_id=tenant_id,                allowed_groups=allowed_groups,
```

This illustrates the orchestration pattern. The actual implementation needs concrete repository interfaces, authorization checks against authoritative permissions, and a reranker that executes in an appropriate worker or inference service.

Security detail: Tenant and ACL filters must already be enforced in both database retrieval operations. The authorization recheck is defense in depth, not a replacement for database filtering. Never allow unauthorized chunks to reach the reranker or context builder.

# 7. How many documents should you rerank?

This is an important production trade-off.

You don't want to retrieve 10,000 documents and rerank every one of them.

A reasonable starting experiment:

| Stage                          | Number of chunks      |
| ------------------------------ | --------------------- |
| BM25 retrieval                 | 30                    |
| Vector retrieval               | 30                    |
| After fusion and deduplication | Up to 50              |
| After reranking                | 5–10                  |
| Final LLM context              | Based on token budget |

Why?

Because reranking requires model inference for every query-document pair.

If you increase the candidate count:

- Retrieval recall may improve.
- Reranking computation increases.
- Query latency increases.
- Memory and GPU requirements increase.

You should benchmark different candidate counts, such as 20, 50, and 100.

# 8. Which reranking models should you consider?

Some options worth benchmarking:

| Model                   | Category      | Consideration             |
| ----------------------- | ------------- | ------------------------- |
| BAAI/bge-reranker-base  | Cross-encoder | Smaller reranking model   |
| BAAI/bge-reranker-large | Cross-encoder | Larger model to benchmark |
| Cohere Rerank           | Hosted API    | Managed reranking service |

The best choice depends on retrieval quality, inference latency, throughput, deployment constraints, and data privacy.

For sensitive financial compliance data, assess whether document text can be sent to an external reranking provider.

# 9. When should you NOT use a reranker?

A reranker isn't always necessary.

Consider these scenarios:

| Scenario                          | Recommendation                          |
| --------------------------------- | --------------------------------------- |
| Small document collection         | Vector search may be sufficient         |
| Very low-latency requirement      | Evaluate without reranking              |
| Exact identifier lookup           | Strong keyword search may be sufficient |
| Complex financial/legal questions | Reranking is worth evaluating           |
| Poor initial retrieval recall     | Fix retrieval first                     |
| High retrieval noise              | Reranking can help                      |

One important limitation:

A reranker cannot recover a document that the initial retrieval stage never retrieved.

If the correct document isn't in the candidate set, the reranker cannot rank it first.

Therefore, retrieval recall must be good before expecting reranking to improve the final result.

# 10. How do you evaluate whether reranking improves your RAG system?

Compare your system with and without reranking using the same evaluation dataset.

| Metric               | Without reranking | With reranking |
| -------------------- | ----------------- | -------------- |
| Recall\@50           | Measure           | Measure        |
| Precision\@5         | Measure           | Measure        |
| MRR\@5               | Measure           | Measure        |
| NDCG\@5              | Measure           | Measure        |
| Answer correctness   | Measure           | Measure        |
| Citation correctness | Measure           | Measure        |
| P95 latency          | Measure           | Measure        |
| Cost per query       | Measure           | Measure        |

Pay particular attention to:

- Precision\@5: Are the top five chunks more relevant?
- MRR: Does the first relevant chunk appear higher?
- NDCG: Is the overall ranking better?
- Answer correctness: Does the final answer actually improve?

Do not evaluate only the reranker score. Evaluate the complete RAG outcome.

# 11. Senior AI Engineer interview answer

If an interviewer asks:

"Why do you need a reranker after vector retrieval?"

You can answer:

> Vector retrieval uses embedding similarity to retrieve potentially relevant documents efficiently. However, embedding similarity doesn't always represent the actual relevance of a document to a particular question.
>
> I use a two-stage retrieval architecture.
>
> First, I use a bi-encoder embedding model to retrieve a broader candidate set from Qdrant. If required, I combine it with BM25 using Reciprocal Rank Fusion.
>
> Next, I use a cross-encoder reranker to evaluate each query-document pair jointly and reorder the candidates based on relevance.
>
> I then select the best chunks, construct the context, and pass it to the LLM.
>
> This improves the quality of retrieved context and can improve answer accuracy, while avoiding the computational expense of applying a cross-encoder to the entire document collection.
>
> I validate the improvement using Precision\@K, MRR, NDCG, answer correctness, and P95 latency.

## 12. Five follow-up interview questions

1. What is the difference between a bi-encoder and cross-encoder?
2. Why can't we use a cross-encoder directly for searching millions of documents?
3. What happens if the correct document is not returned by vector retrieval?
4. How do you reduce reranking latency in a system serving thousands of requests?
5. How do you select the relevance threshold for discarding low-quality chunks?

My recommendation for your production RAG architecture: Use hybrid retrieval → RRF → cross-encoder reranking → context construction → grounded generation. Treat reranking as a separately measurable retrieval stage, not as an automatic guarantee of better answers.