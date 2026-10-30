In a production RAG system, semantic search, keyword search, and hybrid search are three different approaches to finding relevant documents or chunks.

The key difference is how each approach determines whether a document is relevant to a user's query.

- Keyword search: Finds documents based on matching words or terms.
- Semantic search: Finds documents based on meaning and contextual similarity.
- Hybrid search: Combines both approaches to improve retrieval quality.

Let's understand them from a Senior AI Engineer perspective.

# 1. Keyword search

Keyword search retrieves documents by matching the actual words in the user's query against words in the indexed documents.

A common implementation is BM25 (Best Matching 25).

### Example

Suppose your knowledge base contains these documents:

| Document | Content                                       |
| -------- | --------------------------------------------- |
| Doc 1    | How to reset your banking password            |
| Doc 2    | Steps to recover your account credentials     |
| Doc 3    | How to calculate investment returns           |
| Doc 4    | Password security and authentication policies |

User query:

"What is the password reset procedure?"

Keyword search looks for matching terms such as:

- password
- reset
- procedure

It will likely retrieve Doc 1 and Doc 4, depending on their term relevance.

However, Doc 2 might be relevant even though it uses the word "recover" instead of "reset".

This is one limitation of keyword search.

### How BM25 works

BM25 calculates document relevance using factors such as:

1. Term frequency (TF): How frequently a query term appears in a document.
2. Inverse document frequency (IDF): How rare or informative that term is across the corpus.
3. Document length normalization: Prevents longer documents from automatically receiving higher scores.

A simplified BM25 formula:

\\[ BM25(D,Q)=\sum\_{q_i \in Q} IDF(q_i) \cdot \frac{f(q_i,D)(k_1+1)} {f(q_i,D)+k_1(1-b+b\frac{|D|}{avgdl})} \\]

You don't need to memorize the formula for an interview. Understand what it does: it ranks documents based on the importance and frequency of matching query terms.

### Python implementation using BM25

Install:

```
pip install rank-bm25
```

Implementation:

```
from rank_bm25 import BM25Okapidocuments = [    "How to reset your banking password",    "Steps to recover your account credentials",    "How to calculate investment returns",    "Password security and authentication policies",]tokenized_docs = [    document.lower().split()    for document in documents]bm25 = BM25Okapi(tokenized_docs)def keyword_search(query: str, top_k: int = 3):    tokens = query.lower().split()    scores = bm25.get_scores(tokens)    ranked = sorted(        enumerate(scores),        key=lambda item: item[1],        reverse=True    )    return [        {            "document": documents[index],            "score": float(score),        }        for index, score in ranked[:top_k]    ]
```

This is a minimal in-memory example. Production BM25 implementations should also handle tokenization, stemming, phrase matching, indexing, and filtering.

Where keyword search works particularly well:

- Financial policy numbers.
- Transaction IDs.
- Account identifiers.
- Exact product names.
- Legal clauses.
- Error codes.
- Specific technical terms.

For example, if someone searches for `POLICY-2026-104`, semantic search alone might not reliably retrieve the exact policy. Keyword search is very useful here.

# 2. Semantic search

Semantic search retrieves documents based on the meaning of the query rather than requiring exact word matches.

It typically uses an embedding model.

An embedding model converts text into a numerical vector.

For example:

```
"How can I reset my password?"
```

Might become:

```
[0.21, 0.45, -0.12, 0.87, ...]
```

Another sentence:

```
"How do I recover my account credentials?"
```

Will produce another vector.

Although the words are different, their semantic meanings are similar. Their vectors may therefore be close to each other in embedding space.

[Building a Chatbot with RAG: How Retrieval Meets the LLM | Yong Sen](https://images.openai.com/static-rsc-4/AUZunEa0t3gOuzwgtcj0KnGXUeyQk0OghM5JgpAmFhyXesbKuJLy7BkvP-AB74UVPPpfJlEPGsMhW8BFH5cexSgURah4ZKrBgm-wWPdljZBfU2yCfhEHNTqf4z6qGrPiR5WwdRsxamoeTdhzQusvRFdEFWmPYS7kfJDuIJUZkl8?purpose=inline)

[yongsen.space](https://www.yongsen.space/blog/building-a-rag-chatbot-llm-vector-search?utm_source=chatgpt.com)

### How semantic search works

The flow is:

1. Convert documents into chunks.
2. Generate embeddings for each chunk.
3. Store embeddings in a vector database such as Qdrant.
4. Convert the user's query into an embedding.
5. Calculate similarity between the query vector and document vectors.
6. Retrieve the most similar chunks.

### Similarity calculation

A common technique is cosine similarity.

\\[ Similarity(A,B)= \frac{A\cdot B}{\\|A\\|\\|B\\|} \\]

A higher cosine similarity generally indicates greater directional similarity between vectors.

### Python implementation using OpenAI and Qdrant

Install:

```
pip install openai qdrant-client
```

Example:

```
from openai import AsyncOpenAIfrom qdrant_client import QdrantClientfrom qdrant_client.models import Distance, VectorParams, PointStructimport asyncioimport osclient = AsyncOpenAI(    api_key=os.environ["OPENAI_API_KEY"])qdrant = QdrantClient(":memory:")COLLECTION = "documents"DIMENSION = 1536async def create_embedding(text: str):    response = await client.embeddings.create(        model="text-embedding-3-small",        input=text    )    return response.data[0].embeddingasync def index_documents(documents):    qdrant.create_collection(        collection_name=COLLECTION,        vectors_config=VectorParams(            size=DIMENSION,            distance=Distance.COSINE        )    )    points = []
```

This is a runnable-style demonstration, not a complete production indexing service. Production systems need persistent Qdrant storage, stable point IDs, batching, retries, versioning, and authorization filters.

Where semantic search works particularly well:

- Natural-language questions.
- Paraphrased questions.
- Conceptual queries.
- Questions where the document uses different terminology.
- Contextual knowledge retrieval.

For example:

User asks:

"How can an employee take time off?"

The document says:

"Employees are entitled to annual leave under the company's leave policy."

Semantic search can recognize that these sentences are conceptually related.

# 3. Hybrid search

Hybrid search combines keyword search and semantic search.

It uses both:

- BM25 for exact lexical matching.
- Vector similarity for semantic matching.

This is generally a stronger starting point for enterprise RAG systems because enterprise documents often contain both natural-language information and exact identifiers.

[Elasticsearch hybrid search in LangChain - Elasticsearch Labs](https://images.openai.com/static-rsc-4/lET0Lj5I0ia8467esp082WfKKRnL4YHhOmUlv2-tqnQdYIt4YwiqcqF_op5vYAuUGls9KFCT4Oeo4PXxFueaCh26v-szxCh---eQxesKo5TeXH-4ZtgczQaliQ-sAbiJCXJ4NdvOx2x_MdAm6CDLwTFWAAm2RsoEkXILctfK-Yo?purpose=inline)

[elastic.co](https://www.elastic.co/search-labs/blog/langchain-elasticsearch-hybrid-search?utm_source=chatgpt.com)

### Example

Suppose a user asks:

"What are the compliance requirements for policy FIN-2026-104?"

Your knowledge base contains:

| Document | Content                                              |
| -------- | ---------------------------------------------------- |
| Doc A    | Compliance requirements for FIN-2026-104             |
| Doc B    | Financial policy governance and compliance standards |
| Doc C    | Compliance requirements for FIN-2025-104             |
| Doc D    | General investment guidelines                        |

Keyword search is good at identifying the exact policy number.

Semantic search is good at identifying documents discussing compliance requirements, even when the wording differs.

Hybrid search combines these signals to find the most useful results.

## 4. How hybrid search combines results

A common approach is Reciprocal Rank Fusion (RRF).

Suppose the two search systems return:

BM25 results

| Document | Rank |
| -------- | ---- |
| Doc A    | 1    |
| Doc C    | 2    |
| Doc B    | 3    |

Semantic results

| Document | Rank |
| -------- | ---- |
| Doc B    | 1    |
| Doc A    | 2    |
| Doc D    | 3    |

RRF combines their rankings.

Formula:

\\[ RRF(d)=\sum\_{r \in R} \frac{1}{k+rank_r(d)} \\]

Where:

- \\(d\\) = document.
- \\(R\\) = retrieval result lists.
- \\(rank_r(d)\\) = document rank in a result list.
- \\(k\\) = smoothing constant, commonly 60.

For this example:

| Document | BM25 rank | Semantic rank | Combined RRF |
| -------- | --------- | ------------- | ------------ |
| Doc A    | 1         | 2             | 0.03252      |
| Doc B    | 3         | 1             | 0.03227      |
| Doc C    | 2         | —             | 0.01613      |
| Doc D    | —         | 3             | 0.01587      |

Doc A ranks first, narrowly ahead of Doc B.

The advantage is that RRF combines rankings without requiring BM25 and cosine similarity scores to be on the same scale.

### Python implementation of RRF

```
from collections import defaultdictdef reciprocal_rank_fusion(    result_lists: list[list[str]],    k: int = 60,    top_k: int = 5,):    scores = defaultdict(float)    for results in result_lists:        for rank, document_id in enumerate(results, start=1):            scores[document_id] += 1 / (k + rank)    ranked = sorted(        scores.items(),        key=lambda item: item[1],        reverse=True,    )    return ranked[:top_k]bm25_results = ["A", "C", "B"]semantic_results = ["B", "A", "D"]results = reciprocal_rank_fusion(    [bm25_results, semantic_results])print(results)
```

This demonstrates fusion of rankings. A production implementation should preserve document/chunk metadata and deduplicate results using stable IDs.

# 5. Complete comparison

| Feature                  | Keyword search       | Semantic search                | Hybrid search        |
| ------------------------ | -------------------- | ------------------------------ | -------------------- |
| Matching                 | Exact terms          | Meaning                        | Both                 |
| Technology               | BM25, inverted index | Embeddings, vector DB          | BM25 + vector DB     |
| Exact identifiers        | Excellent            | Less reliable                  | Excellent            |
| Paraphrasing             | Limited              | Excellent                      | Excellent            |
| Synonyms                 | Depends on analyzer  | Generally good                 | Good                 |
| Contextual understanding | Limited              | Stronger                       | Stronger             |
| Infrastructure           | Search index         | Embedding model + vector index | Both                 |
| Complexity               | Low–medium           | Medium                         | Higher               |
| Typical RAG use          | Exact lookup         | Conceptual retrieval           | Enterprise retrieval |

# 6. How I would implement this in a production RAG architecture

For an enterprise financial advisor copilot or compliance platform, I would use hybrid retrieval with a reranking stage.

Production retrieval pipeline

User query

"What are the restrictions under policy FIN-2026-104?"

Query processing

Normalize, classify, extract identifiers, establish tenant and ACL filters

Keyword search

BM25 / OpenSearch

Semantic search

Embeddings / Qdrant

RRF fusion

Merge and rank candidate chunks

Cross-encoder reranker

Evaluate query–chunk relevance more deeply

Context builder + LLM

Generate grounded response with citations

### Suggested retrieval configuration

These are starting points, not universal production settings.

| Stage                   | Initial configuration |
| ----------------------- | --------------------- |
| BM25 retrieval          | Top 30                |
| Semantic retrieval      | Top 30                |
| RRF fusion              | Top 30–40             |
| Cross-encoder reranking | Top 10–20             |
| Final context           | Top 5–8 chunks        |

Tune these values against a representative evaluation dataset and latency budget.

### Important production considerations

1. Authorization: Apply tenant and document ACL filters inside both retrieval paths. Never rely on the LLM to enforce access control.
2. Deduplication: Use stable chunk IDs to remove duplicate results.
3. Reranking: RRF combines rankings; a cross-encoder evaluates query–document relevance more directly.
4. Evaluation: Measure Recall\@K, MRR, NDCG\@K, citation correctness, and answer faithfulness.
5. Latency: Run keyword and semantic retrieval concurrently where possible.
6. Fallback: If the vector database is unavailable, keyword-only retrieval can be a degraded mode, provided the response clearly handles the reduced retrieval capability.

# 7. Senior AI Engineer interview questions

These are common follow-up questions you should be prepared to answer.

Q1. Why is semantic search alone insufficient for enterprise RAG?

Because semantic similarity does not guarantee exact matching. It can struggle with policy IDs, account numbers, transaction references, legal clause numbers, and other identifiers. Keyword search complements it.

Q2. Why use RRF instead of simply adding BM25 and cosine similarity scores?

BM25 and cosine similarity have different score distributions and scales. RRF combines rankings instead, avoiding the need to directly normalize incompatible scores.

Q3. What is the difference between RRF and reranking?

- RRF combines results from multiple retrieval systems.
- Reranking uses a more sophisticated relevance model, often a cross-encoder, to reorder the combined candidates.

Q4. Does hybrid search always improve RAG accuracy?

No. It can introduce irrelevant candidates, additional latency, and infrastructure complexity. It must be evaluated using real queries and labeled relevance data.

Q5. What would you do if keyword search retrieves the correct document but semantic search does not?

Keep the keyword result in the fused candidate set, investigate embedding quality and query formulation, and evaluate retrieval recall separately for each search path.

Q6. How would you debug a hybrid search system returning incorrect answers?

Trace each stage independently:

1. Inspect the original query.
2. Inspect BM25 results.
3. Inspect vector search results.
4. Inspect metadata and authorization filters.
5. Inspect RRF ranking.
6. Inspect reranker scores.
7. Inspect final context.
8. Inspect LLM output and citations.

My recommendation: For your enterprise RAG projects, use hybrid search + RRF + cross-encoder reranking as the baseline architecture. Then prove its value through retrieval evaluation rather than assuming that adding more retrieval components automatically improves accuracy.