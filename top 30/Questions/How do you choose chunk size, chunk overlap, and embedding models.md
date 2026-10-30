In a production RAG system, choosing chunk size, chunk overlap, and embedding models is one of the most important architecture decisions because all three directly affect retrieval accuracy, answer quality, latency, and cost.

As a Senior AI Engineer, I would not choose these values based on intuition or simply copy standard configurations such as 500 tokens with 100-token overlap.

I would use this approach:

1. Understand the document structure and query patterns.
2. Select an initial chunking strategy.
3. Select an embedding model based on retrieval requirements.
4. Build a representative evaluation dataset.
5. Benchmark different configurations.
6. Select the configuration that delivers the best retrieval quality within the latency and cost budget.

Let's break this down as we would in an enterprise RAG project.

# 1. How do you choose chunk size?

Chunk size determines how much text is stored in one retrievable unit.

Suppose you have a 20-page financial compliance policy.

You could divide it into:

- 100-token chunks.
- 300-token chunks.
- 500-token chunks.
- 800-token chunks.
- 1,000-token chunks.

Each chunk becomes an individual embedding and retrieval candidate.

[RAG Chatbot with Conversational Memory using LangChain | by Tuan Tran | Medium](https://images.openai.com/static-rsc-4/bSDtDDWCAWqWVU_Yxk4f7vhXN1nUYaOaqgoH-S0XBcZwF8wYg4ueDuQQ4TEjlq1_t7F2onejiXdBtskcQbLGt40wb-wdIyMMrp9_Puj0WO6Pwawh_y2Of2FiWuOM4DuQTtSI3t1ekoh_f8Td7Ho0incYroDfHBBIwC6BEYrxYzw?purpose=inline)

[tuanatran.medium.com](https://tuanatran.medium.com/rag-chatbot-with-conversational-memory-using-langchain-24b5886de244?utm_source=chatgpt.com)

### Why chunk size matters

Consider this document:

> Employees must obtain approval from the compliance department before sharing investment recommendations with external customers. All recommendations must include appropriate risk disclosures and must be reviewed according to the company's investment communication policy.

Suppose the user asks:

"What approval is required before sharing investment recommendations?"

Scenario A: Chunk size = 50 tokens

The relevant information might be split across two chunks.

Chunk 1:

> Employees must obtain approval from the compliance department before sharing investment recommendations...

Chunk 2:

> All recommendations must include appropriate risk disclosures...

This can create fragmented context, depending on where the boundary falls.

Scenario B: Chunk size = 500 tokens

The approval requirement and supporting information may stay together.

However, the chunk might also contain several unrelated policy sections.

This can reduce retrieval precision.

Scenario C: Chunk size = 1,000 tokens

You preserve more context, but the embedding represents a larger amount of text. A specific question may retrieve a large chunk containing substantial irrelevant material.

The trade-off is:

| Chunk size     | Advantages                | Disadvantages                             |
| -------------- | ------------------------- | ----------------------------------------- |
| 100–200 tokens | Precise retrieval         | Context fragmentation                     |
| 300–500 tokens | Good starting balance     | May split complex sections                |
| 600–800 tokens | More surrounding context  | More irrelevant content                   |
| 1,000+ tokens  | Preserves broader context | Less precise retrieval, more context cost |

These are rough starting ranges, not universal thresholds.

# 2. How do you choose chunk overlap?

Chunk overlap means repeating a small portion of text between consecutive chunks.

It helps prevent important information from being lost when a sentence or paragraph crosses a chunk boundary.

For example, consider a document:

```
Chunk 1:
Employees must obtain approval from the compliance
department before sharing investment recommendations.

Chunk 2:
All recommendations must include appropriate risk
disclosures and must be reviewed by the compliance team.
```

Without overlap, the relationship between the two sections might be weakened if the boundary separates related information.

With 50-token overlap, the end of Chunk 1 is repeated at the beginning of Chunk 2.

[RAG Chunking Strategies: How to Split Text for Better Retrieval](https://images.openai.com/static-rsc-4/IOalRuBzRC94ooAV6466Hi6_C-6XcN94FOLAVRuPhtF0UBlVM1_V5wbPDlWUabeUOpnIep-wPEaZtKfoGYJVO0KOkl4Sz6c2C7aOVJuWdOto0I8v9u3KK6zY_xFKKG1o53iDXs-0zVzNKx7or3Noroa5sD0n-9v3iS1T66LWCQY?purpose=inline)

[blog.kalvad.com](https://blog.kalvad.com/rag-deep-dive-series-chunking-strategies/?utm_source=chatgpt.com)

### What overlap should you use?

A reasonable initial range is 10–20% of chunk size.

| Chunk size | Initial overlap |
| ---------- | --------------- |
| 200 tokens | 20–40 tokens    |
| 300 tokens | 30–60 tokens    |
| 400 tokens | 40–80 tokens    |
| 500 tokens | 50–100 tokens   |
| 800 tokens | 80–120 tokens   |

For example, if your chunk size is 400 tokens, start with 60 tokens of overlap.

But don't assume more overlap is always better.

### Problems with excessive overlap

Suppose you have a 10,000-token document.

- With no overlap, you create fewer chunks.
- With 60-token overlap, you create some additional chunks.
- With 200-token overlap, you create considerably more repeated content.

Excessive overlap can cause:

1. Duplicate retrieval results.
2. Increased embedding API cost.
3. Increased vector database storage.
4. Repeated context passed to the LLM.
5. Reduced diversity among retrieved chunks.

My recommendation: Start with 400 tokens and 60 tokens of overlap for ordinary prose documents. Evaluate 300/40, 400/60, and 500/75 configurations rather than assuming one is optimal.

For structured financial documents, preserve section boundaries first. Overlap is not a substitute for understanding document structure.

# 3. How do you choose an embedding model?

An embedding model converts text into a vector representation.

For example:

```
Input:
"How can I recover my banking account?"
```

The embedding model produces something like:

```
[    0.123,    -0.234,    0.456,    0.789,    ...]
```

The vector represents semantic features of the input.

Your RAG system stores document embeddings in a vector database such as Qdrant.

When a user submits a question, the same compatible embedding model converts the query into a vector. The vector database searches for similar document vectors.

[Vector databases for embedding systems | OpenAI](https://images.openai.com/static-rsc-4/ElnbFDS7V-gMynfb2Cl1d_wDxaSDdbXma7D8cHUddiKneCz9iamAzf089UjDQDHuMmvGUyMINMZ9JwDT2DLOUHKOomTTaEwtIF3IX0_Vzd5EXub_04BCPULfOnZjBs743mBolSpmJ8Nx9q4dKYdxgXtBqDG-nOXBtpqWFWIPPvw?purpose=inline)

[campus.datacamp.com](https://campus.datacamp.com/courses/introduction-to-embeddings-with-the-openai-api/vector-databases?ex=1\&utm_source=chatgpt.com)

### What factors should influence model selection?

I would evaluate seven factors.

1\. Retrieval quality

Does the model retrieve the correct document for the user's question?

For example:

- Query: "What are the restrictions on investment recommendations?"
- Document: "Limitations governing financial advisory communications."

The model should understand the relationship despite different wording.

2\. Domain performance

Your embedding model should work well with the language used in your documents.

For financial services, this includes:

- Compliance terminology.
- Financial instruments.
- Investment policies.
- Regulatory terminology.
- Legal clauses.
- Product and account terminology.

3\. Language support

If your system supports English, Hindi, or other languages, check multilingual retrieval quality.

4\. Latency

Measure embedding generation time for both document ingestion and user queries.

5\. Cost

Calculate the cost of embedding the entire document collection and processing expected query volume.

6\. Vector dimensions and storage

A model producing 1,536-dimensional vectors requires a different storage footprint from one producing 3,072-dimensional vectors.

7\. Deployment and privacy

Evaluate whether the model is hosted by a provider or deployed inside your infrastructure. For sensitive financial documents, data residency and provider data-handling terms matter.

### Embedding models worth evaluating

These are representative candidates, not a claim that one is universally best.

| Model                  | Type        | When to evaluate                           |
| ---------------------- | ----------- | ------------------------------------------ |
| text-embedding-3-small | Hosted      | Cost-efficient baseline                    |
| text-embedding-3-large | Hosted      | Higher-capacity retrieval baseline         |
| BGE                    | Open-source | Self-hosted retrieval                      |
| E5                     | Open-source | General-purpose retrieval                  |
| GTE                    | Open-source | General-purpose retrieval and benchmarking |

I would shortlist two or three models and benchmark them against the same evaluation dataset.

Important: A larger embedding model does not automatically produce better retrieval for your particular documents. Measure it.

# 4. Implementing chunking in Python

Let's implement a practical chunking component that can be used in a RAG ingestion pipeline.

Install:

```
pip install langchain-text-splitters tiktoken
```

Use `cl100k_base` as the tokenizer for this example.

```
from dataclasses import dataclassfrom typing import Anyimport tiktokenfrom langchain_text_splitters import RecursiveCharacterTextSplitter@dataclass(frozen=True)class ChunkConfig:    chunk_size: int = 400    chunk_overlap: int = 60    separators: tuple[str, ...] = (        "\n\n",        "\n",        ". ",        " ",        "",    )class DocumentChunker:    def __init__(self, config: ChunkConfig):        if config.chunk_size <= 0:            raise ValueError("chunk_size must be positive")        if not 0 <= config.chunk_overlap < config.chunk_size:            raise ValueError("Invalid chunk_overlap")        self.config = config        self.encoding = tiktoken.get_encoding(            "cl100k_base"        )        self.splitter = RecursiveCharacterTextSplitter(
```

### Usage

```
config = ChunkConfig(    chunk_size=400,    chunk_overlap=60,)chunker = DocumentChunker(config)document_text = """Employees must obtain approval from the compliancedepartment before sharing investment recommendations.All recommendations must include appropriate riskdisclosures and must follow company investment policies."""chunks = chunker.split(    text=document_text,    metadata={        "tenant_id": "tenant_123",        "document_id": "policy_001",        "document_version": 1,        "page_number": 1,        "section": "Investment Compliance",    },)for chunk in chunks:    print(chunk)
```

This is a reusable baseline, but there are important production refinements:

- Split each page or logical section with its source metadata preserved.
- Avoid splitting tables, code blocks, and legal clauses arbitrarily.
- Generate deterministic chunk IDs using document version and chunk identity.
- Store the embedding model and chunking configuration versions.
- Validate chunk lengths and empty chunks.
- Avoid duplicating metadata across unrelated source sections.

The `RecursiveCharacterTextSplitter` attempts to respect the configured separators; it does not guarantee perfect semantic or sentence boundaries.

# 5. Why structure-aware chunking is better than fixed-size chunking

This is particularly important for financial, legal, and compliance documents.

Imagine this policy:

```
Section 1: Investment Recommendations

1.1 Approval Requirements

All investment recommendations must be approved
by the compliance department.

1.2 Risk Disclosures

Every recommendation must include appropriate
risk disclosures.

Section 2: Customer Communications

2.1 External Communications

Employees must follow the approved communication policy.
```

If you blindly split every 400 tokens, you might combine the end of Section 1 with the beginning of Section 2.

You could also separate a policy heading from the paragraph it describes.

Instead, I would use a hierarchy:

Document

Section 1: Investment Recommendations

Chunk 1 — Approval Requirements

Chunk 2 — Risk Disclosures

Section 2: Customer Communications

Chunk 3 — External Communications

Each chunk should preserve metadata such as:

```
{    "tenant_id": "tenant_123",    "document_id": "policy_001",    "document_version": 4,    "chunk_id": "policy_001_v4_chunk_002",    "section": "Investment Recommendations",    "subsection": "Risk Disclosures",    "page_number": 5,    "classification": "confidential",    "embedding_model": "text-embedding-3-small"}
```

Why is this important?

When the user asks:

"What are the approval requirements for investment recommendations?"

You can retrieve the relevant chunk and preserve its source section, page, and version for citation.

### What about tables?

Don't blindly split a financial table into independent text fragments.

For example:

| Investment | Risk level | Maximum allocation |
| ---------- | ---------- | ------------------ |
| Equity     | High       | 40%                |
| Bonds      | Medium     | 50%                |
| Cash       | Low        | 20%                |

A better chunk representation preserves the header and rows together:

```
Section: Portfolio Allocation Policy

Investment | Risk level | Maximum allocation
Equity     | High       | 40%
Bonds      | Medium     | 50%
Cash       | Low        | 20%
```

For large tables, split by rows or logical groups while repeating the header.

For complex tables, consider storing the original table separately and retrieving it through a structured table retrieval mechanism.

# 6. Implementing an embedding provider

Keep embedding generation separate from chunking.

This follows the Strategy/Adapter design principle: your ingestion service should depend on an embedding interface, not a specific provider.

A simple example:

```
from typing import Protocolfrom openai import AsyncOpenAIclass EmbeddingProvider(Protocol):    async def embed_documents(        self,        texts: list[str],    ) -> list[list[float]]:        ...    async def embed_query(        self,        text: str,    ) -> list[float]:        ...class OpenAIEmbeddingProvider:    def __init__(        self,        client: AsyncOpenAI,        model: str = "text-embedding-3-small",    ):        self.client = client        self.model = model    async def embed_documents(        self,        texts: list[str],    ) -> list[list[float]]:        if not texts:            return []
```

Usage:

```
import osfrom openai import AsyncOpenAIclient = AsyncOpenAI(    api_key=os.environ["OPENAI_API_KEY"])embedding_provider = OpenAIEmbeddingProvider(    client=client,    model="text-embedding-3-small",)vectors = await embedding_provider.embed_documents(    [        "Investment recommendations require approval.",        "All customer communications require review.",    ])query_vector = await embedding_provider.embed_query(    "Who approves investment recommendations?")
```

For production, this provider needs additional safeguards:

- Bounded batch sizes and concurrency.
- Provider timeouts and retry budgets.
- Exponential backoff with jitter for transient failures.
- Input token-limit validation.
- Response count and vector-dimension validation.
- Rate limiting and provider error handling.
- Model and configuration version tracking.

One critical rule: the document embedding model and query embedding model must be compatible. Don't index with one unrelated embedding model and search with another.

If you change embedding models, you generally need to re-embed your document collection and migrate the vector index. Do not mix incompatible vector spaces in the same search operation.

# 7. How do you evaluate chunk size and embedding models?

This is where the difference between a regular developer and a senior AI engineer becomes important.

You should not decide that 400 tokens is best just because it worked in one example.

Build a retrieval evaluation dataset.

For example:

| Query                                              | Expected document | Expected chunk |
| -------------------------------------------------- | ----------------- | -------------- |
| Who approves investment recommendations?           | Policy 001        | Chunk 3        |
| What are the risk disclosure requirements?         | Policy 001        | Chunk 5        |
| What is the maximum equity allocation?             | Policy 002        | Chunk 8        |
| What is the procedure for external communications? | Policy 003        | Chunk 2        |

For each query, record the correct supporting chunks.

Then benchmark different configurations.

### Example experiment

| Experiment | Chunk size | Overlap | Embedding model |
| ---------- | ---------- | ------- | --------------- |
| A          | 300        | 40      | Model A         |
| B          | 400        | 60      | Model A         |
| C          | 500        | 75      | Model A         |
| D          | 400        | 60      | Model B         |
| E          | 500        | 75      | Model B         |

Keep the other retrieval settings constant while comparing configurations.

### Metrics to evaluate

1\. Recall\@K

Measures whether the relevant chunk appears in the top K retrieved results.

\\[ Recall\@K = \frac{\text{Relevant chunks retrieved in top K}} {\text{Total relevant chunks}} \\]

2\. Precision\@K

Measures how many retrieved chunks are relevant.

\\[ Precision\@K = \frac{\text{Relevant chunks in top K}} {K} \\]

3\. MRR (Mean Reciprocal Rank)

Measures how high the first relevant result appears.

\\[ MRR = \frac{1}{N} \sum\_{i=1}^{N} \frac{1}{rank_i} \\]

4\. NDCG\@K

Measures ranking quality, taking the relevance grade and position of retrieved documents into account.

5\. Downstream generation quality

Evaluate:

- Answer correctness.
- Faithfulness to retrieved evidence.
- Citation accuracy.
- Abstention when evidence is insufficient.
- Completeness of the answer.

Also measure embedding cost, indexing time, query latency, and vector storage.

### Example experiment runner

You can build a benchmark around your retrieval service:

```
async def evaluate_configuration(    dataset,    chunker,    embedding_provider,    retrieval_service,):    total = len(dataset)    recall_hits = 0    reciprocal_rank_sum = 0.0    for example in dataset:        results = await retrieval_service.search(            query=example["query"],            top_k=5,        )        retrieved_ids = [            result.chunk_id            for result in results        ]        expected_ids = set(            example["relevant_chunk_ids"]        )        if expected_ids.intersection(retrieved_ids):            recall_hits += 1        for rank, chunk_id in enumerate(            retrieved_ids,            start=1,        ):            if chunk_id in expected_ids:                reciprocal_rank_sum += 1.0 / rank
```

This is a simplified benchmark. For a full evaluation, calculate Recall\@K using all relevant chunks, not just whether at least one was found. Also track graded relevance and evaluate on separate tuning and holdout datasets.

# 8. My recommended configuration for an enterprise financial RAG system

For your financial advisor copilot and compliance review architecture, I would begin with this configuration:

Initial configuration

Chunk size

# 400

Tokens

Chunk overlap

# 60

Tokens

Embedding baseline

text-embedding-3-small

Vector database

Qdrant

Retrieval configuration

- Hybrid retrieval: BM25 + semantic search.
- RRF for result fusion.
- Cross-encoder reranking.
- Retrieve 30 candidates from each search path initially.
- Rerank the fused candidates.
- Pass approximately 5–8 relevant chunks to the LLM, subject to the token budget.

These are starting values to benchmark, not final production settings.

## 9. Advanced technique: Parent-child chunking

For complex enterprise documents, I would also evaluate parent-child chunking.

The idea:

- Parent chunk: Larger section containing complete context.
- Child chunk: Smaller, more precise searchable section.

Example:

```
Parent:
Investment Compliance Policy
    |
    |--- Child 1: Approval Requirements
    |
    |--- Child 2: Risk Disclosures
    |
    |--- Child 3: External Communication
```

When a user asks about approval requirements:

1. Search using child chunks.
2. Identify the most relevant child.
3. Retrieve its parent section.
4. Build context using the child and necessary parent information.
5. Generate the answer with citations to the original source.

This gives you precise retrieval without losing surrounding context.

However, don't automatically send the entire parent document to the LLM. That can increase token consumption and introduce irrelevant content.

## 10. Interview-ready answer

If an interviewer asks, "How do you choose chunk size, overlap, and embedding models?", you can answer:

> I don't select chunk size and embedding models using fixed industry defaults. I start by analyzing document structure, document types, and user query patterns.
>
> For ordinary prose documents, I might start with 400 tokens and 60 tokens of overlap. I then evaluate different configurations using a representative retrieval dataset.
>
> I compare embedding models using Recall\@K, Precision\@K, MRR, and NDCG, while also measuring latency, embedding cost, and storage requirements.
>
> For structured financial and compliance documents, I prefer structure-aware chunking that preserves headings, sections, tables, and page metadata.
>
> I also evaluate parent-child retrieval when precise matching and broader contextual understanding are both required.
>
> Finally, I validate the complete RAG pipeline using answer correctness, faithfulness, citation accuracy, and abstention metrics. The final configuration is selected based on measurable retrieval quality and operational constraints.

The most important principle: Chunking and embedding selection are not independent decisions. You should evaluate them together because changing chunk boundaries changes the text represented by each embedding, which can change retrieval quality.