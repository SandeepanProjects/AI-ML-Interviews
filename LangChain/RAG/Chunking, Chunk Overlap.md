# Chunking, Chunk Overlap, Embeddings, and Document Metadata — Staff AI Engineer Guide

In a production RAG system, these concepts are tightly connected. If you choose the wrong chunk size, lose context during splitting, use an unsuitable embedding model, or fail to preserve metadata, your retrieval quality can suffer even when you use a powerful LLM.

Let's use your enterprise financial compliance platform as an example. Suppose a PDF contains this policy:

> Section 4.2 — Performance Claims Marketing materials must not imply guaranteed investment returns. Historical returns must include the applicable measurement period and required risk disclosures. Exceptions require approval from the Compliance team.

A user asks:

"Can our marketing material claim that this investment will generate guaranteed returns?"

Your RAG pipeline must retrieve the relevant policy, preserve its conditions, and cite the correct source page and version.

## 1. First, understand the complete flow

PDF / DOCX source

Original document and page structure

Parse and normalize

Extract text, headings, tables and page numbers

Chunking

Split the content into coherent retrieval units

Embedding

Convert each chunk into a numerical vector

Chunk text

Original content and metadata

Vector index

Embeddings for similarity search

Retrieve → Rerank → Generate answer with citations

Remember the distinctions:

- Chunk size determines how much content goes into each retrieval unit.
- Chunk overlap preserves context between neighboring chunks.
- Embedding represents the meaning of text numerically.
- Metadata tells the system where a chunk came from, which version it belongs to, and who is allowed to access it.

# 2. How do you determine chunk size?

Chunk size is the maximum amount of text you intend to put into one chunk. In most modern RAG systems, it is measured in tokens, not characters.

A token is a piece of text processed by a language model. A token may be a word, part of a word, punctuation, or another text unit. Token counts vary by tokenizer and language.

For example, a 500-token chunk is not necessarily 500 words.

## Three common approaches

[Paper close up.](https://images.openai.com/static-rsc-4/4aOTJl2-w629kI2vHecuTL7ZBixYXxHMJsa9LhLlJxeW7E--zqoDRe1wkWjFmvbU-TMpE877sfJ2QrFp9_-hjtmuc9hx_ovHMgYVEiV8mQiQ7KlGkZvzw64d_PyFIxegAioeFnGLikmlzaOFmnZDe13N6q2y3s_cnXsiWXGLIZOMlZS1tNCsH6LKtWxsZLZU?purpose=inline)

A. Small chunks — 100–250 tokens

Useful for precise facts, short policy clauses, FAQs and highly targeted retrieval.

Risk: a clause may lose its explanation, conditions or exceptions.

[Business Ethics Policy | PDF | Business Ethics | Conflict Of Interest](https://images.openai.com/static-rsc-4/vdaEtKDDnvOpXBKFR8mjSeceIN3JcOfPdJMJczq95zF1ARg0_Gb1Oy12vOkN6sAcC4sne-T07K-HLNjk_-S2jlfgvRwV_UgXeaVOdLZbxxJHboY-Uk7ouJVntd2cT7JNXoLB-yJ2sWczIVex4Aoc8oE4Ll7Bt57A9PCTp4M5-hU?purpose=inline)

B. Medium chunks — 300–700 tokens

A practical starting range for many text-heavy enterprise RAG systems. Often balances focused retrieval with sufficient context.

Risk: still needs evaluation for document structure and query type.

[National Grid 20-F 2020 Redacted Combined Document](https://images.openai.com/static-rsc-4/6xkVHq0qSIyNGo3QUCC5vpH7EM00h8sNR9wZWbWVRzPQnskzPriqNwBBNup9kZCdx_5ijfTp8HxFIhNUtzvSR86shJARzRAyriFr_tmzVyrGxkO4w-XeAhPFBV5aWIAaOCEvyYisSZ67vztvoxVEmd6N6VM8OTTvDGKnkq8Kb4s?purpose=inline)

C. Large chunks — 800–1,500+ tokens

Useful when the answer depends on a longer explanation, related paragraphs or a complete section.

Risk: unrelated content can dilute retrieval relevance and consume more context.

These are starting ranges, not universal standards. Some systems perform better with much smaller chunks, while long-form legal or technical content may benefit from larger parent sections.

## What factors determine the right size?

### 1. Document structure

For a financial policy, the following is a bad split:

```
Chunk 1:
Marketing materials must not imply guaranteed
```

```
Chunk 2:
investment returns. Exceptions require approval
from the Compliance team.
```

The meaning is fragmented.

A better chunk contains the complete rule, including its exception:

```
Section 4.2 — Performance Claims

Marketing materials must not imply guaranteed
investment returns. Exceptions require approval
from the Compliance team.
```

Prefer splitting at meaningful boundaries such as headings, paragraphs, policy clauses and table rows before using arbitrary token boundaries.

### 2. Query type

Different questions need different amounts of context.

| User question                                           | Likely strategy                                       |
| ------------------------------------------------------- | ----------------------------------------------------- |
| "What is the annual fee?"                               | Small, precise chunk                                  |
| "What disclosures are required for historical returns?" | Medium chunk containing the rule and conditions       |
| "Summarize the entire risk policy."                     | Larger parent section or multiple retrieved chunks    |
| "Compare all fees in this table."                       | Table-aware chunks preserving column headers and rows |

### 3. Embedding model limits

The embedding model has a maximum input length. If the input exceeds that limit, the provider may reject it, truncate it, or require you to split it.

The embedding model's input limit is not the same as the LLM's context window. A model that can generate answers using a very long context does not necessarily accept that same amount of text as one embedding input.

### 4. Retrieval evaluation

Ultimately, you determine chunk size experimentally.

Start with a baseline such as 400 tokens and 50 tokens of overlap. Then compare 200, 400, 700 and 1,000-token configurations using the same evaluation questions and retrieval settings.

Measure:

- Recall\@k: Did the relevant evidence appear in the top k results?
- Precision\@k: How many of the retrieved results were relevant?
- MRR or nDCG: How highly did the system rank relevant evidence?
- Citation accuracy: Does the final answer cite the right source and page?
- Latency and cost: How much indexing and query-time work is required?

Do not choose a chunk size solely because the generated answers sound better on a handful of examples.

## 3. What is chunk overlap?

Chunk overlap is the portion of text repeated between adjacent chunks.

Suppose we have 12 tokens and a chunk size of 5 tokens.

Without overlap:

Original text

A B C D E F G H I J K L

Chunk 1: `A B C D E`

Chunk 2: `F G H I J`

Chunk 3: `K L`

With an overlap of 2 tokens:

Chunk 1: `A B C D E`

New content: A–E

Chunk 2: `D E F G H`

Repeated: D, E · New: F, G, H

Chunk 3: `G H I J K`

Repeated: G, H · New: I, J, K

Chunk 4: `J K L`

Repeated: J, K · New: L

This toy example illustrates the principle; real splitters usually try to respect sentence and paragraph boundaries.

### Why is overlap useful?

Imagine the source text says:

> An advisor must disclose all applicable fees before recommending an investment. This requirement applies even when the investment is described as low risk.

If the split occurs between these sentences, a query about the low-risk exception might retrieve only the second sentence, without the original disclosure requirement.

Overlap helps retain neighboring context.

### How much overlap should you use?

A reasonable starting point is often 10–20% of chunk size, but you must validate this for your document collection.

For example:

| Chunk size   | Illustrative overlap |
| ------------ | -------------------- |
| 200 tokens   | 20–40 tokens         |
| 400 tokens   | 40–80 tokens         |
| 700 tokens   | 70–140 tokens        |
| 1,000 tokens | 100–200 tokens       |

Overlap does not have to be fixed. Structure-aware chunking can use paragraph boundaries, and a document parser may retain the entire section heading without duplicating arbitrary amounts of text.

### Code: calculate the chunk stride

For a fixed-window splitter:

\\[ \text{stride} = \text{chunk size} - \text{overlap} \\]

For a 400-token chunk with 60-token overlap, the stride is 340 tokens.

```
def calculate_stride(    chunk_size: int,    chunk_overlap: int,) -> int:    if chunk_size <= 0:        raise ValueError("chunk_size must be positive")    if not 0 <= chunk_overlap < chunk_size:        raise ValueError("Invalid overlap")    return chunk_size - chunk_overlapprint(calculate_stride(400, 60))# 340
```

The stride is the distance between consecutive chunk starting positions in a fixed-token-window algorithm.

### The hidden cost of overlap

If every chunk has 400 tokens and you use 80 tokens of overlap, the approximate stride is 320 tokens.

For long documents, the number of chunks is approximately:

\\[ N \approx \frac{T-O}{C-O} \\]

where:

- \\(T\\) = total token count
- \\(C\\) = chunk size
- \\(O\\) = overlap

This is an approximation that assumes a long document and fixed windows.

For 100,000 tokens, a 400-token chunk and 80-token overlap produce roughly 313 chunks, compared with 250 non-overlapping chunks. That is approximately 25% more chunks to embed and index.

Excessive overlap can cause:

- Higher embedding and storage costs.
- Duplicate search results.
- Lower diversity in the retrieved evidence.
- More redundant text in the LLM context.

Staff-level principle: overlap is a boundary-recovery mechanism, not a substitute for good chunk boundaries.

## 4. What happens if chunks are too small or too large?

Chunk size creates a trade-off between precision and context.

## Too-small chunks

Context fragmentation

Example: 50–100 tokens

Problems

- Conditions and exceptions get separated from the main rule.
- A query may retrieve a sentence that is meaningless without its heading.
- More chunks must be embedded, stored and searched.
- The retriever may return many near-duplicate fragments.

When they can work: FAQs, short clauses, atomic facts, or systems that retrieve a larger parent section after finding a precise child chunk.

## Too-large chunks

Retrieval dilution

Example: 2,000–4,000 tokens

Problems

- One chunk may discuss many unrelated topics.
- The embedding represents a mixture of ideas.
- Relevant evidence may be harder to rank against focused chunks.
- Retrieved context consumes more of the LLM's token budget.
- A large chunk can include irrelevant or conflicting information.

When they can work: Long explanations, broad section summaries, or parent-context retrieval when the answer genuinely depends on several paragraphs.

A critical nuance: large chunks do not automatically have bad embeddings, and small chunks do not automatically have good precision. The result depends on the embedding model, document structure, query distribution, and retrieval method.

### A practical example

Suppose the policy contains:

1. Fee disclosure rules.
2. Investment risk rules.
3. Marketing performance claims.
4. Approval exceptions.

A 2,500-token chunk containing all four topics may match a query about performance claims, but its vector represents the combined content. A 300-token chunk containing the performance-claims section may be more focused.

On the other hand, if the exception appears in the following paragraph, an overly small chunk may retrieve the claim but miss the exception.

This is why chunking is not just text splitting. It is designing the units of evidence that your retrieval system can find and your LLM can reason over.

## 5. Implement chunking properly

Let's build a reusable chunking module with validation, token counting and provenance.

Install the dependencies:

```
pip install langchain-text-splitters tiktoken
```

Create `app/ingestion/chunker.py`:

```
from dataclasses import dataclassfrom typing import Sequenceimport tiktokenfrom langchain_text_splitters import (    RecursiveCharacterTextSplitter,)@dataclass(frozen=True)class SourcePage:    page_number: int    text: str    section: str | None = None@dataclass(frozen=True)class Chunk:    chunk_id: str    document_id: str    text: str    page_start: int    page_end: int    section: str | None    token_count: intclass DocumentChunker:    def __init__(        self,        chunk_size: int = 400,        chunk_overlap: int = 60,    ) -> None:        if chunk_size <= 0:            raise ValueError("chunk_size must be positive")        if not 0 <= chunk_overlap < chunk_size:            raise ValueError("Invalid chunk_overlap")        self.encoder = tiktoken.get_encoding("cl100k_base")        self.splitter = (            RecursiveCharacterTextSplitter.from_tiktoken_encoder(                encoding_name="cl100k_base",                chunk_size=chunk_size,                chunk_overlap=chunk_overlap,                separators=[                    "\n\n",                    "\n",                    ". ",                    " ",                    "",                ],            )        )    def chunk_pages(        self,        document_id: str,        pages: Sequence[SourcePage],    ) -> list[Chunk]:        result: list[Chunk] = []        index = 0        for page in pages:            text = page.text.strip()            if not text:                continue            for piece in self.splitter.split_text(text):                piece = piece.strip()                if not piece:                    continue                result.append(                    Chunk(                        chunk_id=f"{document_id}:chunk:{index}",                        document_id=document_id,                        text=piece,                        page_start=page.page_number,                        page_end=page.page_number,                        section=page.section,                        token_count=len(                            self.encoder.encode(piece)                        ),                    )                )                index += 1        return result
```

Example usage:

```
pages = [    SourcePage(        page_number=1,        section="Performance Claims",        text=(            "Marketing materials must not imply guaranteed "            "investment returns. Historical returns must "            "include the measurement period and applicable "            "risk disclosures."        ),    ),    SourcePage(        page_number=2,        section="Performance Claims",        text=(            "Exceptions require prior approval from "            "the Compliance team."        ),    ),]chunker = DocumentChunker(    chunk_size=400,    chunk_overlap=60,)chunks = chunker.chunk_pages(    document_id="policy-123-v3",    pages=pages,)for chunk in chunks:    print(        chunk.chunk_id,        chunk.page_start,        chunk.token_count,        chunk.text,    )
```

Important limitation: this example chunks each page independently. That is a useful starting point for preserving page citations, but it does not create overlap across page boundaries, and the exception on page 2 may remain separate from the rule on page 1.

For production policy ingestion, I would first reconstruct sections across pages using heading and layout information. I would then chunk those sections while tracking the exact source-page range of every chunk. I would also use dedicated handling for tables, lists and numbered clauses.

The `cl100k_base` tokenizer is an example tokenizer, not a guarantee of the exact token count used by every embedding model. Use the tokenizer appropriate to your selected model when available, and enforce the provider's actual input limits.

# 6. What does it mean to embed chunks?

An embedding is a numerical representation of text, designed so that text with related meanings can often be found near each other in vector space.

Consider these two sentences:

- "The investment has a high risk of losing capital."
- "This portfolio may experience substantial losses."

They use different words but express related concepts. A good embedding model may map their vectors close together.

By contrast:

\- "The office is closed on Sunday."

is likely to be less similar to those investment-risk sentences.

### Conceptual example

Illustrative 3-dimensional vectors only — real embedding models commonly produce hundreds or thousands of dimensions.

Chunk A: "High investment risk"

[0.82, 0.11, -0.34]

Chunk B: "Potential for substantial portfolio losses"

[0.79, 0.15, -0.30]

Chunk C: "Office opening hours"

[-0.12, 0.88, 0.57]

The example values are invented to illustrate the concept, not output from a real model.

The embedding model does not store the original meaning as a human-readable definition. It produces a vector that is useful for similarity-based retrieval.

## 7. How to generate embeddings in Python

For an enterprise application, define an abstraction around the embedding provider so you can switch between hosted models and local models without rewriting your ingestion pipeline.

### Step 1: Define a protocol

Create `app/embeddings/protocols.py`:

```
from typing import Protocolclass EmbeddingProvider(Protocol):    async def embed_documents(        self,        texts: list[str],    ) -> list[list[float]]:        ...    async def embed_query(        self,        text: str,    ) -> list[float]:        ...
```

Why separate `embed_documents` and `embed_query`? Some embedding models use different instructions or encoding modes for documents and queries. Even when both methods use the same underlying endpoint, the interface makes the intended use explicit.

### Step 2: Implement an OpenAI provider

Install the SDK:

```
pip install openai
```

Create `app/embeddings/openai_provider.py`:

```
from openai import AsyncOpenAIclass OpenAIEmbeddingProvider:    def __init__(self, model: str) -> None:        self.client = AsyncOpenAI()        self.model = model    async def _embed(        self,        texts: list[str],    ) -> list[list[float]]:        if not texts:            return []        response = await self.client.embeddings.create(            model=self.model,            input=texts,        )        items = sorted(            response.data,            key=lambda item: item.index,        )        vectors = [item.embedding for item in items]        if len(vectors) != len(texts):            raise RuntimeError(                "Embedding response count mismatch"            )        dimensions = {len(vector) for vector in vectors}        if len(dimensions) != 1:            raise RuntimeError(                "Inconsistent embedding dimensions"            )        return vectors    async def embed_documents(        self,        texts: list[str],    ) -> list[list[float]]:        return await self._embed(texts)    async def embed_query(        self,        text: str,    ) -> list[float]:        vectors = await self._embed([text])        return vectors[0]
```

Configure your API key using a secret manager or environment configuration. Choose a supported embedding model explicitly and record its name and configuration with the indexed data.

This implementation demonstrates the provider boundary. Before using it at scale, add token-aware batching, rate limits, timeouts, bounded retries, cancellation handling and metrics.

### Step 3: Embed the chunks

```
from app.embeddings.openai_provider import (    OpenAIEmbeddingProvider,)async def embed_chunks(chunks):    provider = OpenAIEmbeddingProvider(        model="YOUR_SUPPORTED_EMBEDDING_MODEL"    )    texts = [chunk.text for chunk in chunks]    vectors = await provider.embed_documents(texts)    return [        {            "chunk_id": chunk.chunk_id,            "text": chunk.text,            "vector": vector,            "token_count": chunk.token_count,        }        for chunk, vector in zip(            chunks,            vectors,            strict=True,        )    ]
```

Replace the placeholder with the embedding model you have selected. The exact vector dimensions depend on the model and its configuration.

The key mapping is:

\\[ \text{Chunk}\_i \longrightarrow \text{Embedding}\_i \\]

If you have 500 chunks, you need 500 corresponding vectors. Preserve that association when batching or processing results concurrently.

### Step 4: What happens during retrieval?

At query time, the user's question is embedded using the compatible query-embedding configuration.

For example:

```
question = (    "Can marketing materials promise guaranteed returns?")query_vector = await provider.embed_query(question)
```

The vector database compares this query vector with the stored chunk vectors, typically using cosine similarity, dot product or Euclidean distance, depending on the model and index configuration.

Conceptually:

\\[ \operatorname{cosine}(q,d) = \frac{q\cdot d}{\\|q\\|\\|d\\|} \\]

where \\(q\\) is the query vector and \\(d\\) is a document-chunk vector.

A high similarity score is a signal of semantic relevance, not proof that a policy supports a particular answer. You still need metadata filtering, optional reranking, evidence validation and citation generation.

## 8. What is document metadata?

Metadata is information about a document or chunk, rather than the main content itself.

Think of a chunk as a paragraph in a book. The paragraph is the content; the book title, chapter, page number, publication date and access restrictions are metadata.

In a RAG platform, metadata helps answer questions such as:

- Which PDF did this chunk come from?
- Which page contains the cited policy?
- Is this the current policy version?
- Does this user have permission to retrieve the document?
- Which embedding model created this vector?
- Has the source document changed?
- Is this content a table, paragraph or heading?

### Document-level metadata vs chunk-level metadata

| Document-level metadata     | Chunk-level metadata              |
| --------------------------- | --------------------------------- |
| Document ID                 | Chunk ID                          |
| Original filename           | Chunk index                       |
| Tenant ID                   | Source page or page range         |
| Document version            | Section heading                   |
| Source URI                  | Content type                      |
| SHA-256 file hash           | Chunk content hash                |
| Owner / access policy       | Inherited access labels           |
| Created and effective dates | Embedding model and index version |

Some fields, such as `tenant_id`, `document_version` and `source_uri`, can appear at both levels for convenience. You should still maintain a canonical document record to avoid inconsistent metadata.

### Example: a compliance-policy chunk

```
chunk_record = {    "chunk_id": "policy-123:v3:chunk:17",    "document_id": "policy-123",    "tenant_id": "tenant-42",    "document_version": "3",    "filename": "investment_policy.pdf",    "page_start": 8,    "page_end": 8,    "section": "Performance Claims",    "content_type": "paragraph",    "effective_date": "2026-09-01",    "parser_version": "pdf-parser-v2",    "chunker_version": "chunker-v3",    "embedding_model": "your-selected-model",    "access_labels": ["compliance"],    "text": (        "Marketing materials must not imply "        "guaranteed investment returns."    ),}
```

These are illustrative values.

The `text` field contains the content. The other fields provide identity, provenance, filtering and processing information.

### Why metadata matters in real systems

1\. Citation accuracy

The LLM needs to know which document and page support a claim. Without provenance, you may retrieve relevant text but be unable to provide a trustworthy citation.

2\. Access control

Suppose Tenant A has confidential policy documents that Tenant B must never access. Every retrieval operation must enforce the appropriate tenant and document permissions. Metadata can support these filters, but metadata by itself is not a security boundary: enforce authorization in trusted server-side code and at the storage layer where appropriate.

3\. Version management

Suppose policy version 2 prohibits a particular claim, while version 3 modifies the rule. You must be able to identify which version is effective for the requested review date and prevent stale chunks from silently remaining authoritative.

4\. Reprocessing

If you change your chunking strategy from 400 tokens to 600 tokens, you need to identify which chunks were produced by the old configuration and rebuild the appropriate index.

5\. Filtering and hybrid search

You might search only current policies, restrict results to a business unit, or filter by document type before ranking candidates.

## 9. Store embeddings and metadata together

For your FastAPI + PostgreSQL/pgvector platform, you can store chunk text, metadata and embeddings in PostgreSQL. Alternatively, PostgreSQL can store canonical document records while Qdrant stores vectors and retrieval payloads.

You do not need both vector databases unless you have a concrete reason to operate them together.

Here is a simplified pgvector table design:

```
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    document_version INTEGER NOT NULL,

    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section TEXT,
    content TEXT NOT NULL,

    content_hash TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,

    embedding VECTOR(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Important: `VECTOR(1536)` is only an example. The dimension must match your chosen embedding model's output. If your model returns a different number of dimensions, change the schema accordingly. Model changes that alter dimensions generally require a separate migration or reindexing strategy.

A simplified retrieval query might look like:

```
SELECT
    chunk_id,
    document_id,
    page_start,
    page_end,
    section,
    content,
    embedding <=> :query_embedding AS distance
FROM document_chunks
WHERE tenant_id = :authorized_tenant_id
ORDER BY embedding <=> :query_embedding
LIMIT 10;
```

This illustrates cosine distance using pgvector's `<=>` operator. The query parameter should be supplied through your database driver's parameter binding, not interpolated into SQL text.

For production, this query also needs to enforce document-level access, active-version rules and any applicable effective-date constraints. An index such as HNSW or IVFFlat may improve performance, but it needs to be selected and tuned based on the dataset, recall target and query workload.

For high-security environments, do not rely on a user-supplied tenant ID. Derive authorization scope from the authenticated principal, and consider PostgreSQL row-level security as an additional layer of protection.

## 10. Put everything together in one ingestion service

Here is a simplified service showing how the stages fit together. It assumes the parser, chunker, embedding provider and repository are implemented separately.

```
class DocumentIngestionService:    def __init__(        self,        parser,        chunker,        embedding_provider,        repository,    ):        self.parser = parser        self.chunker = chunker        self.embedding_provider = embedding_provider        self.repository = repository    async def ingest(self, document):        # 1. Parse the source PDF.        parsed_pages = await self.parser.parse(            document.storage_path        )        # 2. Validate extraction quality.        if not parsed_pages:            raise ValueError("Document contains no pages")        usable_pages = [            page for page in parsed_pages            if page.text.strip()        ]        if not usable_pages:            raise ValueError(                "No usable text extracted from document"            )        # 3. Split into provenance-aware chunks.        chunks = self.chunker.chunk_pages(            document_id=document.document_id,            pages=usable_pages,        )        if not chunks:            raise ValueError("No chunks were produced")        # 4. Generate embeddings in bounded batches.        vectors = []        batch_size = 64        for start in range(0, len(chunks), batch_size):            batch = chunks[start:start + batch_size]            batch_vectors = (                await self.embedding_provider.embed_documents(                    [chunk.text for chunk in batch]                )            )            if len(batch_vectors) != len(batch):                raise RuntimeError(                    "Embedding count mismatch"                )            vectors.extend(batch_vectors)        # 5. Persist versioned content and metadata.        records = []        for chunk, vector in zip(            chunks,            vectors,            strict=True,        ):            records.append({                "chunk_id": chunk.chunk_id,                "document_id": document.document_id,                "tenant_id": document.tenant_id,                "document_version": document.version,                "page_start": chunk.page_start,                "page_end": chunk.page_end,                "section": chunk.section,                "content": chunk.text,                "token_count": chunk.token_count,                "embedding": vector,            })        await self.repository.upsert_chunks(records)        # Publication should be a separate validated,        # version-aware operation.        return {            "document_id": document.document_id,            "chunks_created": len(records),            "status": "indexed_pending_validation",        }
```

This is a teaching example, not a complete production implementation. It omits the actual parser and repository implementations, durable job handling, content-hash validation, authorization details, retries and transactional publication.

A robust design should also verify that embeddings are finite, non-empty vectors with the expected dimensions. If the persistence stage fails partway through, the document version should not become active.

### Why separate publication from ingestion?

Imagine a 100-page compliance policy. The first 80 pages parse and embed successfully, but embedding fails on page 81.

You should not publish a partial policy as if it were complete.

Instead:

1. Create a new ingestion job and document version.
2. Parse, chunk and embed all required content.
3. Validate chunk counts, provenance, vector dimensions and quality.
4. Confirm that all required records are persisted.
5. Activate the new version atomically.
6. Retain or retire the previous version according to your versioning policy.

This approach is much safer than deleting the old index at the beginning and hoping the new ingestion succeeds.

## 11. How should you tune chunking in a real project?

I would run a controlled experiment rather than pick one configuration permanently.

## Chunking configuration explorer

Adjust the configuration to see its approximate indexing implications. This is a fixed-window estimate, not a benchmark of actual retrieval quality.

Chunk size

### 400 tokens

100

600

1200

Overlap

### 60 tokens

0

Medium

High

15% overlap relative to chunk size

Total source tokens

10,000 tokens — small document collection

100,000 tokens — medium collection

1,000,000 tokens — larger collection

Approx. chunks

# 294

Stride

# 340

tokens per window step

Actual chunk counts depend on paragraph boundaries, tokenization, page breaks, and the splitter's behavior on short final chunks. Overlap does not guarantee that all chunks are full-sized.

For a proper evaluation, create a fixed set of questions with known supporting passages. Run each candidate chunking configuration through the same embedding model, retrieval method, top-k setting and reranker.

Record:

- Whether the correct passage is retrieved.
- Whether the full rule and its exception are available.
- Whether the citation points to the correct page.
- Index size, ingestion duration and query latency.
- End-to-end answer correctness and groundedness.

Do not change the embedding model, chunk size and reranking strategy simultaneously during the initial experiment; otherwise, you cannot identify which change improved the result.

## 12. Staff-level interview answer

Question: How do you select chunk size, overlap and embedding strategy for a production RAG system?

> I start by understanding the document structure and the types of queries the application must support. I prefer structure-aware chunking over arbitrary fixed windows, preserving headings, policy clauses, table headers and source-page provenance.
>
> I establish a baseline chunk size and modest overlap, then evaluate smaller and larger alternatives against a labeled retrieval dataset. I measure recall\@k, ranking quality, citation accuracy, answer groundedness, indexing cost and latency.
>
> I select the embedding model based on domain retrieval performance, supported languages, input limits, latency, cost and deployment constraints. I embed chunks in bounded batches, preserve the mapping between each chunk and its vector, and store versioned metadata with the original content.
>
> For production reliability, I use idempotent ingestion, tenant-aware access controls, versioned indexes, quality gates and atomic publication. I monitor retrieval quality after deployment and reindex when parser, chunker or embedding-model changes require it.

The most important takeaway is that chunking, embedding and metadata are one retrieval-design problem, not three independent implementation tasks. The goal is to retrieve the right evidence, with enough context to interpret it correctly, and with sufficient provenance to verify and authorize its use.