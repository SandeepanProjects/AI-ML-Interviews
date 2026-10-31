# How would you handle a RAG system returning an incorrect answer when the answer exists in the document?

Senior AI Engineer / Staff Engineer interview answer

If the answer exists in the source document but the RAG system returns an incorrect answer, I would treat this primarily as a retrieval and grounding investigation rather than immediately changing the LLM.

My first objective would be to identify exactly where information is getting lost or corrupted in the RAG pipeline.

A RAG system has several stages:

1. Document ingestion
2. Document parsing
3. Chunking
4. Embedding generation
5. Vector database storage
6. Query embedding
7. Retrieval
8. Reranking
9. Context construction
10. Prompt generation
11. LLM generation
12. Output validation

An error in any one of these stages can produce an incorrect answer.

## 1. First, identify where the failure occurs

Consider this example.

Source document:

```
Fidelity's investment policy states that:

The maximum equity allocation for a conservative
portfolio is 40%.

The maximum equity allocation for a moderate
portfolio is 60%.

The maximum equity allocation for an aggressive
portfolio is 80%.
```

User asks:

"What is the maximum equity allocation for a conservative portfolio?"

Expected answer: 40%.

But the RAG system responds:

"The maximum equity allocation is 60%."

I would investigate the pipeline in the following order.

RAG failure investigation

1. Document parsing

   Was the original information extracted correctly?
2. Chunking

   Was the conservative portfolio information separated from its context?
3. Embedding and indexing

   Were correct embeddings generated and stored?
4. Retrieval

   Did the retriever actually retrieve the chunk containing 40%?
5. Reranking

   Did the reranker incorrectly prioritize the 60% chunk?
6. Context construction

   Did the prompt contain the correct retrieved information?
7. LLM generation

   Did the LLM ignore or misinterpret the correct context?
8. Output validation

   Did the system detect that the answer was unsupported?

The key principle: I would inspect intermediate outputs at every stage instead of blindly changing the prompt or increasing top-K.

## 2. Verify document parsing

The first thing I would check is whether the parser extracted the correct information from the original document.

This is especially important with PDFs containing:

- Tables
- Multiple columns
- Headers and footers
- Scanned pages
- Complex layouts
- Financial statements

For example, the original PDF might contain:

| Portfolio    | Maximum equity |
| ------------ | -------------- |
| Conservative | 40%            |
| Moderate     | 60%            |
| Aggressive   | 80%            |

But the parser might extract:

```
Conservative Moderate Aggressive
40% 60% 80%
```

The information is technically present, but the relationship between each portfolio and its allocation is lost.

Solution: Use layout-aware PDF parsing, preserve table structures, and validate extracted content against the source document.

For example:

```
class DocumentParser:    async def parse(self, file_path: str):        pages = await self.extract_pages(file_path)        if not pages:            raise ValueError("Document extraction failed")        for page in pages:            if not page.text.strip():                raise ValueError(                    f"Empty extracted page: {page.page_number}"                )        return pages
```

In production, I would also retain page numbers, document IDs, table metadata and extraction quality scores.

## 3. Investigate chunking strategy

This is one of the most common causes of incorrect RAG answers.

Suppose we have this original document:

```
Conservative Portfolio

The maximum equity allocation for this portfolio
is 40%.

Moderate Portfolio

The maximum equity allocation for this portfolio
is 60%.
```

A poor chunking strategy might generate:

Chunk 1

```
Conservative Portfolio
```

Chunk 2

```
The maximum equity allocation for this portfolio
is 40%. Moderate Portfolio
```

Chunk 3

```
The maximum equity allocation for this portfolio
is 60%.
```

The relationship between the heading and the answer has been damaged.

The retriever might retrieve the wrong chunk because it lacks the required context.

### My solution

I would use structure-aware chunking rather than relying exclusively on fixed character-based chunking.

Example:

```
from langchain_text_splitters import RecursiveCharacterTextSplittersplitter = RecursiveCharacterTextSplitter(    chunk_size=400,    chunk_overlap=60,    separators=[        "\n\n",        "\n",        ". ",        " ",        ""    ])chunks = splitter.split_text(document_text)
```

For enterprise documents, I would improve this by:

- Preserving section headings.
- Keeping tables together where practical.
- Preserving paragraph boundaries.
- Maintaining page-level metadata.
- Avoiding chunks that separate questions from their answers.

A chunk should be semantically self-contained whenever possible.

## 4. Investigate embedding generation

After verifying parsing and chunking, I would inspect the embedding model.

An embedding model converts text into numerical vectors so semantically similar text can be retrieved.

For example:

```
Query:
Maximum equity allocation for conservative portfolio?

Document:
Conservative portfolio has a maximum equity
allocation of 40%.
```

These two texts should have relatively high semantic similarity.

However, embeddings can fail when:

- The document contains domain-specific terminology.
- The query uses different terminology from the document.
- The embedding model is not suitable for the domain.
- Important numerical or table relationships are poorly represented.
- The wrong embedding model or version is used during querying.

### Solution

I would evaluate embedding models against a representative dataset of real enterprise questions.

I would also verify that:

- The same compatible embedding model is used for ingestion and querying.
- The vector dimensions match.
- The embedding model version is recorded.
- Documents are re-embedded when the model changes.

Example:

```
from openai import AsyncOpenAIclass EmbeddingService:    def __init__(self, client: AsyncOpenAI):        self.client = client    async def embed(self, texts: list[str]) -> list[list[float]]:        response = await self.client.embeddings.create(            model="text-embedding-3-small",            input=texts        )        return [            item.embedding            for item in response.data        ]
```

I would not assume that changing the embedding model will automatically solve the problem. I would first measure retrieval quality using a labeled evaluation dataset.

## 5. Investigate retrieval: the most important step

Now I would inspect what the vector database actually returns.

Suppose the user asks:

```
What is the maximum equity allocation
for a conservative portfolio?
```

The vector database returns:

| Retrieved chunk             | Similarity |
| --------------------------- | ---------- |
| Moderate portfolio: 60%     | 0.91       |
| Aggressive portfolio: 80%   | 0.88       |
| Conservative portfolio: 40% | 0.76       |

The correct answer exists, but the retriever ranks it third.

If I retrieve only the top two chunks, the LLM never sees the correct answer.

This is a retrieval failure, not necessarily an LLM failure.

### Solution 1: Increase retrieval candidates

Instead of retrieving only two chunks, retrieve a larger candidate set.

```
async def retrieve_documents(    vector_store,    query: str,    top_k: int = 10):    results = await vector_store.similarity_search_with_score(        query=query,        k=top_k    )    return results
```

However, increasing top-K alone is not a complete solution. It can introduce irrelevant context and increase token costs.

### Solution 2: Use hybrid search

This is one of my preferred approaches for enterprise RAG.

Hybrid search combines:

- Semantic search: Finds conceptually similar information.
- Keyword search: Finds exact words, phrases, identifiers and numbers.

For example:

```
Query:
Conservative portfolio maximum equity allocation
```

Semantic search might find:

```
Maximum equity exposure for low-risk investors is 40%.
```

Keyword search might find:

```
Conservative portfolio: maximum equity allocation 40%.
```

Combining both improves retrieval in many enterprise use cases.

I would use Reciprocal Rank Fusion (RRF) to combine the rankings.

\\[ RRF(d)=\sum\_{r \in R}\frac{1}{k+rank_r(d)} \\]

Here, \\(k\\) is a ranking constant, and \\(rank_r(d)\\) is the rank of document \\(d\\) in a retrieval result.

### Solution 3: Add metadata filtering

For enterprise financial documents, I would also use metadata filtering.

Example:

```
filters = {    "tenant_id": "tenant_123",    "document_type": "investment_policy",    "document_version": "v3"}
```

This ensures that retrieval is restricted to the correct tenant and relevant document version.

Important: Tenant authorization must be enforced server-side. Never trust tenant IDs supplied directly by the user.

## 6. Introduce a reranking model

Even after hybrid retrieval, I might retrieve 20 candidate chunks.

The problem is that the initial retriever is optimized for candidate recall, not necessarily precise ranking.

I would introduce a cross-encoder reranker.

Retrieval architecture

User query

Vector search

Semantic candidates

BM25 search

Keyword candidates

RRF fusion

Merge candidate rankings

Cross-encoder reranker

Evaluate query-document relevance

Top 5 relevant chunks

LLM generation

A cross-encoder evaluates the query and each candidate document together.

Unlike bi-encoder retrieval, it can evaluate the relationship between the query and candidate text more precisely.

Example:

```
from sentence_transformers import CrossEncoderclass DocumentReranker:    def __init__(self):        self.model = CrossEncoder(            "cross-encoder/ms-marco-MiniLM-L-6-v2"        )    def rerank(self, query, documents, top_n=5):        pairs = [            (query, document.page_content)            for document in documents        ]        scores = self.model.predict(pairs)        ranked = sorted(            zip(documents, scores),            key=lambda item: float(item[1]),            reverse=True        )        return [            {                "document": doc,                "score": float(score)            }            for doc, score in ranked[:top_n]        ]
```

In production, I would load the reranker once during service initialization, not instantiate it for every request.

I would also benchmark reranking latency, throughput and relevance improvements.

## 7. Verify context construction and prompt engineering

Suppose retrieval successfully finds the correct chunk:

```
Conservative portfolio maximum equity allocation: 40%.
```

But the LLM still responds:

```
The maximum allocation is 60%.
```

Now I would investigate context construction and generation.

Possible reasons:

1. The prompt contains contradictory retrieved chunks.
2. The correct chunk is buried in a large context.
3. The LLM is relying on its pretrained knowledge.
4. The prompt does not explicitly require evidence-based answers.
5. The context contains outdated information.

### Solution: Use a grounded generation prompt

```
SYSTEM_PROMPT = """You are an enterprise financial document assistant.Your task is to answer questions using ONLYthe supplied document context.Rules:1. Never use external knowledge.2. Never invent financial values.3. Every factual answer must be supported   by the supplied context.4. If the context contains conflicting values,   explain the conflict.5. If the answer cannot be established,   respond that the information is insufficient.6. Preserve exact numerical values.7. Include document citations and page numbers.Return your answer in the following JSON format:{    "answer": "string",    "citations": [        {            "document_id": "string",            "page_number": 1,            "evidence": "string"        }    ],    "confidence": "high|medium|low"}"""
```

I would also place the most relevant evidence near the end of the context, avoid unnecessary chunks and use a structured output schema.

Important: Prompt engineering improves grounding but does not guarantee correctness. We still need evidence validation.

## 8. Add an answer verification layer

This is something I would strongly recommend in a production RAG system.

After the LLM generates an answer, I would verify whether the answer is actually supported by the retrieved evidence.

The architecture becomes:

```
Retrieve documents
       |
       v
Rerank documents
       |
       v
Generate answer
       |
       v
Evidence verification
       |
       +------ Supported ------> Return answer
       |
       +------ Unsupported ----> Retry retrieval
                                      |
                                      v
                                 Regenerate
                                      |
                                      v
                                Verify again
                                      |
                                      v
                               Return or abstain
```

Example:

```
class AnswerVerifier:    def __init__(self, llm):        self.llm = llm    async def verify(self, question, answer, context):        prompt = f"""        Evaluate whether the answer is supported        by the provided context.        Question:        {question}        Context:        {context}        Answer:        {answer}        Return JSON:        {{            "supported": true,            "reason": "Explanation",            "unsupported_claims": []        }}        Mark supported=false if the answer        contains unsupported factual claims.        """        result = await self.llm.ainvoke(prompt)        return result
```

In actual production code, I would enforce a Pydantic response schema, validate citations against retrieved source IDs and pages, and treat verifier failures as unverified rather than assuming success.

For financial applications, I would additionally validate critical numerical values against source evidence using deterministic checks wherever possible.

## 9. Implement a retry and recovery strategy

What if the verifier detects that the generated answer is unsupported?

I would not keep calling the LLM indefinitely.

I would implement a bounded recovery workflow.

For example:

| Attempt | Action                                    |
| ------- | ----------------------------------------- |
| 1       | Normal hybrid retrieval                   |
| 2       | Expand query and retrieve more candidates |
| 3       | Rerank with additional context            |
| 4       | If evidence remains insufficient, abstain |

A simplified implementation:

```
class RAGService:    def __init__(        self,        retriever,        reranker,        generator,        verifier    ):        self.retriever = retriever        self.reranker = reranker        self.generator = generator        self.verifier = verifier    async def answer(self, question):        for attempt in range(2):            documents = await self.retriever.retrieve(                question,                top_k=10 if attempt == 0 else 20            )            if not documents:                continue            ranked = self.reranker.rerank(                question,                documents,                top_n=5            )            context = "\n\n".join(                item["document"].page_content                for item in ranked            )
```

This is an architectural example; a production implementation would also need typed results, timeouts, retry classification, cancellation, telemetry, tenant authorization, and bounded cost controls.

## 10. Add observability to identify recurring failures

As a Staff AI Engineer, I would not solve only the individual query. I would build observability to identify patterns across thousands or millions of requests.

For every RAG request, I would log the following:

| Component      | What I would monitor               |
| -------------- | ---------------------------------- |
| Parsing        | Extraction failures, empty pages   |
| Chunking       | Chunk count, chunk size            |
| Embeddings     | Model version, latency, failures   |
| Retrieval      | Recall\@K, Precision\@K            |
| Reranking      | Ranking changes, relevance scores  |
| Context        | Token count, source IDs            |
| LLM            | Latency, token usage, model errors |
| Verification   | Groundedness, unsupported claims   |
| Final response | Correctness, citations, abstention |

I would introduce a correlation ID so I can trace one request across the entire pipeline.

For example:

```
{    "request_id": "req_123",    "query": "Maximum equity allocation?",    "retrieved_document_ids": ["doc_1", "doc_2"],    "retrieved_chunk_ids": ["chunk_10", "chunk_20"],    "reranked_chunk_ids": ["chunk_10"],    "embedding_model": "text-embedding-3-small",    "generation_model": "configured-model",    "retrieval_latency_ms": 120,    "reranking_latency_ms": 80,    "generation_latency_ms": 900,    "grounded": True,    "status": "success"}
```

In production, I would use OpenTelemetry for distributed tracing, Prometheus/Grafana for metrics, and structured logs for debugging.

Avoid logging sensitive document contents or personally identifiable information. Use controlled access and redaction.

## 11. Evaluate the RAG system using an evaluation dataset

I would create a golden evaluation dataset containing questions, expected answers, source documents and relevant chunk IDs.

Example:

| Question                 | Expected answer | Relevant chunk |
| ------------------------ | --------------- | -------------- |
| Conservative allocation? | 40%             | Chunk 10       |
| Moderate allocation?     | 60%             | Chunk 20       |
| Aggressive allocation?   | 80%             | Chunk 30       |

I would measure:

1\. Recall\@K

Does retrieval find the correct chunk within the top K results?

\\[ Recall\@K = \frac{\text{Relevant chunks retrieved}} {\text{Total relevant chunks}} \\]

2\. Precision\@K

How many retrieved chunks are actually relevant?

\\[ Precision\@K = \frac{\text{Relevant retrieved chunks}} {\text{Total retrieved chunks}} \\]

3\. MRR

How highly is the first relevant result ranked?

4\. Faithfulness

Is the generated answer supported by the retrieved context?

5\. Answer correctness

Does the generated answer match the expected answer?

I would use RAGAS or a comparable evaluation framework, combined with human-reviewed test cases.

I would also test:

- Questions with exact numerical values.
- Questions involving multiple documents.
- Conflicting document versions.
- Questions with no answer in the documents.
- Questions requiring multiple chunks.
- Adversarial and ambiguous questions.

## 12. My final production architecture

Here is how I would design the complete solution.

## Production RAG architecture

User query

Query understanding and rewriting

Intent, filters, query expansion

Vector retrieval

BM25 retrieval

Hybrid fusion + reranking

Candidate selection and relevance scoring

Context construction

Deduplication, source metadata, token budgeting

LLM generation

Grounded response with structured output

Evidence verification

Claim support, citation validation

Verified

Return answer

Not verified

Bounded retry or abstain

Observability, tracing, evaluation and security apply across all stages.

## 13. How I would explain this in a senior AI engineer interview

If the interviewer asks this question, I would give the following answer:

Interview-ready answer

"If the answer exists in the document but RAG returns an incorrect answer, I would systematically investigate the entire RAG pipeline rather than immediately changing the LLM.

First, I would verify document extraction to ensure that the information is correctly parsed. Then I would inspect chunking to ensure that the relevant information and its surrounding context are preserved.

Next, I would evaluate embedding quality and retrieval results. I would check whether the correct chunk appears in the top-K results using Recall\@K and Precision\@K.

If retrieval quality is poor, I would introduce hybrid search using vector similarity and BM25, combine results using Reciprocal Rank Fusion, and apply a cross-encoder reranker.

If the correct chunk is retrieved but the answer is still incorrect, I would investigate context construction and prompting. I would enforce grounded generation, structured outputs and source citations.

I would then introduce an evidence verification layer to check whether generated claims are supported by retrieved documents. If verification fails, I would trigger a bounded retrieval retry or abstain rather than return an unsupported answer.

Finally, I would add distributed tracing, evaluation datasets and continuous RAG evaluation to prevent the same issue from recurring.

My goal is not simply to make the LLM generate a better answer. My goal is to make the entire retrieval and generation pipeline reliable, measurable and explainable."

## 14. Important follow-up questions you should prepare for

These are natural follow-up questions an interviewer may ask after your answer.

1. What if the correct chunk is retrieved but the LLM ignores it?
2. What if Recall\@10 is high but answer accuracy is low?
3. How do you select the optimal chunk size and overlap?
4. How does hybrid search work internally?
5. Why do you need a reranker if vector search already works?
6. How do you detect hallucinations programmatically?
7. How do you handle conflicting information across multiple document versions?
8. How do you prevent an infinite retrieval and regeneration loop?
9. How would you debug a RAG system with millions of documents?
10. How do you measure whether a new embedding model improves production quality?

My recommendation: For senior/staff-level interviews, pay particular attention to the distinction between retrieval failure, context failure and generation failure. Being able to identify the exact failure stage, support it with metrics, and implement a targeted correction demonstrates much stronger engineering judgment than simply saying "I will increase top-K or improve the prompt."