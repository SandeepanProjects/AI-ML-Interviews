# Parent–Child Retrieval and Contextual Compression in Production RAG

In an enterprise RAG system, retrieving the most semantically similar chunk does not always give the LLM enough information to answer correctly.

Two techniques help solve different parts of this problem:

- Parent–child retrieval finds a precise piece of evidence, then returns a larger section of the original document to preserve context.
- Contextual compression takes retrieved documents and removes irrelevant portions, keeping the information most useful for the current question.

These techniques are complementary. You can use them independently or combine them in a retrieval pipeline.

Let's use your financial compliance platform as the running example.

Suppose a policy contains:

> Section 4.2 — Performance Claims Marketing materials must not imply guaranteed investment returns. Historical returns must include the applicable measurement period and required risk disclosures. Exceptions require prior approval from the Compliance team.

A user asks:

"Can our marketing material claim guaranteed returns, and what disclosures are required?"

A small chunk may contain only the first sentence. A large section may contain the answer alongside several unrelated rules. Parent–child retrieval helps recover the full relevant section; contextual compression helps remove irrelevant material before the LLM generates an answer.

## 1. Parent–child retrieval: the core idea

### Parent document section

Section 4.2 — Performance Claims

The policy prohibits implying guaranteed returns, requires measurement periods and risk disclosures for historical returns, and requires prior Compliance approval for exceptions.

Split into smaller child chunks

Child 1

Prohibition on guaranteed-return claims.

Child 2

Historical-return disclosure requirements.

Child 3

Prior approval for exceptions.

Parent ID

All children link back to Section 4.2.

Embed and search child chunks

### Query matches Child 1

Return the parent section or a bounded portion of it for answer generation.

The central idea is that you index small chunks for precise retrieval but can return larger parent sections for reasoning.

A child chunk is a retrieval unit. A parent is a larger contextual unit, such as a paragraph group, policy section, page range or document chapter.

They do not have to be literal parent and child nodes in a tree database. They can simply be records connected through stable IDs.

### Why not just use large chunks?

Large chunks can contain many unrelated topics, which can make retrieval less precise.

### Why not just use small chunks?

Small chunks can omit conditions, exceptions, definitions or surrounding explanations.

Parent–child retrieval attempts to get both benefits:

- Small child chunks improve retrieval granularity.
- Larger parent chunks restore surrounding context.

One caveat: returning the entire parent is not always the best choice. If a parent contains several pages, returning all of it can reintroduce the same irrelevant content that small chunks were designed to avoid. Production implementations should cap parent size and optionally select a relevant subsection.

## 2. How parent–child retrieval works step by step

A typical implementation has two separate stages: indexing and querying.

### Indexing time

1. Parse the source document.
2. Split it into larger parent sections.
3. Split each parent into smaller child chunks.
4. Generate embeddings for the child chunks.
5. Store child vectors with `parent_id`, `document_id`, page information and access metadata.
6. Store parent text in a parent-document store.

### Query time

1. Embed the user's question.
2. Search the child-vector index.
3. Retrieve the most relevant child chunks.
4. Resolve each child's `parent_id`.
5. Fetch the corresponding parent text.
6. Deduplicate parents and enforce size limits.
7. Pass the selected context to the answer-generation stage.

The following is a simplified architecture:

\#chatgpt-mermaid-\_r_th\_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(237, 237, 237);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-\_r_th\_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_th\_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_th\_ .error-icon{fill:rgb(48, 48, 48);}#chatgpt-mermaid-\_r_th\_ .error-text{fill:rgb(237, 237, 237);stroke:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_th\_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-\_r_th\_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-\_r_th\_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-\_r_th\_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-\_r_th\_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-\_r_th\_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-\_r_th\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_th\_ .marker.cross{stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_th\_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-\_r_th\_ p{margin:0;}#chatgpt-mermaid-\_r_th\_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_th\_ .cluster-label text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_th\_ .cluster-label span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_th\_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-\_r_th\_ .label text,#chatgpt-mermaid-\_r_th\_ span{fill:rgb(237, 237, 237);color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_th\_ .node rect,#chatgpt-mermaid-\_r_th\_ .node circle,#chatgpt-mermaid-\_r_th\_ .node ellipse,#chatgpt-mermaid-\_r_th\_ .node polygon,#chatgpt-mermaid-\_r_th\_ .node path{fill:rgb(9, 23, 44);stroke:rgb(31, 78, 148);stroke-width:1px;}#chatgpt-mermaid-\_r_th\_ .rough-node .label text,#chatgpt-mermaid-\_r_th\_ .node .label text,#chatgpt-mermaid-\_r_th\_ .image-shape .label,#chatgpt-mermaid-\_r_th\_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-\_r_th\_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-\_r_th\_ .rough-node .label,#chatgpt-mermaid-\_r_th\_ .node .label,#chatgpt-mermaid-\_r_th\_ .image-shape .label,#chatgpt-mermaid-\_r_th\_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-\_r_th\_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-\_r_th\_ .root .anchor path{fill:rgb(175, 175, 175)!important;stroke-width:0;stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_th\_ .arrowheadPath{fill:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_th\_ .edgePath .path{stroke:rgb(175, 175, 175);stroke-width:1px;}#chatgpt-mermaid-\_r_th\_ .flowchart-link{stroke:rgb(175, 175, 175);fill:none;}#chatgpt-mermaid-\_r_th\_ .edgeLabel{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_th\_ .edgeLabel p{background-color:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_th\_ .edgeLabel rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_th\_ .labelBkg{background-color:rgba(0, 0, 0, 0.5);}#chatgpt-mermaid-\_r_th\_ .cluster rect{fill:rgb(48, 48, 48);stroke:rgba(255, 255, 255, 0.15);stroke-width:1px;}#chatgpt-mermaid-\_r_th\_ .cluster text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_th\_ .cluster span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_th\_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(48, 48, 48);border:1px solid rgba(255, 255, 255, 0.15);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-\_r_th\_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_th\_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-\_r_th\_ .icon-shape,#chatgpt-mermaid-\_r_th\_ .image-shape{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_th\_ .icon-shape p,#chatgpt-mermaid-\_r_th\_ .image-shape p{background-color:rgb(0, 0, 0);padding:2px;}#chatgpt-mermaid-\_r_th\_ .icon-shape .label rect,#chatgpt-mermaid-\_r_th\_ .image-shape .label rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_th\_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-\_r_th\_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-\_r_th\_ .node .neo-node{stroke:rgb(31, 78, 148);}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].node rect,#chatgpt-mermaid-\_r_th\_ [data-look="neo"].cluster rect,#chatgpt-mermaid-\_r_th\_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-\_r_th\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-\_r_th\_-gradient);stroke-width:1px;}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].node .neo-line path{stroke:rgb(31, 78, 148);filter:none;}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-\_r_th\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-\_r_th\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_th\_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-\_r_th\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_th\_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_th\_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_th\_ .node tspan[font-weight="normal"],#chatgpt-mermaid-\_r_th\_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-\_r_th\_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-width:1px;}#chatgpt-mermaid-\_r_th\_ .node rect,#chatgpt-mermaid-\_r_th\_ .node circle,#chatgpt-mermaid-\_r_th\_ .node ellipse,#chatgpt-mermaid-\_r_th\_ .node polygon,#chatgpt-mermaid-\_r_th\_ .node path{fill:rgb(0, 40, 77);stroke:rgba(255, 255, 255, 0.1);stroke-width:1px;}#chatgpt-mermaid-\_r_th\_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-\_r_th\_ .node.mermaid-decision .label-container{fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-dasharray:2,2;}#chatgpt-mermaid-\_r_th\_ .edgePaths .flowchart-link{stroke:rgb(175, 175, 175);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-\_r_th\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_th\_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}PDF / PolicyParse and normalizeCreate parent sectionsCreate child chunksEmbed childrenVector indexParent document storeUser questionEmbed querySearch child vectorsApply authorization filtersResolve parent IDsDeduplicate and limit contextLLM answer with citations

The parent store might be PostgreSQL, an object store, or a document database. The child index might be pgvector or Qdrant. You can also store parent text and child vectors in PostgreSQL if that meets your scale and operational requirements.

## 3. Production-oriented parent–child implementation

Let's implement the core data model and retrieval logic without tying the design to one vector database.

### Step 1: Define parent and child records

Create `app/retrieval/models.py`.

```
from dataclasses import dataclass@dataclass(frozen=True)class ParentDocument:    parent_id: str    document_id: str    tenant_id: str    document_version: str    page_start: int    page_end: int    section: str    text: str@dataclass(frozen=True)class ChildChunk:    child_id: str    parent_id: str    document_id: str    tenant_id: str    document_version: str    page_start: int    page_end: int    text: str    token_count: int
```

Why maintain both IDs?

- `child_id` identifies the individual searchable chunk.
- `parent_id` connects that chunk to its larger context.
- `document_id` identifies the source document.
- `document_version` prevents the retrieval layer from accidentally mixing versions.
- `tenant_id` supports isolation, but authorization must still be enforced by trusted server-side code.

### Step 2: Build parent and child chunks

For a real system, create parents from the document's actual section hierarchy. Then split each section into children.

Here is a simplified example using the LangChain text splitter:

```
from langchain_text_splitters import (    RecursiveCharacterTextSplitter,)from app.retrieval.models import (    ChildChunk,    ParentDocument,)parent_splitter = RecursiveCharacterTextSplitter(    chunk_size=1800,    chunk_overlap=100,    length_function=len,  # Characters in this example    separators=["\n\n", "\n", ". ", " ", ""],)child_splitter = RecursiveCharacterTextSplitter(    chunk_size=500,    chunk_overlap=75,    length_function=len,    separators=["\n\n", "\n", ". ", " ", ""],)def create_parent_child_records(    *,    document_id: str,    tenant_id: str,    version: str,    page_start: int,    page_end: int,    section: str,    text: str,) -> tuple[list[ParentDocument], list[ChildChunk]]:    parents = []    children = []    parent_texts = parent_splitter.split_text(text)    for parent_index, parent_text in enumerate(parent_texts):        parent_id = f"{document_id}:{version}:p{parent_index}"        parent = ParentDocument(            parent_id=parent_id,            document_id=document_id,            tenant_id=tenant_id,            document_version=version,            page_start=page_start,            page_end=page_end,            section=section,            text=parent_text,        )        parents.append(parent)        child_texts = child_splitter.split_text(parent_text)        for child_index, child_text in enumerate(child_texts):            children.append(                ChildChunk(                    child_id=(                        f"{parent_id}:c{child_index}"                    ),                    parent_id=parent_id,                    document_id=document_id,                    tenant_id=tenant_id,                    document_version=version,                    page_start=page_start,                    page_end=page_end,                    text=child_text,                    token_count=len(child_text.split()),                )            )    return parents, children
```

Important: these settings use character counts, not tokens. The `500` child size means approximately 500 characters, not 500 tokens. The `split()` word count is also only a rough estimate, not a model-token count. For production, use the tokenizer appropriate to your models and choose limits based on evaluation.

This code also assigns a single page range to the entire input section. If the section spans multiple pages, that range is too coarse for precise child-level citations. A production parser should carry page-level spans through splitting and associate each child with its actual source page or page range.

In addition, the example uses positional IDs for clarity. Production IDs should be stable across retries and should account for the document content hash and pipeline version, so reprocessing does not silently overwrite unrelated records.

### Step 3: Define storage interfaces

Separate retrieval logic from the underlying databases.

```
from typing import Protocolfrom app.retrieval.models import (    ChildChunk,    ParentDocument,)class ChildVectorIndex(Protocol):    async def search(        self,        query: str,        *,        tenant_id: str,        document_ids: list[str] | None,        k: int,    ) -> list[tuple[ChildChunk, float]]:        ...class ParentDocumentStore(Protocol):    async def get_many(        self,        parent_ids: list[str],        *,        tenant_id: str,        document_ids: list[str] | None,    ) -> list[ParentDocument]:        ...
```

These are interfaces, not database implementations. Your Qdrant or pgvector adapter implements `ChildVectorIndex`; your PostgreSQL or object-store adapter implements `ParentDocumentStore`.

Notice that the tenant scope is passed into both interfaces. It is not enough to filter children and then fetch parents without authorization checks.

### Step 4: Implement parent retrieval

Create `app/retrieval/parent_child.py`.

```
from dataclasses import dataclassfrom app.retrieval.models import ParentDocumentfrom app.retrieval.models import ChildChunkfrom app.retrieval.models import ChildChunk@dataclass(frozen=True)class RetrievedParent:    parent: ParentDocument    matched_children: tuple[ChildChunk, ...]    best_child_score: floatclass ParentChildRetriever:    def __init__(        self,        child_index,        parent_store,    ) -> None:        self.child_index = child_index        self.parent_store = parent_store    async def retrieve(        self,        query: str,        *,        tenant_id: str,        document_ids: list[str] | None = None,        child_k: int = 10,    ) -> list[RetrievedParent]:        child_hits = await self.child_index.search(            query,            tenant_id=tenant_id,            document_ids=document_ids,            k=child_k,        )        # Group matched child chunks by parent.        grouped = {}        for child, score in child_hits:            entry = grouped.setdefault(                child.parent_id,                {                    "children": [],                    "best_score": float("-inf"),                },            )            entry["children"].append(child)            entry["best_score"] = max(                entry["best_score"],                score,            )        parent_ids = list(grouped)        parents = await self.parent_store.get_many(            parent_ids,            tenant_id=tenant_id,            document_ids=document_ids,        )        parent_by_id = {            parent.parent_id: parent            for parent in parents        }        results = []        for parent_id, entry in grouped.items():            parent = parent_by_id.get(parent_id)            # Never use a parent that failed the store's            # authorization or version checks.            if parent is None:                continue            results.append(                RetrievedParent(                    parent=parent,                    matched_children=tuple(                        entry["children"]                    ),                    best_child_score=entry["best_score"],                )            )        return sorted(            results,            key=lambda result: result.best_child_score,            reverse=True,        )
```

There are two duplicate imports in this example's import section; they are unnecessary and can be reduced to one import of `ChildChunk`. The retrieval algorithm itself illustrates the important flow.

A production implementation must also define score semantics: similarity scores and distances are not interchangeable, and sorting direction depends on the backend. It should preserve the original child rank, apply an explicit parent-ranking policy, and use active-version and access-control filters in the actual database queries.

### What does this implementation achieve?

Suppose the search returns:

| Child | Parent      | Why it matched                         |
| ----- | ----------- | -------------------------------------- |
| `c1`  | Section 4.2 | Guaranteed-return prohibition          |
| `c2`  | Section 4.2 | Required historical-return disclosures |
| `c3`  | Section 7.1 | General approval workflow              |

The retrieval service groups `c1` and `c2` under Section 4.2, then fetches that parent once rather than passing duplicate copies of the same section to the LLM.

That improves context recovery. But it doesn't guarantee that the parent is small enough or that every sentence is relevant. Those are problems contextual compression can help address.

# 4. What is contextual compression?

Contextual compression is a retrieval-time technique that reduces the amount of retrieved text to the parts that are most useful for answering the current question.

Imagine your retriever finds three policy sections:

Retrieved document A — Performance claims

> Marketing materials must not imply guaranteed investment returns. Historical returns must include the applicable measurement period and required risk disclosures. Exceptions require prior approval from the Compliance team.

Retrieved document B — Employee training

> All employees must complete annual compliance training. Managers are responsible for ensuring training completion and maintaining attendance records.

Retrieved document C — Fee disclosures

> Applicable investment fees must be disclosed before a recommendation is finalized. Disclosures must be clear and consistent with approved product documentation.

The question is:

"Can our marketing material promise guaranteed returns?"

A contextual compressor can reduce the retrieved context to:

> Marketing materials must not imply guaranteed investment returns.

The training and fee sections can be excluded if they do not help answer the question.

### 1. Retrieve candidates

A similarity search returns several potentially relevant passages.

Performance

Training

Fees

### 2. Compress the context

A filter, reranker or LLM identifies relevant sentences and removes unrelated content.

### 3. Generate a grounded answer

Use the retained evidence and its original source references to answer the question.

There is an important distinction here: contextual compression does not necessarily mean summarization. Its goal is to reduce irrelevant context while preserving the evidence needed for the answer.

There are several approaches.

| Technique              | How it works                                                                  | Cost           |
| ---------------------- | ----------------------------------------------------------------------------- | -------------- |
| Metadata filtering     | Removes documents that do not match access, date or document-type constraints | Low            |
| Score thresholding     | Drops candidates below a relevance threshold                                  | Low            |
| Reranking              | Scores query–document pairs and keeps the best passages                       | Low to medium  |
| Extractive compression | Keeps relevant sentences from retrieved text                                  | Low to medium  |
| LLM-based compression  | Uses an LLM to extract or condense question-relevant evidence                 | Medium to high |

A reranker and a compressor are not the same thing. A reranker reorders or scores passages; it does not necessarily shorten them. You can use a reranker first, then compress the highest-ranked passages.

## 5. Why contextual compression matters

Suppose your retriever returns 10 chunks, each containing 500 tokens.

That's up to 5,000 tokens of candidate context before the answer-generation step.

If only 1,200 tokens contain relevant evidence, compression may reduce unnecessary context and make the answer easier to ground.

Potential benefits include:

- Less irrelevant information in the LLM context.
- Lower answer-generation token usage when compression actually reduces the prompt.
- Better focus on the user's specific question.
- Less distraction from unrelated passages.
- Easier evidence selection for some question types.

But compression also introduces risks:

- It may remove an important exception or qualification.
- It may omit a negation such as "not permitted."
- It may change the meaning of a legal or financial statement.
- An LLM may introduce unsupported language while summarizing.
- The extra model call can increase latency and cost.

For financial compliance, I would favor extractive compression initially: keep exact source sentences and their provenance. If abstractive compression is required, validate the resulting claims against the original passages.

# 6. Implement contextual compression in Python

Let's build a simple, explicit pipeline with a retriever, a relevance filter and an LLM-based evidence extractor.

The design will enforce three principles:

1. Retrieved text remains untrusted input.
2. Compressed evidence retains its source IDs and page references.
3. The compressor may return only evidence supported by the retrieved passages.

It cannot mathematically guarantee that an LLM will never omit or distort evidence, so the implementation must still be evaluated and monitored.

### Step 1: Define the retrieval result

```
from dataclasses import dataclass@dataclass(frozen=True)class RetrievedChunk:    chunk_id: str    document_id: str    page_number: int    section: str    text: str    score: float@dataclass(frozen=True)class CompressedEvidence:    source_chunk_id: str    document_id: str    page_number: int    section: str    text: str
```

The compressor returns a reference to the original chunk, rather than creating a new untraceable document.

### Step 2: Add a simple score-based filter

```
def filter_by_score(    chunks: list[RetrievedChunk],    *,    threshold: float,    max_chunks: int = 10,) -> list[RetrievedChunk]:    candidates = [        chunk        for chunk in chunks        if chunk.score >= threshold    ]    candidates.sort(        key=lambda chunk: chunk.score,        reverse=True,    )    return candidates[:max_chunks]
```

This is a useful baseline, but there is a critical caveat: the threshold depends on your retrieval model and scoring method. A cosine similarity score, a distance score and a cross-encoder relevance score have different interpretations. Calibrate the threshold against labeled queries rather than assuming `0.7` means "relevant" for every model.

Score filtering can remove irrelevant passages, but it cannot remove irrelevant sentences inside an otherwise relevant passage.

### Step 3: Use an LLM to extract evidence

Install the OpenAI SDK if you want to use the following example:

```
pip install openai pydantic
```

Create `app/retrieval/compressor.py`:

```
import jsonfrom typing import Literalfrom openai import AsyncOpenAIfrom pydantic import BaseModel, Fieldclass ExtractedPassage(BaseModel):    source_chunk_id: str    exact_quote: str = Field(min_length=1)class CompressionResult(BaseModel):    passages: list[ExtractedPassage]class LLMContextCompressor:    def __init__(self, model: str) -> None:        self.client = AsyncOpenAI()        self.model = model    async def compress(        self,        query: str,        chunks: list[RetrievedChunk],    ) -> list[CompressedEvidence]:        if not chunks:            return []        # Use JSON serialization to make source boundaries        # explicit. The document content is untrusted data.        source_payload = [            {                "chunk_id": chunk.chunk_id,                "text": chunk.text,            }            for chunk in chunks        ]
```

This implementation uses JSON-mode output, followed by Pydantic validation and an exact-quote check. It does not use the provider's stricter schema-constrained output mode; you can adopt that when supported by your selected model and SDK version.

The `Literal` import is unused and can be removed.

### What does the validation accomplish?

Suppose the model returns:

```
{
  "passages": [
    {
      "source_chunk_id": "policy-123:chunk:17",
      "exact_quote": "Marketing materials must not imply guaranteed investment returns."
    }
  ]
}
```

The application checks:

1. Does the chunk ID exist in the retrieved candidate set?
2. Is the quote actually present in that chunk?
3. Is the result duplicated?
4. Can the source's document and page information be recovered?

If the model invents a new sentence, the exact-quote validation rejects it.

However, exact-quote validation does not prove that the evidence is sufficient. The model could return only half of a rule, omitting a condition or exception. For compliance, evaluate whether the retained evidence covers the full answer, and include relevant adjacent clauses or the parent section when needed.

# 7. Combine parent–child retrieval with contextual compression

This is where the techniques become especially useful together.

Consider this query:

> Can we claim guaranteed returns, and what disclosures are mandatory?

The child retriever may find two passages in the same policy section. Parent retrieval restores the surrounding rule. Contextual compression then selects the exact sentences needed to answer the question.

A sensible production flow is:

\#chatgpt-mermaid-\_r_105\_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(237, 237, 237);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-\_r_105\_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_105\_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_105\_ .error-icon{fill:rgb(48, 48, 48);}#chatgpt-mermaid-\_r_105\_ .error-text{fill:rgb(237, 237, 237);stroke:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_105\_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-\_r_105\_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-\_r_105\_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-\_r_105\_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-\_r_105\_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-\_r_105\_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-\_r_105\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_105\_ .marker.cross{stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_105\_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-\_r_105\_ p{margin:0;}#chatgpt-mermaid-\_r_105\_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_105\_ .cluster-label text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_105\_ .cluster-label span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_105\_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-\_r_105\_ .label text,#chatgpt-mermaid-\_r_105\_ span{fill:rgb(237, 237, 237);color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_105\_ .node rect,#chatgpt-mermaid-\_r_105\_ .node circle,#chatgpt-mermaid-\_r_105\_ .node ellipse,#chatgpt-mermaid-\_r_105\_ .node polygon,#chatgpt-mermaid-\_r_105\_ .node path{fill:rgb(9, 23, 44);stroke:rgb(31, 78, 148);stroke-width:1px;}#chatgpt-mermaid-\_r_105\_ .rough-node .label text,#chatgpt-mermaid-\_r_105\_ .node .label text,#chatgpt-mermaid-\_r_105\_ .image-shape .label,#chatgpt-mermaid-\_r_105\_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-\_r_105\_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-\_r_105\_ .rough-node .label,#chatgpt-mermaid-\_r_105\_ .node .label,#chatgpt-mermaid-\_r_105\_ .image-shape .label,#chatgpt-mermaid-\_r_105\_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-\_r_105\_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-\_r_105\_ .root .anchor path{fill:rgb(175, 175, 175)!important;stroke-width:0;stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_105\_ .arrowheadPath{fill:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_105\_ .edgePath .path{stroke:rgb(175, 175, 175);stroke-width:1px;}#chatgpt-mermaid-\_r_105\_ .flowchart-link{stroke:rgb(175, 175, 175);fill:none;}#chatgpt-mermaid-\_r_105\_ .edgeLabel{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_105\_ .edgeLabel p{background-color:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_105\_ .edgeLabel rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_105\_ .labelBkg{background-color:rgba(0, 0, 0, 0.5);}#chatgpt-mermaid-\_r_105\_ .cluster rect{fill:rgb(48, 48, 48);stroke:rgba(255, 255, 255, 0.15);stroke-width:1px;}#chatgpt-mermaid-\_r_105\_ .cluster text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_105\_ .cluster span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_105\_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(48, 48, 48);border:1px solid rgba(255, 255, 255, 0.15);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-\_r_105\_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_105\_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-\_r_105\_ .icon-shape,#chatgpt-mermaid-\_r_105\_ .image-shape{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_105\_ .icon-shape p,#chatgpt-mermaid-\_r_105\_ .image-shape p{background-color:rgb(0, 0, 0);padding:2px;}#chatgpt-mermaid-\_r_105\_ .icon-shape .label rect,#chatgpt-mermaid-\_r_105\_ .image-shape .label rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_105\_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-\_r_105\_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-\_r_105\_ .node .neo-node{stroke:rgb(31, 78, 148);}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].node rect,#chatgpt-mermaid-\_r_105\_ [data-look="neo"].cluster rect,#chatgpt-mermaid-\_r_105\_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-\_r_105\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-\_r_105\_-gradient);stroke-width:1px;}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].node .neo-line path{stroke:rgb(31, 78, 148);filter:none;}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-\_r_105\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-\_r_105\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_105\_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-\_r_105\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_105\_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_105\_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_105\_ .node tspan[font-weight="normal"],#chatgpt-mermaid-\_r_105\_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-\_r_105\_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-width:1px;}#chatgpt-mermaid-\_r_105\_ .node rect,#chatgpt-mermaid-\_r_105\_ .node circle,#chatgpt-mermaid-\_r_105\_ .node ellipse,#chatgpt-mermaid-\_r_105\_ .node polygon,#chatgpt-mermaid-\_r_105\_ .node path{fill:rgb(0, 40, 77);stroke:rgba(255, 255, 255, 0.1);stroke-width:1px;}#chatgpt-mermaid-\_r_105\_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-\_r_105\_ .node.mermaid-decision .label-container{fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-dasharray:2,2;}#chatgpt-mermaid-\_r_105\_ .edgePaths .flowchart-link{stroke:rgb(175, 175, 175);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-\_r_105\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_105\_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}User questionAuthorization and querypreparationSearch child chunksFilter and rerank candidatesResolve parent sectionsDeduplicate and enforcecontext limitsExtract relevant evidenceEvidence sufficient?Retrieve additional context orabstainGenerate answer with sourcecitationsValidate answer and citationsNoYes

The important design decision is that compression happens after retrieval, and it must not override access control. A compressor should only receive passages the user is authorized to access.

Also, do not blindly run a second retrieval loop every time the compressor returns little text. First distinguish between genuinely missing evidence, a bad retrieval ranking, and a compressor failure.

## 8. Orchestrate the combined pipeline

Here is a simplified orchestration layer using the interfaces from the earlier examples:

```
class RAGContextBuilder:    def __init__(        self,        parent_child_retriever,        compressor,    ):        self.retriever = parent_child_retriever        self.compressor = compressor    async def build_context(        self,        question: str,        *,        tenant_id: str,        document_ids: list[str] | None = None,    ) -> list[CompressedEvidence]:        # 1. Retrieve matching child chunks and parents.        parent_results = await self.retriever.retrieve(            question,            tenant_id=tenant_id,            document_ids=document_ids,            child_k=12,        )        # 2. Build a deduplicated candidate set from the        # matched children, preserving their provenance.        candidates = []        seen_ids = set()        for result in parent_results:            for child in result.matched_children:                if child.child_id in seen_ids:                    continue                seen_ids.add(child.child_id)                candidates.append(                    RetrievedChunk(                        chunk_id=child.child_id,                        document_id=child.document_id,                        page_number=child.page_start,                        section=result.parent.section,                        text=child.text,                        score=result.best_child_score,                    )                )        # 3. Compress the authorized, retrieved evidence.        evidence = await self.compressor.compress(            question,            candidates,        )        return evidence
```

This example deliberately compresses the matched child passages rather than the entire parent. That preserves precision and avoids expanding the context unnecessarily.

However, there is a trade-off: if a relevant condition is present in the parent but outside the matched children, the compressor cannot recover it. A more robust implementation should pass bounded parent context or adjacent child chunks to the compression stage, while maintaining accurate source-page provenance.

The example also inherits the parent's best child score for all its children. In a production implementation, preserve each child's actual score and rank rather than assigning the parent score to every child.

The final answer-generation step should receive the compressed evidence with source IDs, not just a joined string of text.

# 9. What about LangChain's contextual compression retriever?

LangChain offers a `ContextualCompressionRetriever` abstraction that wraps a base retriever and a document compressor. This is useful when you want to compose existing retrievers and compression components.

Conceptually:

```
# Illustrative LangChain-style composition.# Exact imports and compressor interfaces depend on# the LangChain package versions you install.compression_retriever = ContextualCompressionRetriever(    base_retriever=base_retriever,    base_compressor=base_compressor,)documents = compression_retriever.invoke(    "Can we claim guaranteed returns?")
```

The `base_retriever` finds candidate documents. The `base_compressor` processes those candidates before returning the final documents.

Possible compressors include:

- An LLM-based extractor that retains relevant passages.
- A reranking-based filter that drops low-relevance documents.
- A custom compressor that applies deterministic rules.
- A pipeline that combines reranking and extraction.

The snippet shows the composition pattern, not a complete runnable program: `ContextualCompressionRetriever`, `base_retriever`, and `base_compressor` must be imported, configured and implemented using compatible LangChain packages.

For enterprise applications, I prefer explicit interfaces around retrieval and compression when I need strict control over authorization, source attribution, token budgets, error handling and auditability. LangChain can still provide the underlying components.

## 10. How do you decide which technique to use?

| Situation                                         | Recommended approach                                              |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| Precise factual questions                         | Small child chunks and top-k retrieval                            |
| Questions requiring conditions and exceptions     | Parent–child retrieval                                            |
| Many retrieved passages contain irrelevant text   | Contextual compression                                            |
| Retrieval returns long sections with mixed topics | Child search followed by bounded parent retrieval and compression |
| Legal or financial evidence needs exact citations | Extractive compression with source validation                     |
| Broad questions requiring a complete chapter      | Larger parent context or multiple related sections                |
| Very large retrieval candidate sets               | Metadata filters, reranking, then selective compression           |

For a compliance application, I would start with child retrieval plus parent expansion, then measure whether contextual compression improves answer quality and token efficiency. Compression adds another failure mode and potentially another LLM call, so it should earn its place through evaluation.

## 11. How to evaluate both techniques

Build a test dataset of real questions with manually verified supporting passages.

Compare these configurations:

1. Fixed-size chunks only.
2. Parent–child retrieval.
3. Retrieval followed by reranking.
4. Parent–child retrieval followed by contextual compression.
5. Parent–child retrieval, compression and answer validation.

Track:

- Recall\@k of the required evidence.
- Whether conditions and exceptions survive compression.
- Citation precision and source-page correctness.
- Final answer correctness and groundedness.
- Input tokens passed to the answer-generation LLM.
- Total latency and model cost.
- Failure rates, including empty or invalid compression output.

Do not optimize only for fewer tokens. A compressor that removes 80% of the text but also drops the critical exception is worse than returning the original context.

## 12. Staff-level interview answer

What is the difference between parent–child retrieval and contextual compression?

> Parent–child retrieval addresses the granularity trade-off in RAG. I index smaller child chunks for precise semantic search, then use their parent IDs to retrieve a larger contextual section when the answer requires surrounding rules, definitions or exceptions.
>
> Contextual compression addresses the relevance and context-budget problem. After retrieval, I filter or extract the passages that are useful for the current query, ideally retaining exact source text and provenance.
>
> In a production financial compliance system, I can combine them: retrieve child chunks, rerank the candidates, fetch bounded parent context, compress the relevant evidence and generate an answer with validated citations.
>
> I enforce tenant authorization before retrieval and parent expansion, version the source documents, and ensure compression cannot invent citations. I evaluate the full pipeline on evidence recall, exception preservation, citation accuracy, groundedness, latency and cost.

The key distinction to remember is:

- Parent–child retrieval asks: “How much of the original document should I retrieve after finding a relevant chunk?”
- Contextual compression asks: “How much of that retrieved content is actually useful for answering this question?”

Together, they help a RAG system retrieve focused evidence without losing the context required to interpret it correctly.