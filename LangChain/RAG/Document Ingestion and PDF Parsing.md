# Document Ingestion and PDF Parsing — Senior/Staff AI Engineer Guide

In a production AI system, document ingestion is much more than reading a PDF and extracting text. It is the process of transforming raw documents into validated, searchable, traceable, and retrieval-ready knowledge that an LLM can use reliably.

For example, in your enterprise financial compliance or financial advisor copilot project, you might ingest:

- Financial policies and compliance rulebooks in PDF format.
- Annual reports, research reports, and investment documents.
- Advisor communications and marketing materials.
- Tables containing fees, risk ratings, portfolio allocations, and performance data.
- Scanned PDFs containing images rather than actual text.

The engineering challenge is to extract the right information, preserve its meaning and source location, handle failures, and make the resulting data searchable without losing document-level security.

## 1. Understand the complete ingestion pipeline

1\. Document acquisition

PDF, DOCX, HTML, scanned files, object storage

2\. Validation and security

File type, size limits, malware scanning, tenant authorization

3\. Parsing and extraction

Text, pages, tables, images, OCR, layout and reading order

4\. Quality assessment and normalization

Empty pages, encoding, duplicate text, extraction confidence

5\. Structure-aware chunking

Sections, headings, paragraphs, tables and page boundaries

6\. Embedding and indexing

Embedding model, vector store, keyword index and metadata

7\. Validation and publication

Retrieval tests, version activation, audit trail and monitoring

The most important distinction is:

- Parsing converts a document into structured content.
- Chunking divides the content into retrieval-sized pieces.
- Embedding converts text into vectors.
- Indexing makes the vectors and their metadata searchable.
- Retrieval finds relevant pieces at query time.

These are separate stages with separate failure modes. A good embedding model cannot fix a badly parsed PDF.

## 2. What does PDF parsing actually mean?

A PDF is primarily a page-description format. It is not necessarily a structured document containing paragraphs, headings, and tables in the same way a DOCX file does.

A PDF may contain:

[Searchable PDF vs Scanned PDF: Which Needs OCR?](https://images.openai.com/static-rsc-4/3mzITLG9x2nPxIGGHnMvX14uyP-UmkK0PRB5DhP4JpRL_GveIkFg2FiSLBHspUgO9XS2Va6NHAkS-gUmg7jb5XVpkZ_g0VgYmOUfW24taBIZ18jOe-uvbDKmpyl1cit_AM8kYhv0VMWn6W1oisUOkmj7PPEfSette9HDe1gypgk?purpose=inline)

A. Text-based PDF

The text is encoded in the PDF. A parser can often extract it directly without OCR.

[Automatically correct page orientation in documents | Transloadit](https://images.openai.com/static-rsc-4/EjFxrMKIUEalBqey3r1_YJ8YRLBBMMAVxGA9x3SCQiId-YTUSeB5ZFjjBKgqnSrDeCH5guDLmeBE-W3X3-nAZPrgKgVUzJBNrYUjZNa-7Py2EZFG7AEiNLkedxpVurFJYEc-__8I47GIqLRKgcc7G05P7Tzbu45OKGoxvqiHvCw?purpose=inline)

B. Scanned PDF

Each page is essentially an image. OCR is needed to recognize the characters.

C. Layout-heavy PDF

It contains columns, tables, footnotes, charts, headers, and complex reading order. Plain text extraction may scramble the content.

A production parser should identify which case it is handling and choose an appropriate extraction strategy.

### Common PDF parsing libraries

| Library                    | Best use                                            | Important limitation                                                   |
| -------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| `PyMuPDF` (`fitz`)         | Fast text extraction, page geometry, blocks, images | Complex layouts may require additional processing                      |
| `pdfplumber`               | Word positions, tables, layout inspection           | Table extraction needs tuning for each document family                 |
| `pypdf`                    | Basic PDF text extraction and manipulation          | Not a full OCR or layout-understanding solution                        |
| `PaddleOCR` / Tesseract    | OCR for scanned pages                               | Accuracy depends on scan quality and language                          |
| `Docling`                  | Document structure, layout and tables               | Requires evaluation for your document types and deployment constraints |
| Cloud document AI services | Managed OCR, tables and layout extraction           | Cost, data residency, network and vendor constraints                   |

For a Python-based RAG system, I would start with PyMuPDF for ordinary PDFs, add `pdfplumber` where table extraction matters, and route scanned or complex documents to OCR/layout processing only when needed.

## 3. Build a PDF parser in Python

Let's build this in layers, as you would in a real AI engineering repository.

### Step 1 — Install the dependencies

```
python -m venv .venv
source .venv/bin/activate

pip install pymupdf pdfplumber pydantic
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Step 2 — Extract text page by page

Create `app/ingestion/pdf_parser.py`.

```
from dataclasses import dataclassfrom pathlib import Pathimport fitz  # PyMuPDF@dataclass(frozen=True)class ParsedPage:    page_number: int    text: str    extraction_method: str    char_count: int@dataclass(frozen=True)class ParsedDocument:    filename: str    page_count: int    pages: list[ParsedPage]def parse_pdf(file_path: str) -> ParsedDocument:    path = Path(file_path)    if not path.is_file():        raise FileNotFoundError(f"PDF not found: {path}")    if path.suffix.lower() != ".pdf":        raise ValueError("Only PDF files are supported")    pages: list[ParsedPage] = []    with fitz.open(path) as pdf:        if pdf.is_encrypted:            raise ValueError("Encrypted PDF requires an approved decryption flow")        for index, page in enumerate(pdf):            text = page.get_text("text", sort=True)            text = text.replace("\x00", "").strip()            pages.append(                ParsedPage(                    page_number=index + 1,                    text=text,                    extraction_method="pymupdf",                    char_count=len(text),                )            )        return ParsedDocument(            filename=path.name,            page_count=len(pdf),            pages=pages,        )if __name__ == "__main__":    document = parse_pdf("sample.pdf")    print("File:", document.filename)    print("Pages:", document.page_count)    for page in document.pages:        print(f"\n--- Page {page.page_number} ---")        print(page.text[:1000])
```

### How this code works

1. `fitz.open()` opens the PDF and exposes its pages.
2. `page.get_text("text", sort=True)` extracts text in an approximate reading order.
3. We preserve page numbers so later chunks can cite the original source.
4. The parser returns structured data instead of a single enormous string.

For example, a two-page policy PDF might return:

```
ParsedDocument(    filename="investment_policy.pdf",    page_count=2,    pages=[        ParsedPage(            page_number=1,            text="Investment Risk Policy\n...",            extraction_method="pymupdf",            char_count=1200,        ),        ParsedPage(            page_number=2,            text="Portfolio Restrictions\n...",            extraction_method="pymupdf",            char_count=950,        ),    ],)
```

The counts and text here are illustrative.

Staff-level design decision: preserve document structure and provenance from the beginning. Do not return only `str`, because page-level citations, debugging, reprocessing and access controls become much harder later.

### Step 3 — Extract positioned blocks

Plain text is not always enough. A PDF page may have two columns, a footer, and a table. Extracting text without spatial information can mix unrelated content.

PyMuPDF can extract positioned blocks:

```
import fitzdef extract_page_blocks(pdf_path: str) -> list[dict]:    results = []    with fitz.open(pdf_path) as pdf:        for page_number, page in enumerate(pdf, start=1):            blocks = page.get_text("blocks", sort=True)            for block in blocks:                x0, y0, x1, y1, text, block_no, block_type = block[:7]                if block_type != 0:                    continue                text = text.strip()                if not text:                    continue                results.append({                    "page_number": page_number,                    "block_number": block_no,                    "bbox": [x0, y0, x1, y1],                    "text": text,                })    return results
```

The bounding box records the block's position on the page. This can help reconstruct reading order, distinguish headers and footers, and locate a claim for a reviewer.

However, sorting by position is still a heuristic. For complex multi-column documents, use layout-aware extraction and validate the resulting reading order.

## 4. Handle scanned PDFs using OCR

A common production failure occurs when a PDF opens normally, but the parser returns almost no text.

For example:

```
text = page.get_text("text")if len(text.strip()) < 30:    print("Potential scanned page")
```

This is only a heuristic, not proof. A page with a large chart or a very short title may also have little text.

### Use OCR as a fallback

Install the OCR engine separately:

```
pip install pytesseract pillow
```

You must also install the Tesseract OCR executable on your operating system. On macOS, for example:

```
brew install tesseract
```

Then add an OCR fallback:

```
import ioimport fitzimport pytesseractfrom PIL import Imagedef extract_page_text(page: fitz.Page) -> tuple[str, str]:    # First attempt: native PDF text extraction.    native_text = page.get_text("text", sort=True).strip()    # Tune this threshold using representative documents.    if len(native_text) >= 30:        return native_text, "native"    # Fallback: render the page as an image.    pixmap = page.get_pixmap(        matrix=fitz.Matrix(2, 2),        alpha=False,    )    image = Image.open(io.BytesIO(pixmap.tobytes("png")))    ocr_text = pytesseract.image_to_string(        image,        config="--psm 3",    ).strip()    if ocr_text:        return ocr_text, "ocr"    return native_text, "native_low_text"
```

This implementation is a starting point. A production system should also:

- Limit image resolution and page count to prevent resource exhaustion.
- Record OCR language, version, duration and errors.
- Detect mixed PDFs containing both scanned and digital pages.
- Preserve page numbers and, when available, OCR word coordinates.
- Route poor-quality results to review instead of silently publishing them.

OCR text may contain mistakes in numbers, decimal points, currency symbols and percentages. In financial documents, `8.5%` misread as `85%` is a significant error. Validate critical fields against the source image or use a second extraction method.

## 5. Extract tables correctly

Financial and compliance documents often contain important information in tables. Plain text extraction may produce:

```
Product Fee Risk
Fund A 0.75% High
Fund B 0.25% Medium
```

That might be usable, but more complex tables can lose their column associations.

Use `pdfplumber` to inspect tables:

```
import pdfplumberdef extract_tables(pdf_path: str) -> list[dict]:    extracted = []    with pdfplumber.open(pdf_path) as pdf:        for page_number, page in enumerate(pdf.pages, start=1):            tables = page.extract_tables()            for table_index, table in enumerate(tables):                if not table:                    continue                extracted.append({                    "page_number": page_number,                    "table_index": table_index,                    "rows": table,                })    return extracted
```

For a simple table, the result may resemble:

```
[    {        "page_number": 4,        "table_index": 0,        "rows": [            ["Product", "Annual fee", "Risk"],            ["Fund A", "0.75%", "High"],            ["Fund B", "0.25%", "Medium"],        ],    }]
```

Table detection is not guaranteed. Scanned tables need OCR and layout/table recognition, and borderless tables often require custom extraction settings.

Important design decision: do not embed a table as disconnected cell fragments. Preserve the header and row relationships. One practical approach is to serialize each row with its column names:

```
def table_rows_to_text(rows: list[list[str | None]]) -> list[str]:    if not rows:        return []    headers = [        (value or "").strip()        for value in rows[0]    ]    result = []    for row in rows[1:]:        fields = []        for index, value in enumerate(row):            if index >= len(headers):                break            header = headers[index]            cell = (value or "").strip()            if header and cell:                fields.append(f"{header}: {cell}")        if fields:            result.append("; ".join(fields))    return result
```

Example:

```
Product: Fund A; Annual fee: 0.75%; Risk: High
Product: Fund B; Annual fee: 0.25%; Risk: Medium
```

This is much easier for a retrieval system to interpret than a flat sequence of cells.

## 6. Normalize the extracted content

PDF extraction often produces noisy text:

- Repeated page headers and footers.
- Excessive blank lines.
- Hyphenated words split across lines.
- Broken Unicode characters.
- Repeated text caused by overlapping text layers.
- Page numbers mixed into paragraphs.

Create a normalization layer, but do not blindly remove every newline or hyphen. They may represent meaningful boundaries.

```
import reimport unicodedatadef normalize_text(text: str) -> str:    # Normalize equivalent Unicode representations.    text = unicodedata.normalize("NFC", text)    # Remove null bytes and control characters,    # preserving line breaks and tabs.    text = "".join(        char for char in text        if char in "\n\t" or char.isprintable()    )    # Normalize horizontal whitespace.    text = re.sub(r"[ \t]+", " ", text)    # Reduce excessive blank lines.    text = re.sub(r"\n{3,}", "\n\n", text)    return text.strip()
```

### What about headers and footers?

Suppose every page contains:

```
CONFIDENTIAL — INTERNAL USE ONLY
Investment Policy | Version 2026.3
```

Repeated headers can dominate embedding and retrieval results. You can identify recurring text at similar page positions and remove it from the retrieval text, while retaining the original document and extraction artifacts.

Do not remove confidentiality labels, effective dates or policy versions from your source metadata. A policy's version may be essential to deciding which rule applies.

Also avoid indiscriminately removing numeric lines. They may contain legal clauses, financial figures or section numbers.

## 7. Chunk the document without losing meaning

After parsing, you need to divide the text into chunks suitable for embedding and retrieval.

For example, a 50-page PDF may contain 30,000 tokens. Sending the entire document to an embedding model is often impractical and makes retrieval less precise.

A chunk should ideally represent one coherent idea, such as a policy section or a few related paragraphs.

### Chunking strategies

| Strategy                      | When to use it                                    |
| ----------------------------- | ------------------------------------------------- |
| Fixed character/token windows | Simple baseline or unstructured text              |
| Sliding windows with overlap  | Preserve context across boundaries                |
| Recursive splitting           | Split at paragraph, sentence and word boundaries  |
| Structure-aware chunking      | Preserve PDF headings, sections and tables        |
| Semantic chunking             | Split based on changes in meaning                 |
| Parent-child chunking         | Retrieve small passages but return larger context |

For enterprise RAG, start with structure-aware chunking with a token limit and modest overlap. Semantic chunking can be evaluated later rather than assumed to be superior.

### Example: page-aware recursive chunking

Install:

```
pip install langchain-text-splitters tiktoken
```

Create `app/ingestion/chunker.py`:

```
from dataclasses import dataclassimport tiktokenfrom langchain_text_splitters import RecursiveCharacterTextSplitter@dataclass(frozen=True)class DocumentChunk:    chunk_id: str    document_id: str    page_number: int    chunk_index: int    text: str    token_count: intdef chunk_pages(    document_id: str,    pages: list[dict],    chunk_size: int = 700,    chunk_overlap: int = 100,) -> list[DocumentChunk]:    if chunk_size <= 0:        raise ValueError("chunk_size must be positive")    if not 0 <= chunk_overlap < chunk_size:        raise ValueError("Overlap must be smaller than chunk size")    encoding = tiktoken.get_encoding("cl100k_base")    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(        encoding_name="cl100k_base",        chunk_size=chunk_size,        chunk_overlap=chunk_overlap,        separators=["\n\n", "\n", ". ", " ", ""],    )    chunks: list[DocumentChunk] = []    index = 0    for page in pages:        page_number = page["page_number"]        text = page["text"].strip()        if not text:            continue        for piece in splitter.split_text(text):            piece = piece.strip()            if not piece:                continue            chunks.append(                DocumentChunk(                    chunk_id=f"{document_id}:chunk:{index}",                    document_id=document_id,                    page_number=page_number,                    chunk_index=index,                    text=piece,                    token_count=len(encoding.encode(piece)),                )            )            index += 1    return chunks
```

The `700`-token chunk size and `100`-token overlap are illustrative starting points, not universal best settings.

One important nuance: splitting each page independently preserves page provenance, but it can create poor chunks when a paragraph continues onto the next page. In production, reconstruct document sections across page boundaries where layout and headings allow it, and keep the source-page range for every chunk.

For table chunks, use a separate table-aware strategy so the column headers remain attached to their rows.

### Why overlap matters

Suppose a sentence ends at one chunk's boundary and its conclusion appears in the next chunk. Without overlap, retrieval may return an incomplete statement.

Overlap preserves some surrounding context, but excessive overlap increases embedding costs, duplicate search results and index size. Tune it using retrieval evaluations, not intuition alone.

## 8. Add document metadata and provenance

A senior engineer should treat a chunk as a structured, versioned data object, not merely a string.

For a financial compliance platform, a chunk might have this metadata:

```
{    "chunk_id": "policy-123:v3:chunk:17",    "document_id": "policy-123",    "document_version": "3",    "tenant_id": "tenant-42",    "page_number": 8,    "section": "Marketing Performance Claims",    "source_uri": "s3://private-bucket/policies/policy-123.pdf",    "content_hash": "sha256:...",    "parser_version": "pdf-parser-v2",    "embedding_model": "embedding-model-version",    "access_labels": ["compliance"],}
```

The values are illustrative. Never put sensitive source text or secrets in ordinary application logs.

Important fields include:

- Document identity and version: which source and revision produced the chunk?
- Page and section: where did the information originate?
- Tenant and access labels: who is allowed to retrieve it?
- Content hash: has the source or extracted content changed?
- Pipeline versions: which parser, chunker and embedding model were used?
- Lifecycle state: is the document processing, published, failed or superseded?

A content hash can be generated with Python:

```
import hashlibdef sha256_file(path: str) -> str:    digest = hashlib.sha256()    with open(path, "rb") as file:        for block in iter(            lambda: file.read(1024 * 1024),            b"",        ):            digest.update(block)    return digest.hexdigest()
```

This supports deduplication and change detection. A hash alone is not an authorization mechanism, and identical files may still require separate tenant-scoped records.

## 9. Connect parsing to embeddings and a vector database

Now combine the stages into the ingestion workflow.

Example end-to-end data flow

`investment_policy.pdf`

Parse pages and tables

Normalize and validate

Create provenance-aware chunks

Generate embeddings in batches

PostgreSQL

Metadata, versions, audit state

pgvector / Qdrant

Vector similarity search

Choose pgvector or Qdrant based on your scale and operational requirements. You do not necessarily need both.

### Example embedding code

Here is a provider-independent interface for embedding generation. Your production implementation can call a hosted embedding API or a locally deployed embedding model.

```
from typing import Protocolclass EmbeddingProvider(Protocol):    async def embed_documents(        self,        texts: list[str],    ) -> list[list[float]]:        """Return one vector per input text."""        ...
```

A concrete provider must implement the protocol. The ingestion service should not depend directly on a specific vendor's SDK.

For example, an OpenAI-backed implementation could look like this:

```
from openai import AsyncOpenAIclass OpenAIEmbeddingProvider:    def __init__(self, model: str) -> None:        self.client = AsyncOpenAI()        self.model = model    async def embed_documents(        self,        texts: list[str],    ) -> list[list[float]]:        if not texts:            return []        response = await self.client.embeddings.create(            model=self.model,            input=texts,        )        # Restore input order explicitly.        ordered = sorted(response.data, key=lambda item: item.index)        return [item.embedding for item in ordered]
```

Install the SDK with `pip install openai` and configure its API key through a secret manager or environment configuration.

For a robust implementation, also enforce provider batch-size and token limits, request timeouts, bounded retries, rate limiting, and a dead-letter path for permanently failed batches.

### What happens after embedding?

Suppose a chunk says:

> Marketing materials must not imply guaranteed investment returns unless the claim is specifically permitted by the applicable policy.

The embedding model produces a vector such as:

```
[0.021, -0.183, 0.074, ...]
```

The abbreviated values are illustrative. The full vector depends on the model.

Store the vector alongside the chunk's content, document version and provenance. At query time, embed the user's question, search for relevant chunks, apply authorization filters, and return the original text with citations.

The embedding is not a replacement for the source text.

## 10. Design a production ingestion service

In a real enterprise application, I would separate orchestration from parsing, embedding and persistence.

A recommended repository structure:

```
app/
├── api/
│   └── ingestion_routes.py
├── ingestion/
│   ├── models.py
│   ├── service.py
│   ├── pdf_parser.py
│   ├── ocr.py
│   ├── table_extractor.py
│   ├── normalizer.py
│   ├── chunker.py
│   ├── quality.py
│   └── pipeline.py
├── embeddings/
│   ├── protocols.py
│   └── provider.py
├── repositories/
│   ├── document_repository.py
│   └── chunk_repository.py
├── workers/
│   └── ingestion_tasks.py
└── observability/
    └── metrics.py
```

Each module has a clear responsibility:

| Module                | Responsibility                                                |
| --------------------- | ------------------------------------------------------------- |
| `ingestion_routes.py` | Authenticate requests and enqueue jobs                        |
| `pdf_parser.py`       | Extract native PDF text and page structure                    |
| `ocr.py`              | Extract text from scanned pages                               |
| `table_extractor.py`  | Preserve table structure                                      |
| `quality.py`          | Detect empty pages, suspicious output and extraction failures |
| `chunker.py`          | Produce bounded, provenance-aware chunks                      |
| `provider.py`         | Generate embeddings through a model provider                  |
| `chunk_repository.py` | Persist and retrieve chunks and vectors                       |
| `ingestion_tasks.py`  | Run background jobs with retries and failure handling         |
| `pipeline.py`         | Coordinate the stages and manage job state                    |

### How the API should work

Do not process a large PDF synchronously inside the upload request. Return a job identifier and process it in a background worker.

```
from uuid import UUID, uuid4from fastapi import APIRouter, Depends, HTTPExceptionfrom pydantic import BaseModelrouter = APIRouter()class IngestionAccepted(BaseModel):    job_id: UUID    status: str@router.post(    "/documents/{document_id}/ingestion",    status_code=202,    response_model=IngestionAccepted,)async def start_ingestion(    document_id: UUID,    principal=Depends(require_document_write_access),    # Inject a durable task publisher in the real application.    publisher=Depends(get_ingestion_publisher),):    job_id = uuid4()    # Verify document ownership and authorization, then    # persist the job before returning.    await verify_document_access(        document_id=document_id,        principal=principal,    )    await create_ingestion_job(        job_id=job_id,        document_id=document_id,        tenant_id=principal.tenant_id,        requested_by=principal.user_id,    )    try:        await publisher.publish(            job_id=str(job_id),            document_id=str(document_id),            tenant_id=str(principal.tenant_id),        )    except Exception as exc:        await mark_ingestion_dispatch_failed(job_id)        raise HTTPException(            status_code=503,            detail="Unable to schedule ingestion",        ) from exc    return IngestionAccepted(        job_id=job_id,        status="queued",    )
```

The functions such as `require_document_write_access`, `get_ingestion_publisher` and the persistence functions are application-specific dependencies; they need concrete implementations.

There is also an important reliability issue: saving a database record and publishing a queue message are two separate operations. A crash between them can leave a job stuck in `queued` without a message. For high-reliability ingestion, use a transactional outbox pattern or another durable dispatch mechanism, along with a reconciliation process.

### Recommended worker lifecycle

1\. Claim the job

Validate tenant ownership, acquire a lease and transition the job to `processing`.

2\. Parse and validate

Extract text, tables and OCR content. Record warnings and quality metrics.

3\. Chunk and embed

Use bounded batches, rate limits and retry policies.

4\. Persist results

Write versioned chunks and vectors idempotently, with the appropriate tenant metadata.

5\. Validate and publish

Verify indexing, run quality checks and atomically activate the new document version.

Use Celery with Redis or another suitable queue when you need distributed workers. Redis is useful for queue transport and caching, but it should not automatically be treated as the durable source of truth for document state.

## 11. Critical production engineering concerns

These are the issues that distinguish a demo RAG pipeline from an enterprise implementation.

### A. Idempotency and duplicate ingestion

A worker may execute a task more than once after a timeout or crash.

Use a deterministic ingestion key based on the source hash, document version and pipeline configuration. Make database writes safe to repeat, and use unique constraints to prevent duplicate chunks.

Do not delete the currently published index before the replacement has succeeded. Build the new version separately, validate it, and switch the active version only after successful completion.

### B. Security

PDFs are untrusted inputs. They may be malformed, excessively large or contain hostile embedded content.

- Enforce upload size, page-count and processing-time limits.
- Validate actual file content instead of trusting the extension.
- Scan files for malware and isolate parsing workers.
- Prevent unauthorized access to source files and extracted text.
- Apply tenant and document authorization filters during retrieval, not merely during upload.
- Treat instructions inside documents as untrusted data, not as commands to an agent.
- Keep sensitive document contents out of logs and error messages.

### C. Quality gates

A parser can return text without returning correct text. Measure quality rather than relying only on whether an exception occurred.

Useful metrics include:

| Metric                    | Why it matters                                 |
| ------------------------- | ---------------------------------------------- |
| Text extraction coverage  | Detect pages with missing content              |
| OCR failure rate          | Detect problematic scanned documents           |
| Table extraction accuracy | Preserve financial relationships               |
| Chunk count per page      | Detect unexpectedly empty or fragmented output |
| Embedding failure rate    | Monitor model-provider reliability             |
| Ingestion duration        | Identify slow stages and bottlenecks           |
| Citation accuracy         | Confirm retrieved evidence maps to the source  |
| Retrieval recall\@k       | Test whether expected evidence is found        |

Build a labeled test set containing digital PDFs, scanned PDFs, multi-column reports, tables, footnotes and malformed files. Compare extraction output against manually verified ground truth.

### D. Observability and recovery

Use structured logs and distributed tracing with a correlation ID and ingestion job ID. Track stage durations, retries, failures and provider usage.

Avoid logging raw document content by default.

A production state model could be:

```
QUEUED
  → PROCESSING
  → VALIDATING
  → PUBLISHING
  → SUCCEEDED

PROCESSING → RETRY_WAIT → PROCESSING
PROCESSING → FAILED
VALIDATING → QUARANTINED
```

Make state transitions explicit. Use bounded retries with exponential backoff and jitter for transient failures, and route permanent failures to a dead-letter queue or review workflow.

## 12. How does this connect to RAG and LangGraph?

Document ingestion is usually an offline or asynchronous data pipeline. RAG is the online question-answering pipeline.

### Offline ingestion pipeline

Upload → Parse → OCR → Normalize → Chunk → Embed → Index → Publish

Shared versioned document and vector indexes

### Online RAG / LangGraph workflow

Question → Authorization → Retrieval → Reranking → Evidence validation → LLM answer → Citation validation

In a LangGraph architecture, you might use graph nodes for query rewriting, retrieval, grading and answer generation. But you do not need LangGraph just to parse a PDF. A conventional service pipeline and a background worker are often simpler and more reliable for ingestion.

Use LangGraph when the workflow genuinely benefits from stateful orchestration, conditional routing, retries, human approval or resumable execution.

For your financial compliance project, an additional publication rule is important: a newly ingested policy should not become authoritative until its extraction quality, version metadata and indexing have been validated.

## 13. How to test your PDF parser

At minimum, write tests for ordinary text, scanned pages, empty documents, encrypted PDFs, malformed inputs and table extraction.

Example using `pytest`:

```
from pathlib import Pathimport fitzimport pytestfrom app.ingestion.pdf_parser import parse_pdfdef create_test_pdf(path: Path) -> None:    with fitz.open() as pdf:        page = pdf.new_page()        page.insert_text(            (72, 72),            "Investment Risk Policy",        )        pdf.save(path)def test_parse_pdf_extracts_text(tmp_path: Path) -> None:    path = tmp_path / "policy.pdf"    create_test_pdf(path)    document = parse_pdf(str(path))    assert document.page_count == 1    assert document.pages[0].page_number == 1    assert "Investment Risk Policy" in document.pages[0].textdef test_missing_pdf_raises_error(tmp_path: Path) -> None:    path = tmp_path / "missing.pdf"    with pytest.raises(FileNotFoundError):        parse_pdf(str(path))def test_non_pdf_extension_is_rejected(tmp_path: Path) -> None:    path = tmp_path / "document.txt"    path.write_text("not a PDF")    with pytest.raises(ValueError):        parse_pdf(str(path))
```

Run the tests with:

```
pip install pytest
pytest -q
```

These tests validate the basic parser contract. They do not validate OCR accuracy, reading order, security, table correctness or end-to-end retrieval. Those require additional fixtures and integration tests.

## 14. Staff-level interview answer

Question: How would you design document ingestion for an enterprise RAG platform?

A strong answer would be:

> I design ingestion as an asynchronous, versioned data pipeline rather than a single PDF-to-text function.
>
> First, I validate the source file, enforce authorization and resource limits, and compute a content hash. I use native extraction for digital PDFs and selectively route scanned or layout-heavy pages through OCR and document-layout processing. I preserve page numbers, section boundaries, table relationships and source metadata throughout extraction.
>
> Next, I apply normalization and quality checks, then perform structure-aware chunking with token limits and appropriate overlap. I generate embeddings in bounded batches and persist vectors with document version, tenant, page and access-control metadata.
>
> I use idempotent writes, durable job state, bounded retries, observability and a dead-letter path. I publish a new document version only after parsing, indexing and quality validation succeed, without disrupting the previous active version.
>
> Finally, I evaluate extraction quality and retrieval performance against representative documents, including scanned files and tables. I measure retrieval recall, citation accuracy, latency and failure rates, and use those results to tune the pipeline.

### What makes this a senior/staff answer?

It covers not just the parsing library, but also:

- Reliability and recovery.
- Data quality and retrieval correctness.
- Security and tenant isolation.
- Versioning and safe publication.
- Cost and throughput.
- Testing and measurable outcomes.
- Clear boundaries between ingestion, storage and online retrieval.

The key principle is: the quality of your RAG system is bounded by the quality of the evidence you extract, preserve and retrieve. A sophisticated LLM cannot reliably compensate for missing paragraphs, corrupted tables or incorrect source attribution.