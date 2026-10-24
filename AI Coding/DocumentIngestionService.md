
from __future__ import annotations

import math
from typing import Protocol, Sequence

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionDocument:
    document_id: str
    tenant_id: str
    version: str
    storage_path: str


class DocumentParser(Protocol):
    async def parse(self, storage_path: str) -> Sequence[SourcePage]:
        ...


class Chunker(Protocol):
    def chunk_pages(
        self,
        document_id: str,
        pages: Sequence[SourcePage],
    ) -> list[Chunk]:
        ...


class EmbeddingProvider(Protocol):
    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...


class ChunkRepository(Protocol):
    async def publish_document_version(
        self,
        *,
        tenant_id: str,
        document_id: str,
        version: str,
        records: list[dict],
    ) -> None:
        """
        Publish a complete version idempotently.

        Required implementation guarantees:
        - Enforce tenant/document scope.
        - Make repeated publication of the same version safe.
        - Do not expose a partially published version.
        - Preserve the previous active version if publication fails.
        - Retire old versions only after successful publication.

        If PostgreSQL and Qdrant are both used, coordinate publication
        through an explicit state machine/outbox; do not assume a
        cross-system atomic transaction exists.
        """
        ...


class DocumentIngestionService:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        repository: ChunkRepository,
        *,
        embedding_batch_size: int = 64,
        expected_dimensions: int | None = 1536,
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")

        if expected_dimensions is not None and expected_dimensions <= 0:
            raise ValueError("expected_dimensions must be positive")

        self.parser = parser
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.embedding_batch_size = embedding_batch_size
        self.expected_dimensions = expected_dimensions

    async def ingest(self, document: IngestionDocument) -> int:
        self._validate_document(document)

        # 1. Parse the source document.
        parsed_pages = await self.parser.parse(document.storage_path)

        if not parsed_pages:
            raise ValueError("Document contains no pages")

        usable_pages = [
            page for page in parsed_pages
            if page.text and page.text.strip()
        ]

        if not usable_pages:
            raise ValueError("No usable text extracted from document")

        # 2. Chunk the document.
        chunks = self.chunker.chunk_pages(
            document_id=document.document_id,
            pages=usable_pages,
        )

        if not chunks:
            raise ValueError("No chunks were produced")

        # 3. Generate and validate embeddings.
        vectors: list[list[float]] = []

        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]

            batch_vectors = (
                await self.embedding_provider.embed_documents(
                    [chunk.text for chunk in batch]
                )
            )

            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} embeddings; "
                    f"received {len(batch_vectors)}"
                )

            for vector in batch_vectors:
                if not vector:
                    raise RuntimeError("Received an empty embedding vector")

                if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in vector
                ):
                    raise RuntimeError(
                        "Embedding contains invalid numeric values"
                    )

                if (
                    self.expected_dimensions is not None
                    and len(vector) != self.expected_dimensions
                ):
                    raise RuntimeError(
                        f"Expected {self.expected_dimensions} dimensions; "
                        f"received {len(vector)}"
                    )

            vectors.extend(batch_vectors)

        if len(vectors) != len(chunks):
            raise RuntimeError("Final embedding count mismatch")

        # 4. Build version-scoped records.
        records: list[dict] = []

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors, strict=True)
        ):
            records.append({
                "chunk_id": (
                    f"{document.document_id}:"
                    f"v{document.version}:chunk:{index}"
                ),
                "document_id": document.document_id,
                "tenant_id": document.tenant_id,
                "document_version": document.version,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "text_content": chunk.text,
                "embedding": vector,
            })

        # 5. Publish through a version-aware repository operation.
        # Do not expose a new active version until publication succeeds.
        await self.repository.publish_document_version(
            tenant_id=document.tenant_id,
            document_id=document.document_id,
            version=document.version,
            records=records,
        )

        return len(records)

    @staticmethod
    def _validate_document(document: IngestionDocument) -> None:
        if not document.document_id.strip():
            raise ValueError("document_id must not be empty")

        if not document.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")

        if not document.version.strip():
            raise ValueError("version must not be empty")

        if not document.storage_path.strip():
            raise ValueError("storage_path must not be empty")
