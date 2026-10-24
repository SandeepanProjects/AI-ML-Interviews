
from dataclasses import dataclass
from typing import Sequence

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class SourcePage:
    page_number: int
    text: str
    section: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    section: str | None
    token_count: int


class DocumentChunker:
    """Create token-aware chunks while preserving source-page metadata."""

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and < chunk_size"
            )

        self.encoder = tiktoken.get_encoding("cl100k_base")

        self.splitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                add_start_index=True,
                strip_whitespace=True,
            )
        )

    def chunk_pages(
        self,
        document_id: str,
        pages: Sequence[SourcePage],
    ) -> list[Chunk]:
        if not document_id.strip():
            raise ValueError("document_id must not be empty")

        # Build normalized text and its source-page character ranges.
        page_texts: list[str] = []
        page_mappings: list[tuple[int, int, SourcePage]] = []

        current_pos = 0
        page_separator = "\n"

        for page in pages:
            page_text = page.text.strip()

            if not page_text:
                continue

            start_idx = current_pos
            end_idx = start_idx + len(page_text)

            page_texts.append(page_text)
            page_mappings.append((start_idx, end_idx, page))

            current_pos = end_idx + len(page_separator)

        if not page_texts:
            return []

        full_text = page_separator.join(page_texts)
        docs = self.splitter.create_documents([full_text])

        chunks: list[Chunk] = []

        for index, doc in enumerate(docs):
            chunk_text = doc.page_content.strip()

            if not chunk_text:
                continue

            start_ch = doc.metadata.get("start_index")

            if not isinstance(start_ch, int) or start_ch < 0:
                raise RuntimeError(
                    f"Missing or invalid start_index for chunk {index}"
                )

            # The splitter's content and offsets must remain aligned.
            end_ch = start_ch + len(doc.page_content)

            intersected_pages: list[SourcePage] = []

            for page_start_ch, page_end_ch, page in page_mappings:
                overlaps = (
                    max(start_ch, page_start_ch)
                    < min(end_ch, page_end_ch)
                )

                if overlaps:
                    intersected_pages.append(page)

            # Never fabricate a source page for a chunk.
            if not intersected_pages:
                raise RuntimeError(
                    f"Could not map chunk {index} to a source page"
                )

            # Policy: section identifies the first intersected source page.
            first_page = intersected_pages[0]
            last_page = intersected_pages[-1]

            token_count = len(self.encoder.encode(chunk_text))

            if token_count > self._configured_chunk_size():
                raise RuntimeError(
                    f"Chunk {index} exceeds the configured token limit"
                )

            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:chunk:{index}",
                    document_id=document_id,
                    text=chunk_text,
                    page_start=first_page.page_number,
                    page_end=last_page.page_number,
                    section=first_page.section,
                    token_count=token_count,
                )
            )

        return chunks

    def _configured_chunk_size(self) -> int:
        # Keep the configured limit explicit rather than relying on
        # implementation-specific splitter internals.
        return self.chunk_size




