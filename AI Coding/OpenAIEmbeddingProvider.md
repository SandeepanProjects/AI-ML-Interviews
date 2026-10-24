
from __future__ import annotations

import asyncio
from collections.abc import Sequence

from openai import AsyncOpenAI


class OpenAIEmbeddingProvider:
    """Asynchronous embedding provider for a RAG ingestion pipeline."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        max_batch_size: int = 128,
        max_concurrency: int = 3,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")

        if max_batch_size <= 0 or max_batch_size > 2048:
            raise ValueError("max_batch_size must be between 1 and 2048")

        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.model = model
        self.max_batch_size = max_batch_size
        self.max_concurrency = max_concurrency

        # Configure the client once and reuse it across requests.
        # The SDK handles retryable API failures automatically.
        self.client = AsyncOpenAI(
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._closed = False

    async def _embed_batch(
        self,
        batch: list[str],
    ) -> list[list[float]]:
        async with self._semaphore:
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

        # Validate ordering and response completeness.
        items = sorted(response.data, key=lambda item: item.index)

        if len(items) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings; received {len(items)}"
            )

        expected_indices = list(range(len(batch)))
        actual_indices = [item.index for item in items]

        if actual_indices != expected_indices:
            raise RuntimeError(
                "Embedding response contains missing or invalid indices"
            )

        vectors = [item.embedding for item in items]

        if any(not vector for vector in vectors):
            raise RuntimeError("Received an empty embedding vector")

        dimensions = len(vectors[0])

        if any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError(
                "Embedding vectors have inconsistent dimensions"
            )

        return vectors

    async def _embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if self._closed:
            raise RuntimeError("Embedding provider is closed")

        if not texts:
            return []

        # Reject invalid inputs before making API requests.
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string")

            if not text.strip():
                raise ValueError(f"texts[{index}] must not be empty")

        batches = [
            list(texts[start : start + self.max_batch_size])
            for start in range(0, len(texts), self.max_batch_size)
        ]

        # asyncio.gather preserves the order of its input awaitables.
        # The semaphore bounds the number of active API requests.
        batch_results = await asyncio.gather(
            *(self._embed_batch(batch) for batch in batches)
        )

        vectors = [
            vector
            for batch_vectors in batch_results
            for vector in batch_vectors
        ]

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} vectors; received {len(vectors)}"
            )

        return vectors

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text])
        return vectors[0]

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self.client.close()

    async def __aenter__(self) -> OpenAIEmbeddingProvider:
        if self._closed:
            raise RuntimeError("Embedding provider is closed")
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.aclose()





How to use it


Example A: Embed documents

import asyncio

async def main() -> None:
    async with OpenAIEmbeddingProvider() as provider:
        documents = [
            "Portfolio diversification reduces concentration risk.",
            "Bonds can provide income and help diversify a portfolio.",
            "Liquidity risk occurs when assets cannot be sold quickly."
        ]

        vectors = await provider.embed_documents(documents)

        print("Documents:", len(documents))
        print("Vectors:", len(vectors))
        print("Dimensions:", len(vectors[0]))

asyncio.run(main())





Example B: Embed a user query

import asyncio

async def main() -> None:
    async with OpenAIEmbeddingProvider() as provider:
        vector = await provider.embed_query(
            "What are the risks of portfolio concentration?"
        )

        print("Embedding dimensions:", len(vector))

asyncio.run(main())

