Yes. Below is a **single-file, interview-quality RAG implementation** that improves your original code while keeping it easy to understand.

It uses:

* Wikipedia as the document source
* Sentence Transformers for embeddings
* FAISS for vector retrieval
* Better sentence-aware chunking
* Metadata
* Cosine similarity
* Top-K retrieval
* Optional reranking
* A generative LLM step
* Source citations
* Clear separation of ingestion → retrieval → generation

For a single-file project, this is a good balance between **proper architecture and simplicity**.

### Install

```bash
pip install streamlit wikipedia sentence-transformers faiss-cpu transformers torch
```

### `app.py`

```python
import streamlit as st
import wikipedia
import faiss
import numpy as np
import re

from dataclasses import dataclass
from typing import List, Dict

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Production-Style RAG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Production-Style RAG Pipeline")
st.caption(
    "Wikipedia → Chunking → Embeddings → FAISS → Reranking → LLM"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cross encoder used for reranking retrieved documents
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Small instruction model for demonstration.
# In production you can replace this with an API-based LLM.
GENERATION_MODEL = "google/flan-t5-base"

TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 4

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


# ============================================================
# 3. DATA MODEL
# ============================================================

@dataclass
class DocumentChunk:
    """
    Represents one chunk of a document.
    """

    chunk_id: int
    text: str
    source: str
    title: str


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# 5. LOAD RERANKER
# ============================================================

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        RERANKER_MODEL
    )


# ============================================================
# 6. LOAD GENERATION MODEL
# ============================================================

@st.cache_resource
def load_generation_model():

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL
    )

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
    )

    return generator


# ============================================================
# 7. LOAD WIKIPEDIA DOCUMENT
# ============================================================

def load_wikipedia_document(topic: str):

    try:

        page = wikipedia.page(
            topic,
            auto_suggest=False
        )

        return {
            "title": page.title,
            "text": page.content,
            "url": page.url,
        }

    except wikipedia.exceptions.PageError:

        st.error(
            f"No Wikipedia page found for '{topic}'."
        )

        return None

    except wikipedia.exceptions.DisambiguationError as e:

        st.error(
            "The topic is ambiguous. Try one of these:"
        )

        st.write(e.options[:10])

        return None

    except Exception as e:

        st.error(
            f"Error loading Wikipedia page: {e}"
        )

        return None


# ============================================================
# 8. CLEAN DOCUMENT
# ============================================================

def clean_text(text: str) -> str:

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove excessive newlines
    text = re.sub(
        r"\n+",
        "\n",
        text
    )

    return text.strip()


# ============================================================
# 9. SENTENCE-AWARE CHUNKING
# ============================================================

def chunk_document(
    text: str,
    title: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:

    """
    Split document into approximately chunk_size word chunks.

    Instead of blindly splitting tokens, we try to preserve
    sentence boundaries.
    """

    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []

    current_chunk = []
    current_length = 0

    chunk_id = 0

    for sentence in sentences:

        words = sentence.split()

        sentence_length = len(words)

        # If adding the sentence exceeds chunk size,
        # store current chunk.
        if (
            current_length + sentence_length
            > chunk_size
            and current_chunk
        ):

            chunk_text = " ".join(
                current_chunk
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    title=title,
                )
            )

            chunk_id += 1

            # Keep overlap from previous chunk
            overlap_words = []

            current_words = " ".join(
                current_chunk
            ).split()

            if len(current_words) > overlap:

                overlap_words = current_words[
                    -overlap:
                ]

            current_chunk = overlap_words

            current_length = len(
                overlap_words
            )

        current_chunk.extend(words)

        current_length += sentence_length

    # Add final chunk
    if current_chunk:

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=" ".join(current_chunk),
                source=source,
                title=title,
            )
        )

    return chunks


# ============================================================
# 10. CREATE VECTOR INDEX
# ============================================================

def create_vector_index(
    chunks: List[DocumentChunk],
    embedding_model,
):

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors
    # = cosine similarity
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype(
            np.float32
        )
    )

    return index, embeddings


# ============================================================
# 11. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query: str,
    index,
    chunks: List[DocumentChunk],
    embedding_model,
    top_k: int = TOP_K_RETRIEVAL,
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding.astype(
            np.float32
        ),
        min(top_k, len(chunks)),
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx == -1:
            continue

        results.append(
            {
                "chunk": chunks[idx],
                "score": float(score),
            }
        )

    return results


# ============================================================
# 12. RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    query: str,
    retrieved_documents: List[Dict],
    reranker,
    top_k: int = TOP_K_RERANK,
):

    if not retrieved_documents:

        return []

    pairs = [
        (
            query,
            item["chunk"].text
        )
        for item in retrieved_documents
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    reranked = []

    for item, score in zip(
        retrieved_documents,
        rerank_scores,
    ):

        reranked.append(
            {
                "chunk": item["chunk"],
                "retrieval_score": item["score"],
                "rerank_score": float(score),
            }
        )

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]


# ============================================================
# 13. BUILD RAG CONTEXT
# ============================================================

def build_context(
    documents: List[Dict]
):

    context_parts = []

    for i, item in enumerate(
        documents,
        start=1,
    ):

        chunk = item["chunk"]

        context_parts.append(
            f"""
SOURCE {i}
Title: {chunk.title}
Source: {chunk.source}
Chunk ID: {chunk.chunk_id}

Content:
{chunk.text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# 14. GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    context: str,
    generator,
):

    prompt = f"""
You are a reliable Retrieval-Augmented Generation assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain the answer, say:
   "I don't have enough information in the retrieved documents."
4. Give a concise and clear answer.

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(
        prompt
    )

    answer = result[0]["generated_text"]

    return answer.strip()


# ============================================================
# 15. MAIN RAG PIPELINE
# ============================================================

def run_rag_pipeline(
    topic: str,
    query: str,
):

    # --------------------------------------------------------
    # STEP 1: LOAD DOCUMENT
    # --------------------------------------------------------

    with st.spinner(
        "Loading Wikipedia document..."
    ):

        document = load_wikipedia_document(
            topic
        )

    if not document:

        return None

    # --------------------------------------------------------
    # STEP 2: CHUNK DOCUMENT
    # --------------------------------------------------------

    with st.spinner(
        "Chunking document..."
    ):

        chunks = chunk_document(
            text=document["text"],
            title=document["title"],
            source=document["url"],
        )

    # --------------------------------------------------------
    # STEP 3: LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = load_embedding_model()

    # --------------------------------------------------------
    # STEP 4: CREATE VECTOR INDEX
    # --------------------------------------------------------

    with st.spinner(
        "Creating vector embeddings..."
    ):

        index, _ = create_vector_index(
            chunks,
            embedding_model,
        )

    # --------------------------------------------------------
    # STEP 5: RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "Retrieving relevant chunks..."
    ):

        retrieved_documents = retrieve_documents(
            query=query,
            index=index,
            chunks=chunks,
            embedding_model=embedding_model,
            top_k=TOP_K_RETRIEVAL,
        )

    # --------------------------------------------------------
    # STEP 6: RERANKING
    # --------------------------------------------------------

    with st.spinner(
        "Reranking retrieved documents..."
    ):

        reranker = load_reranker()

        reranked_documents = rerank_documents(
            query=query,
            retrieved_documents=retrieved_documents,
            reranker=reranker,
            top_k=TOP_K_RERANK,
        )

    # --------------------------------------------------------
    # STEP 7: BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        reranked_documents
    )

    # --------------------------------------------------------
    # STEP 8: GENERATION
    # --------------------------------------------------------

    with st.spinner(
        "Generating answer..."
    ):

        generator = load_generation_model()

        answer = generate_answer(
            query=query,
            context=context,
            generator=generator,
        )

    return {
        "answer": answer,
        "chunks": chunks,
        "retrieved": retrieved_documents,
        "reranked": reranked_documents,
        "context": context,
    }


# ============================================================
# 16. STREAMLIT UI
# ============================================================

st.sidebar.header(
    "RAG Configuration"
)

st.sidebar.write(
    f"Embedding model: `{EMBEDDING_MODEL}`"
)

st.sidebar.write(
    f"Retriever Top-K: `{TOP_K_RETRIEVAL}`"
)

st.sidebar.write(
    f"Reranker Top-K: `{TOP_K_RERANK}`"
)

st.sidebar.write(
    f"Chunk size: `{CHUNK_SIZE}`"
)

st.sidebar.write(
    f"Chunk overlap: `{CHUNK_OVERLAP}`"
)


topic = st.text_input(
    "📚 Wikipedia Topic",
    value="Apple Inc.",
)

query = st.text_input(
    "💬 Ask a question",
    placeholder="Who founded Apple?",
)


if st.button(
    "🚀 Run RAG",
    type="primary",
):

    if not topic.strip():

        st.warning(
            "Please enter a topic."
        )

    elif not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        result = run_rag_pipeline(
            topic=topic,
            query=query,
        )

        if result:

            # ==================================================
            # ANSWER
            # ==================================================

            st.subheader(
                "✅ Answer"
            )

            st.write(
                result["answer"]
            )

            # ==================================================
            # SOURCES
            # ==================================================

            st.subheader(
                "📚 Sources"
            )

            for i, item in enumerate(
                result["reranked"],
                start=1,
            ):

                chunk = item["chunk"]

                with st.expander(
                    f"Source {i} — "
                    f"Rerank score: "
                    f"{item['rerank_score']:.4f}"
                ):

                    st.write(
                        f"**Title:** "
                        f"{chunk.title}"
                    )

                    st.write(
                        f"**Source:** "
                        f"{chunk.source}"
                    )

                    st.write(
                        f"**Chunk ID:** "
                        f"{chunk.chunk_id}"
                    )

                    st.write(
                        f"**Vector score:** "
                        f"{item['retrieval_score']:.4f}"
                    )

                    st.write(
                        f"**Rerank score:** "
                        f"{item['rerank_score']:.4f}"
                    )

                    st.write(
                        chunk.text
                    )

            # ==================================================
            # RETRIEVAL DEBUGGING
            # ==================================================

            with st.expander(
                "🔍 Retrieval Debugging"
            ):

                st.write(
                    f"Total chunks: "
                    f"{len(result['chunks'])}"
                )

                st.write(
                    f"Retrieved chunks: "
                    f"{len(result['retrieved'])}"
                )

                st.write(
                    f"Final chunks after reranking: "
                    f"{len(result['reranked'])}"
                )

                st.write(
                    "The final context sent to the "
                    "generation model:"
                )

                st.text(
                    result["context"]
                )
```

### Run it

```bash
streamlit run app.py
```

---

## What makes this better than your original

Your original pipeline was essentially:

```text
Wikipedia
   ↓
Chunk
   ↓
Embedding
   ↓
FAISS
   ↓
Top 3
   ↓
RoBERTa QA
```

The improved version is:

```text
                    INGESTION
                       │
                       ▼
                  Wikipedia
                       │
                       ▼
                 Text Cleaning
                       │
                       ▼
             Sentence-aware Chunking
                       │
                       ▼
                   Embedding
                       │
                       ▼
                  FAISS Index
                       │
                       │
                       ▼
                     QUERY
                       │
                       ▼
                Query Embedding
                       │
                       ▼
              Dense Retrieval
                  Top 10
                       │
                       ▼
                   Reranker
                       │
                       ▼
                   Top 4
                       │
                       ▼
                Context Builder
                       │
                       ▼
                  Prompt
                       │
                       ▼
                Generative LLM
                       │
                       ▼
              Answer + Sources
```

The **most important interview concept** here is that you now have two distinct logical stages:

### Offline/ingestion

```text
Document
 → Clean
 → Chunk
 → Embed
 → Index
```

### Online/query

```text
Query
 → Embed
 → Retrieve
 → Rerank
 → Build Context
 → Generate
```

One caveat: this is still a **single-file demo**, so it rebuilds the Wikipedia index for the selected topic. In a real production system, you would persist the embeddings in **Qdrant/pgvector**, keep ingestion separate from query serving, expose the query pipeline through **FastAPI**, add **Redis caching**, metadata/permission filtering, evaluation, tracing, and monitoring.
