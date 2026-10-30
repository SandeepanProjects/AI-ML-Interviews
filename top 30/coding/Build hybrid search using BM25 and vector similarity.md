pip install rank_bm25 numpy openai


import os
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from openai import OpenAI

# Initialize the OpenAI Client (used to embed incoming runtime queries)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-key-here"))

class HybridSearchEngine:
    def __init__(self, documents: List[Dict[str, Any]], rrf_k: int = 60):
        """
        Initializes the Hybrid Search engine with a corpus of pre-embedded documents.
        
        documents: List of dicts, each containing:
                   - "id": Unique identifier
                   - "content": Raw text string
                   - "embedding": List[float] (pre-computed vector matching the text)
        rrf_k: Constant parameter for Reciprocal Rank Fusion (industry standard is ~60)
        """
        self.documents = documents
        self.rrf_k = rrf_k
        
        # 1. Initialize BM25 (Requires tokenization of the corpus)
        self.corpus_texts = [doc["content"] for doc in documents]
        self.tokenized_corpus = [text.lower().split(" ") for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 2. Extract and format Dense Vectors into a structured NumPy array for fast operations
        self.doc_embeddings = np.array([doc["embedding"] for doc in documents], dtype=np.float32)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Helper to generate dense embedding vector for the search query."""
        response = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _compute_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """Performs vector dot-product similarity calculations across the corpus matrix."""
        # Standard cosine similarity formula: (A dot B) / (||A|| * ||B||)
        dot_product = np.dot(self.doc_embeddings, query_vector)
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        # Avoid division by zero warnings
        norms = doc_norms * query_norm
        norms[norms == 0] = 1e-10
        
        return dot_product / norms

    def search(self, query: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Executes both BM25 and Vector Search, then merges results via Reciprocal Rank Fusion.
        """
        # --- Stream 1: Sparse Retrieval (BM25) ---
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Sort documents by highest BM25 score down
        sparse_ranked_indices = np.argsort(bm25_scores)[::-1]

        # --- Stream 2: Dense Retrieval (Vector Similarity) ---
        query_vector = self._get_query_embedding(query)
        vector_scores = self._compute_cosine_similarity(query_vector)
        # Sort documents by highest similarity score down
        dense_ranked_indices = np.argsort(vector_scores)[::-1]

        # --- Stream 3: Reciprocal Rank Fusion (RRF) ---
        # Initialize an empty fusion score card for all document indices
        rrf_scores = {i: 0.0 for i in range(len(self.documents))}

        # Accumulate scores based on Sparse Rank
        for rank, index in enumerate(sparse_ranked_indices):
            # RRF Formula component: 1 / (k + rank)
            rrf_scores[index] += 1.0 / (self.rrf_k + rank)

        # Accumulate scores based on Dense Rank
        for rank, index in enumerate(dense_ranked_indices):
            rrf_scores[index] += 1.0 / (self.rrf_k + rank)

        # Sort documents globally by their final aggregated RRF score
        sorted_rrf_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        # Construct final payload objects
        final_results = []
        for index, rrf_score in sorted_rrf_results[:top_n]:
            final_results.append({
                "id": self.documents[index]["id"],
                "content": self.documents[index]["content"],
                "rrf_score": round(rrf_score, 5),
                "bm25_score": round(float(bm25_scores[index]), 3),
                "vector_similarity": round(float(vector_scores[index]), 3)
            })

        return final_results






# --- Mock Data & Execution Blueprint ---
if __name__ == "__main__":
    # Setup mock data with text contents and arbitrary 3-dimensional dummy mock vectors
    # Note: In real applications, ensure vector dimensions match your embedding model (e.g., 1536)
    mock_corpus = [
        {
            "id": "doc_1", 
            "content": "Deep learning models require massive computational resources and modern GPUs.",
            "embedding": [0.12, 0.85, -0.43]
        },
        {
            "id": "doc_2", 
            "content": "The recipe calls for organic apples, brown sugar, cinnamon, and fresh lemon juice.",
            "embedding": [-0.76, 0.11, 0.62]
        },
        {
            "id": "doc_3", 
            "content": "Artificial intelligence applications are transforming modern computing and business analytics.",
            "embedding": [0.15, 0.79, -0.39]
        }
    ]

    # Instantiate the search index
    search_index = HybridSearchEngine(documents=mock_corpus, rrf_k=60)
    
    # Overwrite the internal method to use static vectors for validation (avoids active OpenAI API reliance)
    search_index._get_query_embedding = lambda q: np.array([0.14, 0.81, -0.40], dtype=np.float32)

    # Test Search: Query conceptually matches doc 1 & 3, but contains exact keywords from doc 3
    search_query = "artificial intelligence computing"
    hits = search_index.search(query=search_query, top_n=2)

    print(f"Query: '{search_query}'\n")
    for rank, hit in enumerate(hits, 1):
        print(f"Rank {rank}: {hit['id']} (RRF Score: {hit['rrf_score']})")
        print(f" Text: \"{hit['content']}\"")
        print(f" Metrics -> BM25 Score: {hit['bm25_score']} | Vector Sim: {hit['vector_similarity']}\n")



The standard approach to merging these two distinct scoring systems is Reciprocal Rank Fusion (RRF). RRF normalizes and combines the ranks of search results rather than their raw scores, ensuring that neither algorithm overpowers the other.


Why Reciprocal Rank Fusion (RRF) is Used

• Scale Invariance: BM25 scores can range from 0 to 100+ depending on document length and term frequency. Vector similarities usually hover tightly between 0.0 and 1.0. RRF prevents the high BM25 values from drowning out the vector matches.
• Deterministic Tuning: You don't have to manually guess weights (e.g., 0.7 * vector + 0.3 * bm25), which can break whenever your text corpus changes or models update. RRF relies purely on relative placement lists.
