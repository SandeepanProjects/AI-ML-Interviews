# HNSW vs. IVF — Staff-Level AI Engineer Guide

HNSW and IVF are two widely used indexing strategies for Approximate Nearest Neighbour (ANN) search. Both help vector databases retrieve relevant embeddings without exhaustively comparing a query against every stored vector.

The difference is in how they narrow down the search space:

- HNSW (Hierarchical Navigable Small World): Navigates a graph of connected vectors to find promising neighbours.
- IVF (Inverted File Index): Groups vectors into clusters and searches selected clusters rather than the entire collection.

For an enterprise RAG platform using FastAPI, Qdrant, or PostgreSQL with pgvector, understanding this difference helps you make informed decisions about retrieval latency, recall, memory, ingestion, and index maintenance.

## 1. The core difference visually

[How Vector Database Search Works: IVF, HNSW, PQ, and ScaNN](https://images.openai.com/static-rsc-4/fh5jTTnwS2yuErZDdJ1rizI6CU9jqZ4_nIR041ai8-ZdggRv4RbCO6SfcTKwRhhso-ZGQiFEI3fc6PjWgG9Fv51OqLMdNbBhWQAdHZ_t8DLj0cCfhIYgA8Ai8wjHtJ2UpGHFE3b7HxdYyvbmXnRnOJnYiO4_uJ5BVTB1O_rZiKk?purpose=inline)

HNSW: graph traversal

Start at an entry point, move between connected vectors, and refine the search across graph layers.

[IVF Index: Clustering-Based Vector Search & Partitioning - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/9qRuFpmfqVuqWLzPBL49n8ns2oK1pb0QkswY5urM7HzqtSYUlV2nn_I35GIfnhweX5hH0LPghUniaVHAO_JJDk6A3bYknk1CzIQwOsMy7jh6x51Q-E_tK3CLp1-W_Ump4o2IdDYu4e_vFFV6KH3iQEciJMYRPfkkxKqUJmJrKEg?purpose=inline)

IVF: cluster selection

Find nearby cluster centroids, then search the vectors belonging to the selected clusters.

Imagine that your financial compliance platform stores 5 million policy and advisor-communication embeddings.

A query asks:

> Can an advisor guarantee a 15% annual return?

You want the nearest 10 policy passages.

With HNSW, the search traverses a graph toward promising vectors. With IVF, it selects clusters that appear close to the query and searches their contents.

Both can return excellent results, but neither guarantees the true nearest 10 vectors unless the search is configured to perform an exhaustive search or otherwise verify the exact results.

## 2. HNSW: how it works

HNSW stands for Hierarchical Navigable Small World. It is a graph-based ANN algorithm.

Each vector is represented by a node. The index connects nodes to other nodes that are useful for navigating the vector space.

The hierarchy has multiple levels:

- Upper levels have fewer nodes and enable larger navigation steps.
- Lower levels contain more nodes and support more detailed exploration.
- The bottom layer contains the full collection of indexed vectors.

### Search flow

Query embedding

The vector we want to search for

Upper graph layers

Navigate toward promising regions

Lower graph layers

Explore more local candidates

Top-k nearest candidates

Return the best candidates found

The index is constructed ahead of time. At query time, the algorithm navigates the graph rather than scanning every vector.

### Important HNSW parameters

| Parameter         | Purpose                                                 | Trade-off                                                                   |
| ----------------- | ------------------------------------------------------- | --------------------------------------------------------------------------- |
| `M`               | Controls graph connectivity                             | More connections generally use more memory and can improve search quality   |
| `ef_construction` | Controls the breadth of search while building the graph | Higher values generally improve graph quality but increase index-build cost |
| `ef_search`       | Controls the breadth of query-time exploration          | Higher values generally improve recall but increase query latency           |

The exact parameter names and ranges vary across implementations. For example, Qdrant and hnswlib expose related settings through different APIs.

Staff-level insight: HNSW performance is not determined only by the number of vectors. Vector dimensions, graph connectivity, memory locality, query concurrency, update patterns, and the target recall all affect real-world performance.

## 3. HNSW implementation in Python

We'll use `hnswlib`, a library that implements HNSW indexing.

Install the dependencies:

```
pip install numpy hnswlib
```

### Build an index and retrieve nearest neighbours

```
import numpy as npimport hnswlibdef build_hnsw_index(    document_vectors: np.ndarray,    m: int = 16,    ef_construction: int = 200,    ef_search: int = 100,):    document_vectors = np.asarray(        document_vectors, dtype=np.float32    )    if document_vectors.ndim != 2:        raise ValueError("Expected a 2D matrix of embeddings")    if len(document_vectors) == 0:        raise ValueError("Cannot index an empty collection")    if not np.isfinite(document_vectors).all():        raise ValueError("Embeddings contain non-finite values")    if ef_search < 1 or ef_construction < 1 or m < 2:        raise ValueError("Invalid HNSW configuration")    ids = np.arange(len(document_vectors))    index = hnswlib.Index(        space="cosine",        dim=document_vectors.shape[1],    )    index.init_index(        max_elements=len(document_vectors),        M=m,        ef_construction=ef_construction,    )    index.add_items(document_vectors, ids)    index.set_ef(ef_search)    return indexdef search_hnsw(index, query_embedding, k=5):    query = np.asarray(        query_embedding, dtype=np.float32    ).reshape(1, -1)    if not np.isfinite(query).all():        raise ValueError("Query contains non-finite values")    if not 1 <= k <= index.get_current_count():        raise ValueError("k exceeds the number of indexed vectors")    labels, distances = index.knn_query(query, k=k)    return [        {            "document_id": int(doc_id),            "cosine_distance": float(distance),            "cosine_similarity": float(1.0 - distance),        }        for doc_id, distance in zip(labels[0], distances[0])    ]# Example using illustrative embeddings.rng = np.random.default_rng(42)document_vectors = rng.normal(    size=(10_000, 128)).astype(np.float32)query_embedding = rng.normal(    size=128).astype(np.float32)index = build_hnsw_index(document_vectors)results = search_hnsw(index, query_embedding, k=5)for result in results:    print(result)
```

This example builds a cosine-distance HNSW index and retrieves five approximate nearest neighbours.

A few engineering details matter:

- The example uses generated vectors, not real semantic embeddings.
- Document IDs here are row indices. A production system should maintain a reliable mapping to actual document or chunk IDs.
- `k` must not exceed the number of indexed vectors.
- For cosine search, hnswlib returns cosine distances. Lower is better; the similarity conversion shown is `1 - distance`.
- In a real ingestion pipeline, validate embedding dimensions against the model configuration and handle zero or invalid vectors according to your data contract.

### How I would tune HNSW

I would not automatically set `ef_search=100` for every workload. I would benchmark a sweep, for example:

```
for ef_search in [20, 40, 80, 120, 200]:    index.set_ef(ef_search)    # Run the same representative query set.    # Measure Recall@10, p95 latency, and throughput.
```

The appropriate value depends on the corpus and latency target. Also, do not change query-time parameters in a shared index object concurrently without understanding your library's thread-safety guarantees; use an appropriate per-request or per-index configuration strategy.

## 4. IVF: how it works

IVF stands for Inverted File Index. In vector search, it usually means partitioning vectors into clusters and maintaining a list of vectors associated with each cluster.

Unlike HNSW, which navigates graph connections, IVF first determines which clusters are worth searching.

### IVF search flow

Document embeddings

The complete vector collection

Train cluster centroids

Typically using k-means

Assign vectors to clusters

Each vector is associated with an inverted list

Query-time centroid search

Select the nearest nprobe clusters

Search selected inverted lists

Rank candidates and return top-k

Suppose there are 1 million vectors and 1,000 clusters.

The average cluster would contain roughly 1,000 vectors if the assignments were balanced. If a query probes five clusters, it might examine roughly 5,000 vectors instead of one million.

That is only an illustration: clusters are not necessarily balanced, and the centroid-selection step also costs time.

### The critical IVF parameter: `nprobe`

`nprobe` determines how many clusters are searched.

| `nprobe`     | Expected behavior                                                                                                                               |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1            | Search only the closest cluster; potentially very fast, but more likely to miss relevant vectors                                                |
| 5            | Search five selected clusters; broader candidate coverage                                                                                       |
| 20           | Search more clusters; typically higher recall and more computation                                                                              |
| All clusters | Search every cluster; with exact scoring within each cluster, this can recover exact global ranking, subject to implementation details and ties |

IVF also has a build-time parameter, often called `nlist`, that controls the number of clusters.

- More clusters create smaller inverted lists on average.
- Too few clusters can require scanning many vectors per query.
- Too many clusters increase centroid-training and assignment overhead and may make cluster selection less effective.

The best combination depends on corpus size, dimensions, hardware, and recall requirements.

## 5. IVF implementation using FAISS

For practical IVF experiments, FAISS is a useful library. It provides optimized vector indexes and is widely used for similarity search research and applications.

Install:

```
pip install numpy faiss-cpu
```

The following example uses an IVF index with exact L2 distance calculations inside the selected clusters.

```
import numpy as npimport faissrng = np.random.default_rng(42)# Illustrative vectors: 50,000 documents, 128 dimensions.num_documents = 50_000dimensions = 128k = 10document_vectors = rng.normal(    size=(num_documents, dimensions)).astype(np.float32)query_vectors = rng.normal(    size=(5, dimensions)).astype(np.float32)# Number of inverted lists / clusters.nlist = 100# Exact L2 distance calculation for the IVF candidate vectors.quantizer = faiss.IndexFlatL2(dimensions)index = faiss.IndexIVFFlat(    quantizer,    dimensions,    nlist,    faiss.METRIC_L2,)# Train centroids using representative vectors.index.train(document_vectors)# Add vectors to their assigned inverted lists.index.add(document_vectors)# Search a subset of clusters.index.nprobe = 5distances, ids = index.search(query_vectors, k)for query_index in range(len(query_vectors)):    print(f"Query {query_index}:")    for doc_id, distance in zip(        ids[query_index],        distances[query_index],    ):        if doc_id >= 0:            print(                f"Document {doc_id}, "                f"squared L2 distance {float(distance):.4f}"            )
```

This implementation:

1. Creates an IVF index with 100 inverted lists.
2. Trains the clustering structure.
3. Assigns and adds all document vectors.
4. Searches five clusters per query.
5. Returns the ten best candidates found.

FAISS reports squared L2 distances in this example. They preserve the same ranking as ordinary Euclidean distance, but the numeric values differ because the square root is omitted.

### How would I tune IVF?

```
for nprobe in [1, 2, 5, 10, 20, 50, 100]:    index.nprobe = nprobe    # Execute a representative query set.    # Compare Recall@10, p95 latency, and throughput.
```

With `nprobe=nlist`, IVF-Flat examines every cluster and scores all indexed vectors exactly under the selected L2 metric, so it can reproduce the exact L2 ranking. With fewer clusters, it may miss a closer vector assigned to an unprobed cluster.

In production, I would use a held-out evaluation dataset for tuning and account for index build time, query distribution, update patterns, and memory use.

## 6. HNSW vs. IVF: the complete comparison

| Dimension             | HNSW                                                                                 | IVF                                                                               |
| --------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| Core structure        | Navigable graph                                                                      | Clusters and inverted lists                                                       |
| Search mechanism      | Traverses graph connections                                                          | Selects clusters, then searches their vectors                                     |
| Main query parameter  | `ef_search`                                                                          | `nprobe`                                                                          |
| Main build parameters | `M`, `ef_construction`                                                               | `nlist`, training configuration                                                   |
| Index construction    | Builds graph connections                                                             | Trains centroids and assigns vectors                                              |
| Query cost            | Depends on graph traversal and exploration breadth                                   | Depends on centroid search and the size of selected lists                         |
| Memory overhead       | Graph connections can consume substantial memory                                     | Cluster structures add overhead; compressed variants can reduce vector storage    |
| Recall tuning         | Broaden graph exploration                                                            | Probe more clusters                                                               |
| Data updates          | Often supports incremental insertion, with implementation-specific maintenance costs | Supports insertion, but clustering can become less representative as data evolves |
| Compression options   | Depends on implementation                                                            | IVF can be combined with product quantization (IVF-PQ)                            |
| Common use case       | Low-latency, high-recall interactive retrieval                                       | Large collections where partitioning, compression, or scan control is useful      |

A crucial distinction: IVF is not inherently compressed, and HNSW is not inherently exact or approximate based on its name alone. IVF-Flat stores full vectors and calculates exact distances within the selected lists. IVF-PQ compresses vectors and approximates distances, introducing an additional source of error.

Similarly, implementations may offer filtering, quantization, hybrid search, and other optimizations that change the practical trade-offs.

## 7. IVF-Flat vs. IVF-PQ: an important production distinction

You will encounter these two variants in FAISS and other vector-search systems.

IVF-Flat

Stores full vectors in inverted lists. It searches only the selected clusters but calculates exact distances for the vectors it examines. Its main approximation comes from potentially skipping unselected clusters.

IVF-PQ

Uses product quantization to compress vectors into compact codes. It can substantially reduce memory usage, but distances are generally estimated from those codes. Both cluster selection and quantization can affect recall.

At staff level, I would evaluate these as separate choices: first the search structure, then the representation and distance-computation strategy.

For example, IVF-PQ can be attractive when storing full high-dimensional vectors is too expensive. But if retrieval quality is sensitive to small distance differences, its compression error needs careful evaluation.

## 8. How do you choose between HNSW and IVF?

There is no universal rule that one is always faster or more accurate. The choice depends on your data and workload.

## Practical decision guide

Benchmark HNSW first

Interactive enterprise RAG

Consider HNSW when you need low-latency queries, strong recall, and enough RAM for the vectors and graph structure.

Benchmark IVF

Very large collections or compression-sensitive workloads

Consider IVF when you want explicit control over cluster probing, want to explore compressed indexes such as IVF-PQ, or have a workload that benefits from partitioned search.

Use exact search as a baseline

Small collections and evaluation

Exact vectorized search may be simpler and fast enough. It also gives you ground truth for measuring ANN recall.

For your enterprise financial compliance RAG project, I would start by benchmarking HNSW against an exact baseline. I would add IVF to the comparison when corpus size, memory constraints, or workload characteristics make it a plausible alternative.

I would not select an index solely because the corpus contains millions of vectors. Five million vectors at 128 dimensions with a relaxed latency target can have very different requirements from five million vectors at 3,072 dimensions under strict p99 latency requirements.

## 9. How these choices map to Qdrant and pgvector

The algorithm choice also depends on your vector database.

| Platform                 | HNSW                                | IVF                                                          |
| ------------------------ | ----------------------------------- | ------------------------------------------------------------ |
| Qdrant                   | Supports HNSW-based vector indexing | Does not expose FAISS-style IVF as its standard vector index |
| PostgreSQL with pgvector | Supports HNSW indexes               | Supports IVFFlat indexes                                     |
| FAISS                    | Supports HNSW indexes               | Supports IVF-Flat, IVF-PQ, and other variants                |
| hnswlib                  | HNSW implementation                 | Does not provide standard IVF indexing                       |

For example, pgvector supports both HNSW and IVFFlat.

Illustrative HNSW index:

```
CREATE INDEX policy_embedding_hnsw_idx
ON policy_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
```

Illustrative IVFFlat index:

```
CREATE INDEX policy_embedding_ivf_idx
ON policy_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

These are alternative index definitions for a cosine-search workload, assuming the `embedding` column has a compatible pgvector type and extension setup. You would normally benchmark them separately rather than create both without a specific reason.

For pgvector, query-time settings also differ by index type and version. HNSW exposes `hnsw.ef_search`; IVFFlat exposes `ivfflat.probes`. Consult the installed version's documentation for exact syntax and supported tuning options.

For multi-tenant compliance retrieval, also verify the query plan and recall under tenant and policy-version filters. An index that performs well without filters may behave differently when only a small subset of the collection is eligible.

## 10. The staff-level benchmark I would actually run

I would compare HNSW, IVF-Flat, and exact search on the same held-out queries and corpus.

For every configuration, measure:

- Recall\@10 against exact ground truth.
- p50, p95, and p99 query latency.
- Throughput at realistic concurrency.
- Index build duration.
- Resident memory and index size.
- Update and deletion behavior.
- Retrieval quality after authorization and metadata filters.

Then I would select the least expensive configuration that meets the quality and latency objectives.

For example, a hypothetical acceptance criterion might be Recall\@10 of at least 0.95 and p95 latency below 50 ms under a defined load. Those are example targets, not universal production requirements; the appropriate values must come from your application SLOs.

One more important detail: ANN Recall\@10 measures agreement with exact neighbors, not whether the neighbors answer the user's question. I would also maintain a labeled relevance dataset and evaluate Precision\@k, nDCG, and downstream answer faithfulness.

## 11. Interview-ready answer

HNSW and IVF are two ANN indexing strategies that reduce the cost of nearest-neighbour search, but they narrow the search space differently.

HNSW is graph-based. It constructs a hierarchical graph of connected vectors and navigates from upper layers toward promising candidates. Its main query-time tuning parameter is `ef_search`, which controls the breadth of exploration. It is often a strong choice for low-latency, high-recall interactive retrieval, although the graph has memory and construction overhead.

IVF is cluster-based. It partitions vectors around centroids and searches only selected inverted lists. Its main query-time parameter is `nprobe`, which controls how many clusters are examined. IVF-Flat calculates exact distances within the selected clusters, while IVF-PQ adds vector compression and approximate distance calculations.

For a production RAG system, I would benchmark both against an exact-search baseline. I would measure Recall\@k, p95/p99 latency, throughput, index memory, build time, and retrieval quality under real metadata and authorization filters. I would choose based on measured workload requirements rather than assuming one algorithm is universally better.

Remember this: HNSW navigates a graph; IVF selects clusters. `ef_search` controls HNSW's search breadth, while `nprobe` controls how many IVF clusters are searched. Both trade computation for retrieval recall, but their memory use, build costs, and scaling characteristics differ.