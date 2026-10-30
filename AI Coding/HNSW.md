import numpy as np
import hnswlib

rng = np.random.default_rng(42)
num_documents = 50_000
dimensions = 128
k = 10

# 1. Generate random vectors
document_vectors = rng.normal(size=(num_documents, dimensions)).astype(np.float32)
query_vectors = rng.normal(size=(5, dimensions)).astype(np.float32)

# 2. L2-Normalize the vectors for proper Cosine Space calculations
document_vectors /= np.linalg.norm(document_vectors, axis=1, keepdims=True)
query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

# 3. Initialize HNSW Index
index = hnswlib.Index(space="cosine", dim=dimensions)
index.init_index(max_elements=num_documents, M=16, ef_construction=200)

# 4. Add items and query
index.add_items(document_vectors, np.arange(num_documents))
index.set_ef(100)  # Controls query accuracy vs speed

labels, distances = index.knn_query(query_vectors, k=k)

# 5. Output results
for query_index, results in enumerate(labels):
    print(f"\nQuery {query_index}:")
    for doc_id, distance in zip(results, distances[query_index]):
        print(f"  Doc ID: {doc_id:<6} Distance: {float(distance):.4f}")



Here is a step-by-step breakdown of exactly how this code works.
At a high level, this script generates fake data, builds an HNSW (Hierarchical Navigable Small World) graph index, and searches that graph to find the top 10 most similar vectors for a set of queries.

Step 1: Setting Up the Parameters

python
import numpy as np
import hnswlib

rng = np.random.default_rng(42)
num_documents = 50_000
dimensions = 128
k = 10
Use code with caution.
• rng = np.random.default_rng(42): Creates a random number generator. The seed 42 ensures that the "random" numbers generated are identical every time you run the script.
• num_documents = 50_000: The total size of your database (50,000 vectors).
• dimensions = 128: The size of each vector. This mimics real-world text embedding models (like miniature OpenAI or BERT embeddings).
• k = 10: We want to find the top 10 closest matches for our queries.

Step 2: Generating Fake Vectors

python
document_vectors = rng.normal(size=(num_documents, dimensions)).astype(np.float32)
query_vectors = rng.normal(size=(5, dimensions)).astype(np.float32)
Use code with caution.
• document_vectors: A massive matrix of shape (50000, 128). This represents the database of items you want to search through.
• query_vectors: A smaller matrix of shape (5, 128). These represent 5 incoming search queries.
• .astype(np.float32): HNSWlib requires 32-bit floating-point numbers for high-performance memory management and C++ compatibility.

Step 3: Initializing the HNSW Graph

python
index = hnswlib.Index(space="cosine", dim=dimensions)
Use code with caution.
• space="cosine": Tells HNSWlib to calculate similarity using Cosine Distance. In HNSWlib, this is defined mathematically as 1 - cosine_similarity. A distance of 0 means identical vectors; a distance of 1 means orthogonal (90-degree angle) vectors.
• dim=dimensions: Informs the index that every vector passed to it will strictly be 128 elements long.

Step 4: Configuring and Allocating Memory

python
index.init_index(
    max_elements=num_documents,
    M=16,
    ef_construction=200,
)
Use code with caution.
This is where HNSW creates its multi-layered graph structure in memory:
• max_elements=num_documents: Pre-allocates memory for exactly 50,000 vectors. The index cannot grow past this number without being resized.
• M=16: The maximum number of bidirectional connection links each vector node can have to other nodes in the graph layers. Higher numbers mean better accuracy for complex data, but use more RAM. 16 is a standard default.
• ef_construction=200: Controls how hard the algorithm works to find the best connections while building the index. A higher number creates a more accurate graph but makes add_items take longer to finish.

Step 5: Building the Index

python
index.add_items(
    document_vectors,
    np.arange(num_documents),
)
Use code with caution.
• document_vectors: The 50,000 vectors are inserted into the multi-layered graph. HNSW links similar vectors together so the search algorithm can quickly "skip" across unrelated data.
• np.arange(num_documents): Assigns a unique ID token to each vector. In this case, Vector 0 gets ID 0, Vector 1 gets ID 1, up to 49999.

Step 6: Setting Search Accuracy & Querying

python
index.set_ef(100)

labels, distances = index.knn_query(
    query_vectors,
    k=k,
)
Use code with caution.
• index.set_ef(100): Sets the search-time accuracy. During a search, HNSW keeps a dynamic list of the top candidates. An ef of 100 means it tracks 100 potential candidate matches dynamically as it traverses the graph before narrowing down to the absolute top k.
• index.knn_query(...): Executes the K-Nearest Neighbors search. It takes the 5 query vectors, walks the graph, and returns two NumPy arrays:
	• labels: A (5, 10) array containing the integer IDs of the 10 closest document vectors for each query.
	• distances: A (5, 10) array containing the mathematical cosine distance values matching those IDs.

Step 7: Printing the Results

python
for query_index, results in enumerate(labels):
    print(f"Query {query_index}:")
    for doc_id, distance in zip(
        results, distances[query_index]
    ):
        print(doc_id, float(distance))
Use code with caution.
• Loops through each of the 5 queries.
• Matches the discovered Document ID with its Distance score and prints them out sequentially, ordered from closest (smallest distance) to furthest.
Would you like to know how to save this index to a file so you don't have to rebuild it every time, or are you interested in how to add or delete items dynamically later on?
