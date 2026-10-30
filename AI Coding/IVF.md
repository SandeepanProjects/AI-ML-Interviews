import numpy as np
from sklearn.cluster import KMeans

class SimpleIVF:
    def __init__(self, n_clusters=100, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state,
        )
        self.vectors = None
        self.cluster_ids = None

    def fit(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.cluster_ids = self.kmeans.fit_predict(self.vectors)
        return self

    def search(self, query, k=10, nprobe=5):
        query = np.asarray(query, dtype=np.float32)
        
        # Find the closest cluster centroids.
        centroids = self.kmeans.cluster_centers_
        centroid_distances = np.sum((centroids - query) ** 2, axis=1)
        selected_clusters = np.argsort(centroid_distances)[:nprobe]
        
        # Collect vectors from those clusters only.
        candidate_indices = np.flatnonzero(
            np.isin(self.cluster_ids, selected_clusters)
        )
        if len(candidate_indices) == 0:
            return []
            
        candidates = self.vectors[candidate_indices]
        
        # Exact distance calculations inside the candidate subset.
        distances = np.sum((candidates - query) ** 2, axis=1)
        best = np.argsort(distances)[:min(k, len(distances))]
        
        return [
            (
                int(candidate_indices[i]),
                float(np.sqrt(distances[i])),
            )
            for i in best
        ]

# Execution script
rng = np.random.default_rng(42)
vectors = rng.normal(size=(20_000, 64)).astype(np.float32)
query = rng.normal(size=64).astype(np.float32)

index = SimpleIVF(n_clusters=100).fit(vectors)
results = index.search(query, k=10, nprobe=5)
print(results)


This code implements an Inverted File (IVF) index, which is a popular technique used in vector databases (like Milvus or Faiss) for Approximate Nearest Neighbor (ANN) search.
Instead of comparing your query vector against all 20,000 vectors in the database (which gets very slow as data grows), it groups similar vectors into clusters and only searches inside the closest clusters.

1. Initialization (__init__)

python
def __init__(self, n_clusters=100, random_state=42):
Use code with caution.
• What it does: Sets up a Scikit-Learn KMeans clustering model.
• Parameters: n_clusters=100 means the algorithm will partition our high-dimensional space into 100 distinct neighborhoods (buckets).

2. Training Phase (fit)

python
def fit(self, vectors):
    self.vectors = np.asarray(vectors, dtype=np.float32)
    self.cluster_ids = self.kmeans.fit_predict(self.vectors)
    return self
Use code with caution.
• What it does: This maps out your data geography.
• How it works: It takes all 20,000 vectors, runs the K-Means algorithm to find 100 optimal center points (centroids), and assigns every single vector a cluster_id (from 0 to 99) based on whichever center point it is closest to.

3. Query Phase (search)

This is where the approximate search magic happens in three distinct steps:

Step A: Find the closest neighborhoods

python
centroids = self.kmeans.cluster_centers_
centroid_distances = np.sum((centroids - query) ** 2, axis=1)
selected_clusters = np.argsort(centroid_distances)[:nprobe]
Use code with caution.
• How it works: Instead of looking at the 20,000 individual vectors, the code measures the distance from the query to just the 100 cluster center points.
• The nprobe parameter: It sorts these distances and picks the top nprobe=5 closest clusters. We completely ignore the other 95 clusters.

Step B: Gather candidate vectors

python
candidate_indices = np.flatnonzero(np.isin(self.cluster_ids, selected_clusters))
candidates = self.vectors[candidate_indices]
Use code with caution.
• How it works: It filters the global database to extract only the vectors that belong to those 5 chosen clusters. These are our "shortlisted" search candidates.

Step C: Calculate exact distances on the shortlist

python
distances = np.sum((candidates - query) ** 2, axis=1)
best = np.argsort(distances)[:min(k, len(distances))]
Use code with caution.
• How it works: It performs an exact Euclidean distance calculation between the query and our small pool of candidates. It sorts them to find the absolute closest vectors.
• Return Value: It outputs a list of tuples containing the (original_index_position, exact_distance) for the top k=10 items.

💡 The Core Trade-off: Accuracy vs. Speed

• Why do this? If nprobe=5, you might only look at roughly 5% of your total dataset (~1,000 vectors instead of 20,000). This makes the search drastically faster.
• The Catch: Because it ignores 95% of the data, there is a small chance the absolute closest vector was sitting in the 6th or 7th closest cluster and got skipped. This is why it is called an approximate search.
Would you like to see how to modify this code to add a brute-force baseline so you can calculate its exact recall accuracy percentage?
