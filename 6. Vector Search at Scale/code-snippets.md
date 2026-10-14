# Code blocks from: Vector Search at Scale: The Production Engineer's Guide

Code blocks copied from the post, in order. They are fragments, not a runnable notebook.

Source: https://www.mlwhiz.com/p/vector-search-at-scale-the-missing

## 1. The Problem: Brute Force Doesn’t Scale

```python
import numpy as np
import time

def brute_force_search(query, database, k=10):
    """Find k nearest neighbors by computing all distances."""
    distances = np.sum((database - query) ** 2, axis=1)
    return np.argsort(distances)[:k]

# Simulate different scales
for n in [10_000, 100_000, 1_000_000, 10_000_000]:
    d = 128
    database = np.random.randn(n, d).astype(np.float32)
    query = np.random.randn(d).astype(np.float32)

    start = time.time()
    indices = brute_force_search(query, database)
    elapsed = time.time() - start

    memory_gb = (n * d * 4) / (1024**3)
    print(f"n={n:>10,}: {elapsed*1000:>8.1f}ms, memory={memory_gb:.2f}GB")
```

## 1. The Problem: Brute Force Doesn’t Scale

```
n =     10,000:     25.2ms, memory=0.00GB
n =    100,000:     36.7ms, memory=0.05GB
n =  1,000,000:    391.3ms, memory=0.48GB
n = 10,000,000:   4536.0ms, memory=4.77GB
```

## 2. IVF: Inverted File Index

```python
import faiss
import numpy as np
import time

# Simulate different scales
for n in [100_000, 1_000_000, 10_000_000]:
    d = 128
    database = np.random.randn(n, d).astype(np.float32)
    query =  np.random.random((1, d)).astype('float32')

    # Number of Voronoi cells
    nlist = 1024                  
    # The coarse quantizer (how we find centroids)
    quantizer = faiss.IndexFlatL2(d)  
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    # 3. Train & Add
    # Training on the whole set is slow. 
    # FAISS usually trains on a subset (e.g., 10k points).
    index.train(database[:100000])
    
    index.add(database)
    
    start = time.time()
    indices = index.search(x=query,k=10)
    elapsed = time.time() - start
    
    print(f"n={n:>10,}: {elapsed*1000:>8.1f}ms")
```

## 2. IVF: Inverted File Index

```
n=   100,000:      0.2ms
n= 1,000,000:      0.4ms
n=10,000,000:      1.3ms
```

## 3. Product Quantization (PQ): Vector Compression

```
x = [0.23, -1.45, 0.78, ..., 0.45]  # 128 floats × 4 bytes = 512 bytes
```

## 3. Product Quantization (PQ): Vector Compression

```
u¹ = [0.23, -1.45, ..., 0.89]   # dims 1-16
u² = [1.12, 0.34, ..., -0.56]   # dims 17-32
...
u⁸ = [-0.12, 0.67, ..., 0.45]  # dims 113-128
```

## 3. Product Quantization (PQ): Vector Compression

```
u¹ → codebook 1 → nearest centroid: index 42
u² → codebook 2 → nearest centroid: index 189
...
u⁸ → codebook 8 → nearest centroid: index 201
```

## 3. Product Quantization (PQ): Vector Compression

```
pq_code = [42, 189, 7, 255, 91, 123, 56, 201]  # 8 uint8 values
```

## 3. Product Quantization (PQ): Vector Compression

```python
def train_pq_codebooks(training_vectors, m=8, k=256):
    """
    Train Product Quantization codebooks.

    For each subspace, run k-means to find k representative centroids.
    These centroids form the "vocabulary" for encoding subvectors.
    """
    n, d = training_vectors.shape
    d_sub = d // m
    codebooks = []

    for j in range(m):
        # Extract subvectors for this subspace
        start, end = j * d_sub, (j + 1) * d_sub
        subvectors = training_vectors[:, start:end]

        # Run k-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
        kmeans.fit(subvectors)

        codebooks.append(kmeans.cluster_centers_.astype(np.float32))
        print(f"Codebook {j+1}/{m}: {k} centroids of dim {d_sub}")

    return codebooks
```

## 3. Product Quantization (PQ): Vector Compression

```python
def encode_pq(vectors, codebooks):
    """Encode vectors as PQ codes (centroid indices)."""
    n, d = vectors.shape
    m = len(codebooks)
    d_sub = d // m
    k = codebooks[0].shape[0]

    # Use uint8 if k <= 256, else uint16
    dtype = np.uint8 if k <= 256 else np.uint16
    codes = np.zeros((n, m), dtype=dtype)

    for j, codebook in enumerate(codebooks):
        subvectors = vectors[:, j*d_sub:(j+1)*d_sub]

        # Find nearest centroid for each subvector
        distances = np.sum(
            (subvectors[:, np.newaxis] - codebook) ** 2, axis=2
        )
        codes[:, j] = np.argmin(distances, axis=1)

    return codes
```

## 3. Product Quantization (PQ): Vector Compression

```python
import faiss
import numpy as np
import time

# Simulate different scales
for n in [100_000, 1_000_000, 10_000_000]:
    d = 128
    m = 16
    database = np.random.randn(n, d).astype(np.float32)
    query =  np.random.random((1, d)).astype('float32')

    nbits = 8 # Typically 8 (stores 256 sub-centroids per segment)
    index = faiss.IndexPQ(d, m, nbits)
    
    # 3. Train & Add
    # Training on the whole set is slow. 
    # FAISS usually trains on a subset (e.g., 10k points).
    index.train(database[:100000])
    
    index.add(database)
    
    start = time.time()
    indices = index.search(x=query,k=10)
    elapsed = time.time() - start
    
    print(f"n={n:>10,}: {elapsed*1000:>8.1f}ms")
```

## 3. Product Quantization (PQ): Vector Compression

```
n=   100,000:      2.0ms
n= 1,000,000:     11.2ms
n=10,000,000:     93.3ms
```

## 4. IVF + PQ Combined

```python
import faiss
import numpy as np
import time

# Simulate different scales
for n in [100_000, 1_000_000, 10_000_000]:
    d = 128
    m = 16
    nlist = 1024
    database = np.random.randn(n, d).astype(np.float32)
    query =  np.random.random((1, d)).astype('float32')

    nbits = 8 # Typically 8 (stores 256 sub-centroids per segment)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 bits per code

    # 3. Train & Add
    # Training on the whole set is slow. 
    # FAISS usually trains on a subset (e.g., 10k points).
    index.train(database[:100000])
    
    index.add(database)

    index.nprobe = 10
    start = time.time()
    indices = index.search(x=query,k=10)
    elapsed = time.time() - start
    
    print(f"n={n:>10,}: {elapsed*1000:>8.1f}ms")
```

## 4. IVF + PQ Combined

```
n=   100,000:      0.3ms
n= 1,000,000:      0.7ms
n=10,000,000:      5.6ms
```

## Approach 1: Pre-Filtering

```
filtered_ids = db.query("SELECT id FROM products WHERE color='red' AND price < 100")
filtered_vectors = vectors[filtered_ids]
results = brute_force_search(query, filtered_vectors)
```

## Approach 2: Post-Filtering

```
# Fetch more candidates than needed
k_candidates = 100
D, I = index.search(query, k=k_candidates)

# Filter and re-rank
filtered = [(d, i) for d, i in zip(D[0], I[0]) 
            if metadata[i].price < 100][:k] 
```

## Approach 3: Separate Partitioning

```
indexes = {
    'electronics': faiss.IndexIVFPQ(...),
    'clothing': faiss.IndexIVFPQ(...),
    'books': faiss.IndexIVFPQ(...)
}

# Search only relevant index
category = 'electronics'
results = indexes[category].search(query, k=10)
```

## Approach 4: Hybrid Indexes

```
# Milvus example
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 16}},
    limit=10,
    expr='color == "red" and price < 100'  # Filter expression
)

# Pinecone example
results = index.query(
    vector=query_vector,
    top_k=10,
    filter={"color": {"$eq": "red"}, "price": {"$lt": 100}}
)
```

## A. Recall@K - The Primary Quality Metric

```python
def compute_recall(retrieved, ground_truth, k):
    """
    retrieved: indices returned by approximate index [batch_size, k]
    ground_truth: true nearest neighbors [batch_size, k]
    """
    recall_scores = []
    for ret, gt in zip(retrieved, ground_truth):
        # Number of correct results in top-k
        hits = len(set(ret[:k]) & set(gt[:k]))
        recall_scores.append(hits / k)
    return np.mean(recall_scores)
```

## B. QPS (Queries Per Second) - Throughput Metric

```
QPS = 1000 / latency_ms
```

## D. Memory Usage - Runtime and Peak

```
Flat: n × d × 4 bytes
IVFPQ: n × m bytes (where m is subquantizers)
```

## Step 2: Generate Ground Truth Once

```python
import faiss
import numpy as np

# Build exact search index
index_gt = faiss.IndexFlatL2(d)
index_gt.add(database)

# Compute ground truth top-100 for each query
k_gt = 100
gt_distances, gt_indices = index_gt.search(test_queries, k=k_gt)

# Save for reuse
np.save('ground_truth_distances.npy', gt_distances)
np.save('ground_truth_indices.npy', gt_indices)
```

## Step 3: Benchmark Multiple Configurations

```python
import time

configs = [
    {'nlist': 1024, 'nprobe': 8, 'm': 8},
    {'nlist': 1024, 'nprobe': 16, 'm': 8},
    {'nlist': 1024, 'nprobe': 32, 'm': 8},
    {'nlist': 2048, 'nprobe': 16, 'm': 8},
    {'nlist': 1024, 'nprobe': 16, 'm': 16},
]

results = []
for config in configs:
    # Build index
    index = build_ivfpq_index(database, **config)
    
    # Warmup
    index.search(test_queries[:50], k=10)
    
    # Measure latency
    start = time.time()
    D, I = index.search(test_queries, k=10)
    elapsed = time.time() - start
    
    # Compute recall
    recall = compute_recall(I, gt_indices, k=10)
    
    # Compute QPS
    qps = len(test_queries) / elapsed
    
    results.append({
        **config,
        'recall': recall,
        'qps': qps,
        'latency_ms': (elapsed / len(test_queries)) * 1000
    })

# Convert to DataFrame for analysis
import pandas as pd
df = pd.DataFrame(results)
print(df.sort_values('recall', ascending=False))
```

## Step 4: Run Multiple Trials

```python
def benchmark_with_trials(index, queries, k=10, n_trials=5):
    times = []
    for _ in range(n_trials):
        start = time.time()
        D, I = index.search(queries, k=k)
        times.append(time.time() - start)
    
    median_time = np.median(times)
    std_time = np.std(times)
    return median_time, std_time
```

## Step 5: Identify the Pareto Frontier

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['recall'], df['qps'])

for _, row in df.iterrows():
    plt.annotate(
        f"nprobe={row['nprobe']}", 
        (row['recall'], row['qps']),
        fontsize=8
    )

plt.xlabel('Recall@10')
plt.ylabel('QPS')
plt.title('Recall vs Throughput Trade-off')
plt.grid(True, alpha=0.3)
plt.show()
```
