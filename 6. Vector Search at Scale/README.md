# 6. Vector search at scale

Series post 5. A guide to the vector search layer that sits behind retrieval.

## [Vector Search at Scale: The Production Engineer's Guide](https://www.mlwhiz.com/p/vector-search-at-scale-the-missing)

Published 2026-02-03 (series post 5)

The post starts with a story: the author tried to explain IVF-PQ to a colleague, and neither of them could say how the pieces fit. This is the explanation built afterward.

It begins with brute force. On random 128-dimensional vectors, one query over 10 million vectors took about 4.5 seconds in the post's timing, and the cost grows linearly with data size.

IVF clusters the space with k-means into nlist cells and searches only the nprobe closest ones. With 1M vectors, nlist=1024, and nprobe=10, that's about 9,766 comparisons instead of 1,000,000 (roughly 100x fewer). PQ splits each vector into m subvectors, replaces each with a 1-byte centroid ID, and cuts memory by 64x at m=8. Search is asymmetric: build a lookup table of query-to-centroid distances, then sum table lookups. IVF-PQ combines both, and the post reports about 1000x speedup with 30x less memory.

The rest covers four ways to combine filters with vector search, libraries versus self-hosted versus managed databases, and a benchmarking method (recall@K, QPS, latency percentiles, memory, build time, Pareto frontier). It ends with parameter rules of thumb, such as nlist between 1 and 4 times the square root of n.

**Key takeaways**

- IVF cuts search time by looking at fewer vectors. PQ cuts memory by compressing them. You need both at 100M scale.
- Set nlist from the square root of dataset size, start nprobe at nlist/100 to nlist/10, and tune against recall.
- Metadata filters can be pre-filter, post-filter, partitioned indexes, or built into the index. Pre- and post-filtering each fail in some cases.
- Benchmark on realistic queries, against exact ground truth, and report p95 and p99 latency, not only averages.
- Choose the tool by scale: libraries under about 10M vectors, self-hosted above that, managed for small teams.

**Code:** [code-snippets.md](code-snippets.md) has the post's FAISS and NumPy fragments. They are not a runnable notebook.
