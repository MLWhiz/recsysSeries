# 2. Two tower

Series posts 3 and 4. The first lays out the three-stage serving funnel. The second goes deep on the retrieval stage and trains a two-tower model.

## [The 3-Stage Funnel Behind Every Modern Recommender System](https://www.mlwhiz.com/p/the-recommendation-engine-under-the)

Published 2026-01-20 (series post 3)

The post starts with a constraint: return 10 good videos out of 5 billion in 200 ms. If a model takes 10 ms per video, scoring the whole catalog would take 500 days. So the answer is a funnel, not one clever model.

Stage 1 is retrieval, tuned for recall. A two-tower model trained with in-batch negatives (InfoNCE) lets you precompute item embeddings offline and run only the user tower at request time. Approximate nearest neighbour search (HNSW, product quantization for a 64x memory cut) finds ~500 candidates. The post also covers filtered vector search and why production systems run 4 to 10 retrievers in parallel.

Stage 2 is ranking, tuned for precision. A cross-encoder scores user and item features together, which two-tower can't do. MMoE handles multiple objectives. The post spends a section on feature stores and training-serving skew.

Stage 3 is re-ranking: filters, diversity (MMR, DPP), freshness boosts, calibration, and business rules.

It ends with four production problems: evaluation, cold start, exploration, and feedback loops (position bias, IPW). It's aimed at ML engineers who want the whole system picture.

**Key takeaways**

- Split the problem: recall first on billions of items, precision next on hundreds, policy last on dozens.
- Item embeddings don't depend on the user, so compute them offline and serve with ANN search.
- Cross-encoders are more accurate but can't be precomputed, so they only run on the shortlist.
- A feature store keeps training and serving features identical, which avoids skew.
- Position bias and selection bias create feedback loops. Inverse propensity weighting is one fix.

**Code:** Two short fragments (filtered search, a feature vector) are in [code-snippets.md](code-snippets.md). This post has no notebook.

## [How YouTube Finds Your Next Video in Milliseconds](https://www.mlwhiz.com/p/building-youtube-scale-recommendation)

Published 2026-01-26 (series post 4)

This post zooms into retrieval and builds a two-tower model in PyTorch on MovieLens-1M.

It starts from the scale problem: 2 billion videos at 1 ms each is 23 days of compute per request. It then covers the DSSM origin of two-tower models and the features in YouTube's 2016 paper, including average-pooled watch history and the "example age" trick (set to zero at serving time to remove freshness bias).

The training section explains L2-normalised embeddings, in-batch negatives, and temperature. It then explains the popularity bias in in-batch negatives and fixes it with logQ correction. Hard negative mining (in-batch and two-pass) is covered conceptually.

The experiment uses 575,281 positive interactions (rating of 4 or higher), a temporal split, and a model with 648,896 parameters. LogQ gave the biggest gain: the baseline stayed under 10% recall, and logQ at T=1 passed 16%. Combining logQ with T=0.1 reached the best Recall@50 of 24.02%. The post ends with six common pitfalls.

**Key takeaways**

- In-batch negatives give you O(batch size squared) training signals per forward pass. Use batches of 1,024 or more.
- logQ correction stops popular items from being over-penalised as negatives. In this experiment it drove most of the gain.
- Temperature (0.05 to 0.1) sharpens the softmax, but it helped mainly when paired with logQ.
- Always L2-normalise embeddings so dot product equals cosine similarity and temperature is easier to tune.
- Watch for stale item embeddings, training-serving skew, and cold-start items.

**Code:** [two-tower-network-with-temperature-and-logq.ipynb](two-tower-network-with-temperature-and-logq.ipynb)
