# 3. Sequential recommenders

Series post 9a. Both models are trained on the Steam Games dataset.

## [From RNNs to Transformers: Building Sequential Recommenders (Part 1)](https://www.mlwhiz.com/p/rnns-to-transformers-sequential-recommenders)

Published 2026-04-18 (series post 9a)

The post argues that order matters: [Inception, Interstellar, Arrival] says more than the same three titles as an unordered set. Sequential models predict the next item from the sequence so far.

The dataset is Steam Games after 5-core filtering: 56,808 users, 6,382 games, and 2.2M interactions.

GRU4Rec (2016) uses a GRU as session memory, with session-parallel mini-batches and BPR loss with 5 sampled negatives. On Steam it reaches Recall@20 of 0.2107 and NDCG@20 of 0.0873, against 0.0033 and 0.0012 for random. Its limits are sequential processing and a fixed-size hidden state.

SASRec (2018) replaces the GRU with causal self-attention. The post reports a Recall@20 range of 0.24 to 0.27 on this dataset. It also covers the BERT4Rec controversy: a 2023 reproduction found that with the same loss, SASRec matches or beats BERT4Rec.

A long section covers four deployment patterns: score candidates in a second stage, embedding retrieval with an ANN index, one retriever among several, and feature generation for a downstream ranker. It's for engineers who want working sequence models plus a view of how they get served.

**Key takeaways**

- Sequence models capture trajectories that collaborative filtering ignores.
- SASRec is a strong default baseline: it trains in parallel, sees the full history, and matches the causal inference setup.
- BERT4Rec's reported gains came mostly from the loss and training details, not bidirectional attention.
- Training is the same across deployment patterns. Only inference changes (score all items, score candidates, or ANN retrieval).
- Full-catalog scoring is fine for 6K items but too slow for millions.

**Code:** [gru4rec/gru4rec-steam-dataset.ipynb](gru4rec/gru4rec-steam-dataset.ipynb), [sasrec/sasrec-on-steam-games.ipynb](sasrec/sasrec-on-steam-games.ipynb)
