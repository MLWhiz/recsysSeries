# 7. Ranking

Series post 6. The ranking layer, from logistic regression to HSTU.

## [From Candidates to Clicks: The Engineering Anatomy of Ranking](https://www.mlwhiz.com/p/from-candidates-to-clicks-the-engineering)

Published 2026-03-14 (series post 6)

A new engineer asks why you can't just sort by two-tower scores. The answer is that two-tower never sees a user and item together, and a ranker does.

The post frames ranking as a budget problem. Retrieval scores about 100 million items in 50 ms. Ranking sees around 1,000 in 100 ms, which is a 200,000x larger per-item budget, so it can afford much heavier models.

It covers the feature space (dense, sparse ID embeddings, cross features) and then walks the model history in code. Logistic regression and GBDTs come first. Then Wide & Deep, DCN v2 (learned feature crosses), and DLRM (separate dense and sparse paths with pairwise dot-product interactions).

Next are multi-task ranking (shared trunk, task heads, weighted final score, MMoE), a comparison of pointwise, pairwise, and listwise losses, DIN for attention over user history, and HSTU as a sequence-first model.

The best practices section covers hashing for huge ID tables, feature versus model freshness, offline versus online metrics, position-bias correction, and calibration (Platt scaling, isotonic regression, negative downsampling). It's for ML engineers building or reviewing a ranker.

**Key takeaways**

- Retrieval is tuned for recall and ranking for precision. The per-item compute budget is what separates them.
- Embedding tables dominate model size, so rankers are memory-bound.
- Each model fixed one gap: Wide & Deep (memorization and generalization), DCN v2 (manual crosses), DLRM (dense and sparse structure), DIN (static user vectors).
- Training on one signal, like clicks, ranks clickbait first. Use multiple objectives with weights set as a product decision.
- Pointwise loss is a fine default. Offline gains are hypotheses until an A/B test confirms them.

**Code:** [code-snippets.md](code-snippets.md) has the post's PyTorch, LightGBM, and calibration fragments. They are not a runnable notebook.
