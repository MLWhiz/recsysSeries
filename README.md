# Recommendation Systems Series

A comprehensive tutorial series on building recommendation systems from scratch, covering fundamental algorithms to modern deep learning approaches.

## Overview

This repository contains hands-on implementations accompanying the [mlwhiz.com RecSys Series](https://www.mlwhiz.com). Each notebook provides a complete implementation with practical code, making it ideal for ML engineers who want to go beyond theory and build real systems.

## Repository Structure

```
recsysSeries/
├── 1. Recsys Fundamentals/
│   ├── README.md
│   ├── 1. ContentBased.ipynb
│   └── 2. Collaborative Filtering.ipynb
├── 2. Two Tower/
│   ├── README.md
│   ├── code-snippets.md
│   └── two-tower-network-with-temperature-and-logq.ipynb
├── 3. Sequential Recommenders/
│   ├── README.md
│   ├── gru4rec/
│   │   └── gru4rec-steam-dataset.ipynb
│   └── sasrec/
│       └── sasrec-on-steam-games.ipynb
├── 4. Semantic Ids/
│   ├── README.md
│   ├── semantic-ids-steam-1-rqvae.ipynb
│   ├── semantic-ids-steam-2-sasrec.ipynb
│   ├── semantic-ids-steam-3-semantic-sasrec.ipynb
│   └── semantic-ids-steam-4-tiger.ipynb
├── 5. HSTU/
│   ├── README.md
│   ├── code-snippets.md
│   ├── hstu-from-scratch-ml1m.ipynb
│   └── rectools-hstu-ml1m.ipynb
├── 6. Vector Search at Scale/
│   ├── README.md
│   └── code-snippets.md
├── 7. Ranking/
│   ├── README.md
│   └── code-snippets.md
├── 8. Cold Start/
│   ├── README.md
│   └── code-snippets.md
└── 9. Reranking/
    ├── README.md
    └── code-snippets.md
```

Each folder has a README with a summary of its posts.

## Tutorials

### 1. RecSys Fundamentals

**Notebooks:** `1. Recsys Fundamentals/`

**Blog posts:**
- [RecSys Fundamentals: The Art and Science of Digital Matchmaking](https://www.mlwhiz.com/p/the-recommenders-playbook-algorithms) — collaborative filtering, content-based filtering, and hybrid approaches
- [How Recommendation Systems Learned to Think](https://www.mlwhiz.com/p/the-algorithmic-journey-of-recommender) — from Tapestry (1992) to modern neural approaches

Implementations of the two foundational recommendation paradigms: content-based filtering (recommending based on item features) and collaborative filtering (recommending based on user behavior patterns).

---

### 2. Two Tower Networks

**Notebook:** `2. Two Tower/two-tower-network-with-temperature-and-logq.ipynb`

**Blog posts:**
- [How YouTube Finds Your Next Video in Milliseconds](https://www.mlwhiz.com/p/building-youtube-scale-recommendation) — deep dive on Two-Tower architecture, in-batch negatives, temperature scaling, and logQ correction
- [The 3-Stage Funnel Behind Every Modern Recommender System](https://www.mlwhiz.com/p/the-recommendation-engine-under-the) — where Two-Tower fits in the retrieval → ranking → re-ranking pipeline

Two-Tower models are the dominant architecture for large-scale candidate retrieval. By keeping user and item towers independent, item embeddings can be precomputed offline and served via approximate nearest neighbor search for sub-millisecond latency at billion-item scale.

---

### 3. Sequential Recommenders

**Notebooks:** `3. Sequential Recommenders/`

**Blog post:**
- [From RNNs to Transformers: Building Sequential Recommenders (Part 1)](https://www.mlwhiz.com/p/rnns-to-transformers-sequential-recommenders) — GRU4Rec, SASRec, production deployment with FAISS

Implementations of sequential models that treat user history as a sequence to predict the next item:

- **GRU4Rec** (`gru4rec/`) — RNN-based approach using GRUs, the first deep learning model applied to session-based recommendation
- **SASRec** (`sasrec/`) — self-attention based model that captures long-range dependencies, ~17x faster than GRU4Rec+ while achieving better accuracy

Both are trained and evaluated on the Steam Games dataset.

---

### 4. Semantic IDs & Generative Retrieval

**Notebooks:** `4. Semantic Ids/`

**Blog post:**
- [From Random IDs to Semantic IDs: Building a Generative Recommender from Scratch](https://www.mlwhiz.com/p/semantic-ids-rqvae-generative-recommender) — RQ-VAE and TIGER on the Steam Games dataset with Qwen embeddings

A four-part implementation series on generative recommendation using semantic IDs on the Steam Games dataset:

1. **RQ-VAE** (`semantic-ids-steam-1-rqvae.ipynb`) — train a Residual Quantized VAE to encode item metadata into discrete semantic ID tuples
2. **SASRec baseline** (`semantic-ids-steam-2-sasrec.ipynb`) — standard SASRec as the baseline
3. **Semantic SASRec** (`semantic-ids-steam-3-semantic-sasrec.ipynb`) — SASRec trained on semantic IDs instead of arbitrary item IDs
4. **TIGER** (`semantic-ids-steam-4-tiger.ipynb`) — Transformer Index for GEnerative Recommenders: autoregressively generate the semantic IDs of the next item

Semantic IDs replace arbitrary interaction-dependent item IDs with content-aware discrete representations, enabling better generalization and cold-start performance.

---

### 5. HSTU

**Notebooks:** `5. HSTU/`

**Blog posts:**
- [HSTU: How Meta Built a Trillion-Parameter Recommender That Actually Scales](https://www.mlwhiz.com/p/hstu-how-meta-built-a-trillion-parameter) — the architecture, the math, and why it scales
- [HSTU From Scratch in PyTorch - A complete Walkthrough](https://www.mlwhiz.com/p/hstu-from-scratch-in-pytorch-a-complete) — data pipeline, the three sub-layers, retrieval + rating loss, and a benchmark against RecTools' HSTU

- **From scratch** (`hstu-from-scratch-ml1m.ipynb`) — HSTU in plain PyTorch, trained on MovieLens-1M (small config, runs on a free Colab T4)
- **RecTools baseline** (`rectools-hstu-ml1m.ipynb`) — the RecTools HSTU and SASRec on the same data and split, used as the comparison numbers

---

### Posts without notebooks

Four posts are system-design pieces with no notebook: Vector Search at Scale, From Candidates to Clicks, 3 Modern Approaches to Cold Start, and the Reranking post. Each has its own folder (`6. Vector Search at Scale/`, `7. Ranking/`, `8. Cold Start/`, `9. Reranking/`) with a README summary and a `code-snippets.md` file. The snippet files hold the code blocks from the post, in order. They are fragments, not runnable notebooks.

Two more posts have no notebook either: The 3-Stage Funnel (in `2. Two Tower/`) and HSTU: How Meta Built a Trillion-Parameter Recommender (in `5. HSTU/`). Their folders also have a `code-snippets.md`.

---

## Blog Series

All notebooks accompany the **mlwhiz.com RecSys Series**:

| # | Post | Topic |
|---|------|-------|
| 1 | [RecSys Fundamentals: The Art and Science of Digital Matchmaking](https://www.mlwhiz.com/p/the-recommenders-playbook-algorithms) | Content-based & collaborative filtering |
| 2 | [How Recommendation Systems Learned to Think](https://www.mlwhiz.com/p/the-algorithmic-journey-of-recommender) | From collaborative filtering to generative AI |
| 3 | [The 3-Stage Funnel Behind Every Modern Recommender System](https://www.mlwhiz.com/p/the-recommendation-engine-under-the) | Two-Tower, vector databases, cross-encoders |
| 4 | [How YouTube Finds Your Next Video in Milliseconds](https://www.mlwhiz.com/p/building-youtube-scale-recommendation) | Two-tower retrieval, in-batch negatives |
| 5 | [Vector Search at Scale: The Production Engineer's Guide](https://www.mlwhiz.com/p/vector-search-at-scale-the-missing) | IVF, PQ, 100M-vector search |
| 6 | [From Candidates to Clicks: The Engineering Anatomy of Ranking](https://www.mlwhiz.com/p/from-candidates-to-clicks-the-engineering) | Ranking from 1,000 candidates to one item |
| 7 | [3 Modern Approaches to Solving Cold Start in RecSys](https://www.mlwhiz.com/p/cold-start-problem-recsys-modern-approaches) | Bandits, meta-learning, LLMs |
| 8 | [Your Ranking Model Is Right. Your Recommendations Are Wrong](https://www.mlwhiz.com/p/reranking-recsys-diversity-freshness) | Diversity, freshness, business constraints |
| 9 | [From RNNs to Transformers: Building Sequential Recommenders (Part 1)](https://www.mlwhiz.com/p/rnns-to-transformers-sequential-recommenders) | GRU4Rec and SASRec on Steam Games |
| 10 | [From Random IDs to Semantic IDs: Building a Generative Recommender from Scratch](https://www.mlwhiz.com/p/semantic-ids-rqvae-generative-recommender) | RQ-VAE and TIGER on Steam Games |
| 11 | [HSTU: How Meta Built a Trillion-Parameter Recommender That Actually Scales](https://www.mlwhiz.com/p/hstu-how-meta-built-a-trillion-parameter) | HSTU architecture and scaling |
| 12 | [HSTU From Scratch in PyTorch - A complete Walkthrough](https://www.mlwhiz.com/p/hstu-from-scratch-in-pytorch-a-complete) | PyTorch implementation on MovieLens-1M |
