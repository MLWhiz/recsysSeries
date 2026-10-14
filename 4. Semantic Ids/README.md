# 4. Semantic IDs

Series post 9b. A four-notebook pipeline on the Steam Games dataset: embed items, quantize them, then compare three recommenders.

## [From Random IDs to Semantic IDs: Building a Generative Recommender from Scratch](https://www.mlwhiz.com/p/semantic-ids-rqvae-generative-recommender)

Published 2026-05-09 (series post 9b)

Models like SASRec treat items as random integers, so they share no knowledge between similar items and struggle with new ones. Semantic IDs give each item a short token sequence derived from its content, so similar items share prefixes.

The post builds this in three steps. First, embed game descriptions with Qwen3-Embedding-4B (truncated from 2560 to 1024 dimensions using Matryoshka representation learning). Second, an RQ-VAE compresses each embedding to 32 dimensions and quantizes it across 3 codebooks of 256 codes, which gives a 3-token ID. Third, a model generates the next item's ID token by token.

The RQ-VAE section covers the codebook and commitment losses, the straight-through estimator, the rotation trick, and codebook collapse (fixed with k-means initialization, dead-code resets, and a batch size of 32,768).

Three models are compared: baseline SASRec, Semantic SASRec, and TIGER (a decoder-only transformer over a flat token stream, decoded with trie-constrained beam search). On this small, dense catalog, Semantic SASRec lands at roughly 45% of baseline. TIGER beats Semantic SASRec by about 49% on HR@10 and 35% on NDCG@10. The author says this regime isn't where Semantic IDs are meant to win.

**Key takeaways**

- Semantic IDs make item metadata part of the ID, which helps cold start and knowledge sharing.
- RQ-VAE builds the ID in levels, coarse to fine. Training it needs care to avoid codebook collapse.
- TIGER generates items instead of scoring a candidate list, so it can replace the retrieve, rank, rerank pipeline with one model.
- On a static 5K-item catalog, dedicated embeddings still win. Semantic IDs trade some precision for cold-start handling and steerability.

**Code:** [semantic-ids-steam-1-rqvae.ipynb](semantic-ids-steam-1-rqvae.ipynb), [semantic-ids-steam-2-sasrec.ipynb](semantic-ids-steam-2-sasrec.ipynb), [semantic-ids-steam-3-semantic-sasrec.ipynb](semantic-ids-steam-3-semantic-sasrec.ipynb), [semantic-ids-steam-4-tiger.ipynb](semantic-ids-steam-4-tiger.ipynb)
