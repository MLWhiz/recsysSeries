# 1. Recsys fundamentals

Series posts 1 and 2. The first covers the three core recommendation approaches with worked examples. The second traces how the field got from 1990s filtering systems to generative recommenders.

## [RecSys Fundamentals: The Art and Science of Digital Matchmaking](https://www.mlwhiz.com/p/the-recommenders-playbook-algorithms)

Published 2025-09-27 (series post 1)

The post covers the three playbooks behind most recommenders: content-based filtering, collaborative filtering, and hybrids. Each one gets a small worked example you can follow by hand.

Content-based filtering represents items as feature vectors, builds a user profile as a weighted average of liked items, and scores new items by cosine similarity. It works for niche tastes but traps you in a bubble.

Collaborative filtering uses only the user-item rating matrix. The user-based example computes cosine similarity between User1 [5,3,0,4] and User2 [4,0,2,3] (0.84), takes the top 2 neighbours, and predicts a rating of 2.45 for Item 3. The item-based version does the same across items and lands on 3.47.

The post then separates memory-based methods from model-based ones. It derives matrix factorization two ways: Funk SVD trained with SGD, and ALS, which alternates between solving for user and item factors.

Hybrids come last: weighted blending with an alpha knob, switching, and feature combination with LightFM. The goal is to cover cold start (where CF is blind) and filter bubbles (where content-based gets stuck). It's written for readers who are new to recsys and want to code a movie recommender from scratch.

**Key takeaways**

- Content-based filtering needs item features but no other users. Collaborative filtering needs interactions but no features.
- User-based and item-based CF are memory-based and get slow on large matrices. Matrix factorization (Funk SVD, ALS) learns latent factors and predicts with a dot product.
- Each method has a blind spot (cold start for CF, filter bubbles for content-based), so production systems blend them.
- Hybrid options: weighted blend, switch on confidence, or one model with both signals (LightFM).

**Code:** [1. ContentBased.ipynb](1.%20ContentBased.ipynb), [2. Collaborative Filtering.ipynb](2.%20Collaborative%20Filtering.ipynb)

## [How Recommendation Systems Learned to Think](https://www.mlwhiz.com/p/the-algorithmic-journey-of-recommender)

Published 2025-10-04 (series post 2)

This is a history of recommenders, told as a chain of problems and the fix for each one.

It starts with the early systems: Grundy, which asked users questions to solve cold start, then Xerox PARC's Tapestry (which coined "collaborative filtering"), then GroupLens for Usenet. Amazon's item-item CF fixed the O(n²) cost of user similarity by precomputing item similarities offline.

The Netflix Prize ($1M for a 10% improvement) made matrix factorization famous, and it also showed that one metric (RMSE) isn't the same as a good experience.

That led to multi-armed bandits (Thompson Sampling, UCB, LinUCB) and the explore versus exploit framing.

The deep learning section groups models into three ideas. First, neural upgrades to latent factors: NCF, Wide & Deep, and DeepFM. Second, sequence models: GRU4Rec, SASRec, and BERT4Rec. Third, structure and scale: two-tower retrieval and graph networks (GraphSAGE, PinSAGE).

The last part covers generative recommenders: Semantic IDs, TIGER, LIGER, and Kuaishou's OneRec, which the post says gained 1.6% watch time on the main feed. It closes with conversational and explainable recommenders. It's for readers who want the context behind why each approach exists.

**Key takeaways**

- Each era fixed a limit of the last one: manual filters, then CF, then item-item for scale, then factorization, then bandits, then deep nets.
- Netflix Prize lesson: better RMSE doesn't mean a better product, which pushed the field toward multi-objective evaluation.
- Bandits turned recommendation from prediction into decision-making with exploration built in.
- Two-tower retrieval plus ranking became the industrial standard. Generative models with Semantic IDs aim to replace the multi-stage pipeline.
- LLM-based recommenders need grounding (RAG, knowledge graphs) to avoid hallucinated items.

**Code:** This post is conceptual, no code.
