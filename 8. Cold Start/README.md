# 8. Cold start

Series post 7. Three modern approaches to new users and new items.

## [3 Modern Approaches to Solving Cold Start in RecSys](https://www.mlwhiz.com/p/cold-start-problem-recsys-modern-approaches)

Published 2026-03-25 (series post 7)

The post says cold start is three problems, not one: new users, new items, and a new system. It argues that in production the new-item problem is often worse, because an item with no impressions never gets data and so never gets shown. It also points out the warm-start zone of 1 to 5 interactions, where most long-tail users and items live.

Classical fixes come first: popularity ranking, content-based fallback, demographic heuristics, onboarding surveys, and hybrid switching. They ship fast but only guess, and they don't try to learn about the user.

Three modern approaches follow.

- Contextual bandits: Thompson Sampling, UCB, LinUCB, and LLM-generated priors for Beta distributions. Reserve 5 to 10% of slots for exploration.
- Meta-learning: MAML, prototypical networks, and CMML, trained on simulated cold-start episodes from warm users. The post cites 15 to 30% Recall@K gains for users with fewer than 10 interactions.
- LLMs: as feature extractor, reasoning engine, or synthetic data generator.

It then describes how Spotify (audio), TikTok (multi-modal plus fast exploration), and YouTube (side features, example age) layer these ideas, and gives a decision framework.

**Key takeaways**

- Identify which cold start you have before picking a method.
- Bandits fit item exploration. Meta-learning fits new-user personalization in a mature system.
- LLM embeddings of item text are the cheapest, lowest-latency modern option. LLM reasoning at serving time is usually too slow for retrieval.
- Large platforms layer content understanding, first-session exploration, demographics, and fast cold-to-warm learning.
- Cheap exploration (15-second clips) lets TikTok explore more than a platform with 2-hour movies.

**Code:** [code-snippets.md](code-snippets.md) has the post's Python fragments (popularity fallback, Thompson Sampling, sentence-transformer embeddings). They are not a runnable notebook.
