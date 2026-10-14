# 9. Reranking

Series post 8. The final layer: turning a ranked list into a feed.

## [Your Ranking Model Is Right. Your Recommendations Are Wrong](https://www.mlwhiz.com/p/reranking-recsys-diversity-freshness)

Published 2026-04-11 (series post 8)

The post opens with the set problem. A ranker scores items one at a time, so the top 10 can be ten near-identical thrillers. Re-ranking asks what the best collection is, given constraints the ranker doesn't know about.

The mental model is a relevance budget: every diversity rule, freshness boost, or business constraint spends some relevance, and the question is how much.

Diversity comes in three levels: rules (brittle), MMR (greedy, with lambda between 0.5 and 0.7 working well), and DPPs. A three-movie example shows the determinant giving the diverse pair 4.5x higher probability. YouTube's production DPP uses windows of 6 to 12 items and a Gaussian kernel, with quality scaling and kernel width learned from data.

Calibration (Steck, 2018) matches the genre mix to the user's history using KL divergence. Freshness covers time-decay boosts, exploration slots, and example age. Business constraints are ordered as hard filters, boosts, algorithmic re-ranking, then slot assignment.

For multi-objective cases, it covers scalarization, constraint-based selection, and 2D layouts. Uber Eats' MOHR is cited with a $1.5M/week revenue lift. It ends with a playbook, metrics (ILD, calibration error, coverage), and pitfalls.

**Key takeaways**

- Re-ranking works on sets, not items.
- Start with rules, then MMR, then DPP. A well-tuned MMR often beats a poorly tuned DPP.
- Scalarization handles most cases, constraints add guarantees, and hierarchical ranking is for grid layouts.
- Track ILD, calibration error, coverage, and freshness ratio next to NDCG.
- Apply diversity, freshness, and business rules jointly. Sequential passes undo each other.

**Code:** [code-snippets.md](code-snippets.md) has the post's MMR and freshness-boost fragments. They are not a runnable notebook.
