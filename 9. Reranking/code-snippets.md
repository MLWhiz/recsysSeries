# Code blocks from: Your Ranking Model Is Right. Your Recommendations Are Wrong

Code blocks copied from the post, in order. They are fragments, not a runnable notebook.

Source: https://www.mlwhiz.com/p/reranking-recsys-diversity-freshness

## Level 2: Maximal Marginal Relevance (MMR)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr_rerank(relevance_scores, item_embeddings, lambda_param=0.5, top_k=10):
    “”“
    Maximal Marginal Relevance re-ranking.

    Args:
        relevance_scores: array of shape (N,) — ranking model scores
        item_embeddings: array of shape (N, d) — item feature vectors
        lambda_param: trade-off between relevance (1.0) and diversity (0.0)
        top_k: number of items to select

    Returns:
        selected: list of indices in selection order
    “”“
    n_items = len(relevance_scores)
    sim_matrix = cosine_similarity(item_embeddings)

    selected = []
    candidates = list(range(n_items))

    for _ in range(top_k):
        best_score = -np.inf
        best_idx = None

        for idx in candidates:
            # Relevance term
            rel = relevance_scores[idx]

            # Max similarity to any already-selected item
            if selected:
                max_sim = max(sim_matrix[idx][s] for s in selected)
            else:
                max_sim = 0

            # MMR score: balance relevance vs. novelty
            score = lambda_param * rel - (1 - lambda_param) * max_sim

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected
```

## Freshness Boost

```python
def freshness_boost(relevance_score, item_age_hours,
                     boost_strength=0.3, half_life_hours=48):
    “”“
    Apply exponential freshness boost that decays over time.

    A brand-new item gets a 30% score boost.
    After 48 hours, the boost is halved to 15%.
    After 96 hours, it’s down to ~7.5%.
    “”“
    decay = np.exp(-0.693 * item_age_hours / half_life_hours)  # 0.693 = ln(2)
    boost = 1.0 + boost_strength * decay
    return relevance_score * boost
```
