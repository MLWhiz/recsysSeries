# Code blocks from: 3 Modern Approaches to Solving Cold Start in RecSys

Code blocks copied from the post, in order. They are fragments, not a runnable notebook.

Source: https://www.mlwhiz.com/p/cold-start-problem-recsys-modern-approaches

## New User Cold Start

```
r̂(u, i) = μ + q_i^T · p_u
```

## Popularity-Based Ranking

```
score(i) = Σ_u R(u, i)  or  score(i) = count(clicks on i in last 24h)
```

## Content-Based Fallback

```
p_u = (1/|I_u|) Σ_{i ∈ I_u} w_i · f_i

where I_u are the items user interacted with
w_i is weight for item (can be 1 for basic cases, or can be a time decay)
```

## Content-Based Fallback

```
score(u, i) = cos(p_u, f_i) = (p_u · f_i) / (||p_u|| · ||f_i||)
```

## Demographic Heuristics

```
p_u = g(demographics_u)
```

## Hybrid Switching

```
r̂(u, i) = α(u) · r̂_PB(u, i) + (1 - α(u)) · r̂_CF(u, i)
```

## The Ceiling

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cold_start_recommend(user, items, item_features, popular_items, n=10):
    “”“Simple cold-start fallback: content-based if we have ANY
    signal, popularity otherwise.”“”

    if user.interactions:  # even 1 click gives us something
        # Build user profile from interacted item features
        profile = np.mean(
            [item_features[i] for i in user.interactions], axis=0
        )
        scores = cosine_similarity([profile], item_features)[0]
        top_items = np.argsort(scores)[::-1][:n]
        return [items[i] for i in top_items]

    # Zero interactions → fall back to popularity
    return popular_items[:n]
```

## Thompson Sampling for Cold Start

```
θ_k ~ Beta(α_k, β_k)
```

## Thompson Sampling for Cold Start

```
Var(θ_k) = (α_k · β_k) / ((α_k + β_k)² · (α_k + β_k + 1))
```

## Upper Confidence Bound (UCB)

```
UCB_k = μ̂_k + c · √(log(t) / N_k)
```

## Contextual Bandits

```
r(k, x) = f(features_k, context_x)
```

## LLM-Initialized Bandits: The Bleeding Edge

```python
import numpy as np
from scipy.stats import beta

class ThompsonSamplingRanker:
    “”“Rank items using Thompson Sampling — cold-start items
    get natural exploration via wide Beta priors.”“”

    def __init__(self, n_items):
        # Beta(1,1) = uniform prior (maximum uncertainty)
        self.alpha = np.ones(n_items)  # successes + 1
        self.beta_param = np.ones(n_items)  # failures + 1

    def rank(self, candidate_ids, n=10):
        # Sample from each item’s Beta posterior
        samples = np.array([
            beta.rvs(self.alpha[i], self.beta_param[i])
            for i in candidate_ids
        ])
        # Rank by SAMPLED value (not mean estimate!)
        ranked_indices = np.argsort(samples)[::-1][:n]
        return [candidate_ids[i] for i in ranked_indices]

    def update(self, item_id, reward):
        “”“reward = 1 (click/watch) or 0 (skip/ignore)”“”
        self.alpha[item_id] += reward
        self.beta_param[item_id] += (1 - reward)

    def warm_start_from_llm(self, item_id, predicted_ctr, confidence=50):
        “”“Initialize with LLM-predicted CTR as a warm prior.
        confidence controls how much we trust the LLM estimate
        (higher = stronger prior, slower adaptation).”“”
        self.alpha[item_id] = predicted_ctr * confidence + 1
        self.beta_param[item_id] = (1 - predicted_ctr) * confidence + 1
```

## CMML: Industrial-Grade Meta-Learning

```
# Meta-training: learn to adapt quickly
for user in warm_users:
    support = user.interactions[:5]     # pretend this is all we have
    query   = user.interactions[5:20]   # ground truth to evaluate against

    adapted = model.fine_tune(support, steps=1)  # one gradient step
    loss = adapted.evaluate(query)                # how good was that adaptation?
    model.update_init_weights(loss)               # make starting point better

# Serving: adapt to a new cold-start user
adapted = model.fine_tune(new_user.first_clicks, steps=3)
recommendations = adapted.rank(all_items, top_k=10)
```

## Pattern 1: LLM as Feature Extractor

```
e_i = LLM_encode(text_i)
```

## Pattern 2: LLM as Reasoning Engine

```
Prompt: "A user just watched 'Parasite' (2019). What can we infer
about their watching preferences?"

LLM: "This user likely appreciates international cinema, social commentary,
genre-bending narratives. Recommend: Bong Joon-ho's other work,
class-conscious drama (Shoplifters, The Square), tense psychological
thrillers (Uncut Gems, Good Time)..."
```

## Pattern 3: LLM as Data Generator

```
Prompt: "Given this movie description: [description], generate 100 synthetic
user profiles that would likely watch and enjoy this film, along with their
predicted rating (1-5). Include diverse demographics and preference profiles."
```

## Pattern 3: LLM as Data Generator

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# A pre-trained sentence transformer
encoder = SentenceTransformer(”all-MiniLM-L6-v2”)

def embed_cold_start_item(item):
    “”“Generate a rich 384-d embedding for an item with zero
    interactions, using only its text metadata.”“”
    text = f”“”
    Title: {item.title}
    Description: {item.description}
    Genre: {’, ‘.join(item.genres)}
    Cast: {’, ‘.join(item.cast[:5])}
    Tags: {’, ‘.join(item.tags)}
    “”“
    return encoder.encode(text, normalize_embeddings=True)

def add_cold_items_to_index(new_items, faiss_index):
    “”“Add cold-start items to the FAISS index alongside
    warm items — they’re immediately retrievable.”“”
    embeddings = np.array([
        embed_cold_start_item(item) for item in new_items
    ])
    faiss_index.add(embeddings)
    return faiss_index

# Now cold-start items appear in retrieval results
# alongside warm items — no behavioral data needed
```
