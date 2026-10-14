# 5. HSTU

Series posts 9c and 9d. The first explains Meta's HSTU architecture. The second builds it from scratch on MovieLens-1M and benchmarks it against RecTools.

## [HSTU: How Meta Built a Trillion-Parameter Recommender That Actually Scales](https://www.mlwhiz.com/p/hstu-how-meta-built-a-trillion-parameter)

Published 2026-05-18 (series post 9c)

The post asks why scaling a SASRec-style model stopped paying off, and why HSTU doesn't have that problem. It names three mismatches between language and recommendation data. Item vocabularies keep changing, softmax attention erases how strong a user's engagement is, and positional encodings can't express uneven time gaps.

HSTU takes fused (item, action) events as input and treats generation as one model for both retrieval and ranking. Each block has three sub-layers: a pointwise projection that adds a gate (U), spatial aggregation with SiLU instead of softmax plus a relative attention bias over position and log-spaced time buckets, and a gated transformation that replaces the feed-forward block. The post cites an ablation where swapping SiLU for softmax drops HR@10 by 44.7%.

Training combines retrieval, click, like, and watch-time losses. M-FALCON caches history K/V so scoring 10,000 candidates doesn't re-encode an 8,000-event history each time.

Reported results: 1.5 trillion parameters, 12,000+ GPUs, and +12.4% topline engagement. The author also notes that a tuned baseline can match HSTU on academic datasets.

**Key takeaways**

- NLP defaults (softmax, absolute positions, fixed vocabularies) are quietly wrong for recommendation data.
- SiLU attention keeps engagement intensity. Time-bucketed relative bias gives the model a sense of recency.
- M-FALCON makes serving practical by computing history once per request. It works because candidates can't attend to each other.
- The paper shows a power-law relationship between compute and quality, which lets teams plan like LLM teams.

**Code:** Fragments from the post are in [code-snippets.md](code-snippets.md). This post has no notebook. The next post's notebooks implement the architecture.

## [HSTU From Scratch in PyTorch - A complete Walkthrough](https://www.mlwhiz.com/p/hstu-from-scratch-in-pytorch-a-complete)

Published 2026-05-28 (series post 9d)

This is the hands-on follow-up. The model is about 600 lines of PyTorch (D=64, 2 layers) that trains on a free Colab T4, and it's benchmarked against the RecTools HSTU and SASRec on the same data and split.

MovieLens ratings become (item, action, time) events: positive for 4 or higher, neutral for 3, negative below 3. The split is leave-one-out with sequences capped at 200 events. The post walks through the fused input layer, a relative attention bias module, the HSTU block, and two heads (sampled-softmax retrieval and rating prediction).

Results: adding action types lifts HR@10 by 23% relative. The from-scratch model beats SASRec by 6% relative and trails RecTools' HSTU by 10%. The author suggests the gap is the missing learnable positional encoding.

On this small dataset, softmax beat SiLU by about 3 to 4% on test HR@10. That's the opposite of Meta's reported result at production scale. The post also covers FAISS retrieval, ranking with the predicted rating, an M-FALCON benchmark (210x speedup), and a production failure guide.

**Key takeaways**

- Feeding actions alongside items gave the largest measured gain (+23% HR@10).
- The encoder code is simple enough to read end to end.
- SiLU's advantage may need long sequences and large catalogs. Run the ablation on your own data.
- Caching K/V makes candidate scoring much cheaper, and the gap grows with history length.
- Watch for embedding table memory, easy-negative bias, cold items, and serving cost.

**Code:** [hstu-from-scratch-ml1m.ipynb](hstu-from-scratch-ml1m.ipynb), [rectools-hstu-ml1m.ipynb](rectools-hstu-ml1m.ipynb)
