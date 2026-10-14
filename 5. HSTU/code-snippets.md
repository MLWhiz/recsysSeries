# Code blocks from: HSTU: How Meta Built a Trillion-Parameter Recommender That Actually Scales

Code blocks copied from the post, in order. They are fragments, not a runnable notebook.

Source: https://www.mlwhiz.com/p/hstu-how-meta-built-a-trillion-parameter

## 2. Generative recommendation

```python
# Run HSTU over user history (no candidates appended)
user_embedding = hstu_encoder(user_history)[-1]   # (1, D)

# Dot product against item embedding matrix
scores = torch.mm(user_embedding, item_embeddings.t())  # (1, num_items)

# ANN search
top_k_item_ids = scores.topk(k=100).indices
```

## 3. What the model consumes: the fused input format

```
{
  “user_id”: “u_42”,
  “request_time”: 1714000000,
  “history_items”:   [”i_770”, “i_770”, “i_339”, “i_488”, “i_612”],
  “history_actions”: [”click”, “play”,  “launch”, “skip”,  “click”],
  “history_times”:   [t_1, t_2, t_3, t_4, t_5],
  “candidate_items”:     [”i_557”, “i_223”, “i_891”, “i_104”],
  “candidate_positions”: [1, 2, 3, 4],
  “y_click”:  [1, 0, 1, 0],
  “y_like”:   [0, 0, 1, 0],
  “y_follow”: [0, 0, 0, 0],
  “y_watch”:  [47.2, 0.0, 132.5, 3.1]
}
```

## Sub-layer 1: pointwise projection

```
proj = F.silu(self.linear_in(x))    # (B, T, 4D)
K, Q, V, U = proj.chunk(4, dim=-1)  # each (B, T, D)
```

## Sub-layer 2: spatial aggregation

```
rab[0, 1] = E_p[-1] + E_t[bucket(2 hours)]
          = (learned scalar)
```

## Sub-layer 2: spatial aggregation

```
[[0.3, 0,   0  ],
 [0.2, 0.4, 0  ],
 [0.3, 0.7, 0.9]]
```

## Sub-layer 2: spatial aggregation

```python
scores = torch.matmul(Q, K.transpose(-2, -1))     # (B, T, T)
scores = scores + rab_pt                           # add time + position bias
mask   = torch.tril(torch.ones(T, T))
scores = scores * mask                             # causal mask
attn_out = F.silu(scores) @ V                     # (B, T, D)
```

## Sub-layer 3: gated transformation

```
gated  = self.layer_norm(attn_out) * U   # (B, T, D)
output = self.linear_out(gated)          # (B, T, D)
return output + x                        # residual: Y + X
```

## 6. M-FALCON: don’t recompute history per candidate

```
[h₁, h₂, h₃, c₁]   ← full forward pass
[h₁, h₂, h₃, c₂]   ← full forward pass
[h₁, h₂, h₃, c₃]   ← full forward pass
... (×10,000)
```

## 6. M-FALCON: don’t recompute history per candidate

```
proj = F.silu(self.linear_in(x_c1))       # (1, 4D)
q_c1, k_c1, v_c1, u_c1 = proj.chunk(4, dim=-1)  # each (1, D)
```

## 6. M-FALCON: don’t recompute history per candidate

```python
# Concatenate candidate’s K, V onto cached history
K_full = torch.cat([K_hist, k_c1], dim=0)    # (N+1, D)
V_full = torch.cat([V_hist, v_c1], dim=0)    # (N+1, D)

# Candidate query attends to full history + itself
attn_scores = q_c1 @ K_full.T                # (1, N+1)
attn_scores = attn_scores + rab_pt           # add position + time bias
activated   = F.silu(attn_scores)            # (1, N+1)

# Aggregate, gate, project, residual
attn_out  = activated @ V_full               # (1, D)
projected = self.linear_out(self.layer_norm(attn_out) * u_c1)
h_c1      = projected + x_c1                 # candidate representation

# Repeat (or batch) for c₂ … c₁₀₀₀₀
```
