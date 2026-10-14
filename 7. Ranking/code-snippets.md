# Code blocks from: From Candidates to Clicks: The Engineering Anatomy of Ranking

Code blocks copied from the post, in order. They are fragments, not a runnable notebook.

Source: https://www.mlwhiz.com/p/from-candidates-to-clicks-the-engineering

## B. Sparse Features (High-Cardinality IDs)

```python
import torch
import torch.nn as nn

# Each sparse feature gets its own embedding table
embedding_tables = {
    "user_id":    nn.Embedding(num_embeddings=500_000_000, embedding_dim=64),
    "item_id":    nn.Embedding(num_embeddings=1_000_000_000, embedding_dim=64),
    "creator_id": nn.Embedding(num_embeddings=50_000_000, embedding_dim=32),
    "category_id":nn.Embedding(num_embeddings=500, embedding_dim=16),
}

# At inference time: just look up the embedding vector for this user/item pair
user_embedding   = embedding_tables["user_id"](torch.tensor([user_id]))  # shape: [1, 64]
item_embedding   = embedding_tables["item_id"](torch.tensor([item_id]))  # shape: [1, 64]
```

## C. Cross Features

```
# Manual cross feature (the old way)
user_category_hour = f"{user_top_category}_{hour_of_day_bucket}"
# e.g., "cooking_evening" -> one-hot encoded

# Neural approach: learn the crosses automatically (we'll see this with DCN)
```

## Logistic Regression → The Original Ranker

```
P(click) = sigmoid(w1*user_london + w2*item_sports + w3*evening + ...)
```

## Logistic Regression → The Original Ranker

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Simple example: 1M training examples, sparse features
# user_region (50 values), item_category (100 values), hour_bucket (6 values)

# In practice this is millions of features wide (all possible values)
X = one_hot_encode(training_data)   # shape: [1M, ~10K features]
y = training_data["clicked"]

model = LogisticRegression(solver='liblinear')
model.fit(X, y)
```

## Logistic Regression → The Original Ranker

```
# "Feature Engineering Hell" -- an actual practice at early Facebook/Google

cross_features = []
for f1 in dense_features:
    for f2 in dense_features:
        cross_features.append(f"{f1}_AND_{f2}")   # O(n^2) feature explosion

# At 10,000 base features: 100,000,000 potential cross features
# Most are useless. Figuring out which ones aren't -> that's the job.
```

## GBDTs → The Kaggle King

```python
import lightgbm as lgb
import pandas as pd

# Features are now dense numerical values, not one-hot encodings
X_train = pd.DataFrame({
    "user_age": user_ages,
    "user_avg_ctr": user_ctrs,
    "item_category_id": item_categories,   # LightGBM handles categoricals natively
    "hour_of_day": hours,
    "item_age_hours": item_ages,
})

y_train = clicks

dtrain = lgb.Dataset(X_train, label=y_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
}

model = lgb.train(params, dtrain, num_boost_round=500)

# A single tree's decision path might look like:
# if item_category == "sports" AND hour_of_day > 17 AND user_age < 35: CTR += 0.03
```

## 4. Wide & Deep: Google's Breakthrough (2016)

```python
import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    def __init__(self,
                 num_dense_features: int,
                 sparse_feature_dims: dict,  # {feature_name: (vocab_size, embed_dim)}
                 deep_hidden_dims: list = [256, 128, 64]):
        super().__init__()

        # Wide component: linear model on dense + cross features
        self.wide = nn.Linear(num_dense_features, 1)

        # Embedding tables for sparse features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dim)
            for name, (vocab_size, embed_dim) in sparse_feature_dims.items()
        })

        # Deep component: MLP on concatenated embeddings + dense features
        total_embed_dim = sum(d for _, d in sparse_feature_dims.values())
        deep_input_dim = num_dense_features + total_embed_dim

        layers = []
        in_dim = deep_input_dim
        for h_dim in deep_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        self.deep = nn.Sequential(*layers)
        self.deep_output = nn.Linear(in_dim, 1)

    def forward(self, dense_features, sparse_features):
        # Wide path
        wide_output = self.wide(dense_features)  # [B, 1]

        # Deep path: look up embeddings for all sparse features
        embed_list = [
            self.embeddings[name](ids)
            for name, ids in sparse_features.items()
        ]
        deep_input = torch.cat([dense_features] + embed_list, dim=1)  # [B, D]
        deep_output = self.deep_output(self.deep(deep_input))          # [B, 1]

        # Joint output
        logit = wide_output + deep_output
        return torch.sigmoid(logit).squeeze(1)


# Example instantiation
model = WideAndDeep(
    num_dense_features=10,
    sparse_feature_dims={
        "user_id":    (500_000, 64),
        "item_id":    (1_000_000, 64),
        "category_id":(500, 16),
    }
)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Parameters: 96,089,804
```

## 4. Wide & Deep: Google's Breakthrough (2016)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Generate synthetic ranking data
N = 100_000
torch.manual_seed(42)

dense   = torch.randn(N, 10)
user_ids = torch.randint(0, 500_000,   (N,))
item_ids = torch.randint(0, 1_000_000, (N,))
cat_ids  = torch.randint(0, 500,       (N,))

# Ground truth: users click more on items they have affinity for
# (simple synthetic rule)
labels = ((dense[:, 0] + dense[:, 2] > 0) & (cat_ids < 250)).float()

dataset = TensorDataset(dense, user_ids, item_ids, cat_ids, labels)
loader  = DataLoader(dataset, batch_size=2048, shuffle=True)

model     = WideAndDeep(10, {"user_id": (500_000,64), "item_id":(1_000_000,64), "category_id":(500,16)})
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(5):
    total_loss = 0
    for dense_b, uid_b, iid_b, cid_b, y_b in loader:
        pred = model(dense_b, {"user_id": uid_b, "item_id": iid_b, "category_id": cid_b})
        loss = criterion(pred, y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
```

## 4. Wide & Deep: Google's Breakthrough (2016)

```
Epoch 1 | Loss: 0.6891
Epoch 2 | Loss: 0.6204
Epoch 3 | Loss: 0.5877
Epoch 4 | Loss: 0.5701
Epoch 5 | Loss: 0.5589
```

## 5. DCN v2: Automatic Feature Crossing

```
x_l+1 = x_0 * (W_l * x_l + b_l) + x_l
```

## 5. DCN v2: Automatic Feature Crossing

```python
class CrossLayer(nn.Module):
    """One layer of feature crossing. Adds one degree of polynomial interaction."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0: original input [B, D]
        # xl: current layer's input [B, D]
        return x0 * self.W(xl) + xl  # Element-wise: captures 2nd+ order interactions

class DCNv2(nn.Module):
    def __init__(self,
                 num_dense_features: int,
                 sparse_feature_dims: dict,
                 num_cross_layers: int = 3,
                 deep_hidden_dims: list = [256, 128]):
        super().__init__()

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab, dim)
            for name, (vocab, dim) in sparse_feature_dims.items()
        })

        total_embed_dim = sum(d for _, d in sparse_feature_dims.values())
        self.input_dim = num_dense_features + total_embed_dim

        # Cross network: learns explicit feature interactions automatically
        self.cross_layers = nn.ModuleList([
            CrossLayer(self.input_dim) for _ in range(num_cross_layers)
        ])

        # Deep network: learns implicit feature interactions
        layers, in_dim = [], self.input_dim
        for h in deep_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        self.deep = nn.Sequential(*layers)

        # Final output layer combines cross + deep
        self.output = nn.Linear(self.input_dim + in_dim, 1)

    def forward(self, dense_features, sparse_features):
        embed_list = [self.embeddings[k](v) for k, v in sparse_features.items()]
        x = torch.cat([dense_features] + embed_list, dim=1)  # [B, D]

        # Cross network path
        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer(x, x_cross)  # [B, D] -- same shape throughout

        # Deep network path
        x_deep = self.deep(x)  # [B, deep_out_dim]

        # Concatenate and predict
        combined = torch.cat([x_cross, x_deep], dim=1)
        return torch.sigmoid(self.output(combined)).squeeze(1)
```

## 6. DLRM: Meta's Production Architecture

```python
import torch
import torch.nn as nn
from itertools import combinations

class DLRM(nn.Module):
    def __init__(self,
                 dense_input_dim: int,
                 sparse_feature_dims: dict,   # {name: (vocab_size, embed_dim)}
                 bottom_mlp_dims: list = [64, 32],
                 top_mlp_dims: list = [256, 128, 64],
                 interaction: str = "dot"):   # "dot" or "cat"
        super().__init__()
        self.interaction = interaction

        # Embedding tables for sparse features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab, dim)
            for name, (vocab, dim) in sparse_feature_dims.items()
        })
        embed_dim = list(sparse_feature_dims.values())[0][1]  # All must match

        # Bottom MLP: maps dense features -> same dim as embeddings
        bottom_layers, in_dim = [], dense_input_dim
        for h in bottom_mlp_dims:
            bottom_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        assert in_dim == embed_dim, f"Bottom MLP output ({in_dim}) must match embed_dim ({embed_dim})"
        self.bottom_mlp = nn.Sequential(*bottom_layers)

        # Number of features = 1 dense vector + N sparse embedding vectors
        num_features = 1 + len(sparse_feature_dims)

        if interaction == "dot":
            # Pairwise dot products: C(num_features, 2) interaction terms
            num_interactions = num_features * (num_features - 1) // 2
            top_input_dim = embed_dim + num_interactions
        else:
            top_input_dim = embed_dim * num_features

        # Top MLP: final scoring head
        top_layers, in_dim = [], top_input_dim
        for h in top_mlp_dims:
            top_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        self.top_mlp = nn.Sequential(*top_layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, dense_features, sparse_features):
        # Step 1: Process dense features through Bottom MLP -> [B, embed_dim]
        dense_out = self.bottom_mlp(dense_features)  # [B, embed_dim]

        # Step 2: Look up embeddings for all sparse features -> [B, embed_dim] each
        embed_vecs = [dense_out] + [
            self.embeddings[name](ids)
            for name, ids in sparse_features.items()
        ]
        # embed_vecs is a list of [B, embed_dim] tensors

        # Step 3: Interaction layer -- pairwise dot products
        if self.interaction == "dot":
            interaction_terms = []
            for i, j in combinations(range(len(embed_vecs)), 2):
                # Dot product of each pair of vectors -> scalar per sample
                dot = (embed_vecs[i] * embed_vecs[j]).sum(dim=1, keepdim=True)
                interaction_terms.append(dot)
            # Concatenate: dense vector + all pairwise interactions
            x = torch.cat([dense_out] + interaction_terms, dim=1)  # [B, top_input_dim]
        else:
            x = torch.cat(embed_vecs, dim=1)

        # Step 4: Top MLP -> click probability
        return torch.sigmoid(self.output_layer(self.top_mlp(x))).squeeze(1)


# Instantiate and check
model = DLRM(
    dense_input_dim=8,
    sparse_feature_dims={
        "user_id":    (500_000, 32),
        "item_id":    (1_000_000, 32),
        "creator_id": (50_000, 32),
        "category_id":(500, 32),
    },
    bottom_mlp_dims=[64, 32],
    top_mlp_dims=[256, 128, 64],
)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Dummy forward pass
B = 256
dense  = torch.randn(B, 8)
sparse = {
    "user_id":    torch.randint(0, 500_000,   (B,)),
    "item_id":    torch.randint(0, 1_000_000, (B,)),
    "creator_id": torch.randint(0, 50_000,    (B,)),
    "category_id":torch.randint(0, 500,       (B,)),
}
out = model(dense, sparse)
print(f"Output shape: {out.shape}")    # [256]
print(f"Score range: [{out.min():.3f}, {out.max():.3f}]")
```

## 6. DLRM: Meta's Production Architecture

```python
Total parameters: 49,670,881
Output shape: torch.Size([256])
Score range: [0.443, 0.548]
```

## 7. Multi-Task Ranking: Stop Optimizing for Just One Thing

```
Final Score = w_ctr  x P(click)
            + w_cvr  x P(purchase | click)
            + w_wt   x P(long_watch)
            + w_like x P(like)
            - w_skip x P(quick_skip)
            - w_neg  x P(negative_feedback)
```

## Naive Approach: Separate Models

```
# Train one model per objective
model_ctr  = DLRM(...)
model_cvr  = DLRM(...)
model_like = DLRM(...)

# At serving time: run all three, combine scores
p_click    = model_ctr(dense, sparse)
p_purchase = model_cvr(dense, sparse)
p_like     = model_like(dense, sparse)

score = 0.5 * p_click + 0.3 * p_purchase + 0.2 * p_like
```

## Multi-Task Learning: Shared Representations

```python
class MultiTaskRanker(nn.Module):
    def __init__(self,
                 dense_input_dim: int,
                 sparse_feature_dims: dict,
                 shared_dims: list = [256, 128],
                 task_names: list = ["ctr", "cvr", "like"]):
        super().__init__()
        self.task_names = task_names

        # Shared embedding tables (the expensive part -- shared across all tasks)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab, dim)
            for name, (vocab, dim) in sparse_feature_dims.items()
        })

        total_embed_dim = sum(d for _, d in sparse_feature_dims.values())
        input_dim = dense_input_dim + total_embed_dim

        # Shared trunk: learns representations useful for all tasks
        shared_layers, in_dim = [], input_dim
        for h in shared_dims:
            shared_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        self.shared_trunk = nn.Sequential(*shared_layers)

        # Task-specific heads: small MLP per task
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            for task in task_names
        })

    def forward(self, dense_features, sparse_features):
        embed_list = [self.embeddings[k](v) for k, v in sparse_features.items()]
        x = torch.cat([dense_features] + embed_list, dim=1)

        # Shared representation (computed once for all tasks)
        shared = self.shared_trunk(x)

        # Task-specific predictions
        outputs = {
            task: torch.sigmoid(head(shared)).squeeze(1)
            for task, head in self.task_heads.items()
        }
        return outputs


# Training loop: compute loss for each task, combine
model = MultiTaskRanker(
    dense_input_dim=10,
    sparse_feature_dims={"user_id":(100_000,32), "item_id":(500_000,32)},
    task_names=["ctr", "cvr", "like"]
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
task_weights = {"ctr": 1.0, "cvr": 2.0, "like": 0.5}   # tune these for your product

B = 512
dense  = torch.randn(B, 10)
sparse = {"user_id": torch.randint(0,100_000,(B,)), "item_id": torch.randint(0,500_000,(B,))}
labels = {"ctr": torch.randint(0,2,(B,)).float(), "cvr": torch.randint(0,2,(B,)).float(), "like": torch.randint(0,2,(B,)).float()}

predictions = model(dense, sparse)

total_loss = sum(
    task_weights[task] * criterion(predictions[task], labels[task])
    for task in model.task_names
)

total_loss.backward()
optimizer.step()

print("Predictions:")
for task, pred in predictions.items():
    print(f"  {task}: mean={pred.mean():.3f}, std={pred.std():.3f}")
```

## Multi-Task Learning: Shared Representations

```
Predictions:
  ctr: mean=0.522, std=0.007
  cvr: mean=0.473, std=0.008
  like: mean=0.519, std=0.010
```

## 9. Sequential Modeling: What Is This User Interested in Right Now?

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepInterestNetwork(nn.Module):
    """
    DIN: Instead of a fixed user embedding, compute a dynamic
    representation of the user by attending to their history
    w.r.t. the current candidate item.
    """
    def __init__(self,
                 item_vocab_size: int,
                 item_embed_dim: int = 32,
                 history_len: int = 50,
                 dense_input_dim: int = 8,
                 mlp_dims: list = [256, 128, 64]):
        super().__init__()
        self.item_embedding = nn.Embedding(item_vocab_size, item_embed_dim)

        # Attention network: scores how relevant each history item is
        # to the current candidate
        self.attention = nn.Sequential(
            nn.Linear(item_embed_dim * 3, 64),  # [hist_item, target_item, interaction]
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Final MLP: takes attended user vector + dense features -> click prob
        mlp_input_dim = item_embed_dim + dense_input_dim
        layers, in_dim = [], mlp_input_dim
        for h in mlp_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self,
                target_item_id: torch.Tensor,    # [B] -- current candidate
                history_item_ids: torch.Tensor,  # [B, L] -- last L interactions
                history_mask: torch.Tensor,      # [B, L] -- 1=valid, 0=padding
                dense_features: torch.Tensor):   # [B, D] -- user/context features
        B, L = history_item_ids.shape

        # Embed the target item and history items
        target_emb   = self.item_embedding(target_item_id)        # [B, E]
        history_emb  = self.item_embedding(history_item_ids)      # [B, L, E]
        target_exp   = target_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, E]

        # Attention: how much does each history item relate to the target?
        # Concatenate [history_item, target_item, element-wise product]
        attn_input = torch.cat([
            history_emb,                          # [B, L, E]
            target_exp,                           # [B, L, E]
            history_emb * target_exp,             # [B, L, E] -- interaction term
        ], dim=2)                                 # [B, L, 3E]

        attn_scores = self.attention(attn_input).squeeze(2)       # [B, L]

        # Mask padding positions with -inf before softmax
        attn_scores = attn_scores.masked_fill(history_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)              # [B, L]

        # Weighted sum of history embeddings -> dynamic user representation
        user_vector = (attn_weights.unsqueeze(2) * history_emb).sum(dim=1)  # [B, E]

        # Concat with dense features and score
        x = torch.cat([user_vector, dense_features], dim=1)      # [B, E+D]
        return torch.sigmoid(self.output(self.mlp(x))).squeeze(1)


# Example forward pass
model = DeepInterestNetwork(
    item_vocab_size=1_000_000,
    item_embed_dim=32,
    history_len=50,
    dense_input_dim=8,
)

B = 256
target_items   = torch.randint(0, 1_000_000, (B,))
history_items  = torch.randint(0, 1_000_000, (B, 50))
history_mask   = torch.ones(B, 50)
history_mask[:, 30:] = 0   # Simulate users with only 30 history items
dense_feats    = torch.randn(B, 8)

preds = model(target_items, history_items, history_mask, dense_feats)
print(f"Output shape: {preds.shape}")     # [256]
print(f"Score range: [{preds.min():.3f}, {preds.max():.3f}]")
```

## 9. Sequential Modeling: What Is This User Interested in Right Now?

```python
Output shape: torch.Size([256])
Score range: [0.482, 0.520]
```

## 9. Sequential Modeling: What Is This User Interested in Right Now?

```
Static user_id embedding:    "This user likes cooking."
DIN user vector:             "This user just watched 3 cooking videos -- they're in cooking mode right now."
```

## Why This Is a Bigger Deal Than It Sounds

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HSTULayer(nn.Module):
    """
    A simplified HSTU layer. The real HSTU uses a custom relative
    attention mechanism optimised for very long sequences (1000s of actions),
    but the core idea is standard causal self-attention over the action sequence.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None):
        # Causal self-attention: each position can only attend to past positions
        # This is the "generative" part -- predict the next action
        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class HSTU(nn.Module):
    """
    Simplified HSTU for recommendation.

    Core idea: the user's action sequence is the model input.
    No static user_id embedding. No interaction layer.
    The transformer reads the sequence and predicts the next item.
    """
    def __init__(self,
                 item_vocab_size: int,
                 embed_dim: int = 64,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 max_seq_len: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # Each past action is an item embedding
        self.item_embedding = nn.Embedding(item_vocab_size, embed_dim, padding_idx=0)

        # Learnable positional encoding (recency matters in recommendations)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Stack of HSTU transformer layers
        self.layers = nn.ModuleList([
            HSTULayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Upper-triangular mask: position i cannot attend to j > i."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, action_sequence: torch.Tensor) -> torch.Tensor:
        """
        action_sequence: [B, L] -- sequence of item IDs (0 = padding)
        Returns:         [B, L, E] -- contextual representation at each position
        """
        B, L = action_sequence.shape
        positions = torch.arange(L, device=action_sequence.device).unsqueeze(0)

        # Embed actions + add positional encoding
        x = self.item_embedding(action_sequence) + self.pos_embedding(positions)

        # Apply causal mask so position t only sees actions 0..t-1
        causal_mask = self._causal_mask(L, action_sequence.device)

        for layer in self.layers:
            x = layer(x, causal_mask)

        return self.norm(x)  # [B, L, E]

    def score_next_item(self,
                        action_sequence: torch.Tensor,
                        candidate_item_ids: torch.Tensor) -> torch.Tensor:
        """
        Score a set of candidate items as the next action.

        action_sequence:    [B, L]    -- past actions
        candidate_item_ids: [B, K]    -- K candidates to score
        Returns:            [B, K]    -- relevance scores
        """
        # Get the contextual user state at the last real position
        seq_repr = self.forward(action_sequence)       # [B, L, E]
        user_state = seq_repr[:, -1, :]               # [B, E] -- "what is the user thinking now?"

        # Score each candidate via dot product with its embedding
        cand_emb = self.item_embedding(candidate_item_ids)  # [B, K, E]
        scores = torch.bmm(cand_emb, user_state.unsqueeze(2)).squeeze(2)  # [B, K]
        return scores


# Example
model = HSTU(item_vocab_size=1_000_000, embed_dim=64, num_layers=4)

B, L, K = 32, 128, 100  # 32 users, 128 history actions, 100 candidates
action_seq    = torch.randint(1, 1_000_000, (B, L))   # Past actions
candidates    = torch.randint(1, 1_000_000, (B, K))   # Candidate items to rank

scores = model.score_next_item(action_seq, candidates)
print(f"Output shape: {scores.shape}")  # [32, 100]
print(f"Top-1 candidate per user: {scores.argmax(dim=1)}")
```

## Why This Is a Bigger Deal Than It Sounds

```python
Output shape: torch.Size([32, 100])
Top-1 candidate per user: tensor([64, 28, 21, 81, 26, 80, 62, 90, 62,  6,  4, 77, 75, 96, 46, 96,  9,  2, 94, 34, 72, 87, 97, 61, 80, 20, 17, 60, 68, 16, 55,  5])
```

## 11.1 Embedding Tables: The Elephant in the Room

```python
# Hash trick: bound embedding table size at 10M rows regardless of true vocab size
HASH_SIZE = 10_000_000
hashed_user_id = user_id % HASH_SIZE  # Collisions are rare, acceptable trade-off

user_embedding_table = nn.Embedding(HASH_SIZE, 64)
user_emb = user_embedding_table(torch.tensor([hashed_user_id]))
```

## 11.3 Offline vs. Online Evaluation

```
Offline metrics (AUC, logloss) != Online metrics (CTR, Revenue)
```

## 11.3 Offline vs. Online Evaluation

```python
# Minimum detectable effect calculation before running an A/B test
import scipy.stats as stats

def min_sample_size(baseline_rate, min_effect, alpha=0.05, power=0.8):
    """Calculate minimum users per variant for an A/B test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # 1.96 for alpha=0.05
    z_beta  = stats.norm.ppf(power)           # 0.84 for power=0.8

    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_effect)
    p_bar = (p1 + p2) / 2

    n = (z_alpha * (2 * p_bar * (1 - p_bar)) ** 0.5 +
         z_beta  * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5) ** 2 / (p1 - p2) ** 2

    return int(n)

# Example: baseline CTR = 4%, want to detect 5% relative improvement
n = min_sample_size(baseline_rate=0.04, min_effect=0.05)
print(f"Users needed per variant: {n:,}")
# Output: Users needed per variant: 145,828
```

## 11.4 Position Bias and Propensity Correction

```python
# Propensity-weighted loss: down-weight easy top-position clicks

def propensity_weighted_bce(predictions, labels, positions):
    """
    Correct for position bias using inverse propensity scoring.
    p(position_k) is estimated from a randomization experiment or positional model.
    """
    # Empirical propensity: clicks at position k relative to position 1
    propensity_by_position = {1: 1.0, 2: 0.68, 3: 0.51, 4: 0.39, 5: 0.29}
    propensities = torch.tensor([propensity_by_position.get(p.item(), 0.1) for p in positions])

    # Inverse propensity score: rare positions get higher weight
    ips_weights = 1.0 / propensities

    # Weighted binary cross-entropy
    bce = nn.BCELoss(reduction='none')
    loss = bce(predictions, labels)
    return (loss * ips_weights).mean()
```

## 11.5 Calibration: Making P(click) Actually Mean Something

```python
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def check_calibration(y_true, y_pred, n_bins=10):
    """
    Check if model probabilities are calibrated.
    A perfectly calibrated model has fraction_of_positives == mean_predicted_value.
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

    print("Bin    | Predicted | Actual  | Gap")
    print("-------|-----------|---------|----------")
    for pred, actual in zip(mean_pred, fraction_pos):
        gap = abs(pred - actual)
        flag = " <- overconfident" if pred > actual * 1.5 else ""
        print(f"{pred:.3f}  | {pred:.3f}     | {actual:.3f}   | {gap:.3f}{flag}")

    return fraction_pos, mean_pred


# Simulate a downsampled-trained model: scores inflated by ~10x
np.random.seed(42)
true_ctr = 0.001   # Real-world: 0.1% CTR
n_samples = 100_000

y_true = (np.random.rand(n_samples) < true_ctr).astype(int)
# Model trained on 1:10 downsampled negatives -- scores inflated
y_pred_uncalibrated = np.clip(np.random.beta(2, 5, n_samples) * 0.15, 0, 1)

print("=== Uncalibrated model ===")
check_calibration(y_true, y_pred_uncalibrated)
```

## 11.5 Calibration: Making P(click) Actually Mean Something

```
=== Uncalibrated model ===
Bin    | Predicted | Actual  | Gap
-------|-----------|---------|----------
0.013  | 0.013     | 0.001   | 0.012 <- overconfident
0.038  | 0.038     | 0.001   | 0.037 <- overconfident
0.063  | 0.063     | 0.001   | 0.062 <- overconfident
```

## 11.5 Calibration: Making P(click) Actually Mean Something

```python
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def platt_scale(train_scores, train_labels, val_scores):
    """
    Fit a logistic regression on model scores to recalibrate probabilities.
    Simple and effective for most production use cases.
    """
    calibrator = LogisticRegression()
    calibrator.fit(train_scores.reshape(-1, 1), train_labels)
    return calibrator.predict_proba(val_scores.reshape(-1, 1))[:, 1]


def isotonic_calibration(train_scores, train_labels, val_scores):
    """
    Isotonic regression: more flexible than Platt scaling.
    Better when the score-probability relationship is non-monotonic.
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_scores, train_labels)
    return calibrator.predict(val_scores)


# For downsampled training, you can also correct analytically:
def correct_for_downsampling(raw_score, negative_sample_rate):
    """
    If you downsampled negatives at rate r (kept 1/r of negatives),
    correct the predicted probability analytically.
    Reference: McMahan et al. (2013), Ad Click Prediction at Scale.
    """
    # raw_score = p / (p + (1-p)/r)  =>  solve for true p
    return raw_score / (raw_score + (1 - raw_score) / negative_sample_rate)


# Example: model trained with 1:10 ratio instead of true 1:1000
inflated_score = 0.05   # Model thinks 5% CTR
true_rate = 10 / 1000   # We kept 1 in 100 negatives
corrected = correct_for_downsampling(inflated_score, true_rate)
print(f"Raw score: {inflated_score:.4f}  ->  Corrected: {corrected:.4f}")
# Raw score: 0.0500  ->  Corrected: 0.0056
```
