"""Pairwise neural ranker (RankNet-style) in PyTorch.

Small MLP over the engineered features, trained with a pairwise logistic
loss on document pairs whose relevance grade differs within one query.
"""
import numpy as np
import torch
from torch import nn


class RankNet(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_pairs(X, y, group):
    """All (i, j) index pairs per query where y[i] > y[j], capped per query."""
    pairs = []
    start = 0
    for size in group:
        idx = range(start, start + size)
        pos = {}
        for i in idx:
            pos.setdefault(y[i], []).append(i)
        grades = sorted(pos, reverse=True)
        count = 0
        for hi_i, g_hi in enumerate(grades):
            for g_lo in grades[hi_i + 1:]:
                for i in pos[g_hi]:
                    for j in pos[g_lo]:
                        if count < 200:  # cap per query to bound training time
                            pairs.append((i, j))
                            count += 1
        start += size
    return pairs


def train_ranknet(X, y, group, X_val=None, epochs=8, lr=1e-3, seed=7, verbose=True):
    torch.manual_seed(seed)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xs = torch.tensor((X - mean) / std, dtype=torch.float32)
    model = RankNet(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    pairs = make_pairs(X, y, group)
    if verbose:
        print(f"  torch ranker: {len(pairs)} training pairs, {epochs} epochs", flush=True)
    pa = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    pb = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(pairs))
        total = 0.0
        for chunk in perm.split(4096):
            opt.zero_grad()
            diff = model(Xs[pa[chunk]]) - model(Xs[pb[chunk]])
            loss = loss_fn(diff, torch.ones_like(diff))
            loss.backward()
            opt.step()
            total += loss.item() * len(chunk)
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs} pair-loss {total / max(1, len(pairs)):.4f}", flush=True)
    model.eval()
    return model, mean, std


def predict_ranknet(model, X, mean, std):
    with torch.no_grad():
        Xs = torch.tensor((X - mean) / std, dtype=torch.float32)
        return model(Xs).numpy()
