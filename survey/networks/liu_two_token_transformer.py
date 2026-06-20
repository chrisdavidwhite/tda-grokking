#!/usr/bin/env python3
"""Liu et al. two-token decoder-only transformer."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('training')

from datasets import modular_pair_dataset
from experiment_core import ExperimentConfig, ExperimentSpec, print_summary, run_or_load


P = 53
OPERATION = "add"
SPLIT_SEED = 0
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
D_MLP = 1024
DROPOUT = 0.0
LAYER_NORM = True
LR = 1e-3

CONFIG = ExperimentConfig(
    name="liu_two_token_transformer",
    out_dir=Path("runs/liu_two_token_transformer"),
    load_data=True,
    save_data=False,
    runs=5,
    seed=0,
    device="auto",
    num_threads=1,
    grok_train_frac=0.4,
    memorise_train_frac=0.4,
    grok_steps=40_000,
    memorise_steps=40_000,
    grok_weight_decay=1.0,
    memorise_weight_decay=0.0,
    eval_every=500,
    log_every=100,
    n_checkpoints=25,
    include_initial_checkpoint=True,
    stop_early=False,
)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float, layer_norm: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True, bias=False)
        self.ln1 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.ff = nn.Sequential(nn.Linear(d_model, d_mlp, bias=False), nn.ReLU(), nn.Linear(d_mlp, d_model, bias=False))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + self.drop(a))
        return self.ln2(x + self.drop(self.ff(x)))


class LiuTwoTokenTransformer(nn.Module):
    def __init__(self, p: int, d_model: int, n_layers: int, n_heads: int, d_mlp: int, dropout: float, layer_norm: bool):
        super().__init__()
        self.emb = nn.Embedding(p, d_model)
        self.pos = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_mlp, dropout, layer_norm) for _ in range(n_layers)])
        self.head = nn.Linear(2 * d_model, p)

    def forward(self, x):
        h = self.emb(x) + self.pos.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        return self.head(h.reshape(h.size(0), -1))


def make_dataset(train_frac, device, config):
    return modular_pair_dataset(P, OPERATION, train_frac, SPLIT_SEED, device)


def make_model(config):
    return LiuTwoTokenTransformer(P, D_MODEL, N_LAYERS, N_HEADS, D_MLP, DROPOUT, LAYER_NORM)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=weight_decay)


def metadata(config):
    return {"p": P, "operation": OPERATION, "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS, "lr": LR}


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
