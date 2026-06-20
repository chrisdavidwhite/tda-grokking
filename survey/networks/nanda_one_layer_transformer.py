#!/usr/bin/env python3
"""Nanda et al. one-layer transformer for modular addition."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('training')

from datasets import nanda_equation_dataset
from experiment_core import ExperimentConfig, ExperimentSpec, print_summary, run_or_load


P = 97
OPERATION = "add"
SPLIT_SEED = 0
D_MODEL = 128
N_HEADS = 4
D_MLP = 512
USE_BIAS = False
LR = 1e-3

CONFIG = ExperimentConfig(
    name="nanda_one_layer_transformer",
    out_dir=Path("runs/nanda_one_layer_transformer"),
    load_data=True,
    save_data=False,
    runs=5,
    seed=0,
    device="auto",
    num_threads=1,
    grok_train_frac=0.3,
    memorise_train_frac=0.3,
    grok_steps=25_000,
    memorise_steps=25_000,
    grok_weight_decay=1.0,
    memorise_weight_decay=0.0,
    eval_every=500,
    log_every=100,
    n_checkpoints=100,
    include_initial_checkpoint=True,
    stop_early=False
)

class CausalAttentionNoLN(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, use_bias: bool):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.q = nn.Linear(d_model, inner, bias=use_bias)
        self.k = nn.Linear(d_model, inner, bias=use_bias)
        self.v = nn.Linear(d_model, inner, bias=use_bias)
        self.o = nn.Linear(inner, d_model, bias=use_bias)

    def forward(self, x):
        b, t, _ = x.shape

        def split(z):
            return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        score = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = torch.softmax(score.masked_fill(mask, float("-inf")), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, self.n_heads * self.d_head)
        return self.o(y)


class NandaOneLayerTransformer(nn.Module):
    def __init__(self, p: int, d_model: int, n_heads: int, d_mlp: int, use_bias: bool):
        super().__init__()
        self.embed = nn.Embedding(p + 1, d_model)
        self.pos = nn.Parameter(torch.zeros(3, d_model))
        d_head = d_model // n_heads
        self.attn = CausalAttentionNoLN(d_model, n_heads, d_head, use_bias)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=use_bias),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model, bias=use_bias),
        )
        self.unembed = nn.Linear(d_model, p, bias=False)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, idx):
        x = self.embed(idx) + self.pos.unsqueeze(0)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return self.unembed(x[:, -1, :])


def make_dataset(train_frac, device, config):
    return nanda_equation_dataset(P, OPERATION, train_frac, SPLIT_SEED, device)


def make_model(config):
    return NandaOneLayerTransformer(P, D_MODEL, N_HEADS, D_MLP, USE_BIAS)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=weight_decay)


def metadata(config):
    return {"p": P, "operation": OPERATION, "d_model": D_MODEL, "n_heads": N_HEADS, "d_mlp": D_MLP, "lr": LR}


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
