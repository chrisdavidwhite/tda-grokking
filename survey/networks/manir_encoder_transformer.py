#!/usr/bin/env python3
"""Manir and Rupa encoder-only transformer for modular addition."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('training')

from datasets import modular_pair_dataset
from experiment_core import ExperimentConfig, ExperimentSpec, print_summary, run_or_load


P = 97
OPERATION = "add"
SPLIT_SEED = 0
D_MODEL = 512
N_HEADS = 4
DEPTH = 1
D_FF = 2048
DROPOUT = 0.0
LR = 1e-3
GRAD_CLIP = 1.0

CONFIG = ExperimentConfig(
    name="manir_encoder_transformer",
    out_dir=Path("runs/manir_encoder_transformer"),
    load_data=False,
    save_data=True,
    runs=5,
    seed=0,
    device="auto",
    num_threads=1,
    grok_train_frac=0.2,
    memorise_train_frac=0.2,
    grok_steps=100_000,
    memorise_steps=100_000,
    grok_weight_decay=1.0,
    memorise_weight_decay=0.0,
    eval_every=1000,
    log_every=100,
    n_checkpoints=25,
    include_initial_checkpoint=True,
    stop_early=False,
)

class ManirEncoderTransformer(nn.Module):
    def __init__(self, p: int, d_model: int, n_heads: int, depth: int, d_ff: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(p, d_model)
        self.pos = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(d_model, p)

    def forward(self, x):
        h = self.emb(x) + self.pos.unsqueeze(0)
        return self.head(self.encoder(h).mean(dim=1))


def make_dataset(train_frac, device, config):
    return modular_pair_dataset(P, OPERATION, train_frac, SPLIT_SEED, device)


def make_model(config):
    return ManirEncoderTransformer(P, D_MODEL, N_HEADS, DEPTH, D_FF, DROPOUT)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)


def train_step(spec, model, optimizer, xtr, ytr, step, condition, device, config):
    loss = spec.loss_fn(model, xtr, ytr, config)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return loss


def metadata(config):
    return {"p": P, "operation": OPERATION, "d_model": D_MODEL, "n_heads": N_HEADS, "depth": DEPTH, "lr": LR}


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata, train_step=train_step)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
