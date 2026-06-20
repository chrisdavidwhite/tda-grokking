#!/usr/bin/env python3
"""Manir and Rupa residual MLP for modular addition."""

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
WIDTH = 512
DEPTH = 8
RESIDUAL = True
LR = 1e-2
MOMENTUM = 0.9
GRAD_CLIP = 1.0

CONFIG = ExperimentConfig(
    name="manir_residual_mlp",
    out_dir=Path("runs/manir_residual_mlp"),
    load_data=False,
    save_data=True,
    runs=5,
    seed=0,
    device="auto",
    num_threads=1,
    grok_train_frac=0.2,
    memorise_train_frac=0.2,
    grok_steps=150_000,
    memorise_steps=150_000,
    grok_weight_decay=2e-3,
    memorise_weight_decay=0.0,
    eval_every=1000,
    log_every=100,
    n_checkpoints=25,
    include_initial_checkpoint=True,
    stop_early=False,
)

class ResidualBlock(nn.Module):
    def __init__(self, width: int, scale: float = 0.5):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.scale = scale

    def forward(self, h):
        return h + self.scale * self.fc2(F.gelu(self.fc1(self.ln(h))))


class ManirResidualMLP(nn.Module):
    def __init__(self, p: int, width: int, depth: int, residual: bool):
        super().__init__()
        self.emb = nn.Embedding(p, width)
        self.residual = residual
        if residual:
            self.layers = nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
        else:
            self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, p)

    def forward(self, x):
        h = self.emb(x[:, 0]) + self.emb(x[:, 1])
        for layer in self.layers:
            h = layer(h) if self.residual else F.gelu(layer(h))
        return self.head(h)


def make_dataset(train_frac, device, config):
    return modular_pair_dataset(P, OPERATION, train_frac, SPLIT_SEED, device)


def make_model(config):
    return ManirResidualMLP(P, WIDTH, DEPTH, RESIDUAL)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=weight_decay)


def train_step(spec, model, optimizer, xtr, ytr, step, condition, device, config):
    loss = spec.loss_fn(model, xtr, ytr, config)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return loss


def metadata(config):
    return {"p": P, "operation": OPERATION, "width": WIDTH, "depth": DEPTH, "lr": LR, "grad_clip": GRAD_CLIP}


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata, train_step=train_step)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
