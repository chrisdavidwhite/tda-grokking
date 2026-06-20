#!/usr/bin/env python3
"""Salah and Yevick modular-addition embedding-concat MLP."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('training')

from datasets import modular_pair_dataset
from experiment_core import ExperimentConfig, ExperimentSpec, print_summary, run_or_load


P = 43
OPERATION = "add"
SPLIT_SEED = 0
D_EMBED = 128
HIDDEN = 128
INIT_SCALE = 1.0
LR = 3e-4
OPTIMIZER = "adamw"

CONFIG = ExperimentConfig(
    name="salah_embedding_concat_mlp",
    out_dir=Path("runs/salah_embedding_concat_mlp"),
    load_data=False,
    save_data=True,
    runs=25,
    seed=0,
    same_condition_seeds=True,
    device="auto",
    num_threads=1,
    grok_train_frac=0.4,
    memorise_train_frac=0.4,
    grok_steps=150_000,
    memorise_steps=150_000,
    grok_weight_decay=5.0,
    memorise_weight_decay=0.0,
    eval_every=200,
    log_every=200,
    n_checkpoints=100,
    include_initial_checkpoint=True,
    checkpoint_style="log",
    log_checkpoint_min_step=1,
    stop_early=False,
)

class SalahEmbeddingConcatMLP(nn.Module):
    def __init__(self, p: int, d_embed: int, hidden: int, init_scale: float):
        super().__init__()
        self.emb = nn.Embedding(p, d_embed)
        self.fc1 = nn.Linear(2 * d_embed, hidden)
        self.fc2 = nn.Linear(hidden, p)
        self.reset(init_scale)

    def reset(self, scale):
        nn.init.xavier_normal_(self.emb.weight)
        self.emb.weight.data.mul_(scale)
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_normal_(layer.weight)
            layer.weight.data.mul_(scale)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        h = torch.cat([self.emb(x[:, 0]), self.emb(x[:, 1])], dim=-1)
        return self.fc2(F.relu(self.fc1(h)))


def make_dataset(train_frac, device, config):
    return modular_pair_dataset(P, OPERATION, train_frac, SPLIT_SEED, device)


def make_model(config):
    return SalahEmbeddingConcatMLP(P, D_EMBED, HIDDEN, INIT_SCALE)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)


def metadata(config):
    return {"p": P, "operation": OPERATION, "d_embed": D_EMBED, "hidden": HIDDEN, "lr": LR}


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
