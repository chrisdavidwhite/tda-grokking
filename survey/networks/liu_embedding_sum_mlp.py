#!/usr/bin/env python3
"""Liu et al. trainable embeddings plus MLP decoder."""

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
SYMMETRIC = False

D_EMBED = 256
HIDDEN = 512
DEPTH = 2
ACTIVATION = "relu"

EMBEDDING_LR = 1e-3
DECODER_LR = 1e-3
GROK_EMBEDDING_WEIGHT_DECAY = 0.0
GROK_DECODER_WEIGHT_DECAY = 1.0
MEMORISE_EMBEDDING_WEIGHT_DECAY = 0.0
MEMORISE_DECODER_WEIGHT_DECAY = 0.0

CONFIG = ExperimentConfig(
    name="liu_embedding_sum_mlp",
    out_dir=Path("runs/liu_embedding_sum_mlp"),
    load_data=False,
    save_data=True,
    runs=5,
    seed=0,
    device="auto",
    num_threads=1,
    grok_train_frac=0.4,
    memorise_train_frac=0.4,
    grok_steps=10_000,
    memorise_steps=10_000,
    eval_every=500,
    log_every=100,
    n_checkpoints=25,
    include_initial_checkpoint=True,
    stop_early=False,
)

class LiuEmbeddingSumMLP(nn.Module):
    def __init__(self, p: int, d_embed: int, hidden: int, depth: int, activation: str):
        super().__init__()
        self.emb = nn.Embedding(p, d_embed)
        act = nn.ReLU if activation == "relu" else nn.GELU if activation == "gelu" else nn.Tanh
        layers = []
        d = d_embed
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), act()]
            d = hidden
        layers.append(nn.Linear(d, p))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.emb(x[:, 0]) + self.emb(x[:, 1]))


def make_dataset(train_frac, device, config):
    return modular_pair_dataset(P, OPERATION, train_frac, SPLIT_SEED, device, symmetric=SYMMETRIC)


def make_model(config):
    return LiuEmbeddingSumMLP(P, D_EMBED, HIDDEN, DEPTH, ACTIVATION)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    emb_wd = GROK_EMBEDDING_WEIGHT_DECAY if condition == "grok" else MEMORISE_EMBEDDING_WEIGHT_DECAY
    decoder_wd = GROK_DECODER_WEIGHT_DECAY if condition == "grok" else MEMORISE_DECODER_WEIGHT_DECAY
    return torch.optim.AdamW(
        [
            {"params": model.emb.parameters(), "lr": EMBEDDING_LR, "weight_decay": emb_wd},
            {"params": model.decoder.parameters(), "lr": DECODER_LR, "weight_decay": decoder_wd},
        ],
        betas=(0.9, 0.98),
    )


def metadata(config):
    return {
        "p": P,
        "operation": OPERATION,
        "d_embed": D_EMBED,
        "hidden": HIDDEN,
        "depth": DEPTH,
        "activation": ACTIVATION,
        "embedding_lr": EMBEDDING_LR,
        "decoder_lr": DECODER_LR,
    }


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
