from __future__ import annotations

import numpy as np
import torch


def modular_labels(a, b, p: int, operation: str):
    if operation == "add":
        return (a + b) % p
    if operation == "sub":
        return (a - b) % p
    if operation == "mul":
        return (a * b) % p
    if operation == "div":
        inv = np.array([0] + [pow(i, -1, p) for i in range(1, p)], dtype=np.int64)
        return (a * inv[b]) % p
    if operation == "x2_plus_y2":
        return (a * a + b * b) % p
    raise ValueError(operation)


def modular_pairs(p: int, operation: str, symmetric: bool = False):
    pairs = []
    for a in range(p):
        start = a if symmetric else 0
        for b in range(start, p):
            if operation == "div" and b == 0:
                continue
            pairs.append((a, b))
    return np.array(pairs, dtype=np.int64)


def split_pairs(p: int, operation: str, train_frac: float, split_seed: int, symmetric: bool = False):
    pairs = modular_pairs(p, operation, symmetric=symmetric)
    rng = np.random.default_rng(split_seed)
    pairs = pairs[rng.permutation(len(pairs))]
    n_train = int(round(train_frac * len(pairs)))
    return pairs, n_train


def modular_onehot_dataset(p: int, operation: str, train_frac: float, split_seed: int, device: torch.device):
    pairs, n_train = split_pairs(p, operation, train_frac, split_seed)

    a = torch.tensor(pairs[:, 0], dtype=torch.long, device=device)
    b = torch.tensor(pairs[:, 1], dtype=torch.long, device=device)
    idx = torch.arange(len(pairs), device=device)

    x = torch.zeros(len(pairs), 2 * p, device=device)
    x[idx, a] = 1.0
    x[idx, p + b] = 1.0

    y = torch.tensor(modular_labels(pairs[:, 0], pairs[:, 1], p, operation), dtype=torch.long, device=device)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def modular_pair_dataset(p: int, operation: str, train_frac: float, split_seed: int, device: torch.device, symmetric: bool = False):
    pairs, n_train = split_pairs(p, operation, train_frac, split_seed, symmetric=symmetric)
    x = torch.tensor(pairs, dtype=torch.long, device=device)
    y = torch.tensor(modular_labels(pairs[:, 0], pairs[:, 1], p, operation), dtype=torch.long, device=device)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def modular_equation_dataset(p: int, operation: str, train_frac: float, split_seed: int, device: torch.device):
    pairs, n_train = split_pairs(p, operation, train_frac, split_seed)
    op_tok, eq_tok = p, p + 1
    x = np.stack(
        [
            pairs[:, 0],
            np.full(len(pairs), op_tok, dtype=np.int64),
            pairs[:, 1],
            np.full(len(pairs), eq_tok, dtype=np.int64),
        ],
        axis=1,
    )
    y = modular_labels(pairs[:, 0], pairs[:, 1], p, operation)
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def nanda_equation_dataset(p: int, operation: str, train_frac: float, split_seed: int, device: torch.device):
    pairs, n_train = split_pairs(p, operation, train_frac, split_seed)
    eq_tok = p
    x = np.stack([pairs[:, 0], pairs[:, 1], np.full(len(pairs), eq_tok, dtype=np.int64)], axis=1)
    y = modular_labels(pairs[:, 0], pairs[:, 1], p, operation)
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]
