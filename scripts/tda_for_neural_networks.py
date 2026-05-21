"""Topological Data Analysis of Neural Network Weights — Grokking on Modular Arithmetic.

Persistent Homology on the weight matrices of small MLPs trained on the
componentwise modular addition

        G = Z/m_1 × Z/m_2 × ... × Z/m_d ,
        (a, b) ↦ a + b   in G.

For d = 1 this reduces to the canonical grokking task (a + b) mod p. The
generalisation to d ≥ 2 predicts a torus geometry T^d = (S^1)^d for the
grokked solution, with Künneth Betti numbers β_k = C(d, k) — a falsifiable
extension you can test by reading off H_2, H_3, ... at convergence.

Verified hyperparameters that produce reliable grokking on CPU (d = 1):
    MODULI = [47], train_frac = 0.40
    Grokked:   AdamW, weight_decay = 5.0,  12000 epochs -> test_acc ~ 0.97
    Memorised: AdamW, weight_decay = 0.0,  12000 epochs -> test_acc ~ 0.00

Runtime: ~22s per run on CPU for d=1. d=2 raises the cost of compute_ph
(VR up to H_2). For d ≥ 3, expect minutes per checkpoint in §5.
"""

import multiprocessing as mp
import os
import time
import warnings
from collections import defaultdict
from itertools import product as iproduct
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # non-interactive: plt.show() is a no-op

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import ripser
import torch
import torch.nn as nn
import torch.nn.functional as F
from persim import wasserstein
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from scipy.stats import entropy as scipy_entropy, mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

C = {"grokked": "#1d4ed8", "memorised": "#dc2626",
     "h0": "#7c3aed", "h1": "#059669", "accent": "#f59e0b"}

# DEVICE is set after MODULI/TRAIN_FRAC below (the MPS branch needs train_n).

# All figures land here, regardless of cwd when the script is invoked.
# FIG_DIR is set after MODULI/N_EPOCHS/TRAIN_FRAC are known (it embeds them
# in the path so different sweeps don't overwrite each other's plots).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def figpath(name):
    return str(FIG_DIR / name)


# ── §1. Task selector, hyperparameters, derived constants ────────────────
# TASK is the master switch between modular-addition (default), modular-
# multiplication on (Z/p)*, and sparse parity. Set via GROK_TASK env var;
# spawn workers inherit it, so the wrapper scripts only have to set it
# once in the parent process.
TASK = os.environ.get("GROK_TASK", "add").strip().lower()
if TASK not in ("add", "add_flat", "mul", "sparse_parity"):
    raise ValueError(
        f"GROK_TASK must be one of add/add_flat/mul/sparse_parity; got {TASK!r}"
    )

# Hyperparameters — env-var-overridable so wrapper scripts can tune per task
# without editing this file.
MODULI     = [17,17]           # add/mul only. d=2 attempts archived in repo.
TRAIN_FRAC = 0.06           # add/mul only.
HIDDEN     = int(os.environ.get("GROK_HIDDEN",   "256"))
N_EPOCHS   = int(os.environ.get("GROK_N_EPOCHS", "200_000"))
N_RUNS     = int(os.environ.get("GROK_N_RUNS",   "5"))
LR         = float(os.environ.get("GROK_LR",       "1e-3"))
WD_GROK    = float(os.environ.get("GROK_WD_GROK",  "5.0"))
WD_MEM     = float(os.environ.get("GROK_WD_MEM",   "0.0"))

# Sparse-parity-specific hyperparameters (ignored for add/mul).
SP_N_BITS  = int(os.environ.get("SP_N_BITS", "40"))
SP_K       = int(os.environ.get("SP_K",      "3"))
SP_TRAIN   = int(os.environ.get("SP_TRAIN",  "1024"))

# Derived constants. EFFECTIVE_MODULI is the "head-split" passed to
# multi_loss / multi_acc — for sparse parity it's [2] (binary single head);
# for add/mul/add_flat it equals MODULI. IN_DIM differs for add_flat
# because each input element is one-hot over the WHOLE group |G| instead
# of per-factor (so the model can't read the factorisation off the input).
if TASK == "sparse_parity":
    D                = 1
    IN_DIM           = SP_N_BITS
    OUT_DIM          = 2
    EFFECTIVE_MODULI = [2]
    MAX_DIM          = 1
else:
    D                = len(MODULI)
    if TASK == "add_flat":
        IN_DIM       = 2 * int(np.prod(MODULI))     # 2 * |G|
    else:
        IN_DIM       = 2 * sum(MODULI)              # 2 * Σ m_i  (per-factor)
    OUT_DIM          = sum(MODULI)
    EFFECTIVE_MODULI = list(MODULI)
    MAX_DIM          = D

CKPT_EPOCHS = sorted(set(
    [1, 5, 10, 25, 50, 100, 200, 500]
    + list(np.unique(np.logspace(np.log10(500), np.log10(N_EPOCHS), 35).astype(int)))
))


def _pick_device():
    """Choose cuda/mps/cpu. CPU is the default on Mac: on M1 Max the
    eventual sweep parallelism (50 independent runs) beats MPS even at
    d=2, and MPS is strictly worse at d=1. CUDA is used automatically
    when available. Force MPS with GROK_DEVICE=mps if you want to test
    it on a different machine."""
    override = os.environ.get("GROK_DEVICE", "").strip().lower()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _pick_device()
if mp.current_process().name == "MainProcess":
    print(f"[device] {DEVICE}")


def _run_tag():
    """Subdirectory name for this run's figures. Embeds the hyperparameters
    that change between experiments so plots aren't silently overwritten."""
    if TASK == "sparse_parity":
        return f"SP_n{SP_N_BITS}_k{SP_K}_T{SP_TRAIN}_E{N_EPOCHS}"
    moduli_str = "x".join(str(m) for m in MODULI)
    if TASK == "mul":
        prefix = "Mul"
    elif TASK == "add_flat":
        prefix = "Mflat"
    else:
        prefix = "M"
    return f"{prefix}{moduli_str}_E{N_EPOCHS}_F{TRAIN_FRAC:.2f}"


def _task_label():
    """Short human-readable description of the current task, for plot titles."""
    if TASK == "sparse_parity":
        return f"sparse parity n={SP_N_BITS}, k={SP_K}"
    if TASK == "mul":
        return f"MODULI={MODULI} (· mod p)"
    if TASK == "add_flat":
        return f"MODULI={MODULI} (+ mod p, flat input)"
    return f"MODULI={MODULI} (+ mod p)"


FIG_DIR = PROJECT_ROOT / "Figures" / _run_tag()
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _pick_workers():
    """Pool size for parallel runs. CPU-only by default — a single GPU
    context can't run K trainings in parallel, so we don't try.
    Override with GROK_WORKERS=N."""
    override = os.environ.get("GROK_WORKERS", "").strip()
    if override:
        return max(1, int(override))
    if DEVICE.type != "cpu":
        return 1
    return min(N_RUNS, os.cpu_count() or 1)


def _worker_init():
    """Pin BLAS to 1 thread per worker so K workers × T threads
    don't oversubscribe the CPU."""
    torch.set_num_threads(1)


def _train_run_task(args):
    wd, seed = args
    t1 = time.time()
    hist, ckpts = train_run(
        moduli=MODULI, hidden=HIDDEN, n_epochs=N_EPOCHS,
        lr=LR, weight_decay=wd, ckpt_epochs=CKPT_EPOCHS, seed=seed,
    )
    return seed, hist, ckpts, time.time() - t1


def _compute_tda_task(args):
    cond, ri, epoch, W, max_dim = args
    dgms = compute_ph(W, max_dim=max_dim)
    rec  = summarise(dgms)
    rec["epoch"] = epoch
    rec["dgms"]  = dgms
    return cond, ri, epoch, rec


def make_dataset(moduli=None, frac=None, seed=0):
    """Dispatch to the task-appropriate dataset constructor.

    Returns ((X_train, y_train), (X_test, y_test)) for the current TASK.
    Workers re-import this module on spawn and see the same TASK via env
    var, so all replicas agree on which dataset to build."""
    if frac is None:
        frac = TRAIN_FRAC
    if TASK == "sparse_parity":
        return _make_dataset_sparse_parity(seed=seed)
    if TASK == "mul":
        return _make_dataset_mul(moduli=moduli, frac=frac, seed=seed)
    if TASK == "add_flat":
        return _make_dataset_add_flat(moduli=moduli, frac=frac, seed=seed)
    return _make_dataset_add(moduli=moduli, frac=frac, seed=seed)


def _make_dataset_add(moduli=None, frac=TRAIN_FRAC, seed=0):
    """G = Z/m_1 × ... × Z/m_d.  Inputs are pairs (a, b) in G × G,
    encoded as concatenated one-hots over each factor (2 * sum(moduli) dims).
    Labels are (a + b) mod m_i per factor, shape (N, d)."""
    if moduli is None:
        moduli = MODULI
    rng    = np.random.default_rng(seed)
    d      = len(moduli)
    elems  = np.array(list(iproduct(*(range(m) for m in moduli))))   # (|G|, d)
    M      = len(elems)
    pair_i = np.array(list(iproduct(range(M), range(M))))            # (|G|^2, 2)
    a      = elems[pair_i[:, 0]]                                     # (|G|^2, d)
    b      = elems[pair_i[:, 1]]
    labels = (a + b) % np.array(moduli)                              # (|G|^2, d)

    idx = rng.permutation(len(pair_i))
    n   = int(frac * len(pair_i))

    def enc(ix):
        a_oh = [F.one_hot(torch.tensor(a[ix, i]), moduli[i]).float() for i in range(d)]
        b_oh = [F.one_hot(torch.tensor(b[ix, i]), moduli[i]).float() for i in range(d)]
        X    = torch.cat(a_oh + b_oh, dim=1)
        y    = torch.tensor(labels[ix], dtype=torch.long)
        return X, y

    return enc(idx[:n]), enc(idx[n:])


def _make_dataset_add_flat(moduli=None, frac=TRAIN_FRAC, seed=0):
    """Modular addition with per-ELEMENT (rather than per-factor) one-hot
    input encoding. Inputs are pairs (a, b) ∈ G × G with G = Z/m_1 × … ×
    Z/m_d; each of a and b is one-hot encoded over |G| = ∏ m_i classes,
    so IN_DIM = 2·|G| instead of 2·Σ m_i. Labels remain (a+b) per factor,
    so the multi-head loss is unchanged.

    Motivation: the per-factor encoding of _make_dataset_add hands the
    factor decomposition to the model for free at d ≥ 2 (each factor's
    one-hot block is a separate region of the input), which appears to
    kill the memorisation basin and hence grokking. With the flat
    encoding the model sees the whole group G as a single |G|-element
    vocabulary and has to *discover* the factorisation.

    At d=1 (|G| = m_1) this is bit-identical to _make_dataset_add modulo
    RNG order. Useful for sanity-checking the new code path."""
    if moduli is None:
        moduli = MODULI
    rng    = np.random.default_rng(seed)
    elems  = np.array(list(iproduct(*(range(m) for m in moduli))))   # (|G|, d)
    G      = len(elems)
    pair_i = np.array(list(iproduct(range(G), range(G))))            # (|G|^2, 2)
    a      = elems[pair_i[:, 0]]                                     # (|G|^2, d)
    b      = elems[pair_i[:, 1]]
    labels = (a + b) % np.array(moduli)                              # (|G|^2, d)

    idx = rng.permutation(len(pair_i))
    n   = int(frac * len(pair_i))

    def enc(ix):
        a_oh = F.one_hot(torch.tensor(pair_i[ix, 0]), G).float()     # (N, |G|)
        b_oh = F.one_hot(torch.tensor(pair_i[ix, 1]), G).float()
        X    = torch.cat([a_oh, b_oh], dim=1)                        # (N, 2|G|)
        y    = torch.tensor(labels[ix], dtype=torch.long)
        return X, y

    return enc(idx[:n]), enc(idx[n:])


def _make_dataset_mul(moduli=None, frac=TRAIN_FRAC, seed=0):
    """Modular multiplication on (Z/p)*: (a, b) -> a*b mod p with a, b in
    {1, ..., p-1}. Inputs are one-hot pairs of length 2p (with the zero
    index always silent). Single-prime (d=1) only."""
    if moduli is None:
        moduli = MODULI
    if len(moduli) != 1:
        raise ValueError(f"TASK=mul requires len(MODULI)=1, got MODULI={moduli}")
    p = moduli[0]
    rng    = np.random.default_rng(seed)
    pair_i = np.array(list(iproduct(range(1, p), range(1, p))))      # ((p-1)^2, 2)
    a      = pair_i[:, 0]
    b      = pair_i[:, 1]
    labels = (a * b) % p                                              # ((p-1)^2,)

    idx = rng.permutation(len(pair_i))
    n   = int(frac * len(pair_i))

    def enc(ix):
        a_oh = F.one_hot(torch.tensor(a[ix]), p).float()
        b_oh = F.one_hot(torch.tensor(b[ix]), p).float()
        X    = torch.cat([a_oh, b_oh], dim=1)
        y    = torch.tensor(labels[ix], dtype=torch.long).unsqueeze(1)
        return X, y

    return enc(idx[:n]), enc(idx[n:])


def _make_dataset_sparse_parity(seed=0):
    """Sparse parity: random {0,1} vectors of length SP_N_BITS; the label
    is the XOR of SP_K fixed (secret) bits. The secret subset is fixed
    across seeds (so all 50 replicas solve the same task and the
    statistics are over init/data shuffle only)."""
    # Fixed secret subset — same task for every replica.
    secret = np.random.default_rng(0xC051).choice(SP_N_BITS, size=SP_K, replace=False)
    rng    = np.random.default_rng(seed)
    n_test = max(SP_TRAIN, 4096)
    bits   = rng.integers(0, 2, size=(SP_TRAIN + n_test, SP_N_BITS), dtype=np.int64)
    parity = bits[:, secret].sum(axis=1) % 2

    X = torch.tensor(bits, dtype=torch.float32)
    y = torch.tensor(parity, dtype=torch.long).unsqueeze(1)
    return (X[:SP_TRAIN], y[:SP_TRAIN]), (X[SP_TRAIN:], y[SP_TRAIN:])


class GrokMLP(nn.Module):
    """Three-layer MLP with multi-head output. For add/mul, each cyclic
    factor gets its own softmax head (size m_i); for sparse parity, a
    single binary head. The caller can pass explicit in_dim/out_dim to
    override the moduli-derived defaults (required when TASK!='add'/'mul')."""

    def __init__(self, moduli=None, hidden=HIDDEN, in_dim=None, out_dim=None):
        super().__init__()
        if moduli is None:
            moduli = EFFECTIVE_MODULI
        self.moduli = list(moduli)
        if in_dim is None:
            in_dim = IN_DIM
        if out_dim is None:
            out_dim = OUT_DIM
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def weight_matrices(self):
        # .numpy() shares memory with the (CPU) tensor; on CPU, .cpu() is a no-op,
        # so without an explicit copy the optimizer's in-place updates would
        # overwrite every "checkpoint" we ever stored. Force an independent copy.
        return {"W1": self.fc1.weight.detach().cpu().numpy().copy(),
                "W2": self.fc2.weight.detach().cpu().numpy().copy(),
                "W3": self.fc3.weight.detach().cpu().numpy().copy()}

    def weight_vector(self):
        return np.concatenate([p.detach().cpu().numpy().flatten()
                               for p in self.parameters()])


def _split_starts(moduli):
    return np.cumsum([0] + list(moduli))


def multi_loss(logits, y, moduli):
    """Sum of per-factor cross-entropies. logits: (N, sum(moduli)).  y: (N, d)."""
    s = _split_starts(moduli)
    return sum(F.cross_entropy(logits[:, s[i]:s[i+1]], y[:, i]) for i in range(len(moduli)))


def multi_acc(logits, y, moduli):
    """Fraction of examples where ALL d factors are predicted correctly."""
    s = _split_starts(moduli)
    preds = torch.stack(
        [logits[:, s[i]:s[i+1]].argmax(1) for i in range(len(moduli))], dim=1)
    return (preds == y).all(dim=1).float().mean().item()


# ── §2. TDA utilities ─────────────────────────────────────────────────────
def compute_ph(W, max_dim=MAX_DIM):
    """Vietoris–Rips PH on rows of W as a point cloud in R^{ncols}."""
    Dmat = cdist(W, W, metric="euclidean")
    return ripser.ripser(Dmat, maxdim=max_dim, distance_matrix=True)["dgms"]


def persistence_entropy(dgm):
    lt = dgm[:, 1] - dgm[:, 0]
    lt = lt[np.isfinite(lt) & (lt > 0)]
    return float(scipy_entropy(lt / lt.sum())) if len(lt) else 0.0


def total_persistence(dgm, power=1):
    lt = dgm[:, 1] - dgm[:, 0]
    lt = lt[np.isfinite(lt) & (lt > 0)]
    return float(np.sum(lt ** power)) if len(lt) else 0.0


def max_persistence(dgm):
    lt = dgm[:, 1] - dgm[:, 0]
    lt = lt[np.isfinite(lt) & (lt > 0)]
    return float(lt.max()) if len(lt) else 0.0


def betti_at(dgm, threshold):
    born  = dgm[:, 0] <= threshold
    alive = (dgm[:, 1] > threshold) | (~np.isfinite(dgm[:, 1]))
    return int(np.sum(born & alive))


def summarise(dgms):
    out = {}
    for k, dgm in enumerate(dgms):
        out[f"H{k}_entropy"] = persistence_entropy(dgm)
        out[f"H{k}_total"]   = total_persistence(dgm)
        out[f"H{k}_energy"]  = total_persistence(dgm, power=2)
        out[f"H{k}_max"]     = max_persistence(dgm)
        out[f"H{k}_n"]       = len(dgm)
    return out


def plot_barcode(dgm, ax, colour="#1d4ed8", max_bars=40, label=""):
    fin = dgm[np.isfinite(dgm[:, 1])]
    inf = dgm[~np.isfinite(dgm[:, 1])]
    if len(fin):
        order = np.argsort(-(fin[:, 1] - fin[:, 0]))[:max_bars]
        fin   = fin[order]
    cap = fin[:, 1].max() * 1.12 if len(fin) else 1.0
    for i, (b, d) in enumerate(fin):
        ax.plot([b, d], [i, i], lw=2, color=colour, alpha=0.7, solid_capstyle="butt")
    for i, (b, _) in enumerate(inf[:8]):
        ax.plot([b, cap], [len(fin) + i, len(fin) + i],
                lw=2, color=C["accent"], alpha=0.9, solid_capstyle="butt")
        ax.text(cap * 1.01, len(fin) + i, "∞", fontsize=7, va="center", color=C["accent"])
    ax.set_yticks([])
    ax.set_xlabel("Filtration ε", fontsize=9)
    if label:
        ax.set_title(label, fontsize=9)


# ── §3. Training ──────────────────────────────────────────────────────────
def train_run(moduli, hidden, n_epochs, lr, weight_decay, ckpt_epochs, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (Xtr, ytr), (Xte, yte) = make_dataset(moduli=moduli, seed=seed)
    Xtr, ytr = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xte, yte = Xte.to(DEVICE), yte.to(DEVICE)

    # in/out dims are TASK-derived module globals; the `moduli` arg is the
    # head-split (= EFFECTIVE_MODULI), used for the multi_loss / multi_acc
    # call sites below. Pass it through to GrokMLP for label-keeping only.
    model = GrokMLP(moduli=EFFECTIVE_MODULI, hidden=hidden,
                    in_dim=IN_DIM, out_dim=OUT_DIM).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ckpt_set = set(ckpt_epochs)
    history  = defaultdict(list)
    ckpts    = {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        multi_loss(model(Xtr), ytr, EFFECTIVE_MODULI).backward()
        opt.step()

        if epoch % 200 == 0 or epoch in ckpt_set:
            model.eval()
            with torch.no_grad():
                tr_logits = model(Xtr)
                te_logits = model(Xte)
                tr_loss = multi_loss(tr_logits, ytr, EFFECTIVE_MODULI).item()
                te_loss = multi_loss(te_logits, yte, EFFECTIVE_MODULI).item()
                tr_acc  = multi_acc(tr_logits, ytr, EFFECTIVE_MODULI)
                te_acc  = multi_acc(te_logits, yte, EFFECTIVE_MODULI)
            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["test_loss"].append(te_loss)
            history["train_acc"].append(tr_acc)
            history["test_acc"].append(te_acc)

        if epoch in ckpt_set:
            ckpts[epoch] = {
                "weights": model.weight_matrices(),
                "wvec":    model.weight_vector(),
            }

    return dict(history), ckpts


def train_all():
    runs = {"grokked": [], "memorised": []}
    t0   = time.time()
    n_workers = _pick_workers()
    use_pool  = n_workers > 1
    for cond, wd in [("grokked", WD_GROK), ("memorised", WD_MEM)]:
        tag = f"pool={n_workers}" if use_pool else "serial"
        print(f"\n── {cond.upper()}  (weight_decay={wd})  [{tag}] ──")
        tasks = [(wd, i) for i in range(N_RUNS)]
        collected = []
        if use_pool:
            with mp.Pool(n_workers, initializer=_worker_init) as pool:
                done = 0
                for seed, hist, ckpts, dt in pool.imap_unordered(_train_run_task, tasks):
                    done += 1
                    print(f"  [{done:2d}/{N_RUNS}] seed={seed:2d}  "
                          f"train_acc={hist['train_acc'][-1]:.3f}  "
                          f"test_acc={hist['test_acc'][-1]:.3f}  "
                          f"[{dt:.0f}s]")
                    collected.append((seed, hist, ckpts))
        else:
            for i in range(N_RUNS):
                t1 = time.time()
                hist, ckpts = train_run(
                    moduli=MODULI, hidden=HIDDEN, n_epochs=N_EPOCHS,
                    lr=LR, weight_decay=wd,
                    ckpt_epochs=CKPT_EPOCHS, seed=i,
                )
                collected.append((i, hist, ckpts))
                print(f'  run {i+1:2d}/{N_RUNS}  '
                      f'train_acc={hist["train_acc"][-1]:.3f}  '
                      f'test_acc={hist["test_acc"][-1]:.3f}  '
                      f'[{time.time()-t1:.0f}s]')
        collected.sort(key=lambda r: r[0])
        for _, hist, ckpts in collected:
            runs[cond].append({"history": hist, "checkpoints": ckpts})
    print(f"\nTotal training time: {(time.time()-t0)/60:.1f} min")
    return runs


# ── §4. Learning curves ───────────────────────────────────────────────────
def plot_learning_curves(runs, savename="fig_learning_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (metric, ylabel) in zip(axes, [
        ("test_acc",  "Test accuracy"),
        ("train_acc", "Train accuracy"),
        ("test_loss", "Test loss"),
    ]):
        for cond in ("grokked", "memorised"):
            for j, run in enumerate(runs[cond]):
                h = run["history"]
                ax.plot(h["epoch"], h[metric],
                        color=C[cond], alpha=0.45, lw=1.3,
                        label=cond if j == 0 else None)
        ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if "acc" in metric:
            ax.set_ylim(-0.05, 1.05)
        ax.legend()
    fig.suptitle(f"Learning Curves  ({_task_label()}, WD_grok={WD_GROK})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §5. Persistent homology over checkpoints ──────────────────────────────
TDA_LAYER = "W2"


def compute_tda(runs, layer=TDA_LAYER, max_dim=MAX_DIM):
    n_workers = _pick_workers()
    use_pool  = n_workers > 1
    tag = f"pool={n_workers}" if use_pool else "serial"
    print(f"Computing persistence diagrams for {layer}  (max_dim={max_dim})  [{tag}] ...")
    t0 = time.time()

    tasks = []
    for cond in ("grokked", "memorised"):
        for ri, run in enumerate(runs[cond]):
            for epoch in sorted(run["checkpoints"]):
                W = run["checkpoints"][epoch]["weights"][layer]
                tasks.append((cond, ri, epoch, W, max_dim))

    if use_pool:
        with mp.Pool(n_workers, initializer=_worker_init) as pool:
            results = list(pool.imap_unordered(_compute_tda_task, tasks))
    else:
        results = [_compute_tda_task(t) for t in tasks]

    tda = {cond: [[] for _ in runs[cond]] for cond in ("grokked", "memorised")}
    for cond, ri, _, rec in results:
        tda[cond][ri].append(rec)
    for cond in tda:
        for ri in range(len(tda[cond])):
            tda[cond][ri].sort(key=lambda r: r["epoch"])
    print(f"Done in {time.time()-t0:.1f}s")
    return tda


# ── §6. Topological dynamics during training ──────────────────────────────
def _dynamic_metrics(max_dim):
    """Entropy for every dimension, total persistence for k >= 1."""
    specs  = [(f"H{k}_entropy", f"H_{k} Entropy") for k in range(max_dim + 1)]
    specs += [(f"H{k}_total",   f"H_{k} Total Persistence") for k in range(1, max_dim + 1)]
    return specs


def plot_tda_dynamics(tda, runs, layer=TDA_LAYER, savename="fig_tda_dynamics.png"):
    metric_specs = _dynamic_metrics(MAX_DIM)
    nc = len(metric_specs)
    fig, axes = plt.subplots(2, nc, figsize=(3.6 * nc, 8), squeeze=False)
    for row, cond in enumerate(("grokked", "memorised")):
        for col, (metric, label) in enumerate(metric_specs):
            ax  = axes[row, col]
            mat = []
            for run_tda in tda[cond]:
                eps  = [s["epoch"] for s in run_tda]
                vals = [s[metric]  for s in run_tda]
                ax.plot(eps, vals, color=C[cond], alpha=0.35, lw=1.1)
                mat.append(vals)
            ax.plot(eps, np.array(mat).mean(0), color=C[cond], lw=2.5, label="mean")
            if cond == "grokked":
                for run in runs["grokked"]:
                    for ep, acc in zip(run["history"]["epoch"], run["history"]["test_acc"]):
                        if acc >= 0.95:
                            ax.axvline(ep, color="#f59e0b", lw=0.8, alpha=0.5, ls="--")
                            break
            ax.set_xscale("log")
            ax.set_xlabel("Epoch")
            ax.set_title(f"{cond.capitalize()} — {label}")
            if col == 0:
                ax.set_ylabel(label)
            ax.legend(fontsize=8)
    fig.suptitle(f"Topological Complexity vs Epoch  [{layer}]   "
                 f"{_task_label()}  (dashed gold = grokking epoch per run)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §7. Betti trajectories ────────────────────────────────────────────────
def plot_betti_trajectories(tda, runs, layer=TDA_LAYER, savename="fig_betti_trajectories.png"):
    W_ref  = runs["grokked"][0]["checkpoints"][max(runs["grokked"][0]["checkpoints"])]["weights"][layer]
    D_ref  = cdist(W_ref, W_ref, "euclidean")
    THRESH = float(np.median(D_ref[D_ref > 0]))
    print(f"Betti threshold: {THRESH:.4f}")

    dims = list(range(MAX_DIM + 1))
    fig, axes = plt.subplots(1, len(dims), figsize=(5.5 * len(dims), 4.5), squeeze=False)
    axes = axes[0]
    for ax, dim in zip(axes, dims):
        label = f"β_{dim}"
        for cond in ("grokked", "memorised"):
            all_bettis = []
            for run_tda in tda[cond]:
                eps   = [s["epoch"] for s in run_tda]
                betts = [betti_at(s["dgms"][dim], THRESH) for s in run_tda]
                ax.plot(eps, betts, color=C[cond], alpha=0.35, lw=1.0)
                all_bettis.append(betts)
            ax.plot(eps, np.array(all_bettis).mean(0),
                    color=C[cond], lw=2.5, label=cond)
        # Künneth prediction for T^d
        pred = comb(MAX_DIM, dim) if MAX_DIM >= 1 else None
        if pred is not None and MAX_DIM >= 2 and dim >= 1:
            ax.axhline(pred, color="black", lw=0.8, ls=":", alpha=0.5,
                       label=f"T^{MAX_DIM} pred = {pred}")
        ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
    fig.suptitle(f"Betti Numbers at ε={THRESH:.3f}  [{layer}]   {_task_label()}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §8. Grokking transition snapshots (H_1 focused) ───────────────────────
def grok_epoch(history, threshold=0.95):
    for ep, acc in zip(history["epoch"], history["test_acc"]):
        if acc >= threshold:
            return ep
    return None


def plot_grokking_transition(tda, runs, dim=1, savename="fig_grokking_transition.png"):
    if MAX_DIM < dim:
        print(f"Skipping transition snapshots: MAX_DIM={MAX_DIM} < requested dim={dim}")
        return
    grok_ri = next((i for i, r in enumerate(runs["grokked"])
                    if grok_epoch(r["history"]) is not None), None)
    if grok_ri is None:
        print("No run reached test_acc ≥ 95% — try increasing N_EPOCHS.")
        return

    grok_ep = grok_epoch(runs["grokked"][grok_ri]["history"])
    run_tda = tda["grokked"][grok_ri]
    all_eps = [s["epoch"] for s in run_tda]
    after_i  = next((i for i, e in enumerate(all_eps) if e >= grok_ep), len(all_eps) - 1)
    snap_ids = list(range(max(0, after_i - 3), min(len(all_eps), after_i + 4)))
    snap_eps = [all_eps[i] for i in snap_ids]
    print(f"Grokking at epoch ≈ {grok_ep}  |  snapshots: {snap_eps}")

    n = len(snap_ids)
    fig, axes = plt.subplots(3, n, figsize=(2.8 * n, 8))
    run_hist  = runs["grokked"][grok_ri]["history"]
    all_tots  = [s[f"H{dim}_total"]   for s in run_tda]
    all_ents  = [s[f"H{dim}_entropy"] for s in run_tda]

    for col, si in enumerate(snap_ids):
        snap  = run_tda[si]
        ep    = snap["epoch"]
        after = ep >= grok_ep
        clr   = C["grokked"] if after else C["memorised"]
        tag   = " ←GROK" if ep == all_eps[after_i] else ""

        ax0 = axes[0, col]
        fin = snap["dgms"][dim][np.isfinite(snap["dgms"][dim][:, 1])]
        if len(fin):
            lim = fin.max() * 1.12
            ax0.scatter(fin[:, 0], fin[:, 1], s=18, c=clr, alpha=0.7, edgecolors="none")
            ax0.plot([0, lim], [0, lim], "k--", lw=0.7, alpha=0.35)
            ax0.set_xlim(0, lim)
            ax0.set_ylim(0, lim)
        ax0.set_title(f"ep={ep}{tag}", fontsize=8, color=clr, fontweight="bold")
        ax0.tick_params(labelsize=6)
        ax0.set_xlabel("Birth", fontsize=7)
        ax0.set_ylabel("Death", fontsize=7)

        ax1 = axes[1, col]
        plot_barcode(snap["dgms"][dim], ax1, colour=clr, max_bars=20)
        ax1.tick_params(labelsize=6)

        ax2  = axes[2, col]
        near = min(range(len(run_hist["epoch"])),
                   key=lambda k: abs(run_hist["epoch"][k] - ep))
        te_acc   = run_hist["test_acc"][near]
        h_ent_n  = snap[f"H{dim}_entropy"] / (max(all_ents) + 1e-9)
        h_tot_n  = snap[f"H{dim}_total"]   / (max(all_tots) + 1e-9)
        ax2.bar(["test\nacc", f"H_{dim}\nentropy\n(norm)", f"H_{dim} total\n(norm)"],
                [te_acc, h_ent_n, h_tot_n],
                color=[C["accent"], C["h1"], C["h1"]], alpha=0.8)
        ax2.set_ylim(0, 1.15)
        ax2.tick_params(labelsize=6)

    axes[0, 0].set_ylabel(f"H_{dim} Diagram", fontsize=8)
    axes[1, 0].set_ylabel(f"H_{dim} Barcode", fontsize=8)
    axes[2, 0].set_ylabel("Metrics (norm)", fontsize=8)
    fig.suptitle(f"Topological Snapshots Around Grokking Transition (run {grok_ri+1})  "
                 f"H_{dim}   {_task_label()}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §9. Persistence diagrams at convergence (all dims) ────────────────────
def plot_persistence_diagrams(tda, runs, layer=TDA_LAYER, savename="fig_persistence_diagrams.png"):
    # Cap columns so the figure stays viewable; subsample evenly across runs.
    max_cols = 8
    cols     = (list(range(N_RUNS)) if N_RUNS <= max_cols
                else np.linspace(0, N_RUNS - 1, max_cols).round().astype(int).tolist())
    n_show   = len(cols)
    dims     = list(range(MAX_DIM + 1))
    nrows    = 2 * len(dims)             # (grokked × dims) then (memorised × dims)
    fig, axes = plt.subplots(nrows, n_show, figsize=(2.8 * n_show, 2.6 * nrows),
                             squeeze=False)
    row_specs = []
    for cond in ("grokked", "memorised"):
        for d in dims:
            row_specs.append((cond, d, f"H_{d}"))

    for col, run_idx in enumerate(cols):
        for row, (cond, dim, dimlab) in enumerate(row_specs):
            ax   = axes[row, col]
            snap = tda[cond][run_idx][-1]
            dgm  = snap["dgms"][dim]
            fin  = dgm[np.isfinite(dgm[:, 1])]
            ent  = snap[f"H{dim}_entropy"]
            te   = runs[cond][run_idx]["history"]["test_acc"][-1]
            if len(fin):
                lim = fin.max() * 1.1
                ax.scatter(fin[:, 0], fin[:, 1], s=12, c=C[cond],
                           alpha=0.65, edgecolors="none")
                ax.plot([0, lim], [0, lim], "k--", lw=0.6, alpha=0.35)
                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
            ax.set_title(f"{dimlab}  S={ent:.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"{cond}\n{dimlab}", fontsize=8, color=C[cond])
            if row == nrows - 1:
                ax.set_xlabel(f"run {run_idx+1}  te={te:.2f}", fontsize=7)
    fig.suptitle(f"Persistence Diagrams at Convergence  [{layer}]   {_task_label()}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §10. Wasserstein distances & MDS (H_1 by default) ─────────────────────
def plot_wasserstein(tda, dim=1, savename="fig_wasserstein.png"):
    if MAX_DIM < dim:
        print(f"Skipping Wasserstein: MAX_DIM={MAX_DIM} < requested dim={dim}")
        return None, None, None
    final_dgms, final_labels = [], []
    for cond in ("grokked", "memorised"):
        for run_tda in tda[cond]:
            final_dgms.append(run_tda[-1]["dgms"][dim])
            final_labels.append(cond)

    N_tot = len(final_dgms)
    W_mat = np.zeros((N_tot, N_tot))
    print(f"Computing {N_tot}×{N_tot} Wasserstein matrix on H_{dim}...")
    t0 = time.time()
    for i in range(N_tot):
        for j in range(i + 1, N_tot):
            d = wasserstein(final_dgms[i], final_dgms[j])
            W_mat[i, j] = W_mat[j, i] = d
    print(f"Done in {time.time()-t0:.1f}s")

    n_g  = sum(1 for l in final_labels if l == "grokked")
    tklb = [f"G{i+1}" for i in range(n_g)] + [f"M{i+1}" for i in range(N_tot - n_g)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax  = axes[0]
    im  = ax.imshow(W_mat, cmap="magma_r")
    plt.colorbar(im, ax=ax, label=f"Wasserstein-1 (H_{dim})", shrink=0.85)
    ax.set_xticks(range(N_tot))
    ax.set_yticks(range(N_tot))
    ax.set_xticklabels(tklb, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tklb, fontsize=9)
    ax.axvline(n_g - 0.5, color="white", lw=1.5, ls="--", alpha=0.8)
    ax.axhline(n_g - 0.5, color="white", lw=1.5, ls="--", alpha=0.8)
    ax.set_title("Pairwise Wasserstein Distances")

    ax2 = axes[1]
    emb = MDS(n_components=2, dissimilarity="precomputed", random_state=0).fit_transform(W_mat)
    for cond in ("grokked", "memorised"):
        idx = [i for i, l in enumerate(final_labels) if l == cond]
        ax2.scatter(emb[idx, 0], emb[idx, 1],
                    c=C[cond], s=100, label=cond, edgecolors="white", lw=0.5, zorder=3)
        for i in idx:
            ax2.annotate(tklb[i], emb[i], fontsize=8, ha="center", va="bottom",
                         xytext=(0, 6), textcoords="offset points", color=C[cond])
    ax2.set_title("MDS Embedding of Wasserstein Distances")
    ax2.set_xlabel("MDS 1")
    ax2.set_ylabel("MDS 2")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)

    G  = np.array([i for i, l in enumerate(final_labels) if l == "grokked"])
    M  = np.array([i for i, l in enumerate(final_labels) if l == "memorised"])
    gg = W_mat[np.ix_(G, G)]; gg = gg[gg > 0].mean()
    mm = W_mat[np.ix_(M, M)]; mm = mm[mm > 0].mean()
    gm = W_mat[np.ix_(G, M)].mean()
    print(f"Mean intra-Grokked  distance: {gg:.4f}")
    print(f"Mean intra-Memorised distance: {mm:.4f}")
    print(f"Mean cross-condition distance: {gm:.4f}")
    print(f"Cross/intra ratio: {gm/max(gg,mm):.2f}  (>1 means separated)")
    return W_mat, final_labels, tklb


# ── §11. Hierarchical clustering ──────────────────────────────────────────
def plot_dendrogram(W_mat, final_labels, tklb, savename="fig_dendrogram.png"):
    if W_mat is None:
        return
    N_tot = len(final_labels)
    upper = W_mat[np.triu_indices(N_tot, k=1)]
    Z     = linkage(upper, method="ward")
    fig, ax = plt.subplots(figsize=(min(max(9, N_tot * 0.7), 24), 5))
    dend = dendrogram(Z, labels=tklb, ax=ax, leaf_font_size=10, color_threshold=0)
    for lbl, leaf in zip(ax.get_xticklabels(), dend["leaves"]):
        lbl.set_color(C[final_labels[leaf]])
    patches = [mpatches.Patch(color=C[c], label=c) for c in ("grokked", "memorised")]
    ax.legend(handles=patches, loc="upper right")
    ax.set_title("Hierarchical Clustering by H_1 Wasserstein Distance",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Ward linkage distance")
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §12. Layer-wise topology profile ──────────────────────────────────────
def plot_layerwise(runs, savename="fig_layerwise.png"):
    print("Computing layer-wise persistence...")
    layer_data = {}
    for cond in ("grokked", "memorised"):
        for layer in ("W1", "W2", "W3"):
            ents_h0, ents_h1, tots_h1 = [], [], []
            for run in runs[cond]:
                last = max(run["checkpoints"])
                W    = run["checkpoints"][last]["weights"][layer]
                dgms = compute_ph(W, max_dim=max(1, MAX_DIM))
                ents_h0.append(persistence_entropy(dgms[0]))
                ents_h1.append(persistence_entropy(dgms[1]))
                tots_h1.append(total_persistence(dgms[1]))
            layer_data[(cond, layer, "H0_entropy")] = ents_h0
            layer_data[(cond, layer, "H1_entropy")] = ents_h1
            layer_data[(cond, layer, "H1_total")]   = tots_h1
            print(f"  {cond} {layer}  H1_ent={np.mean(ents_h1):.3f}±{np.std(ents_h1):.3f}")

    layers = ["W1", "W2", "W3"]
    xlabs  = ["W_1\n(in→h)", "W_2\n(h→h)", "W_3\n(h→out)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    x, width  = np.arange(3), 0.35
    for ax, (metric, mlabel) in zip(axes, [
        ("H0_entropy", "H_0 Persistence Entropy"),
        ("H1_entropy", "H_1 Persistence Entropy"),
        ("H1_total",   "H_1 Total Persistence"),
    ]):
        for cond, offset in zip(("grokked", "memorised"), [-width / 2, width / 2]):
            means = [np.mean(layer_data[(cond, l, metric)]) for l in layers]
            stds  = [np.std( layer_data[(cond, l, metric)]) for l in layers]
            ax.bar(x + offset, means, width, yerr=stds, color=C[cond], alpha=0.75,
                   label=cond, capsize=4, error_kw={"lw": 1.5})
            rng = np.random.default_rng(42)
            for li, l in enumerate(layers):
                v = layer_data[(cond, l, metric)]
                ax.scatter(x[li] + offset + rng.uniform(-0.05, 0.05, len(v)), v,
                           c=C[cond], s=18, alpha=0.55, zorder=4, edgecolors="none")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabs)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.legend(fontsize=9)
    fig.suptitle("Layer-wise Topology at Convergence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §13. Statistical comparison ───────────────────────────────────────────
def statistical_comparison(tda, layer=TDA_LAYER):
    print("=" * 76)
    print(f"Statistical comparison: Grokked vs Memorised  [{layer} at convergence]  "
          f"{_task_label()}")
    print("=" * 76)
    print(f'{"Metric":<22}  {"Grokked (μ±σ)":>18}  {"Memorised (μ±σ)":>18}  {"p":>8}  Sig')
    print("-" * 76)
    metrics = []
    for k in range(MAX_DIM + 1):
        metrics += [f"H{k}_entropy", f"H{k}_total", f"H{k}_energy",
                    f"H{k}_max",     f"H{k}_n"]
    for metric in metrics:
        g = [tda["grokked"][ri][-1][metric]   for ri in range(N_RUNS)]
        m = [tda["memorised"][ri][-1][metric] for ri in range(N_RUNS)]
        gm, gs = np.mean(g), np.std(g)
        mm, ms = np.mean(m), np.std(m)
        p = mannwhitneyu(g, m, alternative="two-sided")[1] if len(set(g + m)) > 1 else 1.0
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else
               "*"   if p < 0.05  else "."  if p < 0.1  else "ns")
        print(f"{metric:<22}  {gm:>7.3f}±{gs:<8.3f}  {mm:>7.3f}±{ms:<8.3f}  {p:>8.4f}  {sig}")
    print("=" * 76)
    print("Codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1  ns")


# ── §14. PCA of weight trajectories ───────────────────────────────────────
def plot_pca_trajectory(runs, savename="fig_pca_trajectory.png"):
    all_vecs, meta = [], []
    for cond in ("grokked", "memorised"):
        for ri, run in enumerate(runs[cond]):
            for ep in sorted(run["checkpoints"]):
                all_vecs.append(run["checkpoints"][ep]["weights"]["W2"].flatten())
                meta.append({"cond": cond, "run": ri, "epoch": ep})
    all_vecs = np.array(all_vecs)
    pca2  = PCA(n_components=2)
    emb2d = pca2.fit_transform(all_vecs)
    var   = pca2.explained_variance_ratio_
    print(f"PC1: {100*var[0]:.1f}%   PC2: {100*var[1]:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, color_by in zip(axes, ["condition", "epoch"]):
        if color_by == "condition":
            for cond in ("grokked", "memorised"):
                idx = [i for i, m in enumerate(meta) if m["cond"] == cond]
                ax.scatter(emb2d[idx, 0], emb2d[idx, 1],
                           c=C[cond], s=10, alpha=0.4, label=cond, edgecolors="none")
                for ri in range(min(3, N_RUNS)):
                    r_idx = sorted([i for i in idx if meta[i]["run"] == ri],
                                   key=lambda i: meta[i]["epoch"])
                    ax.plot(emb2d[r_idx, 0], emb2d[r_idx, 1],
                            color=C[cond], lw=0.9, alpha=0.55)
            ax.legend()
            ax.set_title("Coloured by condition")
        else:
            sc = ax.scatter(emb2d[:, 0], emb2d[:, 1],
                            c=np.log1p([m["epoch"] for m in meta]),
                            cmap="plasma", s=10, alpha=0.5, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="log(1+epoch)", shrink=0.85)
            ax.set_title("Coloured by epoch")
        ax.set_xlabel(f"PC1 ({100*var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100*var[1]:.1f}%)")
    fig.suptitle("PCA of W_2 Weight Vectors During Training",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §15. TDA metrics vs test accuracy ─────────────────────────────────────
def plot_tda_vs_acc(tda, runs, savename="fig_tda_vs_acc.png"):
    ent_vals, tot_vals, acc_vals, cond_vals = [], [], [], []
    for cond in ("grokked", "memorised"):
        for ri in range(N_RUNS):
            ent_vals.append(tda[cond][ri][-1]["H1_entropy"])
            tot_vals.append(tda[cond][ri][-1]["H1_total"])
            acc_vals.append(runs[cond][ri]["history"]["test_acc"][-1])
            cond_vals.append(cond)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (xvals, xlabel) in zip(axes, [
        (ent_vals, "H_1 Persistence Entropy"),
        (tot_vals, "H_1 Total Persistence"),
    ]):
        for cond in ("grokked", "memorised"):
            idx = [i for i, c in enumerate(cond_vals) if c == cond]
            ax.scatter([xvals[i] for i in idx], [acc_vals[i] for i in idx],
                       c=C[cond], s=80, label=cond, edgecolors="white", lw=0.5, zorder=3)
        rho, p = spearmanr(xvals, acc_vals)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{xlabel}\nvs Test Accuracy  ρ={rho:.3f}  p={p:.3f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── §16. Multi-layer barcodes ─────────────────────────────────────────────
def plot_barcodes(runs, savename="fig_barcodes.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    layers = ["W1", "W2", "W3"]
    xlabs  = ["W_1  (in→h)", "W_2  (h→h)", "W_3  (h→out)"]
    for row, cond in enumerate(("grokked", "memorised")):
        run     = runs[cond][0]
        last_ep = max(run["checkpoints"])
        te_acc  = run["history"]["test_acc"][-1]
        for col, (layer, xlab) in enumerate(zip(layers, xlabs)):
            ax  = axes[row, col]
            W   = run["checkpoints"][last_ep]["weights"][layer]
            dgm = compute_ph(W, max_dim=max(1, MAX_DIM))[1]
            ent = persistence_entropy(dgm)
            plot_barcode(dgm, ax, colour=C[cond], max_bars=40,
                         label=f"{cond}  {xlab}\nte={te_acc:.2f}  S={ent:.2f}")
            ax.tick_params(labelsize=7)
    fig.suptitle("H_1 Persistence Barcodes at Convergence  (run 1 each condition)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath(savename))
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────
class _StdoutTee:
    """Mirror writes to multiple streams so we can capture the run log to
    a file while still showing it on the terminal. Restored on context exit."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()


def main():
    # spawn is the macOS default since 3.8; set explicitly so behaviour
    # is identical on Linux (where fork+torch can deadlock libomp).
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    # Save stdout to FIG_DIR/run.log so each run is self-documenting next
    # to its figures. Workers print via the parent's pool callback, so the
    # parent-only tee captures all the relevant output. Restored on exit.
    import sys
    log_path = FIG_DIR / "run.log"
    log_file = open(log_path, "w", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = _StdoutTee(original_stdout, log_file)
    try:
        _run_main()
    finally:
        sys.stdout = original_stdout
        log_file.close()


def _run_main():
    print(f"Device : {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ripser : {ripser.__version__}")
    print(f"Task   : {TASK}   ({_task_label()})")
    if TASK in ("add", "mul"):
        print(f"MODULI : {MODULI}   (d={D}, |G|={int(np.prod(MODULI))})")
    print(f"In/Out : {IN_DIM} -> {HIDDEN} -> {HIDDEN} -> {OUT_DIM}")
    print(f"MAX_DIM (homology dims computed): {MAX_DIM}")

    (Xtr, ytr), (Xte, yte) = make_dataset()
    print(f"train={len(Xtr)}  test={len(Xte)}  input_dim={Xtr.shape[1]}  labels={tuple(ytr.shape)}")
    print(f"Will train {N_RUNS}×2 = {2*N_RUNS} networks, {N_EPOCHS:,} epochs each.")
    print(f"Checkpoints per run: {len(CKPT_EPOCHS)}")
    print(f"Figures will be written to: {FIG_DIR}")

    runs = train_all()
    plot_learning_curves(runs)

    tda = compute_tda(runs)
    plot_tda_dynamics(tda, runs)
    plot_betti_trajectories(tda, runs)
    plot_grokking_transition(tda, runs)
    plot_persistence_diagrams(tda, runs)

    W_mat, final_labels, tklb = plot_wasserstein(tda)
    plot_dendrogram(W_mat, final_labels, tklb)
    plot_layerwise(runs)
    statistical_comparison(tda)
    plot_pca_trajectory(runs)
    plot_tda_vs_acc(tda, runs)
    plot_barcodes(runs)


if __name__ == "__main__":
    main()
