from __future__ import annotations

import os
import pickle
import random
import re
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

MAX_FILENAME_COMPONENT_LENGTH = 240


@dataclass
class ExperimentConfig:
    name: str
    out_dir: Path
    load_data: bool = False
    save_data: bool = True

    runs: int = 5
    seed: int = 0
    same_condition_seeds: bool = False
    device: str = "auto"
    num_threads: int = 1

    grok_train_frac: float = 0.5
    memorise_train_frac: float = 0.2
    grok_steps: int = 20_000
    memorise_steps: int = 20_000
    grok_weight_decay: float = 0.0
    memorise_weight_decay: float = 0.0

    eval_every: int = 1000
    log_every: int = 100
    eval_batch: int = 8192
    n_checkpoints: int = 25
    include_initial_checkpoint: bool = True
    checkpoint_style: str = "log"
    log_checkpoint_min_step: int = 1
    stop_early: bool = False
    grok_train_threshold: float = 0.99
    grok_test_threshold: float = 0.99
    memorise_train_threshold: float = 0.99
    memorise_test_max: float = 0.30

    quick: bool = False

    @property
    def data_path(self) -> Path:
        return self.out_dir / "training_data.pkl"


@dataclass
class ExperimentSpec:
    make_dataset: Callable
    make_model: Callable
    make_optimizer: Callable
    loss_fn: Callable
    metadata: Callable
    train_step: Callable | None = None
    seed_fn: Callable | None = None


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(model: nn.Module, x, y, batch_size: int) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(x), batch_size):
        pred = model(x[start:start + batch_size]).argmax(dim=-1)
        correct += (pred == y[start:start + batch_size]).sum().item()
    model.train()
    return correct / max(1, len(y))


@torch.no_grad()
def eval_loss(spec: ExperimentSpec, model: nn.Module, x, y, config: ExperimentConfig) -> float:
    model.eval()
    value = spec.loss_fn(model, x, y, config).item()
    model.train()
    return value


@torch.no_grad()
def flat_weights(model: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()]).numpy().astype(np.float32)


@torch.no_grad()
def checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def nearest_unused_step(target_log: float, raw_step: int, max_steps: int, used: set[int]) -> int:
    raw_step = int(np.clip(raw_step, 1, max_steps))
    if raw_step not in used:
        return raw_step

    for radius in range(1, max_steps + 1):
        candidates = [raw_step - radius, raw_step + radius]
        candidates = [step for step in candidates if 1 <= step <= max_steps and step not in used]
        if candidates:
            return min(candidates, key=lambda step: abs(np.log10(step) - target_log))
    raise RuntimeError("Could not find an unused checkpoint step.")


def log_checkpoint_steps(max_steps: int, n_steps: int, min_step: int = 1) -> list[int]:
    if n_steps <= 1:
        return [max_steps]
    min_step = int(np.clip(min_step, 1, max_steps))
    target_logs = np.linspace(np.log10(min_step), np.log10(max_steps), n_steps)
    used = set()
    steps = []
    for target_log in target_logs:
        raw_step = int(round(10 ** target_log))
        step = nearest_unused_step(target_log, raw_step, max_steps, used)
        used.add(step)
        steps.append(step)
    return sorted(steps)


def linear_checkpoint_steps(max_steps: int, n_steps: int) -> list[int]:
    return sorted(set(np.linspace(1, max_steps, n_steps, dtype=int).tolist()))


def hybrid_checkpoint_steps(max_steps: int, n_steps: int) -> list[int]:
    early = [1, 5, 10, 25, 50, 100, 200, 500]
    early = [step for step in early if step < max_steps]
    n_later = max(1, n_steps - len(early))
    later = np.linspace(1000, max_steps, n_later, dtype=int).tolist()
    return sorted(set(early + later))


def checkpoint_steps(max_steps: int, config: ExperimentConfig) -> set[int]:
    n_nonzero = config.n_checkpoints - (1 if config.include_initial_checkpoint else 0)
    n_nonzero = max(1, n_nonzero)

    if config.checkpoint_style == "log":
        steps = log_checkpoint_steps(max_steps, n_nonzero, min_step=config.log_checkpoint_min_step)
    elif config.checkpoint_style == "linear":
        steps = linear_checkpoint_steps(max_steps, n_nonzero)
    elif config.checkpoint_style == "hybrid":
        steps = hybrid_checkpoint_steps(max_steps, n_nonzero)
    else:
        raise ValueError('checkpoint_style must be "log", "linear", or "hybrid"')

    if config.include_initial_checkpoint:
        steps = [0, *steps]
    return set(steps)


def record_checkpoint(spec: ExperimentSpec, model: nn.Module, step: int, xtr, ytr, xte, yte, config: ExperimentConfig):
    return {
        "step": int(step),
        "train_loss": eval_loss(spec, model, xtr, ytr, config),
        "test_loss": eval_loss(spec, model, xte, yte, config),
        "train_acc": accuracy(model, xtr, ytr, config.eval_batch),
        "test_acc": accuracy(model, xte, yte, config.eval_batch),
        "state_dict": checkpoint_state_dict(model),
    }


def default_train_step(spec: ExperimentSpec, model: nn.Module, optimizer, xtr, ytr, step: int, condition: str, device, config: ExperimentConfig):
    loss = spec.loss_fn(model, xtr, ytr, config)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss


def quick_config(config: ExperimentConfig) -> ExperimentConfig:
    if not config.quick:
        return config
    return ExperimentConfig(
        **{
            **config.__dict__,
            "runs": 1,
            "grok_steps": min(config.grok_steps, 100),
            "memorise_steps": min(config.memorise_steps, 100),
        }
    )


def short_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def step_count_k(steps: int) -> str:
    value = steps / 1000
    return f"{value:g}k"


def compact_run_timestamp(config: ExperimentConfig) -> str:
    timestamp = getattr(config, "_filename_timestamp", None)
    if timestamp is None:
        timestamp = time.strftime("%y%m%d-%H%M%S")
        setattr(config, "_filename_timestamp", timestamp)
    return timestamp


def serial_from_filename(path: Path) -> int | None:
    match = re.match(r"^(\d{3})_", path.name)
    return int(match.group(1)) if match else None


def next_experiment_serial(out_dir: Path) -> int:
    serials = [
        serial
        for path in out_dir.glob("*.pkl")
        if (serial := serial_from_filename(path)) is not None
    ]
    return max(serials, default=0) + 1


def experiment_serial(config: ExperimentConfig) -> int:
    serial = getattr(config, "_filename_serial", None)
    if serial is None:
        serial = next_experiment_serial(config.out_dir)
        setattr(config, "_filename_serial", serial)
    return serial


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    return text.strip("-")


def bounded_filename(stem: str, suffix: str, max_component_length: int = MAX_FILENAME_COMPONENT_LENGTH) -> str:
    name = f"{stem}{suffix}"
    if len(name) <= max_component_length:
        return name

    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    keep = max_component_length - len(suffix) - len(digest) - 2
    if keep <= 0:
        raise ValueError("max_component_length is too small for requested suffix")
    return f"{stem[:keep]}_{digest}{suffix}"


def experiment_data_path(config: ExperimentConfig, spec: ExperimentSpec) -> Path:
    meta = spec.metadata(config)
    filename_bits = [
        f"{experiment_serial(config):03d}",
        short_value(meta["p"]),
        short_value(meta["operation"]),
        short_value(config.seed),
        step_count_k(config.grok_steps),
        step_count_k(config.memorise_steps),
        compact_run_timestamp(config),
    ]
    stem = safe_filename("_".join(filename_bits))
    return config.out_dir / bounded_filename(stem, ".pkl")


def load_experiment_data(config: ExperimentConfig, spec: ExperimentSpec):
    path = experiment_data_path(config, spec)
    if path.exists():
        return load_data(path)
    return load_data(config.data_path)


def train_run(spec: ExperimentSpec, condition: str, run_idx: int, device: torch.device, config: ExperimentConfig):
    is_grok = condition == "grok"
    if spec.seed_fn is None:
        condition_offset = 0 if (is_grok or config.same_condition_seeds) else 10_000
        seed = config.seed + condition_offset + run_idx
    else:
        seed = spec.seed_fn(condition, run_idx, config)
    set_seed(seed)
    config.current_condition = condition
    config.current_run_idx = run_idx
    config.current_seed = seed

    train_frac = config.grok_train_frac if is_grok else config.memorise_train_frac
    max_steps = config.grok_steps if is_grok else config.memorise_steps
    weight_decay = config.grok_weight_decay if is_grok else config.memorise_weight_decay

    xtr, ytr, xte, yte = spec.make_dataset(train_frac, device, config)
    model = spec.make_model(config).to(device)
    optimizer = spec.make_optimizer(model, weight_decay, condition, config)

    ckpt_steps = checkpoint_steps(max_steps, config)
    checkpoints = []
    start_time = time.time()
    last_train_acc = 0.0
    last_test_acc = 0.0
    last_loss = float("nan")

    if 0 in ckpt_steps:
        ckpt = record_checkpoint(spec, model, 0, xtr, ytr, xte, yte, config)
        checkpoints.append(ckpt)
        print(f"{condition:8s} run={run_idx} checkpoint={len(checkpoints):02d}/{len(ckpt_steps):02d} step={0:7d}", flush=True)

    for step in range(1, max_steps + 1):
        train_step_fn = spec.train_step or default_train_step
        loss = train_step_fn(spec, model, optimizer, xtr, ytr, step, condition, device, config)
        last_loss = loss.item()

        if config.log_every > 0 and step % config.log_every == 0 and step % config.eval_every != 0:
            print(f"{condition:8s} run={run_idx} step={step:7d} loss_only={loss.item():.6f}", flush=True)

        if step == 1 or step % config.eval_every == 0:
            last_train_acc = accuracy(model, xtr, ytr, config.eval_batch)
            last_test_acc = accuracy(model, xte, yte, config.eval_batch)
            print(
                f"{condition:8s} run={run_idx} step={step:7d} "
                f"loss={loss.item():.6f} train_acc={last_train_acc:.4f} test_acc={last_test_acc:.4f}",
                flush=True,
            )
            if config.stop_early and is_grok and last_train_acc >= config.grok_train_threshold and last_test_acc >= config.grok_test_threshold:
                break
            if config.stop_early and not is_grok and last_train_acc >= config.memorise_train_threshold and last_test_acc <= config.memorise_test_max:
                break

        if step in ckpt_steps:
            ckpt = record_checkpoint(spec, model, step, xtr, ytr, xte, yte, config)
            checkpoints.append(ckpt)
            print(
                f"{condition:8s} run={run_idx} checkpoint={len(checkpoints):02d}/{len(ckpt_steps):02d} "
                f"step={step:7d} train_acc={ckpt['train_acc']:.4f} test_acc={ckpt['test_acc']:.4f}",
                flush=True,
            )

    if step != 1 and step % config.eval_every != 0:
        last_train_acc = accuracy(model, xtr, ytr, config.eval_batch)
        last_test_acc = accuracy(model, xte, yte, config.eval_batch)
        print(
            f"{condition:8s} run={run_idx} step={step:7d} "
            f"final_loss={last_loss:.6f} train_acc={last_train_acc:.4f} test_acc={last_test_acc:.4f}",
            flush=True,
        )

    if not checkpoints or checkpoints[-1]["step"] != step:
        ckpt = record_checkpoint(spec, model, step, xtr, ytr, xte, yte, config)
        checkpoints.append(ckpt)
        last_train_acc = ckpt["train_acc"]
        last_test_acc = ckpt["test_acc"]
        print(
            f"{condition:8s} run={run_idx} checkpoint=final step={step:7d} "
            f"train_acc={ckpt['train_acc']:.4f} test_acc={ckpt['test_acc']:.4f}",
            flush=True,
        )

    return {
        "condition": condition,
        "run": run_idx,
        "seed": seed,
        "steps": step,
        "train_acc": last_train_acc,
        "test_acc": last_test_acc,
        "seconds": time.time() - start_time,
        "checkpoints": checkpoints,
    }


def save_data(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved data to {path}")


def load_data(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    print(f"Loaded data from {path}")
    return data


def run_or_load(config: ExperimentConfig, spec: ExperimentSpec):
    config = quick_config(config)
    torch.set_num_threads(max(1, config.num_threads))
    device = choose_device(config.device)
    data_path = experiment_data_path(config, spec)

    print(f"device={device} experiment={config.name}")
    print(f"data_path={data_path}", flush=True)
    print(
        f"runs={config.runs} seed={config.seed} same_condition_seeds={config.same_condition_seeds} "
        f"gfrac={config.grok_train_frac} mfrac={config.memorise_train_frac} "
        f"gsteps={config.grok_steps} msteps={config.memorise_steps} "
        f"gwd={config.grok_weight_decay} mwd={config.memorise_weight_decay}",
        flush=True,
    )
    if device.type == "mps":
        print('Using Apple Metal MPS GPU via torch.device("mps").', flush=True)

    if config.load_data:
        return load_experiment_data(config, spec)

    results = []
    for condition in ("grok", "memorise"):
        for run_idx in range(config.runs):
            results.append(train_run(spec, condition, run_idx, device, config))

    data = {
        "meta": {
            **spec.metadata(config),
            "name": config.name,
            "runs": config.runs,
            "n_checkpoints": config.n_checkpoints,
            "device": str(device),
        },
        "results": results,
    }
    if config.save_data:
        save_data(data, experiment_data_path(config, spec))
    return data


def print_summary(data) -> None:
    print("\nsummary")
    print("condition,run,seed,steps,train_acc,test_acc,seconds")
    for row in data["results"]:
        print(
            f"{row['condition']},{row['run']},{row['seed']},{row['steps']},"
            f"{row['train_acc']:.6f},{row['test_acc']:.6f},{row['seconds']:.2f}"
        )
