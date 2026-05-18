"""Modular Multiplication on (Z/p)* — second clean grokking example (case B).

(a, b) -> (a * b) mod p, restricted to a, b in {1, ..., p-1}. p is prime,
so (Z/p)* is cyclic of order p-1. Same MLP and training pipeline as the
modular-addition baseline; only the label function and the (a, b) range
differ. Predicts the same S^1 topology at convergence (since (Z/p)* is
cyclic), but reached via a different gradient landscape — the model has
to discover discrete-log features rather than direct additive ones.

Reuses everything in tda_for_neural_networks via the GROK_TASK env var.
Override hyperparameters with the GROK_* / SP_* env vars described in
that file's config block. Figures land in
Figures/Mul<p>_E<epochs>_F<train_frac>/.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("GROK_TASK", "mul")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_for_neural_networks import main


if __name__ == "__main__":
    main()
