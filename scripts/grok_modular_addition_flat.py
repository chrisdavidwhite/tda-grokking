"""Modular Addition with per-ELEMENT (flat) input encoding.

Same arithmetic task as `tda_for_neural_networks.py` (GROK_TASK=add) but
the input encoding treats the WHOLE group G = Z/m_1 × … × Z/m_d as a
single |G|-element vocabulary, rather than laying out one one-hot block
per factor. Inputs are pairs (a, b) ∈ G × G, encoded as
[onehot_|G|(a) | onehot_|G|(b)] of dim 2·|G| = 2·∏ m_i.

Motivation: at d=2, per-factor encoding hands the model the factor
decomposition for free — each factor's one-hot block is a separate
region of the input — and the model trivially solves d copies of the
d=1 problem in parallel. The memorisation basin vanishes and so does
grokking, regardless of HIDDEN, TRAIN_FRAC, or WD. The flat encoding
forces the model to discover that |G| = ∏ m_i factors as a product of
cyclic groups before it can find the Fourier features that solve the
task. This is the direct d-dimensional analog of the d=1 grokking setup.

At d=1 (MODULI=[p]) this encoding is bit-identical to GROK_TASK=add
modulo RNG order — useful as a sanity check before running d ≥ 2
experiments.

Reuses everything in tda_for_neural_networks via GROK_TASK=add_flat.
Figures land in Figures/Mflat<moduli>_E<epochs>_F<train_frac>/.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("GROK_TASK", "add_flat")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_for_neural_networks import main


if __name__ == "__main__":
    main()
