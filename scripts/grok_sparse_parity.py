"""Sparse Parity — third clean grokking example (case A).

Input: a random {0,1}^n vector. Output: parity (XOR) of k secret bits.
Following Power et al. 2022, k=3 hidden bits in a 40-bit input with
~1024 training examples + WD pressure produces robust grokking on a
small MLP.

The interesting *topological* prediction differs sharply from the
modular-addition example: the W2 hidden representation should cluster
into 2^k=8 discrete groups (one per value of the hidden XOR partial
state), i.e. β_0 should rise rather than β_1, and the H_1 signature
should *stay low*. This tests whether the topology-tracks-grokking
finding extends beyond cyclic (S^1) tasks to discrete cluster geometry.

Reuses the pipeline in tda_for_neural_networks via GROK_TASK=sparse_parity.
Default hyperparameters here are tuned for sparse-parity grokking
(smaller WD, more epochs); override individually with env vars.

Figures land in Figures/SP_n<N>_k<K>_T<TRAIN>_E<EPOCHS>/.
"""
import os
import sys
from pathlib import Path

# Task + sparse-parity hyperparameters
os.environ.setdefault("GROK_TASK",  "sparse_parity")
os.environ.setdefault("SP_N_BITS",  "40")
os.environ.setdefault("SP_K",       "3")
os.environ.setdefault("SP_TRAIN",   "1024")

# Sparse-parity grokking needs more epochs and a smaller WD than the
# d=1 modular-addition baseline. These match the regimes reported in
# Power et al. 2022 (their parity grokking section), scaled to a smaller
# hidden width for compute time.
os.environ.setdefault("GROK_HIDDEN",  "256")
os.environ.setdefault("GROK_LR",      "1e-3")
os.environ.setdefault("GROK_WD_GROK", "1.0")
os.environ.setdefault("GROK_WD_MEM",  "0.0")
os.environ.setdefault("GROK_N_EPOCHS","30000")
os.environ.setdefault("GROK_N_RUNS",  "20")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_for_neural_networks import main


if __name__ == "__main__":
    main()
