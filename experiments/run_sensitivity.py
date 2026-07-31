"""
run_sensitivity.py — Entry point for parametric sensitivity analysis.

Runs:
  - One-at-a-time (OAT) sweep
  - Morris screening
  - Monte Carlo sweep (N=300)
  - Ranking stability check (N=200)
  - 2D sweep (ρ_mult_outer × β_DCT)

Usage:
    python experiments/run_sensitivity.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sir_sensitivity import run_full_sensitivity, run_2d_sweep_analysis
