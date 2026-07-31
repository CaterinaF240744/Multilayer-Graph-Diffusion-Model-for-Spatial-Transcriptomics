"""
run_main.py — Entry point for the full analysis pipeline.

Runs all 12 steps of main.py:
  1.  Load Visium
  2.  Spatial graph + random walk
  3.  Single-cell (GSE107585)
  4.  SC-ST alignment + compartments
  5.  Diffusion simulation (single-layer, seed=0.05)
  6.  Parameter validation
  7.  Multilayer diffusion (PT/DCT/TAL, exposure-driven)
  8.  Invasion metrics
  9.  Trans-compartment metrics
  10. D_inter sweep
  10b. Pharmacological modulation
  11. Sensitivity analysis
  12. Abstract figure

Usage:
    python experiments/run_main.py

Prerequisites:
    - data/Mouse_kidney_single_cell_datamatrix.txt (from GSE107585)
    - Internet connection (for Visium auto-download)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Change to data directory for SC file lookup
os.chdir(os.path.join(os.path.dirname(__file__), "..", "data"))

from main import *  # noqa: F401,F403
