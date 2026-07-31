"""
temporal_validation_gse182939.py
----------------------------------
DEPRECATED -- use temporal_validation_FINAL.py instead.

This earlier version had four bugs that are all fixed in the FINAL version:
  6.1  D_H set equal to D_D per macro-type (should be D_H=0.05 uniform)
  6.2  Unstable threshold-based observed-activation estimator (replaced by
       continuous temporal centroid)
  6.3  Only 4 of 5 timepoints used (48h was silently dropped)
  6.4  Wrong dictionary key for KL ("kl_divergence" instead of "kl")

The FINAL pre-registered version is in temporal_validation_FINAL.py.
This file is kept for historical reference only.
"""

import warnings
warnings.warn(
    "temporal_validation_gse182939.py is deprecated. "
    "Use temporal_validation_FINAL.py for the pre-registered analysis.",
    DeprecationWarning,
)
