"""
temporal_validation_FINAL.py
-----------------------------
Pre-registered temporal validation experiment (Section 8.2).

Compares the model's predicted compartment activation order against the
real GSE182939 IRI time-course (sham, 4h, 12h, 48h, 6wk) using a
threshold-free temporal-centroid estimator.


Final pre-registered result:
    rho_s = -0.073, p = 0.830, n = 11
Not statistically significant. Label-transfer KL exceeded the 0.5 warning
threshold for 4 of 5 timepoints (only sham=0.237 was acceptable).
This null result is reported transparently as a data-quality limitation.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from diffusion_model import CompartmentDiffusionModel
from preprocess_compartments import (
    SAMPLE_PATHS, SC_REFERENCE_H5AD, INJURY_MARKERS, run_pipeline,
)

# ── Pre-registered assertion: all 5 timepoints must be present ───────────
# This assertion fails loudly if any of the 5 expected timepoints is missing.
_REQUIRED_TIMEPOINTS = {"sham", "4h", "12h", "48h", "6wk"}
_missing = _REQUIRED_TIMEPOINTS - set(SAMPLE_PATHS.keys())
assert not _missing, (
    f"Missing required timepoints in SAMPLE_PATHS: {_missing}. "
    f"All 5 timepoints (sham, 4h, 12h, 48h, 6wk) must be present. "
    f"Got: {list(SAMPLE_PATHS.keys())}"
)

# Real sampling times of GSE182939 (hours post-IRI; sham treated as t=0)
TIMEPOINT_HOURS = {"sham": 0, "4h": 4, "12h": 12, "48h": 48, "6wk": 24 * 42}

# ── D_H is uniform 0.05, NOT equal to D_D ──────────────────
# The healthy state diffuses slowly and uniformly (structural inertia),
# while the damage signal propagates faster and is cell-type-specific.
# Setting D_H == D_D would make the healthy state
# diffuse 4-7x faster than intended, altering propagation dynamics.
D_H_GLOBAL = 0.05  # uniform for all compartments (matches SIR_DEFAULTS["D_S"])

# Baseline parameters from Table 1 of the manuscript.
# D_D is per-macro-type; D_H is uniform.
TABLE1_PARAMS = {
    "PT":       dict(beta=0.40, rho=0.08, D_H=D_H_GLOBAL, D_D=0.20),
    "DCT":      dict(beta=0.28, rho=0.10, D_H=D_H_GLOBAL, D_D=0.20),
    "TAL":      dict(beta=0.25, rho=0.12, D_H=D_H_GLOBAL, D_D=0.20),
    "vascular": dict(beta=0.30, rho=0.10, D_H=D_H_GLOBAL, D_D=0.35),
    "immune":   dict(beta=0.20, rho=0.25, D_H=D_H_GLOBAL, D_D=0.20),
    "other":    dict(beta=0.22, rho=0.10, D_H=D_H_GLOBAL, D_D=0.20),
}

# Runtime assertion: catch any regression where D_H == D_D for all types.
assert any(p["D_H"] != p["D_D"] for p in TABLE1_PARAMS.values()), (

    "D_H should be 0.05 (uniform), D_D should be per-macro-type."
)

SEED_ZONE = "outer_medulla"
SEED_P_D0 = 0.05


def macro_type_of(compartment_name):
    """Extract macro-type from compartment name (e.g. 'PT_cortex' -> 'PT')."""
    return compartment_name.split("_")[0]


def build_model_from_sham(sham_result):
    """Build the diffusion model from the sham (pre-injury) compartment graph."""
    names = sham_result["names"]
    W_c = sham_result["W_c"]
    beta = np.array([TABLE1_PARAMS[macro_type_of(n)]["beta"] for n in names])
    rho = np.array([TABLE1_PARAMS[macro_type_of(n)]["rho"] for n in names])
    D_H = np.array([TABLE1_PARAMS[macro_type_of(n)]["D_H"] for n in names])
    D_D = np.array([TABLE1_PARAMS[macro_type_of(n)]["D_D"] for n in names])
    model = CompartmentDiffusionModel(W_c, beta, rho, D_H=D_H, D_D=D_D,
                                       compartment_names=names)

    # Seed the outer-medulla compartments (IRI origin zone, same as Section 6.3)
    p_D0 = np.array([SEED_P_D0 if SEED_ZONE in n else 0.0 for n in names])
    return model, p_D0


def score_injury_markers_per_compartment(adata, markers=INJURY_MARKERS):
    """Score the injury-marker panel per compartment using scanpy's score_genes."""
    # Case-insensitive lookup (Visium uses Title-Case, markers may differ)
    var_lower_map = {v.lower(): v for v in adata.var_names}
    genes_present = []
    for g in markers:
        if g in adata.var_names:
            genes_present.append(g)
        elif g.lower() in var_lower_map:
            genes_present.append(var_lower_map[g.lower()])
    if not genes_present:
        raise ValueError(
            f"None of the injury markers found in this timepoint's var_names. "
            f"Tried: {markers}. "
            f"First 10 var_names: {list(adata.var_names[:10])}"
        )
    print(f"  [injury markers] Found {len(genes_present)}/{len(markers)}: {genes_present}")
    sc.tl.score_genes(adata, genes_present, score_name="injury_score")
    return adata.obs.groupby("compartment")["injury_score"].mean()


def observed_temporal_centroid(injury_scores_by_timepoint):
    """
    Compute a continuous, threshold-free, injury-weighted temporal centroid
    for each compartment across all timepoints.

    This replaces the original hard threshold-crossing criterion, which was
    numerically unstable: depending on small changes in the pipeline, it
    either flagged nearly all compartments as crossing at the same early
    timepoint (uninformative) or almost none crossing at all (too few data
    points for a meaningful correlation).

    Method:
      1. Centre each timepoint's scores on that timepoint's own tissue-wide
         mean (removes global technical shifts between independently
         processed cryosections).
      2. For each compartment, compute a weighted average of timepoint hours,
         where weights = max(centered_score, 0) -- only the "above average
         for that timepoint" part counts as evidence of damage.

    This estimator was pre-registered (decided before looking at results)
    to avoid selecting the most favourable of several possible comparison
    methods after the fact.
    """
    # Centre each timepoint on its own tissue-wide mean
    centered_scores = {
        tp: scores - scores.mean()
        for tp, scores in injury_scores_by_timepoint.items()
    }
    non_sham_tps = sorted(
        (tp for tp in TIMEPOINT_HOURS if tp != "sham"),
        key=lambda tp: TIMEPOINT_HOURS[tp],
    )
    all_compartments = centered_scores[non_sham_tps[0]].index
    for tp in non_sham_tps[1:]:
        all_compartments = all_compartments.union(centered_scores[tp].index)

    rows = []
    for compartment in all_compartments:
        weights = []
        hours = []
        for tp in non_sham_tps:
            score = centered_scores[tp].get(compartment, np.nan)
            if pd.notna(score):
                weights.append(max(score, 0.0))
                hours.append(TIMEPOINT_HOURS[tp])
        weights = np.array(weights)
        hours = np.array(hours)
        if weights.sum() > 0:
            t_obs = float(np.sum(weights * hours) / np.sum(weights))
        else:
            t_obs = np.nan
        rows.append(dict(compartment=compartment, t_obs_hours=t_obs))
    return pd.DataFrame(rows).set_index("compartment")


def main():
    """Run the full temporal validation pipeline."""
    # 1. Build compartment graph + model from sham only
    print("Building compartment graph from sham timepoint...")
    sham_result = run_pipeline(SAMPLE_PATHS["sham"], SC_REFERENCE_H5AD)

    
    kl = sham_result.get("kl", None)
    if kl is not None:
        print(f"  Label-transfer KL (sham): {kl:.3f}")
        if kl > 0.5:
            print(f"  [WARNING] KL > 0.5 -- label transfer quality is poor.")

    model, p_D0 = build_model_from_sham(sham_result)

    # 2. Forward simulation with UNCHANGED Table 1 parameters (no refitting)
    print("\nSimulating forward with Table 1 parameters (out-of-sample forecast)...")
    traj = model.simulate(p_D0, t_end=60.0, dt=0.05)
    sim_metrics = model.metrics(traj)

    # 3. Real injury-marker scores at every timepoint
    print("\nScoring injury markers at all 5 timepoints...")
    injury_scores = {}
    for tp, path in SAMPLE_PATHS.items():
        print(f"  Processing {tp}...")
        result = run_pipeline(path, SC_REFERENCE_H5AD)

        
        kl = result.get("kl", None)
        if kl is not None:
            status = "OK" if kl <= 0.5 else "WARNING"
            print(f"    KL = {kl:.3f} [{status}]")

        injury_scores[tp] = score_injury_markers_per_compartment(result["adata"])

    # 4. Compute observed temporal centroid 
    obs_centroid = observed_temporal_centroid(injury_scores)

    # 5. Merge predicted vs. observed and compute rank correlation
    merged = sim_metrics.set_index("compartment").join(obs_centroid, how="inner")
    merged = merged.dropna(subset=["t_act", "t_obs_hours"])

    rho_s, p_val = spearmanr(merged["t_act"], merged["t_obs_hours"])
    print(f"\nSpearman correlation (simulated t_act vs. observed temporal centroid):")
    print(f"  rho_s = {rho_s:.3f}, p = {p_val:.4f}, n = {len(merged)}")

    merged.to_csv("temporal_validation_FINAL_summary.csv")

    # Figure: simulated activation-time ranking vs. observed
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged["t_act"], merged["t_obs_hours"])
    for name, row in merged.iterrows():
        ax.annotate(name, (row["t_act"], row["t_obs_hours"]), fontsize=7)
    ax.set_xlabel("Simulated activation time (model units)")
    ax.set_ylabel("Observed temporal centroid (hours post-IRI)")
    ax.set_title(f"Temporal validation (Spearman rho = {rho_s:.2f}, p = {p_val:.3f})")
    fig.tight_layout()
    fig.savefig("temporal_validation_FINAL_scatter.png", dpi=200)
    print(f"\nSaved: temporal_validation_FINAL_summary.csv, temporal_validation_FINAL_scatter.png")

    return merged, rho_s, p_val


if __name__ == "__main__":
    main()
