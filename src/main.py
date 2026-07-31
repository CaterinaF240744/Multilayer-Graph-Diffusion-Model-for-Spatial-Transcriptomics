"""
main.py  —  Multilayer Graph Diffusion Framework on Spatial Transcriptomics
=============================================================================
Exposure-driven diffusion-reaction model on a compartmental tissue graph
(murine kidney Visium).

State variables (H/D/R/X nomenclature, aligned with the manuscript):
  H(t)  = Healthy fraction        [code alias: S]
  D(t)  = Diseased fraction       [code alias: I]
  R(t)  = Recovered fraction      [code alias: R]
  X(t)  = Dead/irreversible damage [pharmacological model only]

Parameters:
  β_i   = susceptibility to local pathological exposure
  ρ_i   = recovery rate  (code alias: gamma)
  δ_i   = irreversible damage rate  (pharmacological model only)
  D_H   = spatial diffusion of healthy state  (code alias: D_S)
  D_D   = spatial diffusion of diseased state (code alias: D_I)
  R_diff = β/ρ  (diffusive reproduction number)

The infection term is exposure-driven (β·H·λ), NOT bilinear (β·H·D),
where λ_i = Σ_j w̃_ij · D_j is the local disease exposure (Eq. 1-2).

Pipeline
---------
STEP 1  : Load Visium (V1_Mouse_Kidney)
STEP 2  : Spatial graph and random-walk
STEP 3  : Single-cell (GSE107585)
STEP 4  : SC-ST alignment + anatomical compartments
STEP 5  : Diffusion simulation on compartmental graph
STEP 6  : Parameter validation (biologically plausible ranges, murine IRI)
STEP 7  : Multilayer diffusion PT/DCT/TAL
STEP 8  : Invasion metrics (activation_time, peak_time, attack_rate)
STEP 9  : Trans-compartmental metrics (NGM, flux, entropy)
STEP 10 : D_inter sweep — diffusion front robustness
STEP 10b: Pharmacological modulation H/D/D2/R/X + drug (Section 8.6)
STEP 11 : Parametric sensitivity analysis
STEP 12 : Abstract figure
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scanpy as sc
import scipy.sparse as sp

plt.rcParams.update({
    'savefig.dpi':300,
    'figure.dpi':150,
    'lines.linewidth':1.5,
    'lines.markersize':6,
    'axes.linewidth':1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'font.size': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

from preprocessing import (
    load_and_preprocess_visium, convert_sc_to_h5ad,
    annotate_sc, preprocess_sc_for_integration, align_sc_st,
    MACRO_MAP,
)
from graph_utils import (
    build_spatial_graph, random_walk, find_optimal_T,
    invasion_metrics,
)
from compartments import build_compartment2
from sir_compartments import (
    build_default_params_SIR, run_SIR, sir_metrics,
    attach_SIR_to_adata, plot_sir_dynamics, plot_R0_spatial,
    suggest_t_end, sir_temporal_metrics_full,
    print_sir_temporal_summary, plot_sir_temporal,
)
from sir_compartment_graphs import plot_all_compartments
from sir_multilayer import run_SIR_multilayer
from sir_invasion_metrics import run_sir_invasion_analysis
from sir_transcompartment_metrics import (
    run_transcompartment_analysis,
    print_transcompartment_summary,
)
from sir_sensitivity import run_full_sensitivity, run_2d_sweep_analysis
from sir_drug import run_full_drug_analysis

from sir_transition_graphs import (
    plot_single_layer_transition,plot_single_layer_transition_spots,
    plot_multilayer_transition,plot_multilayer_snapshots,plot_drug_transition,
    
)


SEED = 42
np.random.seed(SEED)
sc.settings.verbosity = 1
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Labels for figures (H/D/R/X nomenclature, aligned with manuscript) ────
HDR_LABELS = {
    "H":  "H(t)  Healthy fraction",
    "D":  "D(t)  Diseased fraction",
    "R":  "R(t)  Recovered fraction",
    "X":  "X(t)  Irreversible damage",
    "R0": "R_diff = β/ρ  (diffusive reproduction number)",
}
# Backward-compatible alias
DIFF_LABELS = HDR_LABELS


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Caricamento Visium
# ═══════════════════════════════════════════════════════════════════════════

def step1_visium():
    print("\n" + "="*60)
    print("STEP 1: Loading Visium")
    print("="*60)
    adata_st = load_and_preprocess_visium(
        dataset_name="V1_Mouse_Kidney",
        n_top_genes=2000,
        resolution=0.5,
        random_state=SEED,
    )
    print(f"  Spots: {adata_st.n_obs}  |  HVG genes: {adata_st.n_vars}")
    return adata_st


adata_st = step1_visium()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Grafo spaziale e diffusione random-walk
# ═══════════════════════════════════════════════════════════════════════════

def step2_spatial_graph(adata_st):
    print("\n" + "="*60)
    print("STEP 2: Spatial graph and random-walk diffusion")
    print("="*60)
    g             = build_spatial_graph(adata_st)
    P             = g["P"]
    dist_boundary = g["dist_boundary"]
    x0 = g["is_boundary"].astype(float)
    print(f"  Source nodes (boundary): {int(x0.sum())}")
    T_vals, mse_vals = find_optimal_T(x0, P, dist_boundary,
                                      T_range=range(0, 101, 5))
    T_opt = T_vals[int(np.argmin(mse_vals))]
    print(f"  Optimal random-walk T: {T_opt}")
    x_in = random_walk(x0, P, T_opt)
    adata_st.obs["x0"]   = x0
    adata_st.obs["x_in"] = x_in
    return g, x0, x_in


g, x0, x_in = step2_spatial_graph(adata_st)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Single-cell
# ═══════════════════════════════════════════════════════════════════════════

def step3_single_cell(
    sc_file:   str = "Mouse_kidney_single_cell_datamatrix.txt",
    h5ad_file: str = "GSE107585_sc.h5ad",
):
    print("\n" + "="*60)
    print("STEP 3: Single-cell")
    print("="*60)
    import os
    if os.path.exists(h5ad_file):
        adata_sc = sc.read_h5ad(h5ad_file)
        print(f"  Loaded from cache: {h5ad_file}")
    else:
        adata_sc = convert_sc_to_h5ad(sc_file, outfile=h5ad_file)

    adata_sc.var_names_make_unique()
    adata_sc = annotate_sc(adata_sc)
    adata_sc = preprocess_sc_for_integration(adata_sc)
    print(f"  Cells: {adata_sc.n_obs}  |  Geni: {adata_sc.n_vars}")
    print(f"  Cell types: {adata_sc.obs['cell_identity'].nunique()}")
    sc.pl.umap(adata_sc, color=["cell_identity", "compartment"],
               title=["Cell identity (SC)", "Compartment (SC)"])
    return adata_sc


adata_sc = step3_single_cell(
    sc_file="Mouse_kidney_single_cell_datamatrix.txt",
    h5ad_file="GSE107585_sc.h5ad",
)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Allineamento SC-ST e compartimenti anatomici
# ═══════════════════════════════════════════════════════════════════════════

def step4_alignment(adata_sc, adata_st):
    print("\n" + "="*60)
    print("STEP 4: SC-ST alignment + anatomical compartments")
    print("="*60)

    adata_st2 = align_sc_st(adata_sc, adata_st, obs_key="cell_identity")

    # Label transfer quality: KL divergence between distributions
    sc_dist = adata_sc.obs["cell_identity"].value_counts(normalize=True)
    st_dist = adata_st2.obs["cell_identity"].value_counts(normalize=True)
    common  = sc_dist.index.intersection(st_dist.index)
    kl_div  = float(np.sum(
        sc_dist[common] * np.log(sc_dist[common] / (st_dist[common] + 1e-9))
    ))
    print(f"  Label transfer quality — KL divergence SC→ST: {kl_div:.3f}")
    print(f"  (0 = identical distribution | <0.5 = acceptable)")

    sc.pl.spatial(adata_st2, color=["cell_identity"],
                  title=["Label transfer — tipo cellulare"])

    comp = build_compartment2(
        adata_st2,
        use_anatomical_zones=True,
        fallback_radial=True,
    )
    print(f"  Compartments: {len(comp['comps2'])}  "
          f"  zone method: {comp['zone_method']}")
    for cname in comp["comps2"]:
        n = int((adata_st2.obs["compartment2"] == cname).sum())
        print(f"    {cname}: {n} spots")

    return adata_st2, comp


adata_st2, comp = step4_alignment(adata_sc, adata_st)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Simulazione diffusione su grafo compartimentale
# ═══════════════════════════════════════════════════════════════════════════

def step5_diffusion(comp, adata_st,
                    seed_D: float = 0.05,
                    t_end: float = None,
                    n_steps: int = 800):
    """
    Simulate exposure-driven diffusion on the compartmental graph.

    State variables (H/D/R/X, aligned with manuscript Eq. 18-20):
      H(t) = Healthy fraction   [code: S]
      D(t) = Diseased fraction  [code: I]
      R(t) = Recovered fraction [code: R]

    The seed is placed in outer-medulla compartments: D(0) = seed_D.
    """
    print("\n" + "="*60)
    print("STEP 5: Diffusion on compartmental graph")
    print("="*60)

    comps2        = comp["comps2"]
    Lc2           = comp["Lc2"]
    boundary_mask = comp["boundary_mask"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]
    inv2          = comp["inv2"]

    diff_params = build_default_params_SIR(comps2, macro_of, region_of)

    if t_end is None:
        print("\n  Computing optimal t_end...")
        t_end = suggest_t_end(
            comps2, Lc2, boundary_mask, macro_of, region_of,
            sir_params=diff_params, seed_I=seed_D,
        )

    print(f"\n  Compartments: {len(comps2)}")
    print(f"  Source compartments (boundary): {boundary_mask.sum()}")
    print(f"  seed_D = {seed_D}  |  t_end = {t_end}")

    # Spatial map of R_diff = β/ρ (diffusive reproduction number)
    plot_R0_spatial(diff_params, macro_of, comps2, adata_st, inv2)

    C_fin, phi_fin, A_fin, sol, diff_params = run_SIR(
        comps2, Lc2, boundary_mask, macro_of, region_of,
        sir_params         = diff_params,
        seed_I             = seed_D,
        seed_boundary_only = True,
        use_ivp            = True,
        t_end              = t_end,
        n_steps            = n_steps,
    )

    m = sir_metrics(C_fin, phi_fin, A_fin, macro_of, region_of,
                    diff_params, sol=sol)

    print(f"\n  Attack rate (mean final A)    : {m['global_R']:.3f}")
    print(f"  Residual D mean         : {m['global_I']:.3f}")
    print(f"  Fraction of invaded compartments      : {m['frac_invaded']:.3f}")
    print(f"  Mean global R_diff            : {m['mean_R0']:.2f}")

    print("\n  Per macro_type (diffusion front):")
    for mt, v in m["by_macro"].items():
        print(f"    {mt:12s}  D_mean={v['mean_I']:.3f}  "
              f"A_mean={v['mean_R']:.3f}  R_diff={v['mean_R0']:.2f}  "
              f"invaso={v['frac_invaded']:.0%}")

    # Trajectories with H/D/R labels
    plot_sir_dynamics(sol, macro_of,
                      title=f"Diffusione su grafo — seed_D={seed_D}  t_end={t_end}")

    # Spatial maps at peak of D(t)
    nC = len(comps2)
    phi_traj   = sol.y[nC:2*nC, :]
    t_peak_idx = int(np.argmax(phi_traj.mean(axis=0)))

    C_peak   = np.clip(sol.y[:nC,     t_peak_idx], 0.0, 1.0)
    phi_peak = np.clip(sol.y[nC:2*nC, t_peak_idx], 0.0, 1.0)
    A_peak   = np.clip(sol.y[2*nC:,   t_peak_idx], 0.0, 1.0)
    t_peak_val = float(sol.t[t_peak_idx])

    tot = np.where(C_peak + phi_peak + A_peak > 0,
                   C_peak + phi_peak + A_peak, 1.0)
    C_peak /= tot; phi_peak /= tot; A_peak /= tot

    attach_SIR_to_adata(C_fin,   phi_fin, A_fin,   adata_st, inv2)
    attach_SIR_to_adata(C_peak,  phi_peak, A_peak, adata_st, inv2, tag="peak")

    sc.pl.spatial(
        adata_st,
        color=["SIR_S", "SIR_I", "SIR_R"],
        cmap="viridis",
        title=[f"H(t) final  [t={t_end:.0f}]",
               f"D(t) final  [t={t_end:.0f}]",
               f"R(t) final  [t={t_end:.0f}]"],
    )
    sc.pl.spatial(
        adata_st,
        color=["SIR_S_peak", "SIR_I_peak", "SIR_R_peak"],
        cmap="viridis",
        title=[f"H(t) at peak  [t={t_peak_val:.1f}]",
               f"D(t) at peak  [t={t_peak_val:.1f}]  ← more informative",
               f"R(t) at peak  [t={t_peak_val:.1f}]"],
    )

    return C_fin, phi_fin, A_fin, sol, diff_params, m


C_diff, phi_diff, A_diff, sol_diff, diff_params, diff_m = step5_diffusion(
    comp, adata_st2, seed_D=0.05, t_end=None, n_steps=800
)

# Metriche temporali
tm = sir_temporal_metrics_full(
    sol_diff, comp["macro_of"], comp["region_of"], diff_params,
)
print_sir_temporal_summary(tm, comps2=comp["comps2"])
plot_sir_temporal(
    sol_diff, comp["macro_of"], comp["region_of"], diff_params,
    comps2=comp["comps2"], threshold_I=0.10,
)

plot_all_compartments(
    sol_diff, comp["Wsp"], diff_params,
    comp["comps2"], comp["macro_of"], comp["region_of"],
    comp["boundary_mask"], comp["inv2"], adata_st2,
    n_panels=6,
)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Validazione parametri (range biologici plausibili IRI murino)
# ═══════════════════════════════════════════════════════════════════════════

def step6_validate_params(diff_params, comp):
    """
    Verifica che R_diff = β/ρ per macro_type sia nell'intervallo
    biologicamente plausibile per IRI murino (Gerhardt 2021, Lake 2023).

    Intervalli di riferimento per il diffusive reproduction number:
      PT       : [1.5, 5.0]  — high oxidative susceptibility
      TAL      : [1.0, 3.5]  — OSOM is the primary seed zone
      vascular : [1.2, 4.0]  — corridoio di propagazione
    """
    print("\n" + "="*60)
    print("STEP 6: Parameter validation (biological ranges, murine IRI)")
    print("="*60)

    PLAUSIBLE_R_DIFF = {
        "PT":       (1.5, 5.0),
        "TAL":      (1.0, 3.5),
        "vascular": (1.2, 4.0),
        "DCT":      (1.0, 3.0),
    }

    beta_v  = diff_params["beta"]
    gamma_v = diff_params["gamma"]
    macro_of = comp["macro_of"]
    R_diff   = beta_v / np.where(gamma_v > 0, gamma_v, 1e-12)

    all_ok = True
    for macro, (lo, hi) in PLAUSIBLE_R_DIFF.items():
        mask = macro_of == macro
        if not mask.any():
            continue
        r_mean = float(R_diff[mask].mean())
        status = "✓" if lo <= r_mean <= hi else "⚠️  OUT OF RANGE"
        if "⚠️" in status:
            all_ok = False
        print(f"  {macro:12s}  R_diff={r_mean:.2f}  "
              f"range atteso [{lo:.1f}, {hi:.1f}]  {status}")

    if all_ok:
        print("\n  → All parameters within plausible ranges.")
    else:
        print("\n  → Some parameters out of range: "
              "consider recalibration.")

    # Distribuzione spaziale di R_diff
    adata_st2.obs["R_diff"] = R_diff[comp["inv2"]].astype(np.float32)
    sc.pl.spatial(
        adata_st2,
        color=["R_diff"],
        cmap="RdYlGn_r",
        title=["R_diff = β/ρ per compartimento"],
    )


step6_validate_params(diff_params, comp)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Diffusione multilayer PT/DCT/TAL
# ═══════════════════════════════════════════════════════════════════════════

ml_result = run_SIR_multilayer(
    comp        = comp,
    sir_params  = diff_params,
    adata_st    = adata_st2,
    seed_I      = 0.05,
    D_inter     = 0.05,
    t_end       = float(sol_diff.t[-1]),
    n_steps     = 800,
    threshold_I = 0.10,
)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Metriche di invasione del fronte di diffusione
# ═══════════════════════════════════════════════════════════════════════════

df_invasion = run_sir_invasion_analysis(
    sol=            sol_diff,
    comps2=         comp["comps2"],
    macro_of=       comp["macro_of"],
    region_of=      comp["region_of"],
    boundary_mask=  comp["boundary_mask"],
    adata_st=       adata_st2,
    inv2=           comp["inv2"],
    sir_params=     diff_params,
    threshold_I=    0.10,
)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — Metriche trans-compartimentali
# ═══════════════════════════════════════════════════════════════════════════

sol_ml       = ml_result["sol"]
net          = ml_result["net"]
global_index = net["global_index"]

W_ml = sp.csr_matrix(comp["Wc2"])[np.ix_(global_index, global_index)]
labels_ml    = net["layer_of_global"]
beta_ml_vec  = diff_params["beta"][global_index]
gamma_ml_vec = diff_params["gamma"][global_index]

coords_visium = adata_st2.obsm["spatial"].astype(np.float32)
coords_ml = []
for ci in global_index:
    mask = comp["inv2"] == ci
    if mask.sum() > 0:
        coords_ml.append(coords_visium[mask].mean(axis=0))
    else:
        warnings.warn(
            f"Compartimento {ci} ({comp['comps2'][ci]}) senza spot — "
            "coordinate impostate al centroide tissutale.",
            UserWarning,
        )
        coords_ml.append(coords_visium.mean(axis=0))
coords_ml = np.array(coords_ml)

boundary_global = comp["boundary_mask"][global_index]
seed_idx_ml = int(np.where(boundary_global)[0][0]) if boundary_global.any() else 0

print("\n" + "="*60)
print("STEP 9: Metriche trans-compartimentali")
print("="*60)

tc_results = run_transcompartment_analysis(
    sol      = sol_ml,
    W        = W_ml,
    labels   = labels_ml,
    beta     = beta_ml_vec,
    gamma    = gamma_ml_vec,
    coords   = coords_ml,
    seed_idx = seed_idx_ml,
    plot     = True,
)
print_transcompartment_summary(tc_results)

from fix_interlayer_flux import patch_interlayer_flux
tc_results = patch_interlayer_flux(
    tc_results, ml_result["sol"], ml_result["net"], W_ml, diff_params
)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10b — Modulazione farmacologica (Sezione 8.6, Eq. 11-17)
#
# Estende il modello H/D/R con stati D2 (attenuato), X (danno irrev.)
# e campo farmaco F(t). Il farmaco diffonde attraverso il grafo con
# coefficiente D_F preferenziale nel compartimento vascolare.
#
# t_dose = 20% di t_end → allineato con t_opt = 11.8 ≈ 20% × 60
# (il 25% nel draft era un'approssimazione — il paper riporta 20%)
# ═══════════════════════════════════════════════════════════════════════════

drug_result = run_full_drug_analysis(
    comp          = comp,
    sir_params    = diff_params,
    adata_st      = adata_st2,
    seed_I      = 0.05,
    t_end         = float(sol_diff.t[-1]),
    n_steps       = 800,
    t_dose        = float(sol_diff.t[-1]) * 0.20,   # 20% di t_end (Sez. 8.6)
    dose_amount   = 1.0,
    dose_mode     = "bolus",
    run_sweep     = True,
    sweep_n       = 20,
    threshold_I   = 0.05,
    IC50_beta     = 0.0,    # Hill kinetics disabilitato (Eq. 17: IC50=0 → β_eff=β)
)



# ═══════════════════════════════════════════════════════════════════════════
# STEP 9b — Grafi di transizione
#
# Le ODE sono relegate in appendice. Qui si producono le figure principali
# che mostrano il GRAFO DI TRANSIZIONE sul tessuto:
#   Fig 1: single-layer concentrico (compartimento-level)
#   Fig 2: single-layer spot-level su coordinate Visium reali
#   Fig 3: multilayer statico PT/DCT/TAL
#   Fig 4: multilayer a 3 snapshot (t_dose, t_peak, t_end)
#   Fig 5: drug comparison (se disponibile)
# ═══════════════════════════════════════════════════════════════════════════

from sir_transition_graphs import plot_all_transition_graphs

transition_figs = plot_all_transition_graphs(
    sol_single    = sol_diff,
    Wc2           = comp["Wc2"],
    comps2        = comp["comps2"],
    macro_of      = comp["macro_of"],
    region_of     = comp["region_of"],
    boundary_mask = comp["boundary_mask"],
    sir_params    = diff_params,
    ml_result     = ml_result,
    drug_result   = drug_result if "drug_result" in dir() else None,
    adata_st      = adata_st2,
    inv2          = comp["inv2"],
    Wsp           = comp["Wsp"],
    save_prefix   = "./figures/transition",
    dpi           = 200,
)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 10 — D_inter sweep: multilayer diffusion front robustness
#
# Verifies that the activation sequence PT → DCT → TAL and the layer
# peak_I ranking remain stable as the inter-layer diffusion coefficient
# is varied.
# ═══════════════════════════════════════════════════════════════════════════

def step10_sweep_D_inter(comp, diff_params, adata_st2, sol_diff):
    print("\n" + "="*60)
    print("STEP 10: D_inter sweep — multilayer front robustness")
    print("="*60)

    D_inter_values = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    t_end = float(sol_diff.t[-1])

    results_by_D = {}
    for D in D_inter_values:
        print(f"\n  D_inter = {D:.2f} ...")
        try:
            res = run_SIR_multilayer(
                comp, diff_params, adata_st2,
                seed_I      = 0.05,
                D_inter=D,
                t_end=t_end,
                n_steps=400,
                threshold_I=0.10,
                plot=False,   # nessuna figura durante lo sweep
            )
        except TypeError:
            # Compatibility with sir_multilayer versions without plot parameter
            import matplotlib
            orig_backend=matplotlib.get_backend()
            matplotlib.use("Agg")   # backend non interattivo — sopprime le figure
            res = run_SIR_multilayer(
                comp, diff_params, adata_st2,
                seed_I      = 0.05,
                D_inter=D,
                t_end=t_end,
                n_steps=400,
                threshold_I=0.10,
            )
            matplotlib.use(orig_backend)  # ripristina il backend interattivo
        results_by_D[D] = res["metrics"]["by_layer"]

    # ── Figure: peak_D and t_peak per layer vs D_inter ──────────
    macros = ["PT", "DCT", "TAL"]
    colors = {"PT": "#e74c3c", "DCT": "#2980b9", "TAL": "#27ae60"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for macro in macros:
        peak_I_vals = []
        t_peak_vals = []
        for D in D_inter_values:
            bl = results_by_D[D]
            if macro in bl:
                peak_I_vals.append(bl[macro]["peak_I"])
                t_peak_vals.append(bl[macro]["t_peak"])
            else:
                peak_I_vals.append(np.nan)
                t_peak_vals.append(np.nan)

        col = colors[macro]
        axes[0].plot(D_inter_values, peak_I_vals,
                     "o-", color=col, lw=2, ms=7, label=macro)
        axes[1].plot(D_inter_values, t_peak_vals,
                     "o-", color=col, lw=2, ms=7, label=macro)

    axes[0].set_xlabel("D_inter", fontsize=11)
    axes[0].set_ylabel("Peak D(t) per layer", fontsize=11)
    axes[0].set_title("Peak signal intensity vs D_inter\n"
                      "[ranking PT>DCT>TAL atteso]", fontsize=10)
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("D_inter", fontsize=11)
    axes[1].set_ylabel("t_peak per layer", fontsize=11)
    axes[1].set_title("Ritardo temporale inter-layer vs D_inter\n"
                      "[PT precede DCT precede TAL]", fontsize=10)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Multilayer diffusion front robustness vs D_inter\n"
        "[if ranking stays stable, conclusions are robust]",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    # Stampa tabella riepilogativa
    print("\n  Peak-D table per layer and D_inter:")
    header = f"  {'D_inter':>8s}" + "".join(f"  {m:>8s}" for m in macros)
    print(header)
    print("  " + "-" * (8 + 10 * len(macros)))
    for D in D_inter_values:
        row = f"  {D:>8.3f}"
        for m in macros:
            bl = results_by_D[D]
            v = bl[m]["peak_I"] if m in bl else float("nan")
            row += f"  {v:>8.4f}"
        print(row)

    return results_by_D


results_D_inter = step10_sweep_D_inter(comp, diff_params, adata_st2, sol_diff)




# ═══════════════════════════════════════════════════════════════════════════
# STEP 11 — Parametric sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════

t_end_sensitivity = float(sol_diff.t[-1])

sensitivity_results = run_full_sensitivity(
    comp            = comp,
    base_params     = diff_params,
    t_end           = t_end_sensitivity,
    adata_st        = adata_st2,
    n_oat           = 7,
    n_morris_r      = 15,
    n_mc            = 300,
    n_robust        = 200,
    n_steps_fast    = 150,
    primary_metric  = "global_R",
    run_2d_sweep    = True,
    n_gamma         = 12,
    n_beta          = 12,
    save_prefix     = None,
    verbose         = True,
)

# --- Export raw sensitivity data for Table 9 / Table 11 dispersion stats ---
import os as _os
_os.makedirs("results/tables", exist_ok=True)

sensitivity_results["df_mc"].to_csv(
    "results/tables/sensitivity_mc_raw_N300.csv", index=False
)
sensitivity_results["df_spearman"].to_csv(
    "results/tables/sensitivity_spearman_with_pvalues.csv", index=False
)

_rob = sensitivity_results.get("robustness", {})
if _rob:
    import numpy as _np
    _np.save("results/tables/ranking_stability_corr_inv_N200.npy",
             _rob["corr_inv_series"])
    _np.save("results/tables/ranking_stability_corr_att_N200.npy",
             _rob["corr_att_series"])

print("\n  [export] Raw sensitivity data saved to results/tables/")
# --- end export ---


sweep2d_result = run_2d_sweep_analysis(
    comp        = comp,
    base_params = diff_params,
    adata_st    = adata_st2,
    t_end       = t_end_sensitivity,
    n_gamma     = 12,
    n_beta      = 12,
    seed_I      = 0.05,
    n_steps     = 150,
    save_prefix = None,
)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 12 — Figura (abstract figure)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# STEP 13: Section 7 analysis (edge importance / ablation / seeding)
# Uses the REAL compartment graph built in step4_alignment (same `comp`
# used throughout the rest of the pipeline), not a hardcoded schematic one.
# ═══════════════════════════════════════════════════════════════════════════
import os
from sir_section7 import init_from_compartments, run_section7_analysis

print("\n" + "="*60)
print("STEP 13: Section 7 analysis (real compartment graph)")
print("="*60)

os.makedirs("./results/figures", exist_ok=True)
init_from_compartments(comp)
run_section7_analysis(save_dir="./results/figures", dpi=200)
