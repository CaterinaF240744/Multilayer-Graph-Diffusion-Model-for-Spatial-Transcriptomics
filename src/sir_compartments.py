"""
sir_compartments.py
===================
Modello H/D/R su grafo compartimentale per il rene murino.

Allineamento con l'articolo (Sezioni 5.2, 8.2, Equazioni 2-4)
--------------------------------------------------------------
Le equazioni implementate sono quelle scritte nella Sezione 8.2:

  dp^H_i/dt = -β_i · p^H_i · λ_i(t)  -  D_{H,i} · (L_c p^H)_i
  dp^D_i/dt = +β_i · p^H_i · λ_i(t)  -  ρ_i · p^D_i  -  D_{D,i} · (L_c p^D)_i
  dp^R_i/dt = +ρ_i · p^D_i

where the local exposure is (Eq. 1):

  λ_i(t) = Σ_j w̃_{ij} · p^D_j(t),    w̃_{ij} = w_{ij} / Σ_k w_{ik}

This is the exposure-driven model of Section 5.2. The transition
H→D dipende dalla pressione locale dei vicini malati, non dal termine
bilineare H·D del SIR classico. Le due formulazioni differiscono
quantitatively: the exposure-driven model produces lower peaks and a
more gradual propagation, consistent with a cell-to-cell signalling
locale piuttosto che con un contagio per contatto.

Backward compatibility
----------------------
I nomi pubblici S/I/R nelle firme mappano su H/D/R del paper.
Tutte le funzioni esportate hanno firme invariate.

Funzioni esportate
------------------
build_default_params_SIR   : β, ρ, D_H, D_D per compartimento
simulate_SIR_ivp            : integratore BDF (consigliato, sistemi stiff)
simulate_SIR_euler          : Eulero esplicito con CFL check (per sweep)
run_SIR                     : runner completo
sir_metrics                 : Rdiff, attack rate, invasione per tipo/regione
sir_temporal_metrics        : peak_D, t_peak, auc_D, t_crossing
sir_temporal_metrics_full   : wrapper per tutte le combinazioni macro×region
print_sir_temporal_summary  : stampa riepilogo leggibile
attach_SIR_to_adata         : espande risultati ai nodi spaziali
plot_sir_dynamics           : traiettorie H/D/R per macro_type
plot_sir_temporal           : figura dinamica temporale D(t) per region
plot_R0_spatial             : mappa spaziale di R_diff = β/ρ
suggest_t_end               : auto-detect t_end dalla simulazione pilota
"""

import warnings
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapezoid


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS AND DEFAULT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

SIR_DEFAULTS: dict = dict(
    beta  = 0.30,   # β_i  — local susceptibility
    gamma = 0.10,   # ρ_i  — tasso di recovery
    D_S   = 0.05,   # D_{H,i} — diffusione spaziale H (quasi statica)
    D_I   = 0.20,   # D_{D,i} — diffusione spaziale D (segnale infiammatorio)
)

# Baseline parameters per macro_type — Table 1 of the paper
_MACRO_SIR: dict = {
    "PT":       dict(beta=0.40, gamma=0.08),
    "DCT":      dict(beta=0.28, gamma=0.10),
    "TAL":      dict(beta=0.25, gamma=0.12),
    "vascular": dict(beta=0.30, gamma=0.10, D_I=0.35),
    "immune":   dict(beta=0.20, gamma=0.25),
    "other":    dict(beta=0.22, gamma=0.10),
}

_REGION_SIR: dict = {
    # Anatomical zoning (use_anatomical_zones=True)
    "outer_medulla": dict(beta_mult=1.25, gamma_mult=1.10),
    "cortex":        dict(beta_mult=1.00, gamma_mult=1.00),
    "inner_medulla": dict(beta_mult=0.75, gamma_mult=0.85),
    # Radial zoning legacy (use_anatomical_zones=False)
    "boundary":      dict(beta_mult=1.25, gamma_mult=1.10),
    "mid":           dict(beta_mult=1.00, gamma_mult=1.00),
    "core":          dict(beta_mult=0.80, gamma_mult=0.90),
}

_KNOWN_REGIONS = frozenset(_REGION_SIR.keys())


def build_default_params_SIR(
    comps2:    np.ndarray,
    macro_of:  np.ndarray,
    region_of: np.ndarray,
    beta:  float = None,
    gamma: float = None,
    D_S:   float = None,
    D_I:   float = None,
) -> dict:
    """
    Costruisce i vettori β, ρ (=gamma), D_H (=D_S), D_D (=D_I)
    per compartimento, modulati per macro_type e zona spaziale.

    Returns
    -------
    dict con chiavi: beta (C,), gamma (C,), D_S (C,), D_I (C,)
    """
    nC = len(comps2)

    b0  = beta  if beta  is not None else SIR_DEFAULTS["beta"]
    g0  = gamma if gamma is not None else SIR_DEFAULTS["gamma"]
    ds0 = D_S   if D_S   is not None else SIR_DEFAULTS["D_S"]
    di0 = D_I   if D_I   is not None else SIR_DEFAULTS["D_I"]

    beta_v  = np.full(nC, b0,  dtype=float)
    gamma_v = np.full(nC, g0,  dtype=float)
    D_S_v   = np.full(nC, ds0, dtype=float)
    D_I_v   = np.full(nC, di0, dtype=float)

    for i, m in enumerate(macro_of):
        mp = _MACRO_SIR.get(m, {})
        beta_v[i]  = mp.get("beta",  beta_v[i])
        gamma_v[i] = mp.get("gamma", gamma_v[i])
        D_I_v[i]   = mp.get("D_I",   D_I_v[i])

    unknown_regions = set(region_of) - _KNOWN_REGIONS
    if unknown_regions:
        warnings.warn(
            f"[build_default_params_SIR] Zone non riconosciute in region_of: "
            f"{sorted(unknown_regions)}. "
            f"Zone supportate: {sorted(_KNOWN_REGIONS)}. "
            f"I compartimenti con zona sconosciuta NON vengono modulati "
            f"(beta_mult=1.0, gamma_mult=1.0).",
            UserWarning, stacklevel=2,
        )

    for i, rg in enumerate(region_of):
        rp = _REGION_SIR.get(rg, {})
        beta_v[i]  *= rp.get("beta_mult",  1.0)
        gamma_v[i] *= rp.get("gamma_mult", 1.0)

    return dict(beta=beta_v, gamma=gamma_v, D_S=D_S_v, D_I=D_I_v)


# ═══════════════════════════════════════════════════════════════════════════
# COSTRUZIONE W̃  (matrice di exposure normalizzata per righe)
# ═══════════════════════════════════════════════════════════════════════════

def _build_W_tilde(Lc: sp.spmatrix) -> np.ndarray:
    """
    Ricava W̃ dal Laplaciano normalizzato Lc.

    W_raw_{ij} = diag(Lc)_i · δ_{ij} - Lc_{ij}   (adiacenza pesata, off-diag)
    W̃_{ij}    = W_raw_{ij} / Σ_k W_raw_{ik}       (normalizzazione per riga)

    This is the matrix used to compute the local exposure:
        λ_i(t) = Σ_j W̃_{ij} · p^D_j(t)   (Eq. 1 dell'articolo)
    """
    Lc_arr = np.array(sp.csr_matrix(Lc).toarray(), dtype=float)
    W_raw  = np.diag(np.diag(Lc_arr)) - Lc_arr
    np.fill_diagonal(W_raw, 0.0)
    row_sum = W_raw.sum(axis=1, keepdims=True)
    W_tilde = np.where(row_sum > 1e-12, W_raw / row_sum, 0.0)
    return W_tilde


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATORE BDF  — exposure-driven (Eq. 2-4)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_SIR_ivp(
    Lc:      sp.spmatrix,
    S0:      np.ndarray,   # H_0
    I0:      np.ndarray,   # D_0
    R0:      np.ndarray,   # R_0
    params:  dict,
    t_end:   float = 15.0,
    n_steps: int   = 300,
    method:  str   = "BDF",
    rtol:    float = 1e-6,
    atol:    float = 1e-9,
) -> tuple:
    """
    Integra il sistema H/D/R exposure-driven su grafo compartimentale.

    Equazioni implementate (Sezione 8.2, Eq. 2-4):

      dp^H/dt = -β_i · p^H_i · λ_i(t)  -  D_{H,i} · (L_c @ p^H)_i
      dp^D/dt = +β_i · p^H_i · λ_i(t)  -  ρ_i · p^D_i  -  D_{D,i} · (L_c @ p^D)_i
      dp^R/dt = +ρ_i · p^D_i

    con λ_i(t) = W̃ @ p^D  (Eq. 1)

    Conservation p^H + p^D + p^R = 1 is satisfied by construction.
    Non serve alcun termine di clipping o correzione dell'overflow.

    Parameters
    ----------
    Lc      : sparse (C,C) — Laplaciano normalizzato (Lc2 da build_compartment2)
    S0,I0,R0: ndarray (C,) — condizioni iniziali (H_0, D_0, R_0)
    params  : dict — output di build_default_params_SIR

    Returns
    -------
    S_fin, I_fin, R_fin : ndarray (C,) — stati finali H, D, R
    sol                  : OdeResult
                           sol.y[:C]    → H(t)  shape (C, n_steps)
                           sol.y[C:2C]  → D(t)
                           sol.y[2C:3C] → R(t)
    """
    nC      = len(S0)
    Lc      = sp.csr_matrix(Lc)
    W_tilde = _build_W_tilde(Lc)

    beta_v  = np.asarray(params["beta"],  dtype=float)
    rho_v   = np.asarray(params["gamma"], dtype=float)  # ρ_i
    D_H_v   = np.asarray(params["D_S"],   dtype=float)  # D_{H,i}
    D_D_v   = np.asarray(params["D_I"],   dtype=float)  # D_{D,i}

    y0     = np.concatenate([
        np.asarray(S0, dtype=float),
        np.asarray(I0, dtype=float),
        np.asarray(R0, dtype=float),
    ])
    t_eval = np.linspace(0.0, float(t_end), int(n_steps))

    def rhs(t, y):
        H = np.clip(y[:nC],     0.0, 1.0)
        D = np.clip(y[nC:2*nC], 0.0, 1.0)

        # Exposure locale λ_i(t) = W̃ @ D  (Eq. 1)
        lam = W_tilde @ D

        # ODE — Eq. 2-4
        infect = beta_v * H * lam          # flusso H → D
        recover = rho_v * D                # flusso D → R

        dH = -infect              - D_H_v * (Lc @ H)
        dD = +infect - recover    - D_D_v * (Lc @ D)
        dR = +recover

        return np.concatenate([dH, dD, dR])

    sol = solve_ivp(
        rhs, (0.0, float(t_end)), y0,
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )

    if not sol.success:
        warnings.warn(
            f"simulate_SIR_ivp: solver failed — {sol.message}",
            RuntimeWarning, stacklevel=2,
        )

    # Normalizzazione final (H+D+R deve sommare a 1)
    H_r = np.clip(sol.y[:nC,     -1], 0.0, 1.0)
    D_r = np.clip(sol.y[nC:2*nC, -1], 0.0, 1.0)
    R_r = np.clip(sol.y[2*nC:,   -1], 0.0, 1.0)
    tot = H_r + D_r + R_r
    tot = np.where(tot > 0, tot, 1.0)

    return H_r / tot, D_r / tot, R_r / tot, sol


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATORE EULERO  (per sweep parametrici veloci)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_SIR_euler(
    Lc:      sp.spmatrix,
    S0:      np.ndarray,
    I0:      np.ndarray,
    R0:      np.ndarray,
    params:  dict,
    t_end:   float = 15.0,
    n_steps: int   = 1500,
) -> tuple:
    """
    Integra il sistema H/D/R exposure-driven con Eulero esplicito.
    Usa la stessa formulazione di simulate_SIR_ivp.
    Emette RuntimeWarning se CFL > 0.5.

    Returns
    -------
    S_fin, I_fin, R_fin : ndarray (C,)
    sol_dict            : dict con 't', 'S', 'I', 'R' — traiettorie (C, n_steps+1)
    """
    nC      = len(S0)
    dt      = float(t_end) / int(n_steps)
    Lc      = sp.csr_matrix(Lc)
    W_tilde = _build_W_tilde(Lc)

    beta_v  = np.asarray(params["beta"],  dtype=float)
    rho_v   = np.asarray(params["gamma"], dtype=float)
    D_H_v   = np.asarray(params["D_S"],   dtype=float)
    D_D_v   = np.asarray(params["D_I"],   dtype=float)

    D_max = max(D_H_v.max(), D_D_v.max())
    L_inf = float(abs(Lc).sum(axis=1).max())
    cfl   = dt * D_max * L_inf
    if cfl > 0.5:
        warnings.warn(
            f"simulate_SIR_euler: CFL ≈ {cfl:.2f} > 0.5. "
            "Aumenta n_steps o usa simulate_SIR_ivp.",
            RuntimeWarning, stacklevel=2,
        )

    H = np.asarray(S0, dtype=float).copy()
    D = np.asarray(I0, dtype=float).copy()
    R = np.asarray(R0, dtype=float).copy()

    ts  = np.linspace(0.0, t_end, n_steps + 1)
    H_h = np.empty((nC, n_steps + 1))
    D_h = np.empty((nC, n_steps + 1))
    R_h = np.empty((nC, n_steps + 1))
    H_h[:, 0] = H;  D_h[:, 0] = D;  R_h[:, 0] = R

    for k in range(n_steps):
        H_c = np.clip(H, 0.0, 1.0)
        D_c = np.clip(D, 0.0, 1.0)

        lam     = W_tilde @ D_c
        infect  = beta_v * H_c * lam
        recover = rho_v  * D_c

        dH = -infect              - D_H_v * (Lc @ H_c)
        dD = +infect - recover    - D_D_v * (Lc @ D_c)
        dR = +recover

        H = np.clip(H + dt * dH, 0.0, 1.0)
        D = np.clip(D + dt * dD, 0.0, 1.0)
        R = np.clip(R + dt * dR, 0.0, 1.0)
        H_h[:, k+1] = H;  D_h[:, k+1] = D;  R_h[:, k+1] = R

    return H, D, R, dict(t=ts, S=H_h, I=D_h, R=R_h)


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def run_SIR(
    comps2:             np.ndarray,
    Lc2:                sp.spmatrix,
    boundary_mask:      np.ndarray,
    macro_of:           np.ndarray,
    region_of:          np.ndarray,
    sir_params:         dict  = None,
    seed_I:             float = 0.05,
    seed_boundary_only: bool  = True,
    use_ivp:            bool  = True,
    t_end:              float = 15.0,
    n_steps:            int   = 300,
    method:             str   = "BDF",
) -> tuple:
    """
    Runner completo per il modello H/D/R compartimentale.

    Condizioni iniziali
    -------------------
    seed_boundary_only=True  → D_0[boundary] = seed_I,  H_0 = 1 − D_0
    seed_boundary_only=False → D_0[:] = seed_I
    R_0 = 0 in entrambi i casi.

    Returns
    -------
    S_fin, I_fin, R_fin : ndarray (C,)   (alias H, D, R)
    sol                  : OdeResult o dict (Eulero)
    sir_params           : dict dei parametri usati
    """
    nC = len(comps2)
    if sir_params is None:
        sir_params = build_default_params_SIR(comps2, macro_of, region_of)

    I0 = np.zeros(nC)
    if seed_boundary_only:
        I0[boundary_mask] = seed_I
        seeded_macros = set(macro_of[boundary_mask])
        unseeded      = set(macro_of) - seeded_macros
        if unseeded:
            warnings.warn(
                f"[run_SIR] Macro_type senza seed nella boundary_mask: "
                f"{sorted(unseeded)}. Verranno invasi solo per diffusione. "
                f"If TAL is among these, check the anatomical zoning.",
                UserWarning, stacklevel=2,
            )
    else:
        I0[:] = seed_I

    S0 = np.clip(1.0 - I0, 0.0, 1.0)
    R0 = np.zeros(nC)

    if use_ivp:
        S_fin, I_fin, R_fin, sol = simulate_SIR_ivp(
            Lc2, S0, I0, R0, sir_params,
            t_end=t_end, n_steps=n_steps, method=method,
        )
    else:
        S_fin, I_fin, R_fin, sol = simulate_SIR_euler(
            Lc2, S0, I0, R0, sir_params,
            t_end=t_end, n_steps=n_steps,
        )
    return S_fin, I_fin, R_fin, sol, sir_params


# ═══════════════════════════════════════════════════════════════════════════
# METRICHE
# ═══════════════════════════════════════════════════════════════════════════

def sir_metrics(
    S_fin:       np.ndarray,
    I_fin:       np.ndarray,
    R_fin:       np.ndarray,
    macro_of:    np.ndarray,
    region_of:   np.ndarray,
    sir_params:  dict,
    threshold_I: float = 0.10,
    sol=None,
) -> dict:
    """
    Metriche aggregate per compartimento, macro_type e regione.

    Chiavi restituite
    -----------------
    R0_per_comp  : ndarray (C,) — R_diff = β_i / ρ_i
    mean_R0      : float        — Mean global R_diff
    global_R     : float        — fraction recovered (R final medio)
    global_I     : float        — frazione malata residua
    frac_invaded : float        — frazione compartimenti con D_peak > threshold
    by_macro     : {macro: {mean_I, mean_R, frac_invaded, mean_R0}}
    by_region    : {region: {mean_I, mean_R, frac_invaded}}
    """
    beta_v  = np.asarray(sir_params["beta"],  dtype=float)
    gamma_v = np.asarray(sir_params["gamma"], dtype=float)
    R0_comp = beta_v / np.where(gamma_v > 0, gamma_v, 1e-12)  # R_diff

    if sol is not None:
        nC     = len(I_fin)
        I_peak = sol.y[nC:2*nC, :].max(axis=1)
    else:
        I_peak = I_fin
    invaded = I_peak > threshold_I

    metrics = dict(
        R0_per_comp  = R0_comp,
        mean_R0      = float(R0_comp.mean()),
        global_R     = float(R_fin.mean()),
        global_I     = float(I_fin.mean()),
        frac_invaded = float(invaded.mean()),
    )

    by_macro = {}
    for m in np.unique(macro_of):
        mask = macro_of == m
        by_macro[m] = dict(
            mean_I       = float(I_fin[mask].mean()),
            mean_R       = float(R_fin[mask].mean()),
            frac_invaded = float(invaded[mask].mean()),
            mean_R0      = float(R0_comp[mask].mean()),
        )
    metrics["by_macro"] = by_macro

    by_region = {}
    for rg in np.unique(region_of):
        mask = region_of == rg
        by_region[rg] = dict(
            mean_I       = float(I_fin[mask].mean()),
            mean_R       = float(R_fin[mask].mean()),
            frac_invaded = float(invaded[mask].mean()),
        )
    metrics["by_region"] = by_region

    core_mask = region_of == "core"
    metrics["core_invasion"] = bool(
        I_peak[core_mask].mean() > threshold_I if core_mask.any() else False
    )
    return metrics


def sir_temporal_metrics(
    sol,
    macro_of:      np.ndarray,
    region_of:     np.ndarray,
    sir_params:    dict,
    filter_macro:  tuple = None,
    filter_region: str   = None,
    threshold_I:   float = 0.10,
) -> dict:
    """
    Metriche temporali di D(t) (alias I) per il modello H/D/R.

    Per ogni compartimento nel sottoinsieme selezionato calcola:
    peak_D, t_peak, auc_D, t_crossing, R_diff.

    Parameters
    ----------
    sol           : OdeResult da simulate_SIR_ivp
                    sol.y shape (3C, T): H=[:C], D=[C:2C], R=[2C:3C]
    filter_macro  : tuple o None
    filter_region : str o None
    threshold_I   : soglia per t_crossing e invaded

    Returns
    -------
    dict con peak_I, t_peak, auc_I, t_crossing, R0_per_comp, per_compartment
    """
    t      = sol.t
    nC     = sol.y.shape[0] // 3
    I_traj = sol.y[nC:2*nC, :]   # D(t)

    mask = np.ones(nC, dtype=bool)
    if filter_macro is not None:
        mask &= np.isin(macro_of, list(filter_macro))
    if filter_region is not None:
        mask &= (region_of == filter_region)
    sel_idx = np.where(mask)[0]

    if len(sel_idx) == 0:
        return dict(
            peak_I=0.0, t_peak=None, auc_I=0.0,
            t_crossing=None, R0_per_comp=None,
            per_compartment={}, n_selected=0,
        )

    I_mean     = I_traj[sel_idx, :].mean(axis=0)
    peak_I     = float(I_mean.max())
    t_peak     = float(t[np.argmax(I_mean)])
    auc_I      = float(_trapz(I_mean, t))
    above      = I_mean > threshold_I
    t_crossing = float(t[above][0]) if above.any() else None

    beta_v  = np.asarray(sir_params["beta"],  dtype=float)
    gamma_v = np.asarray(sir_params["gamma"], dtype=float)
    R0_comp = beta_v / np.where(gamma_v > 0, gamma_v, 1e-12)

    per_comp = {}
    for ci in sel_idx:
        I_ci = I_traj[ci, :]
        ab   = I_ci > threshold_I
        per_comp[int(ci)] = dict(
            peak_I     = float(I_ci.max()),
            t_peak     = float(t[np.argmax(I_ci)]),
            auc_I      = float(_trapz(I_ci, t)),
            t_crossing = float(t[ab][0]) if ab.any() else None,
            R0         = float(R0_comp[ci]),
        )

    return dict(
        peak_I          = peak_I,
        t_peak          = t_peak,
        auc_I           = auc_I,
        t_crossing      = t_crossing,
        R0_per_comp     = R0_comp,
        per_compartment = per_comp,
        n_selected      = int(len(sel_idx)),
    )


def sir_temporal_metrics_full(
    sol,
    macro_of:    np.ndarray,
    region_of:   np.ndarray,
    sir_params:  dict,
    threshold_I: float = 0.10,
) -> dict:
    """
    Calcola sir_temporal_metrics per ogni combinazione macro_type × region,
    across global, core, boundary, mid.

    Returns
    -------
    dict con chiavi: "global", "core", "mid", "boundary",
                     "{macro}_core" per ogni macro_type,
                     "per_compartment" (dict unificato)
    """
    results = {}

    results["global"] = sir_temporal_metrics(
        sol, macro_of, region_of, sir_params,
        threshold_I=threshold_I,
    )

    for region in ["core", "mid", "boundary",
                   "cortex", "outer_medulla", "inner_medulla"]:
        results[region] = sir_temporal_metrics(
            sol, macro_of, region_of, sir_params,
            filter_region=region, threshold_I=threshold_I,
        )

    for macro in np.unique(macro_of):
        key = f"{macro}_core"
        results[key] = sir_temporal_metrics(
            sol, macro_of, region_of, sir_params,
            filter_macro=(macro,), filter_region="core",
            threshold_I=threshold_I,
        )

    all_per_comp = {}
    for v in results.values():
        all_per_comp.update(v.get("per_compartment", {}))
    results["per_compartment"] = all_per_comp

    return results


def print_sir_temporal_summary(tm_full: dict,
                                comps2: np.ndarray = None) -> None:
    """Stampa riepilogo leggibile di sir_temporal_metrics_full."""
    keys_global = ["global", "core", "mid", "boundary",
                   "cortex", "outer_medulla", "inner_medulla"]
    print("\n  Metriche temporali H/D/R [compartment-level]:")
    print(f"  {'Sottoinsieme':22s} {'peak_D':8s} {'t_peak':8s} "
          f"{'auc_D':8s} {'t_cross':10s} {'n_comp':6s}")
    print("  " + "-" * 67)

    for k in keys_global:
        m = tm_full.get(k, {})
        if not m or m.get("n_selected", 0) == 0:
            continue
        tc = f"{m['t_crossing']:.1f}" if m["t_crossing"] is not None else "None"
        tp = f"{m['t_peak']:.1f}"     if m["t_peak"]     is not None else "None"
        print(f"  {k:22s} {m['peak_I']:8.3f} {tp:8s} "
              f"{m['auc_I']:8.3f} {tc:10s} {m['n_selected']:6d}")

    print()
    for k in tm_full:
        if not k.endswith("_core") or k in keys_global:
            continue
        m = tm_full[k]
        if m.get("n_selected", 0) == 0:
            continue
        tc = f"{m['t_crossing']:.1f}" if m["t_crossing"] is not None else "None"
        tp = f"{m['t_peak']:.1f}"     if m["t_peak"]     is not None else "None"
        print(f"  {k:22s} {m['peak_I']:8.3f} {tp:8s} "
              f"{m['auc_I']:8.3f} {tc:10s} {m['n_selected']:6d}")

    per_comp = tm_full.get("per_compartment", {})
    if per_comp and comps2 is not None:
        print("\n  Per compartimento:")
        print(f"  {'Nome':25s} {'peak_D':8s} {'t_peak':8s} "
              f"{'auc_D':8s} {'t_cross':10s} {'R_diff':8s}")
        print("  " + "-" * 72)
        for ci, m in sorted(per_comp.items()):
            name = comps2[ci] if ci < len(comps2) else f"comp_{ci}"
            tc   = f"{m['t_crossing']:.1f}" if m["t_crossing"] is not None else "None"
            print(f"  {name:25s} {m['peak_I']:8.3f} {m['t_peak']:8.1f} "
                  f"{m['auc_I']:8.3f} {tc:10s} {m['R0']:8.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZZAZIONE
# ═══════════════════════════════════════════════════════════════════════════

def attach_SIR_to_adata(
    S_fin:    np.ndarray,
    I_fin:    np.ndarray,
    R_fin:    np.ndarray,
    adata_st,
    inv2:     np.ndarray,
    tag:      str = "",
) -> None:
    """
    Espande H/D/R da C compartimenti a N nodi spaziali e attacca
    ad adata_st.obs per sc.pl.spatial.
    Aggiunge: SIR_S{tag}, SIR_I{tag}, SIR_R{tag}
    """
    sfx = f"_{tag}" if tag else ""
    adata_st.obs[f"SIR_S{sfx}"] = S_fin[inv2].astype(np.float32)
    adata_st.obs[f"SIR_I{sfx}"] = I_fin[inv2].astype(np.float32)
    adata_st.obs[f"SIR_R{sfx}"] = R_fin[inv2].astype(np.float32)


def plot_sir_dynamics(
    sol,
    macro_of:         np.ndarray,
    highlight_macros: list  = None,
    figsize:          tuple = (14, 4),
    title:            str   = "",
):
    """
    Traiettorie H/D/R nel tempo, aggregate per macro_type.
    Compatibile con OdeResult (use_ivp=True) e dict Eulero.
    """
    import matplotlib.pyplot as plt

    if hasattr(sol, "t"):
        t   = sol.t
        nC  = sol.y.shape[0] // 3
        S_t = sol.y[:nC,     :]
        I_t = sol.y[nC:2*nC, :]
        R_t = sol.y[2*nC:,   :]
    else:
        t   = sol["t"]
        S_t = sol["S"]
        I_t = sol["I"]
        R_t = sol["R"]

    macros = highlight_macros or list(np.unique(macro_of))
    colors = plt.cm.tab10(np.linspace(0, 1, len(macros)))
    labels = {"S": "H (healthy)", "I": "D (diseased)", "R": "R (recovered)"}

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, (key, traj) in zip(axes, [("S", S_t), ("I", I_t), ("R", R_t)]):
        for color, m in zip(colors, macros):
            idxs = np.where(macro_of == m)[0]
            if len(idxs) == 0:
                continue
            mean = traj[idxs].mean(axis=0)
            std  = traj[idxs].std(axis=0)
            ax.plot(t, mean, label=m, color=color, linewidth=2)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.12)
        ax.set(xlabel="t", ylabel=labels[key],
               title=f"{labels[key]} per macro_type")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle(
        (title or "H/D/R — dinamica temporale") +
        "\n[Eq. 2-4, exposure-driven  |  media ± std per macro_type]",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_R0_spatial(
    sir_params: dict,
    macro_of:   np.ndarray,
    comps2:     np.ndarray,
    adata_st,
    inv2:       np.ndarray,
):
    """
    Mappa spaziale di R_diff = β_i / ρ_i.
    R_diff > 1: susceptibility dominates → amplifier.
    R_diff < 1: recovery domina → assorbente.
    """
    import scanpy as sc

    beta_v  = np.asarray(sir_params["beta"],  dtype=float)
    gamma_v = np.asarray(sir_params["gamma"], dtype=float)
    R0      = beta_v / np.where(gamma_v > 0, gamma_v, 1e-12)

    adata_st.obs["SIR_R0"] = R0[inv2].astype(np.float32)
    sc.pl.spatial(adata_st, color=["SIR_R0"], cmap="RdYlGn_r",
                  title=["R_diff = β/ρ per compartimento  (Eq. 5)"])

    print("\nR_diff per compartimento (ordinato):")
    for i in np.argsort(R0)[::-1]:
        flag = "  ← amplificatore" if R0[i] > 1 else "  ← assorbente"
        print(f"  {comps2[i]:25s}  β={beta_v[i]:.3f}  "
              f"ρ={gamma_v[i]:.3f}  R_diff={R0[i]:.2f}{flag}")


def plot_sir_temporal(
    sol,
    macro_of:    np.ndarray,
    region_of:   np.ndarray,
    sir_params:  dict,
    comps2:      np.ndarray = None,
    threshold_I: float      = 0.10,
    figsize:     tuple      = (16, 5),
):
    """
    Figura dinamica temporale D(t) per il modello H/D/R.
    Pannelli: traiettorie per compartimento, media per region, barchart peak.
    """
    import matplotlib.pyplot as plt

    t      = sol.t
    nC     = sol.y.shape[0] // 3
    I_traj = sol.y[nC:2*nC, :]

    beta_v  = np.asarray(sir_params["beta"],  dtype=float)
    gamma_v = np.asarray(sir_params["gamma"], dtype=float)
    R0_comp = beta_v / np.where(gamma_v > 0, gamma_v, 1e-12)

    macros_uniq = np.unique(macro_of)
    color_map   = {m: plt.cm.tab10(i / max(len(macros_uniq)-1, 1))
                   for i, m in enumerate(macros_uniq)}

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # P1: traiettorie per compartimento
    for ci in range(nC):
        lbl = comps2[ci] if comps2 is not None else f"c{ci}"
        axes[0].plot(t, I_traj[ci, :],
                     color=color_map[macro_of[ci]], lw=1.4, alpha=0.8)
    axes[0].axhline(threshold_I, color="k", ls="--", lw=1.0)
    axes[0].set(xlabel="t", ylabel="D(t)",
                title="D(t) per compartimento\n[exposure-driven, Eq. 3]")

    # P2: media per region
    region_colors = {
        "core":          "#e74c3c", "mid":          "#f39c12",
        "boundary":      "#2980b9", "cortex":       "#8e44ad",
        "outer_medulla": "#c0392b", "inner_medulla": "#16a085",
    }
    for rg, col in region_colors.items():
        mask = region_of == rg
        if mask.sum() == 0:
            continue
        I_rg    = I_traj[mask, :].mean(axis=0)
        pk_idx  = np.argmax(I_rg)
        axes[1].plot(t, I_rg, color=col, lw=2.0, label=rg)
        axes[1].scatter([t[pk_idx]], [I_rg[pk_idx]], color=col, zorder=5, s=50)
        axes[1].annotate(
            f"  pk={I_rg[pk_idx]:.2f}\n  t={t[pk_idx]:.1f}",
            (t[pk_idx], I_rg[pk_idx]), fontsize=7, color=col,
        )
    axes[1].axhline(threshold_I, color="k", ls="--", lw=1.0, alpha=0.6)
    axes[1].set(xlabel="t", ylabel="D media",
                title="D(t) media per region")
    axes[1].legend(fontsize=8)

    # P3: barchart peak_D con R_diff asse duale
    peak_I_comp = I_traj.max(axis=1)
    sort_idx    = np.argsort(peak_I_comp)[::-1]
    names_sorted = ([comps2[i] for i in sort_idx]
                    if comps2 is not None else [f"c{i}" for i in sort_idx])
    bar_colors  = [color_map[macro_of[i]] for i in sort_idx]
    x           = np.arange(len(sort_idx))
    ax3  = axes[2]
    ax3r = ax3.twinx()
    ax3.bar(x, peak_I_comp[sort_idx], color=bar_colors, alpha=0.75)
    ax3r.plot(x, R0_comp[sort_idx], "k--o", ms=4, lw=1.2, label="R_diff")
    ax3r.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax3r.set_ylabel("R_diff = β/ρ", fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("peak D")
    ax3.set_title("Peak D e R_diff per compartimento")
    ax3.axhline(threshold_I, color="k", ls="--", lw=1.0, alpha=0.6)

    fig.suptitle(
        "Dinamica temporale H/D/R  [Eq. 2-4, exposure-driven]\n"
        "[valori compartment-level — broadcast a spot Visium via inv2]",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-DETECTION t_end
# ═══════════════════════════════════════════════════════════════════════════

def suggest_t_end(
    comps2:        np.ndarray,
    Lc2:           sp.spmatrix,
    boundary_mask: np.ndarray,
    macro_of:      np.ndarray,
    region_of:     np.ndarray,
    sir_params:    dict  = None,
    seed_I:        float = 0.05,
    t_probe:       float = 60.0,
    n_steps_probe: int   = 600,
    i_threshold:   float = 1e-3,
    t_min:         float = 5.0,
    t_max:         float = 60.0,
) -> float:
    """
    Suggerisce t_end cercando il primo t in cui D(t) medio scende
    sotto i_threshold nella simulazione pilota.
    """
    if sir_params is None:
        sir_params = build_default_params_SIR(comps2, macro_of, region_of)

    nC = len(comps2)
    I0 = np.zeros(nC);  I0[boundary_mask] = seed_I
    S0 = np.clip(1.0 - I0, 0.0, 1.0)
    R0 = np.zeros(nC)

    _, _, _, sol = simulate_SIR_ivp(
        Lc2, S0, I0, R0, sir_params,
        t_end=t_probe, n_steps=n_steps_probe, method="BDF",
    )

    n_comp = sol.y.shape[0] // 3
    I_mean = sol.y[n_comp:2*n_comp, :].mean(axis=0)

    below  = np.where(I_mean < i_threshold)[0]
    t_end  = float(np.clip(
        sol.t[below[0]] if below.size > 0 else t_max,
        t_min, t_max,
    ))
    print(f"[suggest_t_end] t_end suggerito = {t_end:.2f}")
    return t_end


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES (backward compat)
# ═══════════════════════════════════════════════════════════════════════════

def validate_sir_params_against_literature(sir_params, macro_of):
    """Verifica R_diff per macro_type contro i range biologici IRI murino."""
    PLAUSIBLE_R0 = {
        "PT": (1.5, 5.0), "TAL": (1.0, 3.5), "vascular": (1.2, 4.0),
    }
    beta_v  = sir_params["beta"]
    gamma_v = sir_params["gamma"]
    R0_comp = beta_v / np.where(gamma_v > 0, gamma_v, 1e-12)
    for macro, (lo, hi) in PLAUSIBLE_R0.items():
        mask = macro_of == macro
        if not mask.any():
            continue
        r0_mean = float(R0_comp[mask].mean())
        status  = "✓" if lo <= r0_mean <= hi else "FUORI RANGE"
        print(f"  {macro:12s}  R_diff={r0_mean:.2f}  [{lo},{hi}]  {status}")


def broadcast_sir_with_gene_refinement(
    S_comp, I_comp, R_comp,
    adata_st, inv2,
    stress_gene="havcr1",
):
    """
    Raffina il broadcasting uniforme usando un gene di stress come modulatore.
    D_spot_i = D_comp[inv2[i]] * (1 + α * expr_stress_i_norm)
    """
    I_base    = I_comp[inv2].astype(np.float64)
    var_lower = [v.lower() for v in adata_st.var_names]
    if stress_gene in var_lower:
        idx       = var_lower.index(stress_gene)
        expr      = np.asarray(adata_st.X[:, idx].todense()).flatten()
        expr_norm = (expr - expr.min()) / (expr.max() - expr.min() + 1e-8)
        alpha     = 0.25
        I_spot    = np.clip(I_base * (1 + alpha * expr_norm), 0, 1)
    else:
        I_spot = I_base
    return I_spot

    








