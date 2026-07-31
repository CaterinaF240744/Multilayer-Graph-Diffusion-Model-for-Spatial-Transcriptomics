"""
sir_sensitivity.py
==================
Parametric sensitivity analysis for the compartmental SIR model.

Obiettivo
----------
Dimostrare che le conclusioni qualitative del modello (quali compartimenti
are more vulnerable, the invasion order, the relative role of PT vs
vascular come source) sono ROBUSTE rispetto alle scelte parametriche.
Questo risponde preventivamente alla domanda del revisore:
"Come sono stati scelti i parametri? I risultati cambiano se li modifico?"

Analisi implementate
---------------------
1. one_at_a_time (OAT)
   Varia un parametro alla volta in un range ±p% intorno al valore nominale.
   Misura come cambiano le metriche di output (global_R, frac_invaded,
   invasion_time per macro_type, mean_R0) al variare di ogni parametro.
   Produces tornado plots: which parameters have the most effect.

2. sensitivity_indices (Morris screening)
   Metodo di Morris per stimare l'importanza relativa dei parametri con
   un numero contenuto di simulazioni. Per ogni parametro calcola µ*
   (effetto medio assoluto) e σ (interazione con altri parametri).
   Useful for identifying the most influential parameters before a
   more expensive analysis.

3. monte_carlo_sweep
   Campiona N combinazioni di parametri da distribuzioni uniformi nei
   range specificati. Produce distribuzioni degli output e indici di
   Sobol approssimati (variance-based sensitivity).
   Utile per quantificare l'incertezza totale sulle previsioni.

4. robustness_check
   Verifica specifica per la pubblicazione: dimostra che l'ORDINE dei
   compartments for invasion_time and attack_rate is stable when varying
   the parameters. If the ranking is stable, qualitative conclusions
   sono robuste anche senza calibrazione precisa.

Output
-------
Tutti i risultati sono restituiti come DataFrame pandas e figure matplotlib.
Le figure sono progettate per essere direttamente inseribili in un paper.

Funzioni esportate
-------------------
run_oat_analysis          : analisi one-at-a-time
run_morris_screening      : Morris elementary effects
run_monte_carlo_sweep     : sweep Monte Carlo
run_robustness_check      : ranking stability
plot_tornado              : tornado plot OAT
plot_morris               : scatter µ* vs σ (Morris)
plot_mc_distributions     : distribuzioni output MC
plot_ranking_stability    : ranking stability heatmap
run_full_sensitivity      : runner completo
"""

from __future__ import annotations

import warnings
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# numpy compat
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapezoid


# ═══════════════════════════════════════════════════════════════════════════
# DEFINIZIONE DELLO SPAZIO PARAMETRICO
# ═══════════════════════════════════════════════════════════════════════════

# Parametri da variare e loro range [min_mult, max_mult] rispetto al nominale
# I moltiplicatori sono relativi al valore di default in _MACRO_SIR / _REGION_SIR
PARAM_RANGES: dict[str, tuple[float, float]] = {
    # Parameters β per macro_type
    "beta_PT":       (0.6, 1.6),   # β_PT nominal = 0.40
    "beta_DCT":      (0.6, 1.6),
    "beta_TAL":      (0.6, 1.6),
    "beta_vascular": (0.6, 1.6),

    # Parameters γ per macro_type
    "gamma_PT":       (0.6, 1.6),  # γ_PT nominal = 0.08
    "gamma_DCT":      (0.6, 1.6),
    "gamma_TAL":      (0.6, 1.6),
    "gamma_vascular": (0.6, 1.6),

    # Modulatori di zona
    "beta_mult_outer":  (0.8, 1.6),   # nominale = 1.25
    "beta_mult_inner":  (0.5, 1.2),   # nominale = 0.75
    "gamma_mult_outer": (0.8, 1.4),   # nominale = 1.10
    "gamma_mult_inner": (0.6, 1.2),   # nominale = 0.85

    # Diffusione
    "D_I_global": (0.3, 3.0),         # nominale = 0.20 (default sir_compartments D_I)
    "D_S_global": (0.3, 3.0),         # nominale = 0.05 (default sir_compartments D_S)
}

# Valori nominali (moltiplicatori = 1.0 per tutti)
PARAM_NOMINAL: dict[str, float] = {k: 1.0 for k in PARAM_RANGES}

# Metriche di output da tracciare
OUTPUT_METRICS = [
    "global_R",        # attack rate medio
    "global_I",        # frazione infetta residua
    "global_D",        # danno final (placeholder 0 per SIR puro)
    "frac_invaded",    # frazione compartimenti invasi
    "mean_R0",         # R₀ medio
    "mean_invasion_time",    # tempo medio di invasione
    "pt_attack_rate",        # attack rate PT specifico
    "vascular_attack_rate",  # attack rate vascular
    "pt_invasion_time",      # invasion time PT
]


# ═══════════════════════════════════════════════════════════════════════════
# APPLICAZIONE DEI MOLTIPLICATORI AI PARAMETRI
# ═══════════════════════════════════════════════════════════════════════════

def _apply_multipliers(
    base_params:  dict,
    multipliers:  dict[str, float],
    macro_of:     np.ndarray,
    region_of:    np.ndarray,
) -> dict:
    """
    Applica i moltiplicatori al dizionario di parametri base.

    Parameters
    ----------
    base_params  : output di build_default_params_SIR con valori nominali
    multipliers  : dict {param_name: moltiplicatore} — es. {"beta_PT": 1.2}
    macro_of     : ndarray (C,)
    region_of    : ndarray (C,)

    Returns
    -------
    params_new : dict con stessi campi di base_params, valori modificati
    """
    params = {k: v.copy() if isinstance(v, np.ndarray) else v
              for k, v in base_params.items()}

    beta_v  = params["beta"].copy()
    gamma_v = params["gamma"].copy()
    D_S_v   = params["D_S"].copy()
    D_I_v   = params["D_I"].copy()

    nC = len(beta_v)

    # Moltiplicatori per macro_type
    macro_beta_mults  = {
        "PT":  multipliers.get("beta_PT",       1.0),
        "DCT": multipliers.get("beta_DCT",      1.0),
        "TAL": multipliers.get("beta_TAL",      1.0),
        "vascular": multipliers.get("beta_vascular", 1.0),
    }
    macro_gamma_mults = {
        "PT":  multipliers.get("gamma_PT",       1.0),
        "DCT": multipliers.get("gamma_DCT",      1.0),
        "TAL": multipliers.get("gamma_TAL",      1.0),
        "vascular": multipliers.get("gamma_vascular", 1.0),
    }

    for i, m in enumerate(macro_of):
        beta_v[i]  *= macro_beta_mults.get(m, 1.0)
        gamma_v[i] *= macro_gamma_mults.get(m, 1.0)

    # Modulatori di zona
    zone_beta_mults = {
        "outer_medulla": multipliers.get("beta_mult_outer",  1.0),
        "inner_medulla": multipliers.get("beta_mult_inner",  1.0),
        "boundary":      multipliers.get("beta_mult_outer",  1.0),
    }
    zone_gamma_mults = {
        "outer_medulla": multipliers.get("gamma_mult_outer", 1.0),
        "inner_medulla": multipliers.get("gamma_mult_inner", 1.0),
        "boundary":      multipliers.get("gamma_mult_outer", 1.0),
    }

    for i, rg in enumerate(region_of):
        beta_v[i]  *= zone_beta_mults.get(rg,  1.0)
        gamma_v[i] *= zone_gamma_mults.get(rg, 1.0)

    # Diffusione globale
    d_i_mult = multipliers.get("D_I_global", 1.0)
    d_s_mult = multipliers.get("D_S_global", 1.0)
    D_I_v *= d_i_mult
    D_S_v *= d_s_mult

    params["beta"]  = beta_v
    params["gamma"] = gamma_v
    params["D_S"]   = D_S_v
    params["D_I"]   = D_I_v
    return params


# ═══════════════════════════════════════════════════════════════════════════
# SINGOLA SIMULAZIONE + ESTRAZIONE METRICHE
# ═══════════════════════════════════════════════════════════════════════════

def _run_single(
    comp:         dict,
    base_params:  dict,
    multipliers:  dict[str, float],
    t_end:        float,
    seed_I:       float = 0.10,
    threshold_I:  float = 0.10,
    n_steps:      int   = 200,
) -> dict:
    """
    Esegue una simulazione SIR con i moltiplicatori dati e restituisce
    le metriche di output scalari.
    """
    from sir_compartments import simulate_SIR_ivp

    comps2        = comp["comps2"]
    Lc2           = comp["Lc2"]
    boundary_mask = comp["boundary_mask"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]

    params = _apply_multipliers(base_params, multipliers,
                                macro_of, region_of)

    nC = len(comps2)
    I0 = np.zeros(nC)
    I0[boundary_mask] = seed_I
    S0 = np.clip(1.0 - I0, 0.0, 1.0)
    R0 = np.zeros(nC)

    try:
        S_fin, I_fin, R_fin, sol = simulate_SIR_ivp(
            Lc2, S0, I0, R0, params,
            t_end=t_end, n_steps=n_steps, method="BDF",
            rtol=1e-4, atol=1e-7,   # wider tolerances for speed
        )
    except Exception as e:
        warnings.warn(f"Simulazione fallita: {e}", RuntimeWarning)
        return {m: np.nan for m in OUTPUT_METRICS}

    # Metriche globali
    beta_v  = params["beta"]
    gamma_v = params["gamma"]
    R0_comp = beta_v / np.where(gamma_v > 0, gamma_v, 1e-12)

    I_traj  = sol.y[nC:2*nC, :]
    I_peak  = I_traj.max(axis=1)
    invaded = I_peak > threshold_I

    # Invasion time per compartimento
    t_arr = sol.t
    inv_times = []
    for ci in range(nC):
        above = np.where(I_traj[ci, :] > threshold_I)[0]
        inv_times.append(float(t_arr[above[0]]) if len(above) > 0 else np.nan)
    inv_times = np.array(inv_times)

    # Per macro_type
    pt_mask  = macro_of == "PT"
    vas_mask = macro_of == "vascular"

    out = dict(
        global_R            = float(R_fin.mean()),
        global_I            = float(I_fin.mean()),
        global_D            = float(np.zeros_like(R_fin).mean()),  # SIR non ha D — placeholder 0
        frac_invaded        = float(invaded.mean()),
        mean_R0             = float(R0_comp.mean()),
        mean_invasion_time  = float(np.nanmean(inv_times)),
        pt_attack_rate      = float(R_fin[pt_mask].mean()) if pt_mask.any() else np.nan,
        vascular_attack_rate= float(R_fin[vas_mask].mean()) if vas_mask.any() else np.nan,
        pt_invasion_time    = float(np.nanmean(inv_times[pt_mask])) if pt_mask.any() else np.nan,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 1. ONE-AT-A-TIME (OAT)
# ═══════════════════════════════════════════════════════════════════════════

def run_oat_analysis(
    comp:        dict,
    base_params: dict,
    t_end:       float,
    param_ranges: dict = None,
    n_levels:    int   = 7,
    seed_I:      float = 0.10,
    n_steps:     int   = 200,
    verbose:     bool  = True,
) -> pd.DataFrame:
    """
    Analisi One-At-A-Time: varia un parametro alla volta.

    Per ogni parametro in param_ranges campiona n_levels valori
    equidistanti nel range [min_mult, max_mult] e registra le metriche
    di output. Tutti gli altri parametri restano al valore nominale.

    Parameters
    ----------
    comp         : output di build_compartment2
    base_params  : output di build_default_params_SIR (valori nominali)
    t_end        : float — tempo final simulazione
    param_ranges : dict {nome: (min_mult, max_mult)} — default PARAM_RANGES
    n_levels     : int — numero di livelli per parametro

    Returns
    -------
    pd.DataFrame con colonne:
        param, multiplier, <metriche output>
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    rows = []
    total = len(param_ranges) * n_levels
    done  = 0

    for pname, (lo, hi) in param_ranges.items():
        levels = np.linspace(lo, hi, n_levels)
        for mult in levels:
            mults = {pname: float(mult)}
            out   = _run_single(comp, base_params, mults, t_end,
                                seed_I=seed_I, n_steps=n_steps)
            row = {"param": pname, "multiplier": float(mult)}
            row.update(out)
            rows.append(row)
            done += 1
            if verbose and done % 10 == 0:
                print(f"  OAT: {done}/{total}")

    df = pd.DataFrame(rows)
    print(f"  OAT completato: {len(df)} simulazioni")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. MORRIS SCREENING
# ═══════════════════════════════════════════════════════════════════════════

def run_morris_screening(
    comp:        dict,
    base_params: dict,
    t_end:       float,
    param_ranges: dict = None,
    r:           int   = 10,
    levels:      int   = 4,
    seed_I:      float = 0.10,
    n_steps:     int   = 200,
    verbose:     bool  = True,
) -> pd.DataFrame:
    """
    Morris Elementary Effects method.

    Genera r traiettorie nel spazio parametrico. Per ogni traiettoria,
    calcola l'effetto elementare di ogni parametro:

        EE_i = (y(x + Δe_i) - y(x)) / Δ

    Stima µ* (media degli |EE|) e σ (std degli EE):
    - µ* alto → parametro influente
    - σ alto  → parametro con forti interazioni

    Parameters
    ----------
    r      : int — numero di traiettorie (r × (k+1) simulazioni totali)
    levels : int — numero di livelli per la griglia (tipicamente 4 o 6)

    Returns
    -------
    pd.DataFrame con colonne: param, metric, mu_star, sigma, mu
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    param_names = list(param_ranges.keys())
    k           = len(param_names)
    p           = levels
    delta       = p / (2 * (p - 1))   # passo Morris standard

    # Griglia normalizzata [0,1]
    grid_levels = np.linspace(0, 1, p)

    all_ee: dict[str, dict[str, list]] = {
        pn: {m: [] for m in OUTPUT_METRICS} for pn in param_names
    }

    total_sims = r * (k + 1)
    done = 0

    for _ in range(r):
        # Punto base casuale sulla griglia
        x = np.random.choice(grid_levels, size=k)

        # Trasforma in moltiplicatori reali
        def _to_real(x_norm, pname):
            lo, hi = param_ranges[pname]
            return lo + x_norm * (hi - lo)

        mults_base = {pn: _to_real(x[i], pn) for i, pn in enumerate(param_names)}
        y_base = _run_single(comp, base_params, mults_base, t_end,
                             seed_I=seed_I, n_steps=n_steps)
        done += 1

        # Permutazione casuale degli indici
        perm = np.random.permutation(k)

        x_curr = x.copy()
        y_curr = y_base.copy()

        for idx in perm:
            pname = param_names[idx]
            # Pertuba il parametro idx di ±delta
            sign  = 1.0 if x_curr[idx] <= 0.5 else -1.0
            x_new = x_curr.copy()
            x_new[idx] = np.clip(x_curr[idx] + sign * delta, 0.0, 1.0)

            mults_new = {pn: _to_real(x_new[i], pn)
                         for i, pn in enumerate(param_names)}
            y_new = _run_single(comp, base_params, mults_new, t_end,
                                seed_I=seed_I, n_steps=n_steps)
            done += 1

            # Effetto elementare
            d_real = (_to_real(x_new[idx], pname)
                      - _to_real(x_curr[idx], pname))
            if abs(d_real) > 1e-12:
                for m in OUTPUT_METRICS:
                    v_curr = y_curr.get(m, np.nan)
                    v_new  = y_new.get(m, np.nan)
                    if not (np.isnan(v_curr) or np.isnan(v_new)):
                        ee = (v_new - v_curr) / d_real
                        all_ee[pname][m].append(ee)

            x_curr = x_new
            y_curr = y_new

        if verbose and done % 20 == 0:
            print(f"  Morris: {done}/{total_sims}")

    # Calcola µ*, σ per ogni (param, metric)
    rows = []
    for pname in param_names:
        for m in OUTPUT_METRICS:
            ee_list = all_ee[pname][m]
            if len(ee_list) == 0:
                continue
            ee_arr = np.array(ee_list)
            rows.append(dict(
                param    = pname,
                metric   = m,
                mu_star  = float(np.mean(np.abs(ee_arr))),
                sigma    = float(np.std(ee_arr)),
                mu       = float(np.mean(ee_arr)),
                n_ee     = len(ee_arr),
            ))

    df = pd.DataFrame(rows)
    print(f"  Morris completato: {done} simulazioni")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. MONTE CARLO SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def run_monte_carlo_sweep(
    comp:        dict,
    base_params: dict,
    t_end:       float,
    param_ranges: dict = None,
    n_samples:   int   = 200,
    seed_I:      float = 0.10,
    n_steps:     int   = 200,
    random_seed: int   = 42,
    verbose:     bool  = True,
) -> pd.DataFrame:
    """
    Campiona N combinazioni di parametri da distribuzioni uniformi e
    registra le metriche di output per ciascuna.

    Produce:
    - Distribuzioni degli output (incertezza totale sul modello)
    - Correlazioni di Spearman parametro→output (sensitivity index approssimato)

    Parameters
    ----------
    n_samples : int — numero di simulazioni MC

    Returns
    -------
    pd.DataFrame con n_samples righe, colonne = param_names + output_metrics
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    rng         = np.random.default_rng(random_seed)
    param_names = list(param_ranges.keys())
    rows        = []

    for i in range(n_samples):
        # Campiona uniformemente nel range di ogni parametro
        mults = {
            pn: float(rng.uniform(lo, hi))
            for pn, (lo, hi) in param_ranges.items()
        }
        out  = _run_single(comp, base_params, mults, t_end,
                           seed_I=seed_I, n_steps=n_steps)
        row  = {**mults, **out}
        rows.append(row)

        if verbose and (i+1) % 50 == 0:
            print(f"  MC: {i+1}/{n_samples}")

    df = pd.DataFrame(rows)
    print(f"  MC completato: {len(df)} simulazioni  "
          f"({df[OUTPUT_METRICS].isna().any(axis=1).sum()} fallite)")
    return df


def compute_spearman_indices(df_mc: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman sensitivity indices from a MC DataFrame.

    Returns
    -------
    pd.DataFrame con colonne: param, metric, rho, pvalue
    """
    param_names = [c for c in df_mc.columns if c in PARAM_RANGES]
    rows = []
    for pn in param_names:
        for m in OUTPUT_METRICS:
            mask = df_mc[pn].notna() & df_mc[m].notna()
            if mask.sum() < 10:
                continue
            rho, pval = spearmanr(df_mc.loc[mask, pn], df_mc.loc[mask, m])
            rows.append(dict(param=pn, metric=m, rho=float(rho),
                             pvalue=float(pval)))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ROBUSTNESS CHECK — ranking stability
# ═══════════════════════════════════════════════════════════════════════════

def run_robustness_check(
    comp:        dict,
    base_params: dict,
    t_end:       float,
    param_ranges: dict = None,
    n_samples:   int   = 150,
    seed_I:      float = 0.10,
    n_steps:     int   = 200,
    random_seed: int   = 0,
    verbose:     bool  = True,
) -> dict:
    """
    Verifica che l'ordine dei compartimenti per le metriche chiave sia
    stabile al variare dei parametri.

    Per ogni simulazione MC registra il ranking dei compartimenti per:
    - invasion_time (chi viene invaso prima)
    - attack_rate = R_final (which compartment suffers more damage)

    Restituisce la frequenza con cui ogni compartimento occupa ogni
    posizione del ranking — se un compartimento occupa sempre la stessa
    position, the result is robust.

    Returns
    -------
    dict con chiavi:
        'invasion_time_rank_freq' : DataFrame (C, n_rank_positions)
        'attack_rate_rank_freq'   : DataFrame (C, n_rank_positions)
        'spearman_stability'      : correlazione media tra ranking nominale
                                    e ranking perturbato
        'df_all'                  : DataFrame con tutti i risultati per-compartimento
    """
    from sir_compartments import simulate_SIR_ivp

    if param_ranges is None:
        param_ranges = PARAM_RANGES

    comps2        = comp["comps2"]
    Lc2           = comp["Lc2"]
    boundary_mask = comp["boundary_mask"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]
    nC            = len(comps2)

    rng = np.random.default_rng(random_seed)

    # Simulazione nominale come riferimento
    I0_nom = np.zeros(nC); I0_nom[boundary_mask] = seed_I
    S0_nom = np.clip(1.0 - I0_nom, 0.0, 1.0)
    R0_nom = np.zeros(nC)

    S_nom, I_nom, R_nom, sol_nom = simulate_SIR_ivp(
        Lc2, S0_nom, I0_nom, R0_nom, base_params,
        t_end=t_end, n_steps=n_steps, method="BDF",
        rtol=1e-4, atol=1e-7,
    )

    I_traj_nom = sol_nom.y[nC:2*nC, :]
    t_arr      = sol_nom.t
    threshold  = 0.10

    def _invasion_times(I_traj):
        it = []
        for ci in range(nC):
            above = np.where(I_traj[ci, :] > threshold)[0]
            it.append(float(t_arr[above[0]]) if len(above) > 0 else np.inf)
        return np.array(it)

    inv_nom = _invasion_times(I_traj_nom)
    rank_inv_nom = np.argsort(np.argsort(inv_nom))       # rank invasion
    rank_att_nom = np.argsort(np.argsort(-R_nom))         # rank attack rate

    # Raccoglie ranking per ogni simulazione perturbata
    all_rank_inv = []
    all_rank_att = []
    all_corr_inv = []
    all_corr_att = []

    for i in range(n_samples):
        mults = {pn: float(rng.uniform(lo, hi))
                 for pn, (lo, hi) in param_ranges.items()}
        params_i = _apply_multipliers(base_params, mults, macro_of, region_of)

        try:
            S_i, I_i, R_i, sol_i = simulate_SIR_ivp(
                Lc2, S0_nom, I0_nom, R0_nom, params_i,
                t_end=t_end, n_steps=n_steps, method="BDF",
                rtol=1e-4, atol=1e-7,
            )
        except Exception:
            continue

        I_traj_i   = sol_i.y[nC:2*nC, :]
        inv_i      = _invasion_times(I_traj_i)
        rank_inv_i = np.argsort(np.argsort(inv_i))
        rank_att_i = np.argsort(np.argsort(-R_i))

        all_rank_inv.append(rank_inv_i)
        all_rank_att.append(rank_att_i)

        # Correlazione di Spearman ranking nominale vs perturbato
        rho_inv, _ = spearmanr(rank_inv_nom, rank_inv_i)
        rho_att, _ = spearmanr(rank_att_nom, rank_att_i)
        all_corr_inv.append(float(rho_inv))
        all_corr_att.append(float(rho_att))

        if verbose and (i+1) % 50 == 0:
            print(f"  Robustness: {i+1}/{n_samples}  "
                  f"ρ_inv={np.mean(all_corr_inv):.3f}  "
                  f"ρ_att={np.mean(all_corr_att):.3f}")

    if not all_rank_inv:
        return {}

    # Frequenza di ogni posizione nel ranking per ogni compartimento
    rank_matrix_inv = np.stack(all_rank_inv)   # (n_sim, C)
    rank_matrix_att = np.stack(all_rank_att)

    freq_inv = pd.DataFrame(
        {f"pos_{r}": (rank_matrix_inv == r).mean(axis=0)
         for r in range(nC)},
        index=comps2,
    )
    freq_att = pd.DataFrame(
        {f"pos_{r}": (rank_matrix_att == r).mean(axis=0)
         for r in range(nC)},
        index=comps2,
    )

    spearman_inv = float(np.mean(all_corr_inv))
    spearman_att = float(np.mean(all_corr_att))

    print(f"\n  Robustness check completato: {len(all_rank_inv)} simulazioni")
    print(f"  Spearman invasion_time:  ρ = {spearman_inv:.4f}  "
          f"(1.0 = ranking perfettamente stabile)")
    print(f"  Spearman attack_rate:    ρ = {spearman_att:.4f}")

    return dict(
        invasion_time_rank_freq = freq_inv,
        attack_rate_rank_freq   = freq_att,
        spearman_invasion       = spearman_inv,
        spearman_attack         = spearman_att,
        corr_inv_series         = np.array(all_corr_inv),
        corr_att_series         = np.array(all_corr_att),
        rank_inv_nominal        = rank_inv_nom,
        rank_att_nominal        = rank_att_nom,
        comps2                  = comps2,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def plot_tornado(
    df_oat:   pd.DataFrame,
    metric:   str   = "global_R",
    top_n:    int   = 12,
    figsize:  tuple = (9, 6),
    title:    str   = None,
) -> plt.Figure:
    """
    Tornado plot: per ogni parametro mostra il range di variazione
    dell'output al variare del parametro nel suo range.

    I parametri sono ordinati per ampiezza del range (quelli che
    that change the output most at the top).
    """
    nominal = df_oat.groupby("param")[metric].apply(
        lambda x: x.iloc[len(x)//2]
    )

    ranges = {}
    for pname, grp in df_oat.groupby("param"):
        vals = grp[metric].dropna()
        if len(vals) < 2:
            continue
        ranges[pname] = (float(vals.min()), float(vals.max()))

    if not ranges:
        print(f"  [plot_tornado] nessun dato per metric='{metric}'")
        return None

    # Ordina per ampiezza
    sorted_params = sorted(ranges, key=lambda p: ranges[p][1] - ranges[p][0],
                           reverse=True)[:top_n]

    nom_val = df_oat[df_oat["multiplier"].between(0.95, 1.05)].groupby("param")[metric].mean()

    fig, ax = plt.subplots(figsize=figsize)
    colors_neg = "#d6604d"
    colors_pos = "#4393c3"

    for yi, pname in enumerate(reversed(sorted_params)):
        lo, hi  = ranges[pname]
        nom     = float(nom_val.get(pname, (lo+hi)/2))
        left    = min(lo, hi) - nom
        right   = max(lo, hi) - nom

        ax.barh(yi, right - left, left=left + nom,
                color=colors_pos if right > 0 else colors_neg,
                alpha=0.75, height=0.6, edgecolor="white", lw=0.5)
        # Valori min/max
        ax.text(lo - 0.002, yi, f"{lo:.3f}", ha="right", va="center",
                fontsize=7.5, color="#333333")
        ax.text(hi + 0.002, yi, f"{hi:.3f}", ha="left",  va="center",
                fontsize=7.5, color="#333333")

    ax.set_yticks(range(len(sorted_params)))
    ax.set_yticklabels(list(reversed(sorted_params)), fontsize=9)
    ax.axvline(float(nom_val.mean()), color="k", lw=1.2, ls="--", alpha=0.5,
               label="valore nominale")
    ax.set_xlabel(f"Output: {metric}", fontsize=11)
    ax.set_title(title or f"Tornado plot — sensitivity of {metric}\n"
                 f"(ogni barra = range output al variare del parametro)",
                 fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_morris(
    df_morris: pd.DataFrame,
    metric:    str   = "global_R",
    figsize:   tuple = (7, 6),
    title:     str   = None,
) -> plt.Figure:
    """
    Scatter µ* vs σ (Morris plot).

    Quadranti:
    - µ* alto, σ basso  → effetto lineare importante
    - µ* alto, σ alto   → effetto non lineare / interazioni forti
    - µ* basso          → parametro poco influente
    """
    sub = df_morris[df_morris["metric"] == metric].copy()
    if sub.empty:
        print(f"  [plot_morris] nessun dato per metric='{metric}'")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    max_mu = sub["mu_star"].max() + 1e-9
    sizes  = 30 + 200 * (sub["mu_star"] / max_mu)
    sc = ax.scatter(sub["mu_star"], sub["sigma"],
                    s=sizes, c=sub["mu_star"], cmap="YlOrRd",
                    edgecolors="k", linewidths=0.4, zorder=3)

    for _, row in sub.iterrows():
        ax.annotate(row["param"], (row["mu_star"], row["sigma"]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7, color="#333333")

    # Linea µ* = σ (soglia interazioni)
    lim = max(sub["mu_star"].max(), sub["sigma"].max()) * 1.15
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4,
            label="µ* = σ")

    fig.colorbar(sc, ax=ax, label="µ* (effetto medio assoluto)")
    ax.set_xlabel("µ*  (importanza)", fontsize=11)
    ax.set_ylabel("σ   (interactions / non-linearities)", fontsize=11)
    ax.set_title(title or f"Morris screening — {metric}\n"
                 "(in alto a destra = parametro importante e non-lineare)",
                 fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_mc_distributions(
    df_mc:   pd.DataFrame,
    metrics: list = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Distribuzione degli output dal sweep Monte Carlo.
    Ogni pannello mostra l'istogramma di una metrica con
    mean and 90% credibility interval.
    """
    if metrics is None:
        metrics = [m for m in OUTPUT_METRICS if df_mc[m].notna().sum() > 10]

    n = len(metrics)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize, squeeze=False)

    for idx, metric in enumerate(metrics):
        ax   = axes[idx // ncols][idx % ncols]
        vals = df_mc[metric].dropna()
        if len(vals) == 0:
            ax.axis("off"); continue

        ax.hist(vals, bins=25, color="#4393c3", edgecolor="white",
                alpha=0.8, density=True)

        mu  = float(vals.mean())
        p5  = float(vals.quantile(0.05))
        p95 = float(vals.quantile(0.95))

        ax.axvline(mu,  color="#d6604d", lw=1.8, label=f"media={mu:.3f}")
        ax.axvline(p5,  color="#888888", lw=1.0, ls="--")
        ax.axvline(p95, color="#888888", lw=1.0, ls="--",
                   label=f"90% CI [{p5:.3f}, {p95:.3f}]")
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("valore", fontsize=8)
        ax.legend(fontsize=7)

    # Spegni pannelli vuoti
    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        "Distribuzione degli output — sweep Monte Carlo\n"
        f"N={len(df_mc)} campionamenti uniformi nello spazio parametrico",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


def plot_spearman_heatmap(
    df_spearman: pd.DataFrame,
    figsize:     tuple = (13, 6),
    top_n_params: int  = 12,
) -> plt.Figure:
    """
    Heatmap degli indici di Spearman parametro × metrica.
    Shows which parameters most influence which metrics.
    """
    pivot = df_spearman.pivot(index="param", columns="metric", values="rho")

    # Ordina per importanza assoluta media
    row_importance = pivot.abs().mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[row_importance.index[:top_n_params]]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(val) > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Spearman ρ  (parametro → output)",
                 shrink=0.8)
    ax.set_xlabel("Metrica di output", fontsize=11)
    ax.set_ylabel("Parametro", fontsize=11)
    ax.set_title(
        "Spearman sensitivity indices\n"
        "(rosso = correlazione positiva  |  blu = correlazione negativa  |  "
        "righe ordinate per importanza assoluta)",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


def plot_ranking_stability(
    rob: dict,
    top_n: int = 8,
    figsize: tuple = (13, 5),
) -> plt.Figure:
    """
    Heatmap of ranking stability.
    Per ogni compartimento mostra la frequenza con cui occupa
    le prime top_n posizioni nel ranking di invasion_time e attack_rate.
    A compartment with high frequency at the same position is robust.
    """
    if not rob:
        return None

    comps2   = rob["comps2"]
    freq_inv = rob["invasion_time_rank_freq"]
    freq_att = rob["attack_rate_rank_freq"]
    rho_inv  = rob["spearman_invasion"]
    rho_att  = rob["spearman_attack"]

    # Select top_n compartments by relevance (those with lowest ranking)
    nom_rank_inv = rob["rank_inv_nominal"]
    nom_rank_att = rob["rank_att_nominal"]
    top_idx_inv  = np.argsort(nom_rank_inv)[:top_n]
    top_idx_att  = np.argsort(nom_rank_att)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, freq, top_idx, rho, title_sfx in [
        (axes[0], freq_inv, top_idx_inv, rho_inv, "Invasion time"),
        (axes[1], freq_att, top_idx_att, rho_att, "Attack rate"),
    ]:
        sub   = freq.iloc[top_idx, :top_n]
        names = [comps2[i] for i in top_idx]

        im = ax.imshow(sub.values, cmap="YlOrRd", vmin=0, vmax=1,
                       aspect="auto")
        ax.set_xticks(range(top_n))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels([f"#{r+1}" for r in range(top_n)], fontsize=9)
        ax.set_yticklabels(names, fontsize=9)

        for i in range(len(names)):
            for j in range(top_n):
                val = sub.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if val > 0.5 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Frequenza")
        ax.set_xlabel("Posizione nel ranking", fontsize=10)
        ax.set_title(f"{title_sfx}\nSpearman ρ = {rho:.3f}", fontsize=10)

    fig.suptitle(
        "Compartment ranking stability\n"
        f"(N={len(rob['corr_inv_series'])} perturbazioni parametriche uniformi)\n"
        "Ogni cella = frequenza con cui il compartimento occupa quella posizione",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


def plot_optimal_vs_nominal_curves(
    comp:        dict,
    base_params: dict,
    sweep2d:     dict,
    t_end:       float,
    seed_I:      float = 0.10,
    n_steps:     int   = 300,
    figsize:     tuple = (20, 14),
    save_prefix: str   = None,
) -> plt.Figure:
    """
    Confronto curve temporali I(t), D(t), R(t) per macro_type
    tra scenario nominale e scenario ottimale identificato dallo sweep 2D.

    The optimal scenario is the grid point with minimum global_R
    (gamma_mult_outer massimo, beta_DCT minimo).

    Layout: 2 righe × 3 colonne
      - riga 0 = scenario nominale  (moltiplicatori = 1.0)
      - riga 1 = scenario ottimale  (gamma max, beta_DCT min)
      - col 0  = I(t) — infetti
      - col 1  = D(t) — danno (solo se disponibile, altrimenti R)
      - col 2  = R(t) — recovered

    In ogni cella una curva per macro_type.
    Linea verticale grigia = t_peak_I del nominale per riferimento.

    Parameters
    ----------
    comp, base_params : output di build_compartment2 e build_default_params_SIR
    sweep2d           : output di run_2d_param_sweep
    t_end             : durata simulazione
    seed_I            : seme iniziale
    n_steps           : ODE steps (higher than sweep for smooth curves)
    """
    from sir_compartments import simulate_SIR_ivp

    comps2        = comp["comps2"]
    Lc2           = comp["Lc2"]
    boundary_mask = comp["boundary_mask"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]
    nC            = len(comps2)

    MACRO_COLORS = {
        "PT":       "#27ae60",
        "DCT":      "#2471a3",
        "TAL":      "#922b21",
        "vascular": "#00bcd4",
        "other":    "#888888",
        "immune":   "#e67e22",
    }

    # ── Identifica scenario ottimale dalla griglia ────────────────────────
    gamma_grid = sweep2d["gamma_grid"]
    beta_grid  = sweep2d["beta_grid"]

    # Punto con global_R minimo
    gi_opt, bi_opt = np.unravel_index(
        np.nanargmin(sweep2d["global_R"]),
        sweep2d["global_R"].shape,
    )
    gm_opt = float(gamma_grid[gi_opt])
    bm_opt = float(beta_grid[bi_opt])

    # Punto nominale
    gi_nom = int(np.abs(gamma_grid - 1.0).argmin())
    bi_nom = int(np.abs(beta_grid  - 1.0).argmin())
    R_nom_val = float(sweep2d["global_R"][gi_nom, bi_nom])
    R_opt_val = float(sweep2d["global_R"][gi_opt, bi_opt])

    print(f"  [curves] Nominale:  γ×1.0,  β_DCT×1.0  → global_R={R_nom_val:.3f}")
    print(f"  [curves] Ottimale:  γ×{gm_opt:.2f}, β_DCT×{bm_opt:.2f} "
          f"→ global_R={R_opt_val:.3f}")

    # ── Helper: simula e restituisce sol ─────────────────────────────────
    def _simulate(mults):
        params = _apply_multipliers(base_params, mults, macro_of, region_of)
        I0 = np.zeros(nC)
        I0[boundary_mask] = seed_I
        S0 = np.clip(1.0 - I0, 0.0, 1.0)
        R0v = np.zeros(nC)
        _, _, _, sol = simulate_SIR_ivp(
            Lc2, S0, I0, R0v, params,
            t_end=t_end, n_steps=n_steps,
            method="BDF", rtol=1e-5, atol=1e-8,
        )
        return sol

    print("  [curves] Simulazione nominale...")
    sol_nom = _simulate({"gamma_mult_outer": 1.0, "beta_DCT": 1.0})

    print("  [curves] Simulazione ottimale...")
    sol_opt = _simulate({"gamma_mult_outer": gm_opt, "beta_DCT": bm_opt})

    # t_peak_I nominale
    I_mean_nom = sol_nom.y[nC:2*nC, :].mean(axis=0)
    t_peak_nom = float(sol_nom.t[np.argmax(I_mean_nom)])

    # ── Helper: media per macro_type ──────────────────────────────────────
    macros_present = list(dict.fromkeys(
        m for m in macro_of if m in MACRO_COLORS
    ))

    def _macro_traj(sol, block_idx):
        traj = sol.y[block_idx*nC:(block_idx+1)*nC, :]
        return {
            m: traj[macro_of == m, :].mean(axis=0)
            for m in macros_present
            if (macro_of == m).sum() > 0
        }

    # ── Figura 2 × 3 ─────────────────────────────────────────────────────
    scenarios = [
        (sol_nom, "Nominale\n(γ×1.0, β_DCT×1.0)",  "#555555",
         f"global_R={R_nom_val:.3f}"),
        (sol_opt, f"Ottimale\n(γ×{gm_opt:.2f}, β_DCT×{bm_opt:.2f})",
         "#2980b9",
         f"global_R={R_opt_val:.3f}  (−{R_nom_val-R_opt_val:.3f})"),
    ]

    col_vars = [
        (1, "I(t) — infetti/infiammati"),
        (2, "R(t) — recovered"),
        (0, "S(t) — suscettibili"),
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=figsize,
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )

    y_maxes = {col: 0.0 for col in range(3)}

    for row, (sol, row_label, row_color, row_note) in enumerate(scenarios):
        t_vec = sol.t
        for col, (block_idx, col_title) in enumerate(col_vars):
            ax   = axes[row][col]
            traj = _macro_traj(sol, block_idx)

            for m, vals in traj.items():
                color = MACRO_COLORS.get(m, "#555555")
                ax.plot(t_vec, vals,
                        color=color, lw=2.0, alpha=0.90,
                        label=m if row == 0 else "_")
                y_maxes[col] = max(y_maxes[col], float(vals.max()))

            # Linea t_peak nominale
            ax.axvline(t_peak_nom, color="#aaaaaa",
                       lw=1.0, ls="--", alpha=0.6,
                       label=f"t_peak nom ({t_peak_nom:.1f})"
                       if row == 0 and col == 0 else "_")

            ax.set_xlim(0, t_end)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("t", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)

            if row == 0:
                ax.set_title(col_title, fontsize=10,
                             fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(
                    f"{row_label}\n{row_note}",
                    fontsize=8, labelpad=8,
                    color=row_color, fontweight="bold",
                )

    # Uniforma scala Y per ogni colonna
    for col in range(3):
        for row in range(2):
            axes[row][col].set_ylim(0, min(y_maxes[col] * 1.12, 1.05))

    # Legenda macro_type (prima cella)
    macro_lines = [
        Line2D([0], [0], color=MACRO_COLORS.get(m, "#555"),
               lw=2.0, label=m)
        for m in macros_present
    ]
    peak_line = Line2D([0], [0], color="#aaaaaa", lw=1.0,
                       ls="--", label=f"t_peak nom ({t_peak_nom:.1f})")
    fig.legend(
        handles=macro_lines + [peak_line],
        loc="lower center",
        ncol=len(macro_lines) + 1,
        fontsize=8, framealpha=0.95,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        "Confronto curve temporali — scenario nominale vs ottimale\n"
        "[identified by 2D sweep gamma_mult_outer × beta_DCT]",
        fontsize=12, y=1.01,
    )

    if save_prefix:
        sp_ = f"{save_prefix}_optimal_vs_nominal_curves.png"
        fig.savefig(sp_, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {sp_}")

    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SWEEP 2D — gamma_mult_outer × beta_DCT
# ═══════════════════════════════════════════════════════════════════════════

def run_2d_param_sweep(
    comp:         dict,
    base_params:  dict,
    t_end:        float,
    n_gamma:      int   = 12,
    n_beta:       int   = 12,
    gamma_range:  tuple = (0.7, 2.0),
    beta_range:   tuple = (0.6, 1.8),
    seed_I:       float = 0.10,
    n_steps:      int   = 150,
    threshold_I:  float = 0.10,
) -> dict:
    """
    Sweep 2D su gamma_mult_outer × beta_DCT.

    Per ogni coppia (gamma_mult, beta_DCT_mult) esegue una simulazione
    SIR e raccoglie le metriche di output. Produce una griglia di
    n_gamma × n_beta simulazioni.

    The two parameters were identified by the sensitivity analysis
    as the most influential with opposite effects:
      - beta_DCT   aumenta global_R (ρ=+0.458)
      - gamma_mult_outer  lo diminuisce  (ρ=−0.273)

    Parameters
    ----------
    comp, base_params : output di build_compartment2 e build_default_params_SIR
    t_end             : durata simulazione
    n_gamma           : numero di livelli per gamma_mult_outer
    n_beta            : numero di livelli per beta_DCT
    gamma_range       : (min_mult, max_mult) per gamma_mult_outer
    beta_range        : (min_mult, max_mult) per beta_DCT
    seed_I            : frazione iniziale infetti ai boundary
    n_steps           : passi ODE per ogni simulazione
    threshold_I       : soglia per invasion_time

    Returns
    -------
    dict con chiavi:
        gamma_grid   : ndarray (n_gamma,) — valori moltiplicatore gamma
        beta_grid    : ndarray (n_beta,)  — valori moltiplicatore beta_DCT
        global_R     : ndarray (n_gamma, n_beta) — attack rate final
        global_D     : ndarray (n_gamma, n_beta) — danno final medio
        frac_invaded : ndarray (n_gamma, n_beta) — frazione compartimenti invasi
        mean_inv_time: ndarray (n_gamma, n_beta) — tempo medio invasione
        df           : DataFrame long-format per analisi successiva
    """
    gamma_grid = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    beta_grid  = np.linspace(beta_range[0],  beta_range[1],  n_beta)

    macro_of  = comp["macro_of"]
    region_of = comp["region_of"]

    global_R      = np.full((n_gamma, n_beta), np.nan)
    global_D      = np.full((n_gamma, n_beta), np.nan)
    frac_invaded  = np.full((n_gamma, n_beta), np.nan)
    mean_inv_time = np.full((n_gamma, n_beta), np.nan)

    rows = []
    total = n_gamma * n_beta
    done  = 0

    print(f"  [2D sweep] {total} simulazioni "
          f"({n_gamma} gamma × {n_beta} beta_DCT)...")

    for gi, gm in enumerate(gamma_grid):
        for bi, bm in enumerate(beta_grid):
            mults = {
                "gamma_mult_outer": gm,
                "beta_DCT":         bm,
            }
            try:
                res = _run_single(
                    comp, base_params, mults, t_end,
                    seed_I=seed_I,
                    threshold_I=threshold_I,
                    n_steps=n_steps,
                )
                global_R[gi, bi]      = res.get("global_R",          np.nan)
                global_D[gi, bi]      = res.get("global_D",          np.nan)
                frac_invaded[gi, bi]  = res.get("frac_invaded",       np.nan)
                mean_inv_time[gi, bi] = res.get("mean_invasion_time", np.nan)
                rows.append({
                    "gamma_mult_outer": gm,
                    "beta_DCT_mult":    bm,
                    **res,
                })
            except Exception as e:
                warnings.warn(f"[2D sweep] errore a gamma={gm:.2f} "
                              f"beta={bm:.2f}: {e}")

            done += 1
            if done % 20 == 0 or done == total:
                print(f"    {done}/{total} completate")

    df = pd.DataFrame(rows)
    return dict(
        gamma_grid    = gamma_grid,
        beta_grid     = beta_grid,
        global_R      = global_R,
        global_D      = global_D,
        frac_invaded  = frac_invaded,
        mean_inv_time = mean_inv_time,
        df            = df,
    )


def plot_2d_sweep(
    sweep2d:     dict,
    base_params: dict,
    comp:        dict,
    figsize:     tuple = (20, 14),
    save_prefix: str   = None,
) -> plt.Figure:
    """
    Figura 2×2 con heatmap della griglia 2D gamma_mult_outer × beta_DCT.

    Pannelli:
      (0,0) global_R      — attack rate final
      (0,1) global_D      — danno irreversibile final
      (1,0) frac_invaded  — frazione compartimenti invasi
      (1,1) mean_inv_time — tempo medio di invasione

    Su ogni heatmap:
      - Stella nera = punto nominale (moltiplicatori = 1.0)
      - Contorno bianco = isovalue corrispondente al nominale
      - Frecce nelle label degli assi indicano direzione protettiva

    Parameters
    ----------
    sweep2d     : output di run_2d_param_sweep
    base_params : parametri nominali (per marcatore punto base)
    comp        : output di build_compartment2
    """
    gamma_grid = sweep2d["gamma_grid"]
    beta_grid  = sweep2d["beta_grid"]

    # global_D non esiste nel SIR puro — usiamo (1 - global_R) come proxy
    # of "potential damage" = fraction not recovered at final time
    damage_proxy = np.where(
        np.isnan(sweep2d["global_R"]),
        np.nan,
        1.0 - sweep2d["global_R"],
    )

    panels = [
        ("global_R",     sweep2d["global_R"],
         "Attack rate final (global_R)",
         "YlOrRd", None, None),
        ("damage_proxy", damage_proxy,
         "Danno potenziale (1 − global_R)",
         "Reds",   None, None),
        ("frac_invaded", sweep2d["frac_invaded"],
         "Frazione compartimenti invasi",
         "YlOrRd", 0.0,  1.0),
        ("mean_inv_time", sweep2d["mean_inv_time"],
         "Time medio di invasione",
         "YlGn_r", None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    for idx, (key, data, title, cmap, vmin, vmax) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]

        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)

        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            extent=[beta_grid[0], beta_grid[-1],
                    gamma_grid[0], gamma_grid[-1]],
        )
        fig.colorbar(im, ax=ax, shrink=0.80, pad=0.02)

        # Contorni iso-valore
        try:
            cs = ax.contour(
                beta_grid, gamma_grid, data,
                levels=7, colors="white",
                linewidths=0.7, alpha=0.55,
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f",
                      colors="white")
        except Exception:
            pass

        # Punto nominale (moltiplicatori = 1.0)
        ax.plot(1.0, 1.0, marker="*", markersize=14,
                color="black", zorder=6,
                label="nominal (×1.0)")

        # Isocontorno del valore nominale
        nom_val = float(np.nanmean(
            data[np.abs(gamma_grid - 1.0).argmin(),
                 np.abs(beta_grid  - 1.0).argmin():]
            [:1]
        ))
        try:
            ax.contour(
                beta_grid, gamma_grid, data,
                levels=[nom_val],
                colors=["black"], linewidths=1.5,
                linestyles=["--"],
            )
        except Exception:
            pass

        ax.set_xlabel(
            "beta_DCT mult  →  more aggressive",
            fontsize=9,
        )
        ax.set_ylabel(
            "gamma_mult_outer  →  more protective",
            fontsize=9,
        )
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    fig.suptitle(
        "2D sweep: gamma_mult_outer × beta_DCT\n"
        "[stella = punto nominale  |  -- = isocontorno nominale  |"
        "  colore = valore metrica]",
        fontsize=12, y=1.01,
    )

    if save_prefix:
        sp_ = f"{save_prefix}_2d_sweep.png"
        fig.savefig(sp_, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {sp_}")

    plt.tight_layout()
    plt.show()
    plt.close()
    return fig


def plot_2d_sweep_spatial(
    sweep2d:     dict,
    comp:        dict,
    adata_st,
    t_end:       float,
    base_params: dict,
    n_steps:     int   = 200,
    seed_I:      float = 0.10,
    save_prefix: str   = None,
) -> None:
    """
    Mappa spaziale sul tessuto Visium della differenza di danno D
    tra scenario nominale e scenario ottimale (gamma_mult_outer massimo,
    beta_DCT minimo).

    Mostra dove nel tessuto il farmaco ipotetico (modulazione di
    gamma_mult_outer) avrebbe il massimo effetto protettivo.

    Parameters
    ----------
    sweep2d     : output di run_2d_param_sweep
    comp        : output di build_compartment2
    adata_st    : AnnData Visium con obsm['spatial']
    t_end       : durata simulazione
    base_params : parametri nominali
    n_steps     : passi ODE
    seed_I      : seme iniziale
    """
    import scanpy as sc
    from sir_compartments import run_SIR, attach_SIR_to_adata

    comps2        = comp["comps2"]
    Lc2           = comp["Lc2"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]
    boundary_mask = comp["boundary_mask"]
    inv2          = comp["inv2"]

    gamma_grid = sweep2d["gamma_grid"]
    beta_grid  = sweep2d["beta_grid"]

    # Scenario ottimale: gamma_mult_outer massimo, beta_DCT minimo
    gi_opt = len(gamma_grid) - 1
    bi_opt = 0
    gm_opt = float(gamma_grid[gi_opt])
    bm_opt = float(beta_grid[bi_opt])

    print(f"  [spatial] Scenario ottimale: "
          f"gamma_mult={gm_opt:.2f}, beta_DCT_mult={bm_opt:.2f}")
    print(f"  [spatial] global_R ottimale: "
          f"{sweep2d['global_R'][gi_opt, bi_opt]:.3f}  "
          f"(nominale: {sweep2d['global_R'][np.abs(gamma_grid-1.0).argmin(), np.abs(beta_grid-1.0).argmin()]:.3f})")

    # Simulazione nominale
    mults_nom = {"gamma_mult_outer": 1.0, "beta_DCT": 1.0}
    params_nom = _apply_multipliers(base_params, mults_nom, macro_of, region_of)
    S_nom, I_nom, R_nom, _, _ = run_SIR(
        comps2, Lc2, boundary_mask, macro_of, region_of,
        sir_params=params_nom, seed_I=seed_I,
        seed_boundary_only=True, use_ivp=True,
        t_end=t_end, n_steps=n_steps,
    )

    # Simulazione ottimale
    mults_opt = {"gamma_mult_outer": gm_opt, "beta_DCT": bm_opt}
    params_opt = _apply_multipliers(base_params, mults_opt, macro_of, region_of)
    S_opt, I_opt, R_opt, _, _ = run_SIR(
        comps2, Lc2, boundary_mask, macro_of, region_of,
        sir_params=params_opt, seed_I=seed_I,
        seed_boundary_only=True, use_ivp=True,
        t_end=t_end, n_steps=n_steps,
    )

    # Differenza R (beneficio) e differenza I (riduzione infetti)
    delta_R = R_opt - R_nom   # positive = more recovered with optimal params
    delta_I = I_nom - I_opt   # positivo = meno infetti con param ottimali

    # Espandi agli spot Visium
    delta_R_spot = delta_R[inv2].astype(np.float32)
    delta_I_spot = delta_I[inv2].astype(np.float32)
    R_nom_spot   = R_nom[inv2].astype(np.float32)
    R_opt_spot   = R_opt[inv2].astype(np.float32)

    adata_st.obs["SWEEP_delta_R"]  = delta_R_spot
    adata_st.obs["SWEEP_delta_I"]  = delta_I_spot
    adata_st.obs["SWEEP_R_nom"]    = R_nom_spot
    adata_st.obs["SWEEP_R_opt"]    = R_opt_spot

    sc.pl.spatial(
        adata_st,
        color=["SWEEP_R_nom", "SWEEP_R_opt"],
        cmap="viridis",
        title=[
            "R final — nominale",
            f"R final — ottimale (γ×{gm_opt:.1f}, β_DCT×{bm_opt:.1f})",
        ],
        show=True,
    )
    sc.pl.spatial(
        adata_st,
        color=["SWEEP_delta_R", "SWEEP_delta_I"],
        cmap="RdYlGn",
        title=[
            "ΔR = R_opt − R_nom  (verde = beneficio)",
            "ΔI = I_nom − I_opt  (verde = riduzione infetti)",
        ],
        show=True,
    )

    if save_prefix:
        # Salva anche come figura matplotlib per controllo
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, vals, title, cmap in [
            (axes[0], delta_R_spot,
             f"ΔR (ottimale − nominale)\nγ×{gm_opt:.1f} β_DCT×{bm_opt:.1f}",
             "RdYlGn"),
            (axes[1], delta_I_spot,
             "ΔI (riduzione infetti)",
             "RdYlGn"),
        ]:
            coords = adata_st.obsm["spatial"]
            sc_ = ax.scatter(coords[:, 0], coords[:, 1],
                             c=vals, cmap=cmap, s=4, alpha=0.8)
            plt.colorbar(sc_, ax=ax, shrink=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.suptitle("Mappa spaziale beneficio sweep 2D", fontsize=11)
        plt.tight_layout()
        sp_ = f"{save_prefix}_2d_sweep_spatial.png"
        fig.savefig(sp_, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {sp_}")
        plt.close()


def run_2d_sweep_analysis(
    comp:         dict,
    base_params:  dict,
    adata_st,
    t_end:        float,
    n_gamma:      int   = 12,
    n_beta:       int   = 12,
    gamma_range:  tuple = (0.7, 2.0),
    beta_range:   tuple = (0.6, 1.8),
    seed_I:       float = 0.10,
    n_steps:      int   = 150,
    save_prefix:  str   = None,
) -> dict:
    """
    Runner completo per lo sweep 2D gamma_mult_outer × beta_DCT.

    Esegue in sequenza:
    1. Griglia 2D di simulazioni SIR
    2. Heatmap 2×2 con le 4 metriche principali
    3. Mappa spaziale della differenza di danno sul tessuto Visium

    Parameters
    ----------
    comp, base_params : output di build_compartment2 e build_default_params_SIR
    adata_st          : AnnData Visium
    t_end             : durata simulazione
    n_gamma, n_beta   : dimensioni della griglia
    gamma_range       : range moltiplicatore gamma_mult_outer
    beta_range        : range moltiplicatore beta_DCT
    seed_I            : seme iniziale
    n_steps           : passi ODE per ogni simulazione della griglia
    save_prefix       : prefisso per salvare le figure

    Returns
    -------
    dict con chiavi:
        sweep2d : output di run_2d_param_sweep
        fig_2d  : figura heatmap
    """
    print(f"\n{'='*60}")
    print("SWEEP 2D — gamma_mult_outer × beta_DCT")
    print(f"{'='*60}")
    print(f"  Grid: {n_gamma} × {n_beta} = {n_gamma*n_beta} simulations")
    print(f"  gamma_mult_outer: [{gamma_range[0]:.2f}, {gamma_range[1]:.2f}]")
    print(f"  beta_DCT mult:    [{beta_range[0]:.2f},  {beta_range[1]:.2f}]")

    # ── 1. Griglia simulazioni ────────────────────────────────────────────
    sweep2d = run_2d_param_sweep(
        comp, base_params, t_end,
        n_gamma=n_gamma, n_beta=n_beta,
        gamma_range=gamma_range, beta_range=beta_range,
        seed_I=seed_I, n_steps=n_steps,
    )

    # ── Riepilogo testuale ────────────────────────────────────────────────
    gamma_grid = sweep2d["gamma_grid"]
    beta_grid  = sweep2d["beta_grid"]
    gi_nom = np.abs(gamma_grid - 1.0).argmin()
    bi_nom = np.abs(beta_grid  - 1.0).argmin()
    gi_opt = len(gamma_grid) - 1
    bi_opt = 0

    R_nom  = sweep2d["global_R"][gi_nom, bi_nom]
    R_opt  = sweep2d["global_R"][gi_opt, bi_opt]
    R_worst= sweep2d["global_R"][0, -1]  # gamma basso, beta alto

    print(f"\n  global_R nominal (×1.0, ×1.0):   {R_nom:.3f}")
    print(f"  global_R ottimale (γ max, β min):  {R_opt:.3f}  "
          f"(riduzione: {R_nom-R_opt:.3f})")
    print(f"  global_R peggiore (γ min, β max):  {R_worst:.3f}  "
          f"(aumento: {R_worst-R_nom:.3f})")

    D_nom   = sweep2d["global_D"][gi_nom, bi_nom]
    D_opt   = sweep2d["global_D"][gi_opt, bi_opt]
    if not np.isnan(D_nom) and not np.isnan(D_opt):
        print(f"  global_D nominale:  {D_nom:.3f}")
        print(f"  global_D ottimale:  {D_opt:.3f}  "
              f"(riduzione: {D_nom-D_opt:.3f})")

    # ── 2. Heatmap 2D ────────────────────────────────────────────────────
    fig_2d = plot_2d_sweep(
        sweep2d, base_params, comp,
        save_prefix=save_prefix,
    )

    # ── 3. Mappa spaziale ─────────────────────────────────────────────────
    if adata_st is not None:
        plot_2d_sweep_spatial(
            sweep2d, comp, adata_st, t_end, base_params,
            n_steps=n_steps * 2,
            seed_I=seed_I,
            save_prefix=save_prefix,
        )
    else:
        print("  [2D sweep] adata_st non fornito — mappa spaziale saltata.")

    # ── 4. Confronto curve temporali nominale vs ottimale ─────────────────
    print("\n  [2D sweep] Confronto curve temporali nominale vs ottimale...")
    fig_curves = plot_optimal_vs_nominal_curves(
        comp        = comp,
        base_params = base_params,
        sweep2d     = sweep2d,
        t_end       = t_end,
        seed_I      = seed_I,
        n_steps     = n_steps * 2,
        save_prefix = save_prefix,
    )

    return dict(sweep2d=sweep2d, fig_2d=fig_2d, fig_curves=fig_curves)


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def run_full_sensitivity(
    comp:        dict,
    base_params: dict,
    t_end:       float,
    adata_st     = None,
    param_ranges: dict  = None,
    n_oat:       int    = 7,
    n_morris_r:  int    = 15,
    n_mc:        int    = 300,
    n_robust:    int    = 200,
    seed_I:      float  = 0.10,
    n_steps_fast: int   = 150,
    primary_metric: str = "global_R",
    run_2d_sweep: bool  = True,
    n_gamma:     int    = 12,
    n_beta:      int    = 12,
    save_prefix: str    = None,
    verbose:     bool   = True,
) -> dict:
    """
    Runs the full sensitivity analysis in sequence:
    1. OAT  → tornado plot
    2. Morris screening → µ*-σ scatter
    3. Monte Carlo → distribuzioni output + Spearman heatmap
    4. Robustness check → ranking stability
    5. Sweep 2D gamma_mult_outer × beta_DCT (opzionale)

    Parameters
    ----------
    comp, base_params : output di build_compartment2 e build_default_params_SIR
    t_end             : float — tempo final
    adata_st          : AnnData Visium (necessario per mappa spaziale sweep 2D)
    n_oat             : livelli per parametro nell'OAT
    n_morris_r        : traiettorie Morris
    n_mc              : campioni Monte Carlo
    n_robust          : simulazioni robustness check
    n_steps_fast      : ODE steps for sensitivity simulations
    primary_metric    : metrica principale per i plot OAT e Morris
    run_2d_sweep      : se True esegue lo sweep 2D gamma × beta_DCT
    n_gamma, n_beta   : dimensioni griglia sweep 2D
    save_prefix       : prefisso per salvare le figure (None = non salva)

    Returns
    -------
    dict con chiavi:
        df_oat, df_morris, df_mc, df_spearman, robustness,
        sweep_2d (se run_2d_sweep=True),
        figures: {nome: matplotlib.Figure}
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    np.random.seed(42)
    results = {}
    figs    = {}

    total_sims = (n_oat * len(param_ranges)
                  + n_morris_r * (len(param_ranges) + 1)
                  + n_mc + n_robust)
    print(f"\n{'='*60}")
    print(f"PARAMETRIC SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    print(f"  Parametri: {len(param_ranges)}")
    print(f"  Simulazioni totali stimate: ~{total_sims}")
    print(f"  Metrica primaria: {primary_metric}")
    print(f"  n_steps per simulazione: {n_steps_fast}")

    # ── 1. OAT ──────────────────────────────────────────────────────────
    print(f"\n[1/4] One-At-A-Time ({n_oat} levels × {len(param_ranges)} params)...")
    df_oat = run_oat_analysis(
        comp, base_params, t_end,
        param_ranges=param_ranges,
        n_levels=n_oat, seed_I=seed_I,
        n_steps=n_steps_fast, verbose=verbose,
    )
    results["df_oat"] = df_oat

    # Tornado plot per la metrica primaria + invasion_time + pt_attack_rate
    for met in [primary_metric, "mean_invasion_time", "pt_attack_rate"]:
        fig = plot_tornado(df_oat, metric=met, top_n=12)
        if fig:
            figs[f"tornado_{met}"] = fig
            if save_prefix:
                fig.savefig(f"{save_prefix}_tornado_{met}.png",
                            dpi=150, bbox_inches="tight")

    # ── 2. Morris ────────────────────────────────────────────────────────
    print(f"\n[2/4] Morris screening (r={n_morris_r} traiettorie)...")
    df_morris = run_morris_screening(
        comp, base_params, t_end,
        param_ranges=param_ranges,
        r=n_morris_r, seed_I=seed_I,
        n_steps=n_steps_fast, verbose=verbose,
    )
    results["df_morris"] = df_morris

    for met in [primary_metric, "mean_invasion_time"]:
        fig = plot_morris(df_morris, metric=met)
        if fig:
            figs[f"morris_{met}"] = fig
            if save_prefix:
                fig.savefig(f"{save_prefix}_morris_{met}.png",
                            dpi=150, bbox_inches="tight")

    # ── 3. Monte Carlo ───────────────────────────────────────────────────
    print(f"\n[3/4] Monte Carlo sweep (N={n_mc})...")
    df_mc = run_monte_carlo_sweep(
        comp, base_params, t_end,
        param_ranges=param_ranges,
        n_samples=n_mc, seed_I=seed_I,
        n_steps=n_steps_fast, verbose=verbose,
    )
    results["df_mc"] = df_mc

    df_spearman = compute_spearman_indices(df_mc)
    results["df_spearman"] = df_spearman

    fig_mc = plot_mc_distributions(df_mc)
    figs["mc_distributions"] = fig_mc
    if save_prefix:
        fig_mc.savefig(f"{save_prefix}_mc_distributions.png",
                       dpi=150, bbox_inches="tight")

    fig_sp = plot_spearman_heatmap(df_spearman)
    figs["spearman_heatmap"] = fig_sp
    if save_prefix:
        fig_sp.savefig(f"{save_prefix}_spearman_heatmap.png",
                       dpi=150, bbox_inches="tight")

    # ── 4. Robustness ────────────────────────────────────────────────────
    print(f"\n[4/4] Robustness check (N={n_robust})...")
    rob = run_robustness_check(
        comp, base_params, t_end,
        param_ranges=param_ranges,
        n_samples=n_robust, seed_I=seed_I,
        n_steps=n_steps_fast, verbose=verbose,
    )
    results["robustness"] = rob

    if rob:
        fig_rob = plot_ranking_stability(rob)
        if fig_rob:
            figs["ranking_stability"] = fig_rob
            if save_prefix:
                fig_rob.savefig(f"{save_prefix}_ranking_stability.png",
                                dpi=150, bbox_inches="tight")

    results["figures"] = figs

    # ── 5. 2D sweep gamma_mult_outer × beta_DCT ──────────────────────────
    if run_2d_sweep:
        if adata_st is None:
            warnings.warn(
                "[run_full_sensitivity] adata_st non fornito — "
                "mappa spaziale sweep 2D saltata, solo heatmap.",
                UserWarning,
            )
        print(f"\n[5/5] Sweep 2D gamma_mult_outer × beta_DCT "
              f"({n_gamma}×{n_beta} = {n_gamma*n_beta} simulazioni)...")
        sweep2d_result = run_2d_sweep_analysis(
            comp        = comp,
            base_params = base_params,
            adata_st    = adata_st,
            t_end       = t_end,
            n_gamma     = n_gamma,
            n_beta      = n_beta,
            seed_I      = seed_I,
            n_steps     = n_steps_fast,
            save_prefix = save_prefix,
        )
        results["sweep_2d"] = sweep2d_result
        figs["2d_sweep"] = sweep2d_result.get("fig_2d")

    # ── Riepilogo testuale ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*60}")

    if not df_oat.empty:
        # Parametri con maggiore effetto su global_R
        oat_range = (df_oat.groupby("param")[primary_metric]
                     .apply(lambda x: x.max() - x.min())
                     .sort_values(ascending=False))
        print(f"\n  Top 5 parametri per effetto su '{primary_metric}' (OAT):")
        for pn, rng in oat_range.head(5).items():
            print(f"    {pn:30s}  range = {rng:.4f}")

    if not df_spearman.empty:
        top_sp = (df_spearman[df_spearman["metric"] == primary_metric]
                  .assign(abs_rho=lambda d: d["rho"].abs())
                  .sort_values("abs_rho", ascending=False)
                  .head(5))
        print(f"\n  Top 5 parametri per Spearman su '{primary_metric}' (MC):")
        for _, r in top_sp.iterrows():
            print(f"    {r['param']:30s}  ρ = {r['rho']:+.3f}")

    if rob:
        print(f"\n  Ranking stability (mean Spearman):")
        print(f"    invasion_time : ρ = {rob['spearman_invasion']:.4f}")
        print(f"    attack_rate   : ρ = {rob['spearman_attack']:.4f}")
        if rob["spearman_invasion"] > 0.85 and rob["spearman_attack"] > 0.85:
            print("    → CONCLUSIONE: ranking ROBUSTO (ρ > 0.85 per entrambe)")
        else:
            print("    → ATTENZIONE: ranking parzialmente instabile")

    if not df_mc.empty:
        print(f"\n  Incertezza output da MC (90% CI):")
        for m in [primary_metric, "mean_invasion_time", "frac_invaded"]:
            vals = df_mc[m].dropna()
            if len(vals) > 0:
                p5, p95 = vals.quantile([0.05, 0.95])
                print(f"    {m:30s}  [{p5:.3f}, {p95:.3f}]  "
                      f"(media={vals.mean():.3f})")

    return results