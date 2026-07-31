"""
sir_drug.py
===========
Modello farmacologico esteso sul grafo compartimentale.

Allineamento con l'articolo (Sezione 8.6, Equazioni 11-17)
-----------------------------------------------------------
The base model is the H/D/R/X exposure-driven system described in
Sezioni 5 e 8 dell'articolo. Le variabili corrispondono a:

  H  → Healthy           (p^H_i)
  D  → Diseased          (p^D_i)    ← stato attivo
  D2 → Diseased-att.     (p^{D2}_i) ← stato attenuato dal farmaco
  R  → Recovered         (p^R_i)
  X  → Dead/damage       (p^X_i)    ← danno irreversibile (δ_i)
  F  → Drug conc.        (F_i)

Equazioni implementate (Eq. 11-16 dell'articolo):

  dp^H /dt  = -β_eff_i · p^H_i · λ_i(t)  - D_{H,i} · (L_c @ p^H)_i
  dp^D /dt  = +β_eff_i · p^H_i · λ_i(t)  - ρ_i · p^D_i
              - δ_i · p^D_i - k_F · F_i · p^D_i
              - D_{D,i} · (L_c @ p^D)_i
  dp^{D2}/dt = +k_F · F_i · p^D_i          - ρ^{(2)}_i · p^{D2}_i
               - D_{D,i} · (L_c @ p^{D2})_i
  dp^R /dt  = +ρ_i · p^D_i + ρ^{(2)}_i · p^{D2}_i
  dp^X /dt  = +δ_i · p^D_i
  dF   /dt  =  σ(t) - k_e · F_i - k_F · F_i · p^D_i
              - D_{F,i} · (L_c @ F)_i

where the local exposure is (Eq. 1):

  λ_i(t) = Σ_j w̃_{ij} · p^D_j(t)

and susceptibility is modulated by the drug via Hill kinetics (Eq. 17):

  β_eff_i = β_i / (1 + F_i / IC50)   se IC50 > 0
           = β_i                       se IC50 = 0 (default, backward-compat)

Conservazione
-------------
p^H + p^D + p^{D2} + p^R + p^X = 1 is satisfied by construction:
ogni flusso che esce da uno stato entra esattamente in un altro.
Non serve alcun termine di drenaggio artificiale.

Backward compatibility
----------------------
Firme pubbliche invariate rispetto alla versione precedente.
Internamente S/I/I2/R/D/F mappano su H/D/D2/R/X/F del paper.
"""

import warnings
import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapezoid

import scipy.sparse as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════════════

DRUG_DEFAULTS = dict(
    delta        = 0.02,   # tasso danno irreversibile δ_i
    k_F          = 0.50,   # efficacia farmaco (D → D2)
    gamma2_mult  = 1.5,    # ρ^{(2)} = gamma2_mult × ρ_i
    k_e          = 0.15,   # eliminazione PK del farmaco
    D_F          = 0.30,   # diffusione spaziale del farmaco (default)
    dose_amount  = 1.0,
    IC50_beta    = 0.0,    # IC50 for β reduction via Hill kinetics (0 = disabled)
)

# δ_i per macro_type
_MACRO_DELTA = {
    "PT":       0.040,
    "DCT":      0.020,
    "TAL":      0.015,
    "vascular": 0.010,
    "immune":   0.005,
    "other":    0.015,
}

# Moltiplicatori δ per zona anatomica
_REGION_DELTA_MULT = {
    "outer_medulla": 1.30,
    "cortex":        1.00,
    "inner_medulla": 0.70,
    # Zonazione radiale legacy
    "boundary":      1.30,
    "mid":           1.00,
    "core":          0.70,
}

# D_{F,i} per macro_type — vascular is the drug delivery corridor (Sec. 8.6)
_MACRO_DF = {
    "vascular": 0.60,
    "PT":       0.30,
    "DCT":      0.25,
    "TAL":      0.25,
    "immune":   0.20,
    "other":    0.25,
}


# ═══════════════════════════════════════════════════════════════════════════
# COSTRUZIONE PARAMETRI
# ═══════════════════════════════════════════════════════════════════════════

def build_drug_params(
    comps2:      np.ndarray,
    macro_of:    np.ndarray,
    region_of:   np.ndarray,
    sir_params:  dict,
    delta:       float = None,
    k_F:         float = None,
    gamma2_mult: float = None,
    k_e:         float = None,
    D_F:         float = None,
    IC50_beta:   float = None,
) -> dict:
    """
    Costruisce i vettori di parametri per il modello H/D/D2/R/X + farmaco.

    Parameters
    ----------
    sir_params  : output di build_default_params_SIR (fornisce β, ρ, D_H, D_D)
    delta       : override globale per δ_i
    k_F         : efficacia farmaco (D → D2)
    gamma2_mult : ρ^{(2)} = gamma2_mult × ρ_i
    k_e         : eliminazione PK
    D_F         : diffusione farmaco (None → per-macro da _MACRO_DF)
    IC50_beta   : IC50 per Hill kinetics (0 = disabilitato)

    Returns
    -------
    dict con chiavi: beta, gamma (=ρ), D_S (=D_H), D_I (=D_D),
                     delta, gamma2, k_F, k_e, D_F, IC50_beta
    """
    nC = len(comps2)
    params = {k: np.asarray(v, dtype=float).copy()
              for k, v in sir_params.items()}

    # δ_i
    d0 = delta if delta is not None else DRUG_DEFAULTS["delta"]
    delta_v = np.full(nC, d0, dtype=float)
    for i, m in enumerate(macro_of):
        delta_v[i] = _MACRO_DELTA.get(m, d0)
    for i, rg in enumerate(region_of):
        delta_v[i] *= _REGION_DELTA_MULT.get(rg, 1.0)

    # ρ^{(2)}_i
    gm = gamma2_mult if gamma2_mult is not None else DRUG_DEFAULTS["gamma2_mult"]
    gamma2_v = params["gamma"] * gm

    params["delta"]     = delta_v
    params["gamma2"]    = gamma2_v
    params["k_F"]       = float(k_F  if k_F  is not None else DRUG_DEFAULTS["k_F"])
    params["k_e"]       = float(k_e  if k_e  is not None else DRUG_DEFAULTS["k_e"])
    params["IC50_beta"] = float(
        IC50_beta if IC50_beta is not None else DRUG_DEFAULTS["IC50_beta"]
    )

    df0 = float(D_F if D_F is not None else DRUG_DEFAULTS["D_F"])
    D_F_v = np.full(nC, df0, dtype=float)
    for i, m in enumerate(macro_of):
        D_F_v[i] = _MACRO_DF.get(m, df0)
    params["D_F"] = D_F_v

    return params


# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONI DI SOMMINISTRAZIONE σ(t)  — Sezione 8.6
# ═══════════════════════════════════════════════════════════════════════════

def sigma_bolus(t: float, t_dose: float, dose_amount: float,
                width: float = 0.5) -> float:
    """Impulso gaussiano normalizzato: ∫σ dt = dose_amount."""
    norm = dose_amount / (width * np.sqrt(2.0 * np.pi))
    return norm * np.exp(-0.5 * ((t - t_dose) / width) ** 2)


def sigma_infusion(t: float, t_dose: float, dose_amount: float,
                   duration: float = 5.0) -> float:
    """Infusione continua su [t_dose, t_dose + duration]."""
    return float(dose_amount) if t_dose <= t <= t_dose + duration else 0.0


def sigma_repeated(t: float, t_dose: float, dose_amount: float,
                   n_doses: int = 3, interval: float = 10.0,
                   width: float = 0.5) -> float:
    """n_doses impulsi gaussiani normalizzati separati da interval."""
    norm = dose_amount / (width * np.sqrt(2.0 * np.pi))
    total = 0.0
    for k in range(n_doses):
        t_k = t_dose + k * interval
        total += norm * np.exp(-0.5 * ((t - t_k) / width) ** 2)
    return total


SIGMA_FUNCTIONS = {
    "bolus":    sigma_bolus,
    "infusion": sigma_infusion,
    "repeated": sigma_repeated,
}


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATORE  H/D/D2/R/X + FARMACO
# ═══════════════════════════════════════════════════════════════════════════

def simulate_SIRD_drug(
    Lc:           sp.spmatrix,
    S0:           np.ndarray,   # H_0
    I0:           np.ndarray,   # D_0
    I2_0:         np.ndarray,   # D2_0
    R0:           np.ndarray,   # R_0
    D0:           np.ndarray,   # X_0
    F0:           np.ndarray,   # F_0
    params:       dict,
    t_end:        float = 80.0,
    n_steps:      int   = 800,
    method:       str   = "BDF",
    rtol:         float = 1e-6,
    atol:         float = 1e-9,
    t_dose:       float = 0.0,
    dose_amount:  float = None,
    dose_mode:    str   = "bolus",
    dose_kwargs:  dict  = None,
    target_mask:  np.ndarray = None,
) -> tuple:
    """
    Integra il sistema H/D/D2/R/X + farmaco (Eq. 11-16 dell'articolo).

    Vettore di stato (6·C): y = [H | D | D2 | R | X | F]
    I nomi S/I/I2/R/D/F nella firma sono alias per backward compatibility.

    The H→D transition is driven by the local exposure λ_i(t) = W̃ @ D,
    non dal termine bilineare H·D del SIR classico. Questo allinea
    il codice con la Sezione 5.2 e le Equazioni 11-16 dell'articolo.
    """
    nC  = len(S0)
    Lc  = sp.csr_matrix(Lc)
    dkw = dose_kwargs or {}

    beta_v   = np.asarray(params["beta"],   dtype=float)
    rho_v    = np.asarray(params["gamma"],  dtype=float)
    delta_v  = np.asarray(params["delta"],  dtype=float)
    rho2_v   = np.asarray(params["gamma2"], dtype=float)
    D_H_v    = np.asarray(params["D_S"],    dtype=float)
    D_D_v    = np.asarray(params["D_I"],    dtype=float)
    k_F      = float(params["k_F"])
    k_e      = float(params["k_e"])
    IC50     = float(params.get("IC50_beta", 0.0))

    _df   = params["D_F"]
    D_F_v = (np.full(nC, float(_df), dtype=float) if np.isscalar(_df)
             else np.asarray(_df, dtype=float))
    if D_F_v.shape[0] != nC:
        D_F_v = np.full(nC, float(np.mean(_df)), dtype=float)

    da       = float(dose_amount if dose_amount is not None
                     else DRUG_DEFAULTS["dose_amount"])
    sigma_fn = SIGMA_FUNCTIONS.get(dose_mode, sigma_bolus)
    targeting = (np.where(np.asarray(target_mask, dtype=bool), 1.0, 0.0)
                 if target_mask is not None else np.ones(nC, dtype=float))

    # W̃_{ij} per il calcolo di λ_i = Σ_j w̃_{ij} · D_j  (Eq. 1)
    # Ricaviamo l'adiacenza pesata dal Laplaciano: W = diag(L) - L
    Lc_arr = Lc.toarray()
    W_raw  = np.diag(np.diag(Lc_arr)) - Lc_arr
    np.fill_diagonal(W_raw, 0.0)
    row_sum = W_raw.sum(axis=1, keepdims=True)
    W_tilde = np.where(row_sum > 1e-12, W_raw / row_sum, 0.0)

    y0 = np.concatenate([
        np.asarray(S0,   dtype=float),
        np.asarray(I0,   dtype=float),
        np.asarray(I2_0, dtype=float),
        np.asarray(R0,   dtype=float),
        np.asarray(D0,   dtype=float),
        np.asarray(F0,   dtype=float),
    ])
    t_eval = np.linspace(0.0, float(t_end), int(n_steps))

    def rhs(t, y):
        H  = np.clip(y[0*nC:1*nC], 0.0, 1.0)
        D  = np.clip(y[1*nC:2*nC], 0.0, 1.0)
        D2 = np.clip(y[2*nC:3*nC], 0.0, 1.0)
        R  = np.clip(y[3*nC:4*nC], 0.0, 1.0)
        X  = np.clip(y[4*nC:5*nC], 0.0, 1.0)
        F  = np.clip(y[5*nC:6*nC], 0.0, None)

        # Exposure locale λ_i(t) — Eq. 1
        lam = W_tilde @ D

        # β_eff with Hill kinetics — Eq. 17
        beta_eff = beta_v / (1.0 + F / IC50) if IC50 > 0.0 else beta_v

        # Flussi tra stati
        infect   = beta_eff * H  * lam       # H → D   (Eq. 11, 12)
        recover1 = rho_v   * D               # D → R   (Eq. 12, 14)
        damage   = delta_v * D               # D → X   (Eq. 12, 15)
        drug_act = k_F     * F * D           # D → D2  (Eq. 12, 13)
        recover2 = rho2_v  * D2              # D2 → R  (Eq. 13, 14)
        sig      = float(sigma_fn(t, t_dose, da, **dkw)) * targeting

        # ODE — Eq. 11-16
        dH  = -infect                              - D_H_v * (Lc @ H)
        dD  = +infect - recover1 - damage - drug_act - D_D_v * (Lc @ D)
        dD2 = +drug_act - recover2                 - D_D_v * (Lc @ D2)
        dR  = +recover1 + recover2
        dX  = +damage
        dF  =  sig - k_e * F - drug_act            - D_F_v * (Lc @ F)

        return np.concatenate([dH, dD, dD2, dR, dX, dF])

    sol = solve_ivp(
        rhs, (0.0, float(t_end)), y0,
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )

    if not sol.success:
        warnings.warn(
            f"simulate_SIRD_drug: solver failed — {sol.message}",
            RuntimeWarning, stacklevel=2,
        )

    def _fin(block):
        return np.clip(sol.y[block*nC:(block+1)*nC, -1], 0.0, None)

    H_r  = np.clip(_fin(0), 0.0, 1.0)
    D_r  = np.clip(_fin(1), 0.0, 1.0)
    D2_r = np.clip(_fin(2), 0.0, 1.0)
    R_r  = np.clip(_fin(3), 0.0, 1.0)
    X_r  = np.clip(_fin(4), 0.0, 1.0)
    F_r  = np.clip(_fin(5), 0.0, None)

    # Normalizzazione: H + D + D2 + R + X deve sommaire a 1
    pop = H_r + D_r + D2_r + R_r + X_r
    pop = np.where(pop > 0, pop, 1.0)
    finals = dict(
        S  = H_r  / pop,    # H  (alias S per backward compat)
        I  = D_r  / pop,    # D  (alias I)
        I2 = D2_r / pop,    # D2 (alias I2)
        R  = R_r  / pop,    # R
        D  = X_r  / pop,    # X  (alias D)
        F  = F_r,
    )
    return sol, finals


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def run_SIRD_drug(
    comps2:             np.ndarray,
    Lc2:                sp.spmatrix,
    boundary_mask:      np.ndarray,
    macro_of:           np.ndarray,
    region_of:          np.ndarray,
    sir_params:         dict,
    drug_params:        dict  = None,
    seed_I:             float = 0.10,
    seed_boundary_only: bool  = True,
    t_end:              float = 80.0,
    n_steps:            int   = 800,
    t_dose:             float = 20.0,
    dose_amount:        float = 1.0,
    dose_mode:          str   = "bolus",
    dose_kwargs:        dict  = None,
    run_nodrug:         bool  = True,
    target_mask:        np.ndarray = None,
) -> dict:
    """
    Runner completo: costruisce le CI, esegue la simulazione con e senza
    farmaco, restituisce sol + finals + parametri.
    """
    nC = len(comps2)

    if drug_params is None:
        drug_params = build_drug_params(
            comps2, macro_of, region_of, sir_params
        )

    D0 = np.zeros(nC)
    if seed_boundary_only:
        D0[boundary_mask] = seed_I
    else:
        D0[:] = seed_I

    H0   = np.clip(1.0 - D0, 0.0, 1.0)
    ic   = dict(S0=H0, I0=D0, I2_0=np.zeros(nC),
                R0=np.zeros(nC), D0=np.zeros(nC), F0=np.zeros(nC))

    mode_str = "targeted" if target_mask is not None else "globale"
    print(f"\n  [DRUG]  t_dose={t_dose}  dose_amount={dose_amount}"
          f"  mode={dose_mode}  dose_mode={mode_str}"
          f"  k_F={drug_params['k_F']:.2f}"
          f"  k_e={drug_params['k_e']:.2f}"
          f"  IC50_beta={drug_params.get('IC50_beta', 0.0):.3f}")

    sol_drug, finals_drug = simulate_SIRD_drug(
        Lc2, **ic, params=drug_params,
        t_end=t_end, n_steps=n_steps,
        t_dose=t_dose, dose_amount=dose_amount,
        dose_mode=dose_mode, dose_kwargs=dose_kwargs,
        target_mask=target_mask,
    )

    sol_nodrug = finals_nodrug = None
    if run_nodrug:
        print("  [NO-DRUG] simulazione senza farmaco (controllo)...")
        sol_nodrug, finals_nodrug = simulate_SIRD_drug(
            Lc2, **ic, params=drug_params,
            t_end=t_end, n_steps=n_steps,
            t_dose=t_dose, dose_amount=0.0,
            dose_mode=dose_mode, dose_kwargs=dose_kwargs,
        )

    return dict(
        sol_drug      = sol_drug,
        finals_drug   = finals_drug,
        sol_nodrug    = sol_nodrug,
        finals_nodrug = finals_nodrug,
        drug_params   = drug_params,
        comps2        = comps2,
        macro_of      = macro_of,
        region_of     = region_of,
        boundary_mask = boundary_mask,
        target_mask   = target_mask,
    )


# ═══════════════════════════════════════════════════════════════════════════
# METRICHE
# ═══════════════════════════════════════════════════════════════════════════

def drug_metrics(
    sol:         object,
    finals:      dict,
    macro_of:    np.ndarray,
    region_of:   np.ndarray,
    drug_params: dict,
    threshold_I: float = 0.05,
) -> dict:
    """
    Calcola le metriche di outcome.
    Chiavi: global_D (=X, danno), global_R (recovered), global_I (D residuo),
            global_I2 (D2 residuo), peak_F, t_peak_F, by_macro, by_region,
            per_comp.
    """
    nC     = sol.y.shape[0] // 6
    t      = sol.t
    I_traj = sol.y[1*nC:2*nC, :]   # D(t)
    F_traj = sol.y[5*nC:6*nC, :]

    peak_F   = float(F_traj.mean(axis=0).max())
    t_peak_F = float(t[np.argmax(F_traj.mean(axis=0))])

    metrics = dict(
        global_D   = float(finals["D"].mean()),
        global_R   = float(finals["R"].mean()),
        global_I   = float(finals["I"].mean()),
        global_I2  = float(finals["I2"].mean()),
        peak_F     = peak_F,
        t_peak_F   = t_peak_F,
    )

    by_macro = {}
    for m in np.unique(macro_of):
        mask = macro_of == m
        I_pk = I_traj[mask, :].max(axis=1)
        by_macro[m] = dict(
            mean_D    = float(finals["D"][mask].mean()),
            mean_R    = float(finals["R"][mask].mean()),
            mean_I    = float(finals["I"][mask].mean()),
            mean_I2   = float(finals["I2"][mask].mean()),
            frac_dead = float((finals["D"][mask] > threshold_I).mean()),
            peak_I    = float(I_pk.mean()),
        )
    metrics["by_macro"] = by_macro

    by_region = {}
    for rg in np.unique(region_of):
        mask = region_of == rg
        by_region[rg] = dict(
            mean_D    = float(finals["D"][mask].mean()),
            mean_R    = float(finals["R"][mask].mean()),
            frac_dead = float((finals["D"][mask] > threshold_I).mean()),
        )
    metrics["by_region"] = by_region

    per_comp = {}
    for i in range(nC):
        I_i = I_traj[i, :]
        per_comp[i] = dict(
            peak_I   = float(I_i.max()),
            t_peak_I = float(t[np.argmax(I_i)]),
            auc_I    = float(_trapz(I_i, t)),
            D_fin    = float(finals["D"][i]),
            R_fin    = float(finals["R"][i]),
            I2_fin   = float(finals["I2"][i]),
            invasion = bool((I_i > threshold_I).any()),
        )
    metrics["per_comp"] = per_comp
    return metrics


def print_drug_summary(result: dict, m_drug: dict, m_nodrug: dict = None) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print("H/D/D2/R/X + FARMACO — RIEPILOGO  (Sezione 8.6, Eq. 11-16)")
    print(sep)

    dp = result["drug_params"]
    print(f"\n  Parametri farmaco:")
    print(f"    k_F       = {dp['k_F']:.3f}   (D → D2, efficacia)")
    print(f"    k_e       = {dp['k_e']:.3f}   (eliminazione PK)")
    print(f"    IC50_beta = {dp.get('IC50_beta', 0.0):.3f}   "
          f"(riduzione β Hill; 0=disabilitata)")
    _df = dp["D_F"]
    if np.isscalar(_df):
        print(f"    D_F  = {float(_df):.3f}   (diffusione spaziale farmaco)")
    else:
        print(f"    D_F  = {float(np.mean(_df)):.3f} (media)  "
              f"[{float(np.min(_df)):.3f}–{float(np.max(_df)):.3f}]")

    print(f"\n  Esito globale [con farmaco]:")
    print(f"    X (danno irrev.) final : {m_drug['global_D']:.4f}")
    print(f"    R (recovered) final    : {m_drug['global_R']:.4f}")
    print(f"    D (diseased) residuo    : {m_drug['global_I']:.4f}")
    print(f"    Peak F (farmaco, medio) : {m_drug['peak_F']:.4f}"
          f"  @ t={m_drug['t_peak_F']:.1f}")

    if m_nodrug is not None:
        x_drug = m_drug["global_D"];  x_no = m_nodrug["global_D"]
        r_drug = m_drug["global_R"];  r_no = m_nodrug["global_R"]
        avert  = (x_no - x_drug) / (x_no + 1e-12) * 100
        rgain  = (r_drug - r_no) / (r_no + 1e-12) * 100
        print(f"\n  Confronto farmaco vs no-farmaco:")
        print(f"    X (danno) ridotto del   : {avert:.1f}%"
              f"  ({x_no:.4f} → {x_drug:.4f})")
        print(f"    R aumentato del         : {rgain:.1f}%"
              f"  ({r_no:.4f} → {r_drug:.4f})")

        # Dettaglio per macro_type
        print(f"\n  ΔX per macro_type (danno evitato dal farmaco):")
        for m in np.unique(result["macro_of"]):
            if m in m_nodrug["by_macro"] and m in m_drug["by_macro"]:
                x_nd  = m_nodrug["by_macro"][m]["mean_D"]
                x_d   = m_drug["by_macro"][m]["mean_D"]
                delta_pct = (x_nd - x_d) / (x_nd + 1e-12) * 100
                print(f"    {m:10s}  ΔX = {x_nd - x_d:.4f}  ({delta_pct:.1f}%)")

    print(f"\n  Per macro_type [con farmaco]:")
    print(f"  {'macro':10s} {'X':8s} {'R':8s} {'D':8s} {'D2':8s} {'frac_X':10s}")
    print("  " + "-" * 60)
    for m, v in m_drug["by_macro"].items():
        print(f"  {m:10s} {v['mean_D']:8.4f} {v['mean_R']:8.4f} "
              f"{v['mean_I']:8.4f} {v['mean_I2']:8.4f} {v['frac_dead']:10.2%}")

    print(f"\n  Per regione:")
    for rg, v in m_drug["by_region"].items():
        print(f"  {rg:10s}  X={v['mean_D']:.4f}  R={v['mean_R']:.4f}  "
              f"frac_X={v['frac_dead']:.2%}")
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════
# SWEEP t_dose  — finestra terapeutica ottimale (Sezione 8.6)
# ═══════════════════════════════════════════════════════════════════════════

def sweep_t_dose(
    comps2:        np.ndarray,
    Lc2:           sp.spmatrix,
    boundary_mask: np.ndarray,
    macro_of:      np.ndarray,
    region_of:     np.ndarray,
    sir_params:    dict,
    drug_params:   dict  = None,
    t_dose_grid:   np.ndarray = None,
    seed_I:        float = 0.10,
    t_end:         float = 80.0,
    n_steps:       int   = 400,
    dose_amount:   float = 1.0,
    dose_mode:     str   = "bolus",
    dose_kwargs:   dict  = None,
) -> dict:
    """
    Sweep sulla finestra terapeutica: t_dose ∈ [0, 0.75·t_end].
    Registra X_fin e R_fin per ogni t_dose per identificare la finestra ottimale.
    """
    if t_dose_grid is None:
        t_dose_grid = np.linspace(0.0, t_end * 0.75, 20)
    if drug_params is None:
        drug_params = build_drug_params(
            comps2, macro_of, region_of, sir_params
        )

    nC    = len(comps2)
    D0    = np.zeros(nC);  D0[boundary_mask] = seed_I
    H0    = np.clip(1.0 - D0, 0.0, 1.0)
    zeros = np.zeros(nC)
    X_arr  = np.zeros(len(t_dose_grid))
    R_arr  = np.zeros(len(t_dose_grid))
    I2_arr = np.zeros(len(t_dose_grid))

    print(f"  Sweep t_dose: {len(t_dose_grid)} punti ...")
    for k, td in enumerate(t_dose_grid):
        _, finals = simulate_SIRD_drug(
            Lc2, S0=H0, I0=D0, I2_0=zeros.copy(),
            R0=zeros.copy(), D0=zeros.copy(), F0=zeros.copy(),
            params=drug_params, t_end=t_end, n_steps=n_steps,
            t_dose=td, dose_amount=dose_amount,
            dose_mode=dose_mode, dose_kwargs=dose_kwargs,
        )
        X_arr[k]  = float(finals["D"].mean())
        R_arr[k]  = float(finals["R"].mean())
        I2_arr[k] = float(finals["I2"].mean())
        if k % 5 == 0:
            print(f"    t_dose={td:.1f}  X={X_arr[k]:.4f}  R={R_arr[k]:.4f}")

    t_opt_X = float(t_dose_grid[np.argmin(X_arr)])
    t_opt_R = float(t_dose_grid[np.argmax(R_arr)])
    print(f"  t_dose ottimale (min danno X) : {t_opt_X:.1f}")
    print(f"  t_dose ottimale (max R)       : {t_opt_R:.1f}")

    return dict(
        t_dose_grid = t_dose_grid,
        D_fin       = X_arr,       # alias D per backward compat
        R_fin       = R_arr,
        I2_fin      = I2_arr,
        t_opt_D     = t_opt_X,     # alias per backward compat
        t_opt_R     = t_opt_R,
    )


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZZAZIONI
# ═══════════════════════════════════════════════════════════════════════════

def plot_drug_dynamics(sol, macro_of, comps2=None, title="", figsize=(18, 10)):
    """Traiettorie H/D/D2/R/X/F per macro_type (media ± std)."""
    nC   = sol.y.shape[0] // 6
    t    = sol.t
    blks = {
        "H (healthy)":     sol.y[0*nC:1*nC, :],
        "D (diseased)":    sol.y[1*nC:2*nC, :],
        "D2 (attenuato)":  sol.y[2*nC:3*nC, :],
        "R (recovered)":   sol.y[3*nC:4*nC, :],
        "X (danno irrev.)":sol.y[4*nC:5*nC, :],
        "F (farmaco)":     sol.y[5*nC:6*nC, :],
    }
    macros = list(np.unique(macro_of))
    colors = plt.cm.tab10(np.linspace(0, 1, len(macros)))
    cmap   = {m: c for m, c in zip(macros, colors)}
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
    for ax, (lbl, traj) in zip(axes.flatten(), blks.items()):
        for m in macros:
            idx = np.where(macro_of == m)[0]
            if len(idx) == 0:
                continue
            mean = traj[idx].mean(axis=0)
            std  = traj[idx].std(axis=0)
            ax.plot(t, mean, label=m, color=cmap[m], lw=2.0)
            ax.fill_between(t, mean - std, mean + std,
                            color=cmap[m], alpha=0.12)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("t", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        if "F" not in lbl:
            ax.set_ylim(-0.02, 1.05)
    fig.suptitle(
        (title or "H/D/D2/R/X + Farmaco — dinamica temporale") +
        "\n[media ± std per macro_type  |  Eq. 11-16]",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_drug_vs_nodrug(sol_drug, sol_nodrug, macro_of, title="", figsize=(16, 5)):
    """Confronto D(t), X(t), R(t) con e senza farmaco."""
    nC     = sol_drug.y.shape[0] // 6
    t      = sol_drug.t
    slots  = {"D (diseased)": 1, "X (danno irrev.)": 4, "R (recovered)": 3}
    macros = list(np.unique(macro_of))
    colors = plt.cm.tab10(np.linspace(0, 1, len(macros)))
    cmap   = {m: c for m, c in zip(macros, colors)}
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    for ax, (lbl, bi) in zip(axes, slots.items()):
        traj_d  = sol_drug.y[bi*nC:(bi+1)*nC, :]
        traj_nd = sol_nodrug.y[bi*nC:(bi+1)*nC, :]
        for m in macros:
            idx = np.where(macro_of == m)[0]
            if len(idx) == 0:
                continue
            ax.plot(t, traj_d[idx].mean(axis=0),
                    color=cmap[m], lw=2.0, label=f"{m} +drug")
            ax.plot(t, traj_nd[idx].mean(axis=0),
                    color=cmap[m], lw=1.5, ls="--", alpha=0.6)
        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylim(-0.01, None)
    from matplotlib.lines import Line2D
    legend_els = (
        [Line2D([0], [0], color=cmap[m], lw=2, label=m) for m in macros] +
        [Line2D([0], [0], color="k", lw=2,   label="+farmaco"),
         Line2D([0], [0], color="k", lw=1.5, ls="--", label="−farmaco")]
    )
    axes[0].legend(handles=legend_els, fontsize=8, loc="upper right")
    fig.suptitle(
        (title or "H/D/D2/R/X — confronto farmaco vs no-farmaco") +
        "\n[continua = +drug  |  tratteggiata = −drug]",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_sweep_results(sweep, figsize=(11, 4)):
    """Figura finestra terapeutica: X_fin e R_fin vs t_dose."""
    tg     = sweep["t_dose_grid"]
    X_arr  = sweep["D_fin"]
    R_arr  = sweep["R_fin"]
    t_optX = sweep["t_opt_D"]
    t_optR = sweep["t_opt_R"]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, arr, color, t_opt, ylabel, ttl in [
        (axes[0], X_arr, "#c0392b", t_optX,
         "X final (danno irrev. medio)", "Danno irreversibile vs t_dose"),
        (axes[1], R_arr, "#27ae60", t_optR,
         "R final (recovered medio)", "Recovered vs t_dose"),
    ]:
        ax.plot(tg, arr, color=color, lw=2.0)
        ax.axvline(t_opt, color=color, ls="--", lw=1.5,
                   label=f"t_opt = {t_opt:.1f}")
        ax.set(xlabel="t_dose", ylabel=ylabel, title=ttl)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Finestra terapeutica ottimale — sweep t_dose  (Sezione 8.6)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


def plot_drug_spatial(finals_drug, finals_nodrug, adata_st, inv2, tag="drug"):
    """Mappe spaziali Visium: D, D2, R, X, F e danno evitato ΔX."""
    import scanpy as sc
    vars_drug = {
        f"DRUG_D_{tag}":  finals_drug["I"],
        f"DRUG_D2_{tag}": finals_drug["I2"],
        f"DRUG_R_{tag}":  finals_drug["R"],
        f"DRUG_X_{tag}":  finals_drug["D"],
        f"DRUG_F_{tag}":  finals_drug["F"],
    }
    for col, vals in vars_drug.items():
        adata_st.obs[col] = vals[inv2].astype(np.float32)
    sc.pl.spatial(
        adata_st, color=list(vars_drug.keys()), cmap="viridis",
        title=["D (+drug)", "D2 (attenuati)", "R (+drug)",
               "X (danno irrev.)", "F (farmaco residuo)"],
    )
    if finals_nodrug is not None:
        x_diff = np.clip(finals_nodrug["D"] - finals_drug["D"], 0.0, None)
        adata_st.obs["DRUG_X_averted"] = x_diff[inv2].astype(np.float32)
        sc.pl.spatial(
            adata_st, color=["DRUG_X_averted"], cmap="RdYlGn",
            title=["Danno evitato ΔX = X_nodrug − X_drug"],
        )


def attach_drug_to_adata(finals, adata_st, inv2, tag=""):
    """
    Scrive H/D/D2/R/X/F come colonne obs su adata_st.
    I nomi seguono la nomenclatura dell'articolo (H, D, D2, R, X, F).
    Mantiene anche gli alias SIRD_S/I/I2/R/D/F per backward compat.
    """
    sfx = f"_{tag}" if tag else ""
    paper_map = {"H": "S", "D": "I", "D2": "I2", "R": "R", "X": "D", "F": "F"}
    for paper_name, key in paper_map.items():
        adata_st.obs[f"SIRD_{paper_name}{sfx}"] = (
            finals[key][inv2].astype(np.float32)
        )


# ═══════════════════════════════════════════════════════════════════════════
# MASTER RUNNER — pipeline completa Sezione 8.6
# ═══════════════════════════════════════════════════════════════════════════

def run_full_drug_analysis(
    comp:          dict,
    sir_params:    dict,
    adata_st,
    seed_I:        float = 0.10,
    t_end:         float = 80.0,
    n_steps:       int   = 800,
    t_dose:        float = 20.0,
    dose_amount:   float = 1.0,
    dose_mode:     str   = "bolus",
    dose_kwargs:   dict  = None,
    run_sweep:     bool  = True,
    sweep_n:       int   = 20,
    threshold_I:   float = 0.05,
    IC50_beta:     float = 0.0,
) -> dict:
    """
    Pipeline completa: parametri → simulazione → metriche → figure → sweep.

    Corrisponde alla Sezione 8.6 dell'articolo. Il t_dose ottimale viene
    identificato come il punto in cui si minimizza X_fin (danno irrev.)
    e si massimizza R_fin, che corrisponde a circa 25% del tempo di
    simulazione secondo la Sezione 8.6.
    """
    comps2        = comp["comps2"]
    Lc2           = comp["Lc2"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]
    boundary_mask = comp["boundary_mask"]
    inv2          = comp["inv2"]

    print("\n" + "="*60)
    print("STEP DRUG — Modulazione farmacologica  (Sezione 8.6)")
    print("="*60)

    drug_params = build_drug_params(
        comps2, macro_of, region_of, sir_params,
        IC50_beta=IC50_beta,
    )

    result = run_SIRD_drug(
        comps2, Lc2, boundary_mask, macro_of, region_of,
        sir_params=sir_params, drug_params=drug_params,
        seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        t_dose=t_dose, dose_amount=dose_amount,
        dose_mode=dose_mode, dose_kwargs=dose_kwargs,
        run_nodrug=True,
    )

    m_drug   = drug_metrics(result["sol_drug"],   result["finals_drug"],
                            macro_of, region_of, drug_params, threshold_I)
    m_nodrug = drug_metrics(result["sol_nodrug"], result["finals_nodrug"],
                            macro_of, region_of, drug_params, threshold_I)

    print_drug_summary(result, m_drug, m_nodrug)
    plot_drug_dynamics(
        result["sol_drug"], macro_of, comps2,
        title=f"H/D/D2/R/X + Farmaco  t_dose={t_dose}  mode={dose_mode}",
    )
    plot_drug_vs_nodrug(
        result["sol_drug"], result["sol_nodrug"], macro_of,
        title=f"Confronto  t_dose={t_dose}",
    )
    attach_drug_to_adata(result["finals_drug"],   adata_st, inv2, tag="drug")
    attach_drug_to_adata(result["finals_nodrug"], adata_st, inv2, tag="nodrug")
    plot_drug_spatial(result["finals_drug"], result["finals_nodrug"],
                      adata_st, inv2)

    sweep_result = None
    if run_sweep:
        print("\n  Sweep finestra terapeutica (Sezione 8.6) ...")
        sweep_result = sweep_t_dose(
            comps2, Lc2, boundary_mask, macro_of, region_of,
            sir_params=sir_params, drug_params=drug_params,
            t_dose_grid=np.linspace(0.0, t_end * 0.75, sweep_n),
            seed_I=seed_I, t_end=t_end, n_steps=n_steps // 2,
            dose_amount=dose_amount, dose_mode=dose_mode,
            dose_kwargs=dose_kwargs,
        )
        plot_sweep_results(sweep_result)

    return dict(
        result         = result,
        metrics_drug   = m_drug,
        metrics_nodrug = m_nodrug,
        sweep          = sweep_result,
        drug_params    = drug_params,
    )