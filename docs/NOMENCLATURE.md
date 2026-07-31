# Nomenclature Mapping: Code ↔ Manuscript

The manuscript uses **H/D/R/X** (Healthy, Diseased, Recovered, Dead) throughout.
The codebase historically used **S/I/R** (Susceptible, Infected, Recovered) as
public API names. This document maps between the two and explains the
convention adopted in the refactored code.

## State Variables

| Manuscript | Code (internal) | Code (alias) | Description |
|------------|-----------------|--------------|-------------|
| H(t)       | `H` / `pH`      | `S`          | Healthy fraction |
| D(t)       | `D` / `pD`      | `I`          | Diseased fraction |
| R(t)       | `R` / `pR`      | `R`          | Recovered fraction |
| X(t)       | `X` / `pX`      | `D` (drug model) | Irreversible damage / dead |
| D2(t)      | `D2` / `pD2`    | `I2`         | Attenuated diseased (drug model) |
| F(t)       | `F`             | `F`          | Drug concentration |

## Parameters

| Manuscript | Code variable | Description |
|------------|---------------|-------------|
| β_i        | `beta`        | Susceptibility to local pathological exposure |
| ρ_i        | `gamma`       | Recovery rate |
| δ_i        | `delta`       | Irreversible damage rate (drug model) |
| D_{H,i}    | `D_S`         | Spatial diffusion of healthy state |
| D_{D,i}    | `D_I`         | Spatial diffusion of diseased state |
| D_{F,i}    | `D_F`         | Drug spatial diffusion |
| k_F        | `k_F`         | Drug efficacy (D → D2) |
| k_e        | `k_e`         | Drug elimination (PK) |
| R_diff     | `R0`          | Diffusive reproduction number = β/ρ |
| λ_i(t)     | `lam`         | Local disease exposure (Eq. 1) |

## Key Equations (Manuscript ↔ Code)

### Exposure-driven infection (Eq. 1-2)
```
Manuscript:  λ_i(t) = Σ_j w̃_ij · D_j(t)           (Eq. 1)
             dH/dt = -β_i · H_i · λ_i(t) - ...     (Eq. 18)
             dD/dt = +β_i · H_i · λ_i(t) - ρ·D - ... (Eq. 19)

Code:        lam = W_tilde @ D                     (sir_compartments.py)
             infect = beta * H * lam               (exposure-driven, NOT bilinear)
```

### Multilayer exposure (Eq. 8/21)
```
Manuscript:  λ^[α]_i = Σ_β κ_αβ Σ_j w̃^[α,β]_ij · D^[β]_j  (Eq. 8)

Code:        lam_l = Wt @ I_l + lam_inter[s:e]      (sir_multilayer.py)
             lam_inter = W_inter @ I                (inter-layer exposure)
```

### Pharmacological model (Eq. 25-30)
```
Manuscript:  dD/dt = +β_eff·H·λ - ρ·D - δ·D - k_F·F·D - D_D·(L·D)  (Eq. 26)

Code:        infect = beta_eff * H * lam
             dD = +infect - recover1 - damage - drug_act - D_D_v*(Lc@D)
```

## Convention

- **Public function names** retain `SIR` for backward compatibility
  (e.g., `run_SIR`, `simulate_SIR_ivp`, `build_default_params_SIR`).
- **Internal variables** use `H/D/R` in docstrings and comments.
- **Plot labels and print statements** use `H(t)/D(t)/R(t)`.
- **The ODE state vector** is ordered `[H | D | R]` (or `[H | D | D2 | R | X | F]`
  in the drug model), matching the manuscript.

## Modules

| Module | Nomenclature status |
|--------|-------------------|
| `main.py` | Fully H/D/R/X (docstring, labels, print, plot titles) |
| `sir_compartments.py` | Docstrings use H/D/R; internal vars use S/I/R (documented aliases) |
| `sir_multilayer.py` | Docstrings use H/D/R; exposure-driven equations (Eq. 1-2-4) |
| `sir_drug.py` | Already uses H/D/D2/R/X internally; API aliases documented |
| `sir_section7.py` | Uses pH/pD/pR (already H/D/R compatible) |
| `sir_invasion_metrics.py` | Uses I_traj (alias for D); documented |
| `sir_transcompartment_metrics.py` | Uses S/I in helpers (aliases for H/D) |
| `sir_sensitivity.py` | Uses SIR metric names (documented aliases) |
