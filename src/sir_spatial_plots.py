"""
sir_spatial_plots.py
====================
Mappe spaziali per pubblicazione scientifica.

Ogni specie biologica ha una colormap dedicata e semanticamente coerente:

  Modello SIR
  -----------
  S (Suscettibili) → Blues       blu = tessuto sano, intatto
  I (Infetti)      → Reds        rosso = infiammazione attiva
  R (Recovered)    → Greens      verde = risoluzione / guarigione

  Modello SIRD + Farmaco
  ----------------------
  S  (Suscettibili)  → Blues
  I  (Infetti)       → Reds
  T  (Treated = I₂)  → Oranges   arancione = in trattamento
  R  (Recovered)     → Greens
  D  (Danno/Morti)   → Greys     grigio = danno irreversibile

Funzioni esportate
------------------
plot_sir_spatial_species        : mappa SIR con 3 colormaps (S/I/R)
plot_sird_spatial_species       : mappa SIRD+farmaco con 5 colormaps
plot_sir_spatial_peak           : confronto final vs picco
plot_sird_spatial_comparison    : farmaco vs no-farmaco (5 specie × 2 condizioni)
attach_and_plot_sir             : wrapper completo per SIR
attach_and_plot_sird            : wrapper completo per SIRD+farmaco

Compatibility
-------------
Tutte le funzioni accettano sia OdeResult (solve_ivp) sia dict (Eulero).
Usano sc.pl.spatial internamente ma aggiungono sempre le colonne
direttamente in adata_st.obs — compatibile con il broadcasting inv2.
"""

from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import ColorbarBase


# ═══════════════════════════════════════════════════════════════════════════
# PALETTE SPECIE  (colormap + label + vmin/vmax biologici)
# ═══════════════════════════════════════════════════════════════════════════

# Ogni entry: (colormap_name, label_completa, descrizione_breve)
SPECIES_PALETTE = {
    "S":  ("Blues",   "S — Suscettibili",       "Tessuto sano"),
    "I":  ("Reds",    "I — Infetti/Infiammati",  "Infiammazione attiva"),
    "T":  ("Oranges", "T — Trattati (I₂)",       "In trattamento farmacologico"),
    "R":  ("Greens",  "R — Recovered",           "Guarigione / risoluzione"),
    "D":  ("Greys",   "D — Danno irreversibile",  "Morte cellulare / fibrosi"),
}

# Ordine di visualizzazione per ciascun modello
SIR_SPECIES  = ["S", "I", "R"]
SIRD_SPECIES = ["S", "I", "T", "R", "D"]


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _expand_to_spots(values: np.ndarray, inv2: np.ndarray) -> np.ndarray:
    """Broadcasting compartimento → spot Visium."""
    return values[inv2].astype(np.float32)


def _attach_species(adata_st, arrays: dict[str, np.ndarray],
                    inv2: np.ndarray, prefix: str = "") -> list[str]:
    """
    Attacca le specie ad adata_st.obs con il prefisso dato.
    Restituisce la lista delle chiavi inserite.
    """
    keys = []
    for sp, arr in arrays.items():
        col = f"{prefix}{sp}"
        adata_st.obs[col] = _expand_to_spots(arr, inv2)
        keys.append(col)
    return keys


def _safe_vmax(arr: np.ndarray, q: float = 0.99) -> float:
    """vmax robusto agli outlier: percentile q dell'array."""
    v = float(np.nanquantile(arr, q))
    return max(v, 1e-6)


def _fig_label(species_key: str) -> tuple[str, str, str]:
    """Restituisce (cmap_name, label, descrizione) per la specie."""
    return SPECIES_PALETTE.get(species_key, ("viridis", species_key, ""))


# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONE BASE: scatter spaziale con colormap dedicata
# ═══════════════════════════════════════════════════════════════════════════

def _plot_spatial_panel(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    cmap_name: str,
    title: str,
    vmin: float = 0.0,
    vmax: float = None,
    spot_size: float = 3.0,
    colorbar: bool = True,
    cbar_label: str = "",
) -> None:
    """
    Disegna un singolo pannello scatter spaziale su ax fornito.

    Parameters
    ----------
    coords    : (N, 2) coordinate Visium
    values    : (N,)  valore per ogni spot
    cmap_name : colormap matplotlib
    title     : titolo del pannello
    vmin/vmax : range colorbar (vmax=None → percentile 99)
    spot_size : dimensione marker scatter
    """
    if vmax is None:
        vmax = _safe_vmax(values)

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=values, cmap=cmap, norm=norm,
        s=spot_size, linewidths=0,
        rasterized=True,          # essenziale per PDF/SVG da paper
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=4, fontweight="bold")

    if colorbar:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02,
                            fraction=0.046)
        cbar.set_label(cbar_label or title, fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        # 3 tick puliti
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}",
                             f"{(vmin+vmax)/2:.2f}",
                             f"{vmax:.2f}"])


# ═══════════════════════════════════════════════════════════════════════════
# SIR — MAPPA SPAZIALE PER SPECIE
# ═══════════════════════════════════════════════════════════════════════════

def plot_sir_spatial_species(
    S_vals:    np.ndarray,
    I_vals:    np.ndarray,
    R_vals:    np.ndarray,
    inv2:      np.ndarray,
    adata_st,
    tag:       str   = "",
    t_label:   str   = "",
    figsize:   tuple = None,
    spot_size: float = 3.0,
    save_path: str   = None,
    dpi:       int   = 300,
) -> plt.Figure:
    """
    Mappa spaziale SIR con colormap dedicata per ogni specie.

    Genera una figura con 3 pannelli affiancati:
      [S — Blues]  [I — Reds]  [R — Greens]

    Ogni pannello ha la propria colorbar calibrata sull'intervallo
    effettivo dei valori (vmax = percentile 99 per robustezza agli outlier).

    Parameters
    ----------
    S_vals, I_vals, R_vals : ndarray (C,) — valori compartimento
    inv2       : ndarray (N,) — broadcast compartimento → spot
    adata_st   : AnnData con obsm['spatial']
    tag        : suffisso per i nomi colonna in adata_st.obs
    t_label    : stringa temporale per il titolo (es. "t = 12.5")
    figsize    : se None, calcolato automaticamente
    save_path  : percorso output (PNG/SVG/PDF)
    dpi        : risoluzione (300 dpi per paper)

    Returns
    -------
    matplotlib.Figure
    """
    coords  = adata_st.obsm["spatial"].astype(np.float32)
    sfx     = f"_{tag}" if tag else ""

    # Espandi agli spot
    S_spot = _expand_to_spots(S_vals, inv2)
    I_spot = _expand_to_spots(I_vals, inv2)
    R_spot = _expand_to_spots(R_vals, inv2)

    # Save to adata_st.obs for pipeline compatibility
    adata_st.obs[f"SIR_S{sfx}"] = S_spot
    adata_st.obs[f"SIR_I{sfx}"] = I_spot
    adata_st.obs[f"SIR_R{sfx}"] = R_spot

    # Figura
    if figsize is None:
        figsize = (14, 4.5)

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={"wspace": 0.35})

    species_data = [
        (axes[0], S_spot, "S", "S — Suscettibili"),
        (axes[1], I_spot, "I", "I — Infetti / Infiammati"),
        (axes[2], R_spot, "R", "R — Recovered"),
    ]

    for ax, vals, sp_key, sp_label in species_data:
        cmap_name, _, descr = _fig_label(sp_key)
        # vmin per S parte da valore minimo (non sempre 0)
        vmin = float(np.nanquantile(vals, 0.01))
        vmax = float(np.nanquantile(vals, 0.99))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-4
        _plot_spatial_panel(
            ax, coords, vals,
            cmap_name=cmap_name,
            title=sp_label,
            vmin=vmin, vmax=vmax,
            spot_size=spot_size,
            cbar_label=f"[0–1]  {descr}",
        )

    t_str = f"  —  {t_label}" if t_label else ""
    fig.suptitle(
        f"Dinamica SIR — distribuzione spaziale per specie{t_str}\n"
        "[ogni pannello ha colormap e scala dedicate — "
        "S=blu · I=rosso · R=verde]",
        fontsize=11, y=1.02,
    )

    _finalize_figure(fig, save_path, dpi)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIR — CONFRONTO FINALE vs PICCO
# ═══════════════════════════════════════════════════════════════════════════

def plot_sir_spatial_peak(
    sol,
    inv2:      np.ndarray,
    adata_st,
    figsize:   tuple = None,
    spot_size: float = 3.0,
    save_path: str   = None,
    dpi:       int   = 300,
) -> plt.Figure:
    """
    Figura 2 × 3: stato at peak di I (riga superiore) e stato final
    (riga inferiore). Scala colorbar CONDIVISA tra le due righe per
    rendere il confronto diretto — standard per paper.

    Layout
    ------
    riga 0 [picco]:  S_pk | I_pk | R_pk
    riga 1 [final]: S_fin| I_fin| R_fin
    """
    coords = adata_st.obsm["spatial"].astype(np.float32)

    # Estrai traiettorie
    if hasattr(sol, "t"):
        t_arr  = sol.t
        nC     = sol.y.shape[0] // 3
        S_t    = sol.y[:nC,     :]
        I_t    = sol.y[nC:2*nC, :]
        R_t    = sol.y[2*nC:,   :]
    else:
        t_arr  = sol["t"]
        S_t    = sol["S"]
        I_t    = sol["I"]
        R_t    = sol["R"]
        nC     = S_t.shape[0]

    t_peak_idx = int(np.argmax(I_t.mean(axis=0)))
    t_peak_val = float(t_arr[t_peak_idx])
    t_end_val  = float(t_arr[-1])

    S_pk  = np.clip(S_t[:, t_peak_idx], 0, 1)
    I_pk  = np.clip(I_t[:, t_peak_idx], 0, 1)
    R_pk  = np.clip(R_t[:, t_peak_idx], 0, 1)
    S_fin = np.clip(S_t[:, -1], 0, 1)
    I_fin = np.clip(I_t[:, -1], 0, 1)
    R_fin = np.clip(R_t[:, -1], 0, 1)

    if figsize is None:
        figsize = (14, 9)

    fig, axes = plt.subplots(2, 3, figsize=figsize,
                             gridspec_kw={"wspace": 0.35, "hspace": 0.35})

    # Range condivisi per specie (confronto diretto)
    ranges = {}
    for sp_key, pk_arr, fin_arr in [("S", S_pk, S_fin),
                                     ("I", I_pk, I_fin),
                                     ("R", R_pk, R_fin)]:
        all_vals = np.concatenate([
            _expand_to_spots(pk_arr, inv2),
            _expand_to_spots(fin_arr, inv2),
        ])
        vmin = float(np.nanquantile(all_vals, 0.01))
        vmax = float(np.nanquantile(all_vals, 0.99))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-4
        ranges[sp_key] = (vmin, vmax)

    rows = [
        (0, S_pk,  I_pk,  R_pk,  f"Picco I  (t = {t_peak_val:.1f})"),
        (1, S_fin, I_fin, R_fin, f"Finale    (t = {t_end_val:.1f})"),
    ]

    for row_idx, S_v, I_v, R_v, row_label in rows:
        for col_idx, (sp_key, vals, sp_label) in enumerate([
            ("S", S_v, "S — Suscettibili"),
            ("I", I_v, "I — Infetti"),
            ("R", R_v, "R — Recovered"),
        ]):
            ax = axes[row_idx][col_idx]
            spot_vals = _expand_to_spots(vals, inv2)
            cmap_name, _, _ = _fig_label(sp_key)
            vmin, vmax = ranges[sp_key]
            _plot_spatial_panel(
                ax, coords, spot_vals,
                cmap_name=cmap_name,
                title=sp_label if row_idx == 0 else "",
                vmin=vmin, vmax=vmax,
                spot_size=spot_size,
                cbar_label="[0–1]",
            )
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9, labelpad=8,
                              fontweight="bold")

    fig.suptitle(
        "Dinamica SIR — confronto Picco di I vs Stato final\n"
        "[scala colorbar condivisa per riga di specie  ·  S=blu · I=rosso · R=verde]",
        fontsize=11, y=1.02,
    )

    _finalize_figure(fig, save_path, dpi)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIRD+FARMACO — MAPPA SPAZIALE PER SPECIE
# ═══════════════════════════════════════════════════════════════════════════

def plot_sird_spatial_species(
    finals:    dict,
    inv2:      np.ndarray,
    adata_st,
    tag:       str   = "drug",
    t_label:   str   = "",
    figsize:   tuple = None,
    spot_size: float = 3.0,
    save_path: str   = None,
    dpi:       int   = 300,
) -> plt.Figure:
    """
    Mappa spaziale SIRD+farmaco con colormap dedicata per 5 specie.

    Genera una figura con 5 pannelli affiancati:
      [S — Blues] [I — Reds] [T — Oranges] [R — Greens] [D — Greys]

    T = specie trattata (I₂ nel codice originale, rinominata per chiarezza).
    F (drug) is excluded from biological maps by design.

    Parameters
    ----------
    finals : dict con chiavi S, I, I2, R, D (ndarray C,)
             output di simulate_SIRD_drug o run_SIRD_drug
    tag    : suffisso per le colonne obs (es. "drug" o "nodrug")
    """
    coords = adata_st.obsm["spatial"].astype(np.float32)
    sfx    = f"_{tag}" if tag else ""

    # Mappa interna: I2 → T (Treated)
    species_map = {
        "S": finals["S"],
        "I": finals["I"],
        "T": finals["I2"],   # I₂ = Treated
        "R": finals["R"],
        "D": finals["D"],
    }

    # Salva in adata_st.obs
    obs_keys = {}
    for sp, arr in species_map.items():
        col = f"SIRD_{sp}{sfx}"
        adata_st.obs[col] = _expand_to_spots(arr, inv2)
        obs_keys[sp] = col

    if figsize is None:
        figsize = (22, 4.5)

    fig, axes = plt.subplots(1, 5, figsize=figsize,
                             gridspec_kw={"wspace": 0.38})

    labels_full = {
        "S": "S — Suscettibili",
        "I": "I — Infetti",
        "T": "T — Trattati (I₂)",
        "R": "R — Recovered",
        "D": "D — Danno irreversibile",
    }

    for ax, sp_key in zip(axes, SIRD_SPECIES):
        spot_vals = adata_st.obs[obs_keys[sp_key]].values
        cmap_name, _, descr = _fig_label(sp_key)
        vmin = float(np.nanquantile(spot_vals, 0.01))
        vmax = float(np.nanquantile(spot_vals, 0.99))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-4
        _plot_spatial_panel(
            ax, coords, spot_vals,
            cmap_name=cmap_name,
            title=labels_full[sp_key],
            vmin=vmin, vmax=vmax,
            spot_size=spot_size,
            cbar_label=f"[0–1]  {descr}",
        )

    t_str    = f"  —  {t_label}" if t_label else ""
    cond_str = "con farmaco" if "drug" in tag else ("senza farmaco" if "nodrug" in tag else tag)
    fig.suptitle(
        f"Dinamica SIRD+Farmaco — distribuzione spaziale per specie{t_str}"
        f"  [{cond_str}]\n"
        "[S=blu · I=rosso · T=arancio · R=verde · D=grigio]",
        fontsize=11, y=1.02,
    )

    _finalize_figure(fig, save_path, dpi)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIRD+FARMACO — CONFRONTO DRUG vs NO-DRUG
# ═══════════════════════════════════════════════════════════════════════════

def plot_sird_spatial_comparison(
    finals_drug:   dict,
    finals_nodrug: dict,
    inv2:          np.ndarray,
    adata_st,
    figsize:       tuple = None,
    spot_size:     float = 3.0,
    save_path:     str   = None,
    dpi:           int   = 300,
) -> plt.Figure:
    """
    Figura 2 × 5 per confronto diretto farmaco vs no-farmaco.

    Layout
    ------
    riga 0 [+farmaco]:   S | I | T | R | D
    riga 1 [-farmaco]:   S | I | T | R | D

    Scala colorbar CONDIVISA per colonna (stessa specie) → confronto
    diretto e quantitativo tra le due condizioni, standard paper.

    Parameters
    ----------
    finals_drug, finals_nodrug : dict {S,I,I2,R,D} — stati finali
    """
    coords = adata_st.obsm["spatial"].astype(np.float32)

    if figsize is None:
        figsize = (22, 9)

    fig, axes = plt.subplots(2, 5, figsize=figsize,
                             gridspec_kw={"wspace": 0.38, "hspace": 0.30})

    labels_full = {
        "S": "S — Suscettibili",
        "I": "I — Infetti",
        "T": "T — Trattati",
        "R": "R — Recovered",
        "D": "D — Danno",
    }

    # Calcola range condivisi per colonna (per specie)
    key_map = {"S": "S", "I": "I", "T": "I2", "R": "R", "D": "D"}
    shared_ranges = {}
    for sp_key, fin_key in key_map.items():
        arr_d  = _expand_to_spots(finals_drug[fin_key],   inv2)
        arr_nd = _expand_to_spots(finals_nodrug[fin_key], inv2)
        all_v  = np.concatenate([arr_d, arr_nd])
        vmin   = float(np.nanquantile(all_v, 0.01))
        vmax   = float(np.nanquantile(all_v, 0.99))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-4
        shared_ranges[sp_key] = (vmin, vmax)

    row_configs = [
        (0, finals_drug,   "+farmaco"),
        (1, finals_nodrug, "−farmaco"),
    ]

    for row_idx, finals, cond_label in row_configs:
        for col_idx, sp_key in enumerate(SIRD_SPECIES):
            ax        = axes[row_idx][col_idx]
            fin_key   = key_map[sp_key]
            spot_vals = _expand_to_spots(finals[fin_key], inv2)
            cmap_name, _, descr = _fig_label(sp_key)
            vmin, vmax = shared_ranges[sp_key]

            _plot_spatial_panel(
                ax, coords, spot_vals,
                cmap_name=cmap_name,
                title=labels_full[sp_key] if row_idx == 0 else "",
                vmin=vmin, vmax=vmax,
                spot_size=spot_size,
                cbar_label="[0–1]",
            )
            if col_idx == 0:
                ax.set_ylabel(cond_label, fontsize=10,
                              fontweight="bold", labelpad=8,
                              color="#c0392b" if row_idx == 0 else "#555555")

    fig.suptitle(
        "Dinamica SIRD+Farmaco — confronto +farmaco vs −farmaco\n"
        "[scala colorbar condivisa per specie (colonna)  ·  "
        "S=blu · I=rosso · T=arancio · R=verde · D=grigio]",
        fontsize=11, y=1.02,
    )

    _finalize_figure(fig, save_path, dpi)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIRD — DELTA: DANNO EVITATO
# ═══════════════════════════════════════════════════════════════════════════

def plot_sird_spatial_delta(
    finals_drug:   dict,
    finals_nodrug: dict,
    inv2:          np.ndarray,
    adata_st,
    figsize:       tuple = None,
    spot_size:     float = 3.0,
    save_path:     str   = None,
    dpi:           int   = 300,
) -> plt.Figure:
    """
    Mappa spaziale delle differenze farmaco − no-farmaco per 3 specie
    chiave (I, R, D).

    Colormap divergente (RdYlGn):
      verde = beneficio del farmaco (↓I, ↑R, ↓D)
      rosso = danno residuo o riduzione recovery

    Interpretazione biologica:
      ΔI = I_nodrug − I_drug   (positivo = farmaco riduce infetti)
      ΔR = R_drug − R_nodrug   (positivo = farmaco aumenta recovery)
      ΔD = D_nodrug − D_drug   (positivo = farmaco riduce danno)
    """
    coords = adata_st.obsm["spatial"].astype(np.float32)

    delta_I = _expand_to_spots(finals_nodrug["I"]  - finals_drug["I"],  inv2)
    delta_R = _expand_to_spots(finals_drug["R"]    - finals_nodrug["R"], inv2)
    delta_D = _expand_to_spots(finals_nodrug["D"]  - finals_drug["D"],  inv2)

    adata_st.obs["SIRD_delta_I"] = delta_I.astype(np.float32)
    adata_st.obs["SIRD_delta_R"] = delta_R.astype(np.float32)
    adata_st.obs["SIRD_delta_D"] = delta_D.astype(np.float32)

    if figsize is None:
        figsize = (16, 5)

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={"wspace": 0.38})

    panels = [
        (axes[0], delta_I, "ΔI = I_nodrug − I_drug\n(verde = meno infetti)"),
        (axes[1], delta_R, "ΔR = R_drug − R_nodrug\n(green = more recovery)"),
        (axes[2], delta_D, "ΔD = D_nodrug − D_drug\n(verde = meno danno)"),
    ]

    for ax, vals, title in panels:
        vabs = float(np.nanquantile(np.abs(vals), 0.99)) + 1e-6
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=vals, cmap="RdYlGn",
            vmin=-vabs, vmax=vabs,
            s=spot_size, linewidths=0, rasterized=True,
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9, pad=4, fontweight="bold")
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_ticks([-vabs, 0, vabs])
        cbar.set_ticklabels([f"−{vabs:.3f}", "0", f"+{vabs:.3f}"])

    fig.suptitle(
        "Beneficio del farmaco — differenze spaziali (+farmaco − senza farmaco)\n"
        "[verde = beneficio  ·  rosso = danno residuo / riduzione]",
        fontsize=11, y=1.02,
    )

    _finalize_figure(fig, save_path, dpi)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# WRAPPER COMPLETI
# ═══════════════════════════════════════════════════════════════════════════

def attach_and_plot_sir(
    S_fin:     np.ndarray,
    I_fin:     np.ndarray,
    R_fin:     np.ndarray,
    sol,
    inv2:      np.ndarray,
    adata_st,
    t_end:     float = None,
    spot_size: float = 3.0,
    save_prefix: str = None,
    dpi:       int   = 300,
) -> dict:
    """
    Wrapper completo per le mappe spaziali SIR.

    Genera:
    1. Mappa stato final (S/I/R — colormap dedicate)
    2. Confronto picco vs final (scala condivisa)

    Parameters
    ----------
    S_fin, I_fin, R_fin : valori finali (C,)
    sol                  : OdeResult o dict con traiettorie
    inv2                 : broadcast compartimento → spot
    adata_st             : AnnData Visium
    save_prefix          : se fornito, salva PNG con questo prefisso

    Returns
    -------
    dict {fig_final, fig_peak}
    """
    t_str = f"t = {t_end:.1f}" if t_end is not None else "final"

    print("\n  [SIR SPATIAL] Figura 1/2 — stato final per specie...")
    fig_final = plot_sir_spatial_species(
        S_fin, I_fin, R_fin,
        inv2=inv2, adata_st=adata_st,
        tag="final", t_label=t_str,
        spot_size=spot_size,
        save_path=f"{save_prefix}_sir_final.png" if save_prefix else None,
        dpi=dpi,
    )

    print("  [SIR SPATIAL] Figura 2/2 — confronto picco vs final...")
    fig_peak = plot_sir_spatial_peak(
        sol, inv2=inv2, adata_st=adata_st,
        spot_size=spot_size,
        save_path=f"{save_prefix}_sir_peak_vs_final.png" if save_prefix else None,
        dpi=dpi,
    )

    return dict(fig_final=fig_final, fig_peak=fig_peak)


def attach_and_plot_sird(
    result:    dict,
    inv2:      np.ndarray,
    adata_st,
    spot_size: float = 3.0,
    save_prefix: str = None,
    dpi:       int   = 300,
) -> dict:
    """
    Wrapper completo per le mappe spaziali SIRD+farmaco.

    Genera:
    1. Mappa stato final — condizione +farmaco (5 specie)
    2. Mappa stato final — condizione −farmaco (5 specie)
    3. Confronto +farmaco vs −farmaco (scala condivisa per specie)
    4. Delta: danno evitato (ΔI, ΔR, ΔD)

    Parameters
    ----------
    result : output di run_SIRD_drug o run_full_drug_analysis["result"]

    Returns
    -------
    dict {fig_drug, fig_nodrug, fig_comparison, fig_delta}
    """
    finals_drug   = result["finals_drug"]
    finals_nodrug = result.get("finals_nodrug")

    print("\n  [SIRD SPATIAL] Figura 1/4 — stato final +farmaco per specie...")
    fig_drug = plot_sird_spatial_species(
        finals_drug, inv2=inv2, adata_st=adata_st,
        tag="drug", t_label="+farmaco",
        spot_size=spot_size,
        save_path=f"{save_prefix}_sird_drug.png" if save_prefix else None,
        dpi=dpi,
    )

    fig_nodrug = fig_comparison = fig_delta = None

    if finals_nodrug is not None:
        print("  [SIRD SPATIAL] Figura 2/4 — stato final −farmaco per specie...")
        fig_nodrug = plot_sird_spatial_species(
            finals_nodrug, inv2=inv2, adata_st=adata_st,
            tag="nodrug", t_label="−farmaco",
            spot_size=spot_size,
            save_path=f"{save_prefix}_sird_nodrug.png" if save_prefix else None,
            dpi=dpi,
        )

        print("  [SIRD SPATIAL] Figura 3/4 — confronto +drug vs −drug...")
        fig_comparison = plot_sird_spatial_comparison(
            finals_drug, finals_nodrug,
            inv2=inv2, adata_st=adata_st,
            spot_size=spot_size,
            save_path=f"{save_prefix}_sird_comparison.png" if save_prefix else None,
            dpi=dpi,
        )

        print("  [SIRD SPATIAL] Figura 4/4 — delta (danno evitato)...")
        fig_delta = plot_sird_spatial_delta(
            finals_drug, finals_nodrug,
            inv2=inv2, adata_st=adata_st,
            spot_size=spot_size,
            save_path=f"{save_prefix}_sird_delta.png" if save_prefix else None,
            dpi=dpi,
        )
    else:
        print("  [SIRD SPATIAL] Nessun risultato no-drug — skip figure 2/3/4.")

    return dict(
        fig_drug       = fig_drug,
        fig_nodrug     = fig_nodrug,
        fig_comparison = fig_comparison,
        fig_delta      = fig_delta,
    )


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES FINALE
# ═══════════════════════════════════════════════════════════════════════════

def _finalize_figure(
    fig:       plt.Figure,
    save_path: str  = None,
    dpi:       int  = 300,
) -> None:
    """Salva (se richiesto) e mostra la figura."""
    plt.tight_layout()
    if save_path:
        # Supporta PNG, PDF, SVG — tutti adatti per paper
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        print(f"    Salvato: {save_path}")
    plt.show()
    plt.close(fig)
