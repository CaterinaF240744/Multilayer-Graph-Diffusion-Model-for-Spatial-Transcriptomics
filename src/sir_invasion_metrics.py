"""
sir_invasion_metrics.py
========================
Invasion metrics per compartment (H/D/R nomenclature, manuscript Eq. 18-20).

Code aliases: I = D (diseased), R = R (recovered), S = H (healthy).

Per ogni compartimento i vengono calcolate:

  1. invasion_time  — primo istante t in cui D_i(t) > threshold  [activation time]
  2. t_peak         — istante at peak di D_i(t)
  3. auc_I          — area sotto curva D_i(t) (integrale trapezoidale)
  4. attack_rate    — R_i final (frazione recovered = danno cumulato)
  5. peak_velocity  — max(dD_i/dt) over time (maximum velocity)

Additional aggregate temporal metrics (peak_I, t_crossing) via
sir_temporal_metrics_full da sir_compartments.py.

NOTA: tutti i valori sono compartment-level.
Spatial maps reflect inv2 broadcasting, not real intra-compartment
intra-compartimento reale.

Funzioni esportate
-------------------
compute_invasion_metrics_per_compartment : calcola tutte le metriche
plot_invasion_heatmap                    : heatmap macro_type × region
plot_invasion_bars                       : bar chart per compartimento
plot_invasion_spatial                    : mappa spaziale su tessuto Visium
invasion_summary                         : tabella riassuntiva
run_sir_invasion_analysis                : wrapper completo con figure
"""

import numpy as np

# ── numpy compat shim ────────────────────────────────────────────────────────
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapezoid
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# CALCOLO METRICHE
# ═══════════════════════════════════════════════════════════════════════════

def compute_invasion_metrics_per_compartment(
    sol,
    comps2:      np.ndarray,
    macro_of:    np.ndarray,
    region_of:   np.ndarray,
    boundary_mask: np.ndarray,
    threshold_I: float = 0.10,
) -> pd.DataFrame:
    """
    Calcola invasion time, attack rate e peak velocity per ogni compartimento.

    Definizioni
    -----------
    invasion_time : primo t in cui I_i(t) > threshold_I
                    (NaN se il compartimento non viene mai invaso)
    attack_rate   : R_i(t_fin) — frazione recovered final
                    (proxy del danno cumulato)
    peak_velocity : max(dI_i/dt) calcolato con np.gradient
                    (maximum propagation velocity in the compartment)

    Parameters
    ----------
    sol          : OdeResult (use_ivp=True) o dict (use_ivp=False)
    comps2       : ndarray (C,) — nomi compartimenti
    macro_of     : ndarray (C,) — tipo cellulare per compartimento
    region_of    : ndarray (C,) — regione per compartimento
    boundary_mask: ndarray bool (C,) — compartimenti seed
    threshold_I  : float — soglia per invasion_time

    Returns
    -------
    pd.DataFrame con colonne:
        compartment, macro_type, region, is_boundary,
        invasion_time, t_peak, auc_I,
        attack_rate, peak_velocity, max_I, final_I
    """
    # Estrai traiettorie
    if hasattr(sol, "t"):
        t_arr  = sol.t
        nC     = sol.y.shape[0] // 3
        I_traj = sol.y[nC:2*nC,   :]   # (C, T)
        R_traj = sol.y[2*nC:3*nC, :]
    else:
        t_arr  = sol["t"]
        I_traj = sol["I"]
        R_traj = sol["R"]
        nC     = I_traj.shape[0]

    rows = []
    for i, comp_name in enumerate(comps2):

        I_i = I_traj[i, :]
        R_i = R_traj[i, :]

        # 1. Invasion time (crossing time)
        crossed = np.where(I_i > threshold_I)[0]
        if len(crossed) > 0:
            inv_time = float(t_arr[crossed[0]])
        else:
            inv_time = np.nan

        # 2. Time at peak di I
        t_peak_i = float(t_arr[np.argmax(I_i)])

        # 3. AUC di I(t)
        auc_i = float(_trapz(I_i, t_arr))

        # 4. Attack rate (R final)
        attack_rate = float(R_i[-1])

        # 5. Peak velocity — dI/dt massimo
        dI_dt         = np.gradient(I_i, t_arr)
        peak_velocity = float(dI_dt.max())

        rows.append(dict(
            compartment   = comp_name,
            macro_type    = macro_of[i],
            region        = region_of[i],
            is_boundary   = bool(boundary_mask[i]),
            invasion_time = inv_time,
            t_peak        = t_peak_i,
            auc_I         = auc_i,
            attack_rate   = attack_rate,
            peak_velocity = peak_velocity,
            max_I         = float(I_i.max()),
            final_I       = float(I_i[-1]),
        ))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TESTUALE
# ═══════════════════════════════════════════════════════════════════════════

def invasion_summary(df: pd.DataFrame) -> None:
    """
    Stampa una tabella riassuntiva ordinata per invasion_time.
    I compartimenti non invasi (NaN) appaiono in fondo.
    Include le metriche temporali: t_peak, auc_I.
    """
    df_sorted = df.sort_values("invasion_time", na_position="last")

    print("\n" + "═"*100)
    print(f"{'Compartimento':<25} {'macro':<10} {'region':<10} "
          f"{'inv_time':>9} {'t_peak':>7} {'auc_I':>7} "
          f"{'attack_R':>9} {'peak_dI/dt':>11} {'max_I':>7}")
    print("─"*100)

    for _, r in df_sorted.iterrows():
        seed  = " [SEED]" if r["is_boundary"] else ""
        it    = f"{r['invasion_time']:.2f}" if not np.isnan(r["invasion_time"]) else "  —"
        tp    = f"{r['t_peak']:.2f}"        if "t_peak" in r else "  —"
        auc   = f"{r['auc_I']:.3f}"         if "auc_I"  in r else "  —"
        print(f"  {r['compartment']:<23}{seed:<8} {r['macro_type']:<10} "
              f"{r['region']:<10} {it:>9} {tp:>7} {auc:>7} "
              f"{r['attack_rate']:>9.3f} {r['peak_velocity']:>11.4f} "
              f"{r['max_I']:>7.3f}")

    print("═"*100)
    n_inv = df["invasion_time"].notna().sum()
    print(f"  Compartimenti invasi: {n_inv}/{len(df)}")
    print(f"  Invasion time medio (invasi): "
          f"{df['invasion_time'].mean():.2f}")
    if "t_peak" in df.columns:
        print(f"  T_peak medio:                 "
              f"{df['t_peak'].mean():.2f}")
    if "auc_I" in df.columns:
        print(f"  AUC_I medio:                  "
              f"{df['auc_I'].mean():.3f}")
    print(f"  Attack rate medio globale:    "
          f"{df['attack_rate'].mean():.3f}")
    print(f"  Peak velocity medio:          "
          f"{df['peak_velocity'].mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# HEATMAP  macro_type × region
# ═══════════════════════════════════════════════════════════════════════════

def plot_invasion_heatmap(
    df:      pd.DataFrame,
    metric:  str   = "invasion_time",
    figsize: tuple = (9, 5),
    cmap:    str   = "YlOrRd_r",
    title:   str   = None,
):
    """
    Heatmap della metrica scelta su griglia macro_type × region.

    Supporta sia la zonazione anatomica (cortex / outer_medulla /
    inner_medulla) sia quella radiale legacy (boundary / mid / core).
    Column order is determined automatically from the content
    del DataFrame, con un ordine preferenziale biologico quando i nomi
    corrispondono.

    Parameters
    ----------
    metric : 'invasion_time' | 'attack_rate' | 'peak_velocity' |
             'max_I' | 't_peak' | 'auc_I'
    cmap   : 'YlOrRd_r' per invasion_time (verde=presto), altrimenti 'YlOrRd'
    """
    pivot = df.pivot_table(
        index="macro_type", columns="region",
        values=metric, aggfunc="mean",
    )

    # Ordine preferenziale: anatomico prima, radiale come fallback
    preferred_order = [
        "outer_medulla", "cortex", "inner_medulla",   # anatomico
        "boundary", "mid", "core",                    # radiale legacy
    ]
    col_order = [c for c in preferred_order if c in pivot.columns]
    col_order += [c for c in pivot.columns if c not in col_order]

    # Se sono presenti zone anatomiche, rimuovi le zone radiali legacy
    # per evitare colonne miste che confondono la lettura
    anatomical = {"outer_medulla", "cortex", "inner_medulla"}
    radial     = {"boundary", "mid", "core"}
    present    = set(pivot.columns)
    if present & anatomical:   # almeno una zona anatomica presente
        col_order = [c for c in col_order if c not in radial]

    if not col_order:
        col_order = list(pivot.columns)
    pivot = pivot[col_order]

    valid = pivot.values[~np.isnan(pivot.values)]
    if valid.size == 0:
        print(f"  [plot_invasion_heatmap] nessun valore valido per '{metric}' — skip.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticklabels(pivot.index,   fontsize=11)

    threshold = valid.mean()
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if (not np.isnan(val) and val > threshold)
                    else "black")

    fig.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel("Zona spaziale", fontsize=11)
    ax.set_ylabel("Macro type", fontsize=11)
    ax.set_title(title or f"Heatmap invasione — {metric}", fontsize=13)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# BAR CHART per compartimento
# ═══════════════════════════════════════════════════════════════════════════

def plot_invasion_bars(
    df:      pd.DataFrame,
    figsize: tuple = (14, 10),
):
    """
    Tre bar chart verticali: invasion_time, attack_rate, peak_velocity.
    I compartimenti sono ordinati per invasion_time crescente.
    I seed (boundary) sono evidenziati in rosso scuro.
    """
    df_sorted = df.sort_values("invasion_time", na_position="last").copy()
    labels    = df_sorted["compartment"].tolist()
    x         = np.arange(len(labels))
    colors    = ["#8B0000" if b else "#4878CF"
                 for b in df_sorted["is_boundary"]]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Invasion time
    it_vals = df_sorted["invasion_time"].fillna(0).values
    it_cols = ["#cccccc" if np.isnan(v) else c
               for v, c in zip(df_sorted["invasion_time"], colors)]
    bars = axes[0].bar(x, it_vals, color=it_cols, edgecolor="white", width=0.7)
    axes[0].set_ylabel("Invasion time (t)", fontsize=10)
    axes[0].set_title("Invasion time per compartimento\n"
                      "(NaN → compartimento non invaso → barra grigia)", fontsize=10)
    # Annotazione NaN
    for xi, (_, row) in zip(x, df_sorted.iterrows()):
        if np.isnan(row["invasion_time"]):
            axes[0].text(xi, 0.2, "n.i.", ha="center", va="bottom",
                         fontsize=7, color="#888888")

    # Attack rate
    axes[1].bar(x, df_sorted["attack_rate"].values,
                color=colors, edgecolor="white", width=0.7)
    axes[1].set_ylabel("Attack rate R(t_fin)", fontsize=10)
    axes[1].set_title("Attack rate final per compartimento", fontsize=10)
    axes[1].set_ylim(0, df_sorted["attack_rate"].max() * 1.15)

    # Peak velocity
    axes[2].bar(x, df_sorted["peak_velocity"].values,
                color=colors, edgecolor="white", width=0.7)
    axes[2].set_ylabel("Peak dI/dt", fontsize=10)
    axes[2].set_title("Maximum propagation velocity", fontsize=10)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=40, ha="right", fontsize=9)

    # Legenda
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#8B0000", label="Boundary (seed)"),
        Patch(facecolor="#4878CF", label="Core / Mid"),
        Patch(facecolor="#cccccc", label="Non invaso"),
    ]
    axes[0].legend(handles=legend_els, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MAPPA SPAZIALE su tessuto Visium
# ═══════════════════════════════════════════════════════════════════════════

def plot_invasion_spatial(
    df:       pd.DataFrame,
    comps2:   np.ndarray,
    adata_st,
    inv2:     np.ndarray,
    metrics:  list = None,
):
    """
    Mappa spaziale delle metriche di invasione su tessuto Visium.

    Metriche e colormap scelte biologicamente:
      invasion_time  → YlOrRd_r (rosso=presto, giallo=tardi) — intuitivo
      attack_rate    → YlOrRd, clippato a [0,1]
      t_peak         → YlOrRd_r (red=early peak) — more informative than
                       peak_velocity which has low spatial variability
    """
    import scanpy as sc

    if metrics is None:
        metrics = ["invasion_time", "attack_rate", "t_peak"]

    comp_to_idx = {name: i for i, name in enumerate(comps2)}

    # Colormap per metrica
    metric_cmaps = {
        "invasion_time": "YlOrRd_r",   # invertita: rosso=invaso presto
        "attack_rate":   "YlOrRd",
        "t_peak":        "YlOrRd_r",   # invertita: rosso=picco precoce
        "peak_velocity": "YlOrRd",
        "auc_I":         "YlOrRd",
        "max_I":         "YlOrRd",
    }

    for metric in metrics:
        vals = np.full(len(comps2), np.nan)
        for _, row in df.iterrows():
            idx = comp_to_idx.get(row["compartment"])
            if idx is not None:
                v = row[metric]
                # Clippa attack_rate a [0,1]
                if metric == "attack_rate":
                    v = float(np.clip(v, 0.0, 1.0))
                vals[idx] = v

        vals_vis = np.where(np.isnan(vals), 0.0, vals)
        col = f"SIR_inv_{metric}"
        adata_st.obs[col] = vals_vis[inv2].astype(np.float32)

    titles = {
        "invasion_time": "Invasion time (rosso=presto)",
        "attack_rate":   "Attack rate [0-1]",
        "t_peak":        "Time at peak I (rosso=precoce)",
        "peak_velocity": "Maximum velocity dI/dt",
        "auc_I":         "AUC di I(t)",
        "max_I":         "Peak I",
    }

    for metric in metrics:
        cmap = metric_cmaps.get(metric, "YlOrRd")
        sc.pl.spatial(
            adata_st,
            color=[f"SIR_inv_{metric}"],
            cmap=cmap,
            title=[titles.get(metric, metric)],
        )


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def run_sir_invasion_analysis(
    sol,
    comps2:       np.ndarray,
    macro_of:     np.ndarray,
    region_of:    np.ndarray,
    boundary_mask: np.ndarray,
    adata_st,
    inv2:         np.ndarray,
    sir_params:   dict = None,
    threshold_I:  float = 0.10,
) -> pd.DataFrame:
    """
    Wrapper completo: metriche di invasione + metriche temporali + figure.

    Aggiunto rispetto alla versione precedente:
    - sir_temporal_metrics_full: peak_I, t_peak, auc_I, t_crossing
      per ogni combinazione macro_type × region
    - print_sir_temporal_summary: tabella riepilogativa
    - plot_sir_temporal: figura dinamica I(t)

    Returns
    -------
    df : pd.DataFrame con tutte le metriche per compartimento
         (include t_peak e auc_I)
    """
    from sir_compartments import (
        sir_temporal_metrics_full,
        print_sir_temporal_summary,
        plot_sir_temporal,
    )

    print("\n" + "="*60)
    print("SIR — Metriche di invasione per compartimento")
    print("="*60)

    # ── Metriche per compartimento (con t_peak e auc_I) ──────────────────
    df = compute_invasion_metrics_per_compartment(
        sol, comps2, macro_of, region_of, boundary_mask,
        threshold_I=threshold_I,
    )
    invasion_summary(df)

    # ── Metriche temporali aggregate ─────────────────────────────────────
    if sir_params is not None:
        print("\n" + "="*60)
        print("SIR — Metriche temporali (peak_I, t_peak, auc_I, t_crossing)")
        print("="*60)
        tm_full = sir_temporal_metrics_full(
            sol, macro_of, region_of, sir_params,
            threshold_I=threshold_I,
        )
        print_sir_temporal_summary(tm_full, comps2=comps2)

        # Figura dinamica temporale
        plot_sir_temporal(
            sol, macro_of, region_of, sir_params,
            comps2=comps2, threshold_I=threshold_I,
        )
    else:
        print("\n  [NOTA] Passa sir_params per ottenere metriche temporali "
              "e plot_sir_temporal.")

    # ── Figure spaziali ──────────────────────────────────────────────────
    for metric, cmap, ttl in [
        ("invasion_time",  "YlOrRd_r", "Invasion time (basso = prima)"),
        ("attack_rate",    "YlOrRd",   "Attack rate final R(t_fin)"),
        ("peak_velocity",  "YlOrRd",   "Maximum velocity dI/dt"),
        ("t_peak",         "YlOrRd",   "Time at peak di I"),
        ("auc_I",          "YlOrRd",   "AUC di I(t)"),
    ]:
        if metric in df.columns:
            plot_invasion_heatmap(df, metric=metric, cmap=cmap, title=ttl)

    plot_invasion_bars(df)
    plot_invasion_spatial(df, comps2, adata_st, inv2)

    return df