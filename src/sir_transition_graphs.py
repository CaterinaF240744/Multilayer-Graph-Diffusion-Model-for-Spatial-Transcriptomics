"""
sir_transition_graphs.py  —  v4  (edge-weight + font + legend fixes)
=====================================================================
Changes vs v3
-------------
1. EDGE THICKNESS   — _INTRA_LW_SCALE 0.75→2.2, _INTRA_ALPHA 0.28→0.45
                      _INTER_LW 0.85→2.0,        _INTER_ALPHA 0.22→0.40
2. TITLE FONT       — _FS_TITLE 18→20, _FS_LEGEND 12→13
3. LEGEND ORDER     — all legends now follow:
                       PT kNN → DCT kNN → TAL kNN →
                       Inter-layer → Boundary seed → Size∝βᵢ
                       (drug figure appends High-pD / High-pX / Healthy)
"""

from __future__ import annotations
import os
import numpy as np
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.lines import Line2D

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapezoid

# ── Palette ──────────────────────────────────────────────────────────────────
MACRO_COLOR = {
    "PT":       "#c0392b",
    "DCT":      "#2471a3",
    "TAL":      "#1e8449",
    "vascular": "#7d3c98",
    "immune":   "#d35400",
    "other":    "#717d7e",
}
LAYER_COLORS = {"PT": "#c0392b", "DCT": "#2471a3", "TAL": "#1e8449"}
_FIGBG = "#FFFFFF"

# ── Tuning constants ──────────────────────────────────────────────────────────
_INTRA_LW_SCALE  = 2.5       # ← was 0.75
_INTRA_ALPHA     = 0.50       # ← was 0.28
_INTRA_MAX_EDGES = 120

_INTER_N_SAMPLE  = 8
_INTER_LW        = 2.0       # ← was 0.85
_INTER_ALPHA     = 0.45      # ← was 0.22

_NODE_MIN   = 6.0
_NODE_MAX   = 20.0
_SEED_BONUS = 18.0

_FS_BADGE  = 14.0
_FS_TITLE  = 20.0             # ← was 18.0
_FS_LEGEND = 15.0             # ← was 12.0
_FS_CB     = 14.0

_CMAP_LOAD = mcolors.LinearSegmentedColormap.from_list(
    "DiseaseLoad",
    ["#f0faf2", "#f9e4b7", "#e67e22", "#c0392b", "#7b241c"],
    N=512,
)


# ════════════════════════════════════════════════════════════════════════════
# DISPLAY / SAVE HELPER
# ════════════════════════════════════════════════════════════════════════════

def _show(fig, save_path: str | None, dpi: int = 300):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=_FIGBG)
        print(f"    → saved: {save_path}")
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if any(b in backend for b in ("agg", "svg", "pdf", "cairo", "gdk")):
        pass
    else:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _to_dense(W):
    if sp.issparse(W):
        return W.toarray().astype(np.float64)
    return np.asarray(W, dtype=np.float64)


def _rgba_to_hex(rgba: np.ndarray) -> list[str]:
    return [mcolors.to_hex(row[:3]) for row in np.asarray(rgba)]


def _adaptive_node_size(values, coords,
                         base_min=_NODE_MIN, base_max=_NODE_MAX) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
    if len(coords) > 1:
        from scipy.spatial import KDTree
        dists, _ = KDTree(coords).query(coords, k=2)
        med_nn = float(np.median(dists[:, 1]))
        coord_range = float(np.ptp(coords[:, 0]) + np.ptp(coords[:, 1])) / 2
        cap = (med_nn / (coord_range + 1e-9)) * 1400
        base_max = min(base_max, max(cap, base_min + 2.0))
    return base_min + (base_max - base_min) * v_norm


def _draw_knn_edges(ax, coords, Wsp_csr, mask, x_off=0.0,
                    color="#aaaaaa",
                    lw_scale=_INTRA_LW_SCALE,
                    alpha=_INTRA_ALPHA,
                    max_edges=_INTRA_MAX_EDGES,
                    zorder=2):
    idx     = np.where(mask)[0]
    idx_set = set(idx.tolist())
    coo     = Wsp_csr.tocoo()
    w_max   = float(coo.data.max()) + 1e-9
    drawn   = 0
    for si, sj, w in zip(coo.row, coo.col, coo.data):
        if drawn >= max_edges:
            break
        if si >= sj:
            continue
        if si not in idx_set or sj not in idx_set:
            continue
        lw = 0.50 + lw_scale * (w / w_max)   # raised baseline
        ax.plot(
            [coords[si, 0] + x_off, coords[sj, 0] + x_off],
            [coords[si, 1],          coords[sj, 1]],
            color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round",
        )
        drawn += 1


def _draw_inter_edges(ax, coords, macro_spot, ma, mb,
                      x_off_a, x_off_b, Wsp_csr,
                      n_sample=_INTER_N_SAMPLE,
                      color="#c8c8c8", lw=_INTER_LW,
                      alpha=_INTER_ALPHA, zorder=1):
    sa = np.where(macro_spot == ma)[0]
    sb = np.where(macro_spot == mb)[0]
    if len(sa) == 0 or len(sb) == 0:
        return
    rng = np.random.default_rng(42)
    n   = min(n_sample, len(sa), len(sb))
    for si, sj in zip(rng.choice(sa, n, replace=False),
                      rng.choice(sb, n, replace=False)):
        ax.plot(
            [coords[si, 0] + x_off_a, coords[sj, 0] + x_off_b],
            [coords[si, 1],            coords[sj, 1]],
            color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round",
        )


def _layer_badge(ax, xc, yc, label, color, r_diff=None, peak=None):
    lines = [label]
    if r_diff is not None:
        lines.append(
            f"$R_{{\\mathrm{{diff}}}}$={r_diff:.2f}  "
            f"$p^D_{{\\mathrm{{peak}}}}$={peak:.3f}"
        )
    ax.text(xc, yc, "\n".join(lines),
            ha="center", va="bottom",
            fontsize=_FS_BADGE, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=color, alpha=0.88, lw=1.0),
            zorder=10, clip_on=False)


def _layer_gap(coords, gap_frac=0.45) -> float:
    return float(np.ptp(coords[:, 0])) * (1.0 + gap_frac)


def _add_colorbar(fig, ax_ref, cmap, norm, label, shrink=0.38, pad=0.015):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_ref, shrink=shrink, pad=pad, location="right")
    cb.set_label(label, fontsize=_FS_CB)
    cb.ax.tick_params(labelsize=8.5)
    return cb


# ── legend builder (canonical order) ─────────────────────────────────────────

def _make_legend_handles(macros, extra=None):
    """
    Return legend handles in canonical order:
      PT kNN → DCT kNN → TAL kNN → Inter-layer → Boundary seed → Size∝βᵢ
    followed by any items in `extra` (list of handle objects).
    """
    handles = [
        mpatches.Patch(facecolor=LAYER_COLORS[m], edgecolor="none",
                       label=f"{m} kNN")
        for m in macros if m in LAYER_COLORS
    ]
    handles += [
        Line2D([0],[0], color="#c8c8c8", lw=2.5, label="Inter-layer"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="#cc0000", markeredgewidth=2.0,
               markersize=10, label="Boundary seed"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
               markersize=7, label=r"Size$\propto\beta_i$"),
    ]
    if extra:
        handles += extra
    return handles


# ════════════════════════════════════════════════════════════════════════════
# RADIAL LAYOUT / RELAX
# ════════════════════════════════════════════════════════════════════════════

def _make_radial_layout(coords, macro_s, macros, x_off) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    new = coords.copy()
    base_range = float(np.ptp(coords[:, 0]) + np.ptp(coords[:, 1]) + 1e-9)
    base_r = 0.22 * base_range
    for m in macros:
        idx = np.where(macro_s == m)[0]
        if len(idx) == 0:
            continue
        ang = np.linspace(0, 2*np.pi, len(idx), endpoint=False)
        r = base_r
        new[idx, 0] = x_off[m] + r * np.cos(ang)
        new[idx, 1] = r * np.sin(ang)
    return new


def _relax(coords, iterations=20, min_dist=5.0, strength=0.002):
    coords = coords.copy().astype(float)
    N = len(coords)
    min_dist2 = min_dist * min_dist
    for _ in range(iterations):
        for i in range(N):
            for j in range(i+1, N):
                dx = coords[i,0] - coords[j,0]
                dy = coords[i,1] - coords[j,1]
                dist2 = dx*dx + dy*dy + 1e-9
                if dist2 < min_dist2:
                    force = strength / dist2
                    coords[i,0] += dx * force
                    coords[i,1] += dy * force
                    coords[j,0] -= dx * force
                    coords[j,1] -= dy * force
    return coords


# ════════════════════════════════════════════════════════════════════════════
# SHARED PANEL DRAW
# ════════════════════════════════════════════════════════════════════════════

def _draw_one_panel(ax, coords, macro_spot, bnd_spot, beta_spot,
                    value_spot, cmap, norm, macros, x_off,
                    Wsp_csr, n_inter, y_range,
                    show_badges=True, badge_kw=None):
    badge_kw = badge_kw or {}

    for i_m, ma in enumerate(macros):
        for mb in macros[i_m+1:]:
            _draw_inter_edges(ax, coords, macro_spot,
                              ma, mb, x_off[ma], x_off[mb], Wsp_csr,
                              n_sample=n_inter, zorder=1)

    for m in macros:
        _draw_knn_edges(ax, coords, Wsp_csr,
                        macro_spot == m, x_off=x_off[m],
                        color=LAYER_COLORS.get(m, "#888888"), zorder=2)

    for m in macros:
        idxs = np.where(macro_spot == m)[0]
        if len(idxs) == 0:
            continue
        xs       = coords[idxs, 0] + x_off[m]
        ys       = coords[idxs, 1]
        vals     = value_spot[idxs]
        b_m      = beta_spot[idxs]
        bnd      = bnd_spot[idxs]
        base_sz  = _adaptive_node_size(b_m, np.column_stack([xs, ys]))
        node_hex = _rgba_to_hex(cmap(norm(vals)))
        ec_layer = LAYER_COLORS.get(m, "#444444")

        nsel = ~bnd
        if nsel.any():
            ax.scatter(xs[nsel], ys[nsel],
                       s=base_sz[nsel],
                       facecolors=[node_hex[i] for i in np.where(nsel)[0]],
                       edgecolors=ec_layer,
                       linewidths=0.90, alpha=0.93, zorder=3)
        bsel = bnd
        if bsel.any():
            ax.scatter(xs[bsel], ys[bsel],
                       s=base_sz[bsel] + _SEED_BONUS,
                       facecolors=[node_hex[i] for i in np.where(bsel)[0]],
                       edgecolors="#aa0000",
                       linewidths=3.2, alpha=0.97, zorder=5)

    if show_badges:
        for m in macros:
            idxs = np.where(macro_spot == m)[0]
            if len(idxs) == 0:
                continue
            xs = coords[idxs, 0] + x_off[m]
            ys = coords[idxs, 1]
            _layer_badge(ax,
                         float(xs.mean()),
                         float(ys.max()) + y_range * 0.10,
                         m, color=LAYER_COLORS.get(m, "#333333"),
                         **badge_kw.get(m, {}))


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — SINGLE-LAYER CONCENTRIC
# ════════════════════════════════════════════════════════════════════════════

def plot_single_layer_transition(
    sol, Wc2, comps2, macro_of, region_of, boundary_mask,
    sir_params, figsize=(14, 11), save_path=None, dpi=300,
):
    nC     = len(comps2)
    beta_v = np.asarray(sir_params["beta"],  dtype=float)
    gam_v  = np.asarray(sir_params["gamma"], dtype=float)
    R_diff = beta_v / np.where(gam_v > 0, gam_v, 1e-12)
    t      = sol.t
    n_comp = sol.y.shape[0] // 3
    I_traj = sol.y[n_comp:2*n_comp, :]
    peak_D = I_traj.max(axis=1)

    Wc2_arr = _to_dense(Wc2)
    S_traj  = sol.y[:n_comp, :]
    Kdyn    = np.zeros((nC, nC))
    for i in range(nC):
        for j in range(nC):
            if i == j or Wc2_arr[i, j] <= 0:
                continue
            Kdyn[i, j] = float(_trapz(
                beta_v[i]*Wc2_arr[i,j]*S_traj[j,:]*I_traj[i,:], t))

    ZONE = {"boundary":0,"outer_medulla":1,"cortex":2,
            "inner_medulla":3,"core":4,"mid":3}
    zi   = np.array([ZONE.get(r, 2) for r in region_of])
    rad  = np.linspace(1.0, 5.0, 5)
    pos  = np.zeros((nC, 2))
    for ring in range(5):
        ix = np.where(zi == ring)[0]
        if len(ix) == 0:
            continue
        ang = np.linspace(0, 2*np.pi, len(ix), endpoint=False)
        pos[ix, 0] = rad[ring]*np.cos(ang)
        pos[ix, 1] = rad[ring]*np.sin(ang)
    pos += np.random.default_rng(0).uniform(-0.10, 0.10, pos.shape)

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FIGBG)
    plt.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.05)
    ax.set_facecolor(_FIGBG); ax.set_aspect("equal"); ax.axis("off")

    zlabels = {0:"Boundary (seed)",1:"Outer medulla",2:"Cortex",
               3:"Inner medulla",4:"Core"}
    for ring in range(5):
        ax.add_patch(plt.Circle((0,0), rad[ring], color="#e4e4e4",
                                fill=False, lw=0.6, ls="--", zorder=0))
        a0 = -np.pi/5
        ax.text(rad[ring]*np.cos(a0)+0.05, rad[ring]*np.sin(a0),
                zlabels[ring], fontsize=6.0, color="#b0b0b0",
                ha="center", va="center", zorder=0)

    kv = Kdyn[Kdyn > 0]
    if len(kv):
        thr = np.percentile(kv, 55)
        km  = float(kv.max())
        for i in range(nC):
            for j in range(nC):
                if Kdyn[i,j] < thr:
                    continue
                wn  = (Kdyn[i,j]-thr)/(km-thr+1e-9)
                col = MACRO_COLOR.get(macro_of[i], "#888888")
                ax.annotate("", xy=pos[j], xytext=pos[i],
                            arrowprops=dict(
                                arrowstyle="->,head_width=0.14,head_length=0.18",
                                color=col, lw=0.65+3.0*wn,
                                alpha=0.20+0.65*wn,
                                connectionstyle="arc3,rad=0.12"), zorder=2)

    cmap_rd = matplotlib.colormaps["RdYlGn_r"]
    norm_rd = mcolors.Normalize(vmin=R_diff.min(), vmax=R_diff.max())
    pd_lo, pd_hi = peak_D.min(), max(peak_D.max(), 1e-6)
    for i in range(nC):
        x, y  = pos[i]
        rn    = (peak_D[i]-pd_lo)/(pd_hi-pd_lo+1e-9)
        sz    = 55 + 520*rn
        col_h = mcolors.to_hex(cmap_rd(norm_rd(R_diff[i])))
        ec    = "#cc0000" if boundary_mask[i] else "#333333"
        ew    = 2.0       if boundary_mask[i] else 0.6
        ax.scatter(x, y, s=sz, facecolors=col_h,
                   edgecolors=ec, linewidths=ew, zorder=4, alpha=0.91)

    sm = plt.cm.ScalarMappable(cmap=cmap_rd, norm=norm_rd)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.40, pad=0.01)
    cb.set_label(r"$R_\mathrm{diff}=\alpha/\kappa$", fontsize=_FS_CB)
    cb.ax.tick_params(labelsize=7.5)

    macros_present = list(np.unique(macro_of))
    leg = (
        [Line2D([0],[0], color=MACRO_COLOR.get(m,"#888"), lw=2.0,
                label=f"{m} outflow") for m in macros_present] +
        [Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                markeredgecolor="#cc0000", markeredgewidth=2.0,
                markersize=8, label="Seed"),
         Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                markersize=3, label=r"Small=low $p^D$"),
         Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                markersize=9, label=r"Large=high $p^D$")]
    )
    ax.legend(handles=leg, bbox_to_anchor=(0.8,0.7), fontsize=_FS_LEGEND, 
              framealpha=0.90, facecolor=_FIGBG, edgecolor= "#444444", borderpad=0.4, ncol=2, 
              
              title="Cell type / Node encoding", title_fontsize=12)
    ax.set_title(
        "Single-layer diffusion graph — concentric layout\n"
        r"Colour=$R_\mathrm{diff}$  ·  Size=peak $p^D$  ·  Edge=$K_{AB}$",
        fontsize=_FS_TITLE, pad=10)

    _show(fig, save_path, dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — SINGLE-LAYER SPOT-LEVEL
# ════════════════════════════════════════════════════════════════════════════

def plot_single_layer_transition_spots(
    sol, comp, sir_params, adata_st,
    figsize=(12, 10), save_path=None, dpi=300,
):
    coords   = adata_st.obsm["spatial"].astype(np.float32)
    inv2     = comp["inv2"]
    macro_of = comp["macro_of"]
    bnd_mask = comp["boundary_mask"]
    Wsp      = comp["Wsp"]

    beta_v = np.asarray(sir_params["beta"],  dtype=float)
    gam_v  = np.asarray(sir_params["gamma"], dtype=float)
    R_diff = beta_v / np.where(gam_v > 0, gam_v, 1e-12)

    R_spot   = R_diff[inv2]
    macro_s  = macro_of[inv2]
    bnd_s    = bnd_mask[inv2]
    cmap_rd  = matplotlib.colormaps["RdYlGn_r"]
    norm_rd  = mcolors.Normalize(vmin=R_diff.min(), vmax=R_diff.max())
    sz       = _adaptive_node_size(R_spot, coords, base_min=3.0, base_max=20.0)
    Wsp_csr  = sp.csr_matrix(Wsp)
    node_hex = _rgba_to_hex(cmap_rd(norm_rd(R_spot)))

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FIGBG)
    plt.subplots_adjust(left=0.02, right=0.88, top=0.91, bottom=0.04)
    ax.set_facecolor(_FIGBG); ax.set_aspect("equal"); ax.axis("off")

    for m in np.unique(macro_s):
        _draw_knn_edges(ax, coords, Wsp_csr, macro_s == m,
                        color=MACRO_COLOR.get(m, "#888888"),
                        lw_scale=2.0, alpha=0.45, max_edges=400, zorder=2)

    nsel = ~bnd_s
    if nsel.any():
        ax.scatter(coords[nsel,0], coords[nsel,1], s=sz[nsel],
                   facecolors=[node_hex[i] for i in np.where(nsel)[0]],
                   edgecolors="#555555", linewidths=0.30,
                   alpha=0.91, zorder=3)
    bsel = bnd_s
    if bsel.any():
        ax.scatter(coords[bsel,0], coords[bsel,1], s=sz[bsel]+_SEED_BONUS,
                   facecolors=[node_hex[i] for i in np.where(bsel)[0]],
                   edgecolors="#cc0000", linewidths=1.6,
                   alpha=0.97, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap_rd, norm=norm_rd)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.01)
    cb.set_label(r"$R_\mathrm{diff}=\alpha/\kappa$", fontsize=_FS_CB)
    cb.ax.tick_params(labelsize=7.5)

    macros_present = [m for m in np.unique(macro_s) if m in MACRO_COLOR]
    leg = _make_legend_handles(
        macros_present,
        extra=[Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                      markersize=3.5, label=r"Size$\propto R_\mathrm{diff}$")]
    )
    ax.legend(handles=leg, bbox_to_anchor=(0.8,1.0), fontsize=_FS_LEGEND,
              framealpha=0.90, facecolor=_FIGBG, edgecolor='#444444', borderpad=0.4, ncol=2)
    ax.set_title(
        "Single-layer transition graph — spot-level Visium layout\n"
        r"Colour=$R_\mathrm{diff}$  ·  Size$\propto R_\mathrm{diff}$  ·  Edge=$w_{ij}$",
        fontsize=_FS_TITLE, pad=10)

    _show(fig, save_path, dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — MULTILAYER STATIC
# ════════════════════════════════════════════════════════════════════════════

def plot_multilayer_transition(
    sol, net, W_inter, sir_params,
    comps2, macro_of, region_of, boundary_mask,
    adata_st=None, inv2=None, Wsp=None,
    figsize=None, n_inter_edges=_INTER_N_SAMPLE,
    save_path=None, dpi=300,
):
    layers      = net["layers"]
    node_ranges = net["node_ranges"]
    global_idx  = net["global_index"]
    n_total     = net["n_total"]
    macros      = [m for m in ("PT","DCT","TAL") if m in layers]

    beta_g = np.asarray(sir_params["beta"],  dtype=float)
    gam_g  = np.asarray(sir_params["gamma"], dtype=float)

    if adata_st is None:
        raise ValueError("adata_st required")

    coords_raw  = adata_st.obsm["spatial"].astype(np.float32)
    step    = np.ptp(coords_raw[:,0]) * 0.65
    y_range = float(np.ptp(coords_raw[:, 1]))
    x_off   = {m: i*step for i, m in enumerate(macros)}

    macro_s = np.full(len(inv2), 'other', dtype=object)
    for m in macros:
        macro_s[np.isin(inv2, layers[m]['nodes'])] = m

    coords = coords_raw.copy().astype(float)
    for m in macros:
        idx = np.where(macro_s==m)[0]
        coords[idx]= _relax(coords[idx], iterations=25, min_dist=6.0, strength=0.003)

    bnd_s  = boundary_mask[inv2]
    beta_s = beta_g[inv2]

    I_fin = np.zeros(len(comps2))
    for m in macros:
        s, e = node_ranges[m]
        I_fin[global_idx[s:e]] = sol.y[n_total+s:n_total+e, -1]

    I_spot = np.clip(I_fin[inv2], 0, 1)
    _i_vals = I_spot[I_spot > 0]
    _vmax = float(np.percentile(_i_vals, 95)) if len(_i_vals) > 0 else 0.40
    _vmax = max(_vmax * 1.1, 0.05)
    norm_D = mcolors.Normalize(vmin=0.0, vmax=_vmax)

    pk_global = sol.y[n_total:2*n_total, :].max(axis=1)
    badge_kw = {}
    for m in macros:
        s, e = node_ranges[m]
        gi   = global_idx[s:e]
        badge_kw[m] = {
            "r_diff": float((beta_g[gi]/np.where(gam_g[gi]>0,gam_g[gi],1e-12)).mean()),
            "peak":   float(pk_global[s:e].mean()),
        }

    cmap_D  = matplotlib.colormaps["YlOrRd"]
    Wsp_csr = sp.csr_matrix(Wsp)

    if figsize is None:
        figsize = (7*len(macros)+2, 8)

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FIGBG)
    plt.subplots_adjust(left=0.01, right=0.87, top=0.88, bottom=0.09)
    ax.set_facecolor(_FIGBG); ax.set_aspect("equal"); ax.axis("off")

    all_x = np.concatenate([coords[:,0]+x_off[m] for m in macros])
    x_pad = float(np.ptp(coords[:,0]))*0.10
    y_pad = float(np.ptp(coords[:,0]))*0.10
    ax.set_xlim(all_x.min()-x_pad, all_x.max()+x_pad)
    ax.set_ylim(coords[:,1].min()-y_pad,
                coords[:,1].max()+y_pad+y_range*0.18)

    _draw_one_panel(ax, coords, macro_s, bnd_s, beta_s,
                    I_spot, cmap_D, norm_D, macros, x_off,
                    Wsp_csr, n_inter_edges, y_range,
                    show_badges=True, badge_kw=badge_kw)

    _add_colorbar(fig, ax, cmap_D, norm_D,
                  r"$p^D$ final — diseased fraction",
                  shrink=0.42, pad=0.02)

    # canonical legend order: PT→DCT→TAL→Inter-layer→Seed→Size
    leg = _make_legend_handles(macros)
    ax.legend(handles=leg, loc="lower center", fontsize=_FS_LEGEND,
              framealpha=0.90, facecolor=_FIGBG,edgecolor='#444444', borderpad=0.4, ncol=len(leg),
              bbox_to_anchor=(0.50, -0.12))
    ax.set_title(
        "Multilayer compartment transition graph — PT / DCT / TAL\n"
        r"Colour=$p^D$ (final)  ·  Size$\propto\beta_i$  ·"
        "  Coloured=intra-layer kNN  ·  Grey=inter-layer diffusion",
        fontsize=_FS_TITLE, pad=12)

    _show(fig, save_path, dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — MULTILAYER 3 SNAPSHOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_multilayer_snapshots(
    sol, net, sir_params,
    comps2, macro_of, region_of, boundary_mask,
    adata_st, inv2, Wsp,
    t_dose=None, figsize=None, n_inter_edges=_INTER_N_SAMPLE,
    save_path=None, dpi=300,
):
    layers      = net["layers"]
    node_ranges = net["node_ranges"]
    global_idx  = net["global_index"]
    n_total     = net["n_total"]
    macros      = [m for m in ("PT","DCT","TAL") if m in layers]

    beta_g = np.asarray(sir_params["beta"], dtype=float)
    t      = sol.t
    I_traj = sol.y[n_total:2*n_total, :]

    t_pk  = int(np.argmax(I_traj.mean(axis=0)))
    t_end = len(t)-1
    t_dos = int(np.argmin(np.abs(t-t_dose))) if t_dose is not None else len(t)//5
    snaps = [
        (t_dos, f"$t={t[t_dos]:.1f}$  [dosing]"),
        (t_pk,  f"$t={t[t_pk]:.1f}$  [peak $p^D$]"),
        (t_end, f"$t={t[t_end]:.1f}$  [end]"),
    ]

    coords_raw  = adata_st.obsm["spatial"].astype(np.float32)
    step    = np.ptp(coords_raw[:,0]) * 1.10
    y_range = float(np.ptp(coords_raw[:, 1]))
    x_off   = {m: i*step for i, m in enumerate(macros)}

    macro_s = np.full(len(inv2), "other", dtype=object)
    for m in macros:
        macro_s[np.isin(inv2, layers[m]["nodes"])] = m

    coords=coords_raw.copy().astype(float)
    for m in macros:
        idx=np.where(macro_s==m)[0]
        coords[idx]=_relax(coords[idx], iterations=25, min_dist=6.0, strength=0.003)

    bnd_s  = boundary_mask[inv2]
    beta_s = beta_g[inv2]

    Wsp_csr = sp.csr_matrix(Wsp)
    cmap_D  = matplotlib.colormaps["YlOrRd"]
    norm_D  = mcolors.Normalize(vmin=0.0, vmax=0.50)

    all_x = np.concatenate([coords[:,0]+x_off[m] for m in macros])
    x_pad = float(np.ptp(coords[:,0]))*0.10
    y_pad = float(np.ptp(coords[:,0]))*0.10
    x_lim = (all_x.min()-x_pad, all_x.max()+x_pad)
    y_lim = (coords[:,1].min()-y_pad,
             coords[:,1].max()+y_pad+y_range*0.18)

    if figsize is None:
        figsize = (9*len(macros)+4, 10)

    fig, axes = plt.subplots(3, 1, figsize=figsize, facecolor=_FIGBG)
    plt.subplots_adjust(left=0.01, right=0.87, top=0.88,
                        bottom=0.18, wspace=0.04)

    for col_idx, (ax_col, (t_idx, t_label)) in enumerate(zip(axes, snaps)):
        ax_col.set_facecolor(_FIGBG); ax_col.set_aspect("equal")
        ax_col.axis("off")
        ax_col.set_xlim(x_lim); ax_col.set_ylim(y_lim)

        I_t = np.zeros(len(comps2))
        for m in macros:
            s, e = node_ranges[m]
            I_t[global_idx[s:e]] = np.clip(sol.y[n_total+s:n_total+e, t_idx], 0, 1)

        _draw_one_panel(ax_col, coords, macro_s, bnd_s, beta_s,
                        I_t[inv2], cmap_D, norm_D, macros, x_off,
                        Wsp_csr, n_inter_edges, y_range,
                        show_badges=True, badge_kw={})

        ax_col.set_title(t_label, fontsize=10, fontweight="bold", pad=6)

    sm = plt.cm.ScalarMappable(cmap=cmap_D, norm=norm_D)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.tolist(), shrink=0.52, pad=0.02,
                      location="right")
    cb.set_label(r"$p^D(t)$ — diseased fraction", fontsize=_FS_CB)
    cb.ax.tick_params(labelsize=7.5)

    # canonical legend order
    leg = _make_legend_handles(macros)
    axes[-1].legend(handles=leg, bbox_to_anchor=(0.6,-0.2),
                   fontsize=_FS_LEGEND, framealpha=0.90, facecolor=_FIGBG, edgecolor="#444444", borderpad=0.4,
                   ncol=len(leg))

    fig.suptitle(
        r"Multilayer transition  ·  PT/DCT/TAL  ·  Three time snapshots   "
        r"[Colour=$p^D(t)$  ·  Size$\propto\beta_i$]",
        fontsize=_FS_TITLE, y=0.97)

    _show(fig, save_path, dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — DRUG COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def plot_drug_transition(
    sol_drug, sol_nodrug, Wc2,
    comps2, macro_of, region_of, boundary_mask, sir_params,
    net=None, adata_st=None, inv2=None, Wsp=None,
    figsize=None, n_inter_edges=_INTER_N_SAMPLE,
    save_path=None, dpi=300,
):
    if net is not None:
        layers      = net["layers"]
        node_ranges = net["node_ranges"]
        global_idx  = net["global_index"]
        n_total     = net["n_total"]
        macros      = [m for m in ("PT","DCT","TAL") if m in layers]
        use_ml = True
    else:
        use_ml = False
        macros = list(np.unique(macro_of))

    macro_s = np.full(len(inv2), 'other', dtype=object)
    if use_ml:
        for m in macros:
            macro_s[np.isin(inv2, layers[m]["nodes"])] = m
    else:
        macro_s = macro_of[inv2]

    beta_g = np.asarray(sir_params["beta"], dtype=float)

    def _load(sol_obj):
        y = sol_obj.y
        n = y.shape[0]//6
        return np.clip(y[n:2*n,-1]+y[4*n:5*n,-1], 0, 1)

    ld_drug  = _load(sol_drug)
    ld_ndrug = _load(sol_nodrug)

    if adata_st is None or inv2 is None:
        raise ValueError("adata_st and inv2 required")

    coords_raw  = adata_st.obsm["spatial"].astype(np.float32)
    step    = np.ptp(coords_raw[:,0]) * 0.55
    y_range = float(np.ptp(coords_raw[:, 1]))
    x_off   = {m: i*step for i, m in enumerate(macros)}

    coords = coords_raw.copy().astype(float)
    for m in macros:
        idx = np.where(macro_s == m)[0]
        coords[idx] = _relax(coords[idx], iterations=25, min_dist=6.0, strength=0.003)

    bnd_s  = boundary_mask[inv2]
    beta_s = beta_g[inv2]
    Wsp_csr = sp.csr_matrix(Wsp) if Wsp is not None else None

    v_max  = max(float(np.percentile(np.concatenate([ld_drug,ld_ndrug]),98)), 0.25)
    norm_l = mcolors.Normalize(vmin=0.0, vmax=v_max)

    def _to_spot(lc):
        out = np.zeros(len(comps2))
        if use_ml:
            for m in macros:
                s, e = node_ranges[m]
                out[global_idx[s:e]] = lc[s:e]
        else:
            out = lc
        return out[inv2]

    all_x = np.concatenate([coords[:,0]+x_off[m] for m in macros])
    x_pad = float(np.ptp(coords[:,0]))*0.10
    y_pad = float(np.ptp(coords[:,1]))*0.10
    x_lim = (all_x.min()-x_pad, all_x.max()+x_pad)
    y_lim = (coords[:,1].min()-y_pad,
             coords[:,1].max()+y_pad+y_range*0.18)

    if figsize is None:
        pw = max(7*len(macros)//3+3, 10)
        figsize = (2*pw+1, 8)

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=_FIGBG)
    plt.subplots_adjust(left=0.01, right=0.87, top=0.86,
                        bottom=0.20, wspace=0.05)

    for ax, (label, label_col, ld_spot) in zip(
            axes,
            [("+Drug",   "#27ae60", _to_spot(ld_drug)),
             ("No Drug", "#c0392b", _to_spot(ld_ndrug))]):

        ax.set_facecolor(_FIGBG); ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)

        _draw_one_panel(ax, coords, macro_s, bnd_s, beta_s,
                        ld_spot, _CMAP_LOAD, norm_l, macros, x_off,
                        Wsp_csr, n_inter_edges, y_range,
                        show_badges=True, badge_kw={})

        ax.text(float(np.mean([coords[:,0].min()+x_off[macros[0]],
                               coords[:,0].max()+x_off[macros[-1]]])),
                y_lim[1],
                label, ha="center", va="top",
                fontsize=13, fontweight="bold", color=label_col,
                zorder=20, clip_on=False)

    sm = plt.cm.ScalarMappable(cmap=_CMAP_LOAD, norm=norm_l)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.tolist(), shrink=0.55, pad=0.02,
                      location="right")
    cb.set_label(r"Disease load  $p^D+p^X$", fontsize=_FS_CB)
    cb.ax.tick_params(labelsize=7.5)

    # canonical order + drug-specific extras
    drug_extra = [
        mpatches.Patch(facecolor="#c0392b", edgecolor="none", label=r"High $p^D$"),
        mpatches.Patch(facecolor="#7b241c", edgecolor="none", label=r"High $p^X$"),
        mpatches.Patch(facecolor="#f0faf2", edgecolor="#aaaaaa", label="Healthy"),
    ]
    leg = _make_legend_handles(macros, extra=drug_extra)
    axes[0].legend(handles=leg, bbox_to_anchor=(0.85,-0.12),
                   
                   fontsize=_FS_LEGEND, framealpha=0.90, edgecolor="#444444", borderpad=0.4, facecolor=_FIGBG,
                   ncol=2, title="Node encoding", title_fontsize=10.0)

    fig.suptitle(
        r"Drug modulation — Multilayer comparison   "
        r"[Colour=$p^D+p^X$  ·  Size$\propto\beta_i$  ·"
        r"  Coloured=intra-layer  ·  Grey=inter-layer]",
        fontsize=_FS_TITLE, y=0.97)

    _show(fig, save_path, dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════════════

def plot_all_transition_graphs(
    sol_single, Wc2, comps2, macro_of, region_of, boundary_mask,
    sir_params, ml_result,
    drug_result=None, adata_st=None, inv2=None, Wsp=None,
    save_prefix=None, dpi=300,
):
    """
    Generate all transition graphs and save them to disk.

    Parameters
    ----------
    save_prefix : str
        Path prefix for output files, e.g. "results/kidney_sir".
    """
    if save_prefix is None:
        print("  ⚠  WARNING: save_prefix is None — figures will NOT be saved to disk.")

    figs = {}
    print("\n"+"="*60+"\nSTEP 9b: Transition graphs\n"+"="*60)

    def _sp(suffix):
        return f"{save_prefix}{suffix}" if save_prefix else None

    print("\n  [1/5] Single-layer (concentric)...")
    figs["fig_single"] = plot_single_layer_transition(
        sol_single, Wc2, comps2, macro_of, region_of, boundary_mask,
        sir_params, save_path=_sp("_T1_single_concentric.pdf"), dpi=dpi)

    if adata_st is not None and inv2 is not None and Wsp is not None:
        print("\n  [2/5] Single-layer (spot-level)...")
        figs["fig_single_spots"] = plot_single_layer_transition_spots(
            sol_single,
            dict(inv2=inv2, macro_of=macro_of, region_of=region_of,
                 boundary_mask=boundary_mask, Wsp=Wsp, comps2=comps2),
            sir_params, adata_st,
            save_path=_sp("_T2_single_spots.pdf"), dpi=dpi)
    else:
        print("\n  [2/5] Skip (adata_st/inv2/Wsp not provided)")

    print("\n  [3/5] Multilayer static...")
    figs["fig_multi"] = plot_multilayer_transition(
        ml_result["sol"], ml_result["net"], ml_result["W_inter"],
        sir_params, comps2, macro_of, region_of, boundary_mask,
        adata_st=adata_st, inv2=inv2, Wsp=Wsp,
        save_path=_sp("_T3_multilayer_static.pdf"), dpi=dpi)

    if adata_st is not None and inv2 is not None and Wsp is not None:
        t_dose_val = float(ml_result["sol"].t[-1])*0.20 if drug_result is None else None
        print("\n  [4/5] Multilayer snapshots...")
        figs["fig_snapshots"] = plot_multilayer_snapshots(
            ml_result["sol"], ml_result["net"], sir_params,
            comps2, macro_of, region_of, boundary_mask,
            adata_st=adata_st, inv2=inv2, Wsp=Wsp, t_dose=t_dose_val,
            save_path=_sp("_T4_multilayer_snapshots.pdf"), dpi=dpi)

    if drug_result is not None:
        r = drug_result["result"]
        if r.get("sol_nodrug") is not None:
            print("\n  [5/5] Drug comparison...")
            figs["fig_drug"] = plot_drug_transition(
                r["sol_drug"], r["sol_nodrug"],
                Wc2, comps2, macro_of, region_of, boundary_mask,
                sir_params, net=ml_result["net"],
                adata_st=adata_st, inv2=inv2, Wsp=Wsp,
                save_path=_sp("_T5_drug_comparison.pdf"), dpi=dpi)
        else:
            print("\n  [5/5] Skip (sol_nodrug not available)")
    else:
        print("\n  Drug figure skipped (drug_result not provided)")

    return figs