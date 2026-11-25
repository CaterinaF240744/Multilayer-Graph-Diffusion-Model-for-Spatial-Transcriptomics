"""slideseq_r3.py -- Slide-seqV2 case study (Section 8.6) with the canonical model.

Same protocol as the R2 regeneration script (slideseq_regen.py):
  * marker-based macro-types (neuron / glia / vascular / immune / other);
  * three concentric zones by distance-from-centroid terciles
    (surface / middle / deep);
  * symmetric kNN bead graph (k = 6), disconnected pieces joined by their
    nearest bead pair;
  * parameters mapped from Table 1 (neuron <- PT, glia <- DCT, ...);
  * surface seeding with p^D(0) = 0.05.

Change for R3: the state equations are the canonical exposure-driven model of
Section 5.2 (no Laplacian transport of H and D, so no D_H / D_D):

    lambda = W~ p^D,   W~ = D^-1 W (row-normalised)
    dH = -beta H lambda
    dD =  beta H lambda - rho D
    dR =  rho D

Integration: classical RK4 with a fixed step (default dt = 0.02). The system
has ~10^5 beads, too many for the implicit BDF solver used elsewhere; RK4
preserves the linear invariant H + D + R = 1 exactly up to round-off and the
step is far below the stability limit (beta + rho <= 0.5). The simplex is
checked at every saved time point with the same tolerance as the rest of the
pipeline.

Usage
-----
python experiments/rebuttal/slideseq_r3.py --h5ad data/slideseq_cerebellum.h5ad \
       --tissue cerebellum
Outputs go to results/r3/slideseq/<tissue>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import model_core as mc  # noqa: E402

MARKERS = {
    "neuron":   ["Slc17a7", "Gad1", "Gad2", "Slc32a1", "Car8", "Calb1", "Calb2",
                 "Rbfox3", "Atp1a3", "Grm1"],
    "glia":     ["Mbp", "Plp1", "Mog", "Sox10", "Aqp4", "Mfge8", "Gfap", "Aldh1l1"],
    "vascular": ["Pecam1", "Cldn5", "Vwf", "Acta2", "Kdr", "Flt1"],
    "immune":   ["Ptprc", "Cx3cr1", "Hexb", "C1qa", "Lyz2", "Ctss"],
}
MACROS = ["neuron", "glia", "vascular", "immune", "other"]
ZONES = ["surface", "middle", "deep"]
# mapped from Table 1 (unchanged from R2)
BETA = {"neuron": 0.40, "glia": 0.28, "vascular": 0.30, "immune": 0.20, "other": 0.22}
RHO = {"neuron": 0.08, "glia": 0.10, "vascular": 0.10, "immune": 0.25, "other": 0.10}
SEED_D, THR, K = 0.05, 0.10, 6


def annotate(a):
    """Marker-score macro-types (log1p of raw counts, as in R2)."""
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    Xl = np.log1p(X)
    S, used = [], {}
    for m, genes in MARKERS.items():
        g = [x for x in genes if x in a.var_names]
        used[m] = g
        S.append(Xl[:, [a.var_names.get_loc(x) for x in g]].mean(1) if g
                 else np.full(a.n_obs, -1e9))
    S = np.vstack(S).T
    macro = np.array(list(MARKERS))[S.argmax(1)].astype(object)
    macro[S.max(1) <= 0] = "other"
    return macro, used


def zones(xy):
    d = np.linalg.norm(xy - xy.mean(0), axis=1)
    q1, q2 = np.quantile(d, [1 / 3, 2 / 3])
    return np.where(d > q2, "surface", np.where(d > q1, "middle", "deep")).astype(object)


def knn_graph(xy, k=K):
    n = len(xy)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((np.ones(n * k), (rows, idx[:, 1:].ravel())), shape=(n, n))
    W = W.maximum(W.T).tolil()
    n_comp, lab = connected_components(W, directed=False)
    for c in range(1, n_comp):                      # join pieces (separate folia)
        src = np.where(lab == c)[0]
        tree = cKDTree(xy[lab < c])
        other = np.where(lab < c)[0]
        dist, j = tree.query(xy[src])
        i = int(np.argmin(dist))
        u, v = src[i], other[j[i]]
        W[u, v] = W[v, u] = 1.0
    return W.tocsr(), n_comp


def rk4(Wn, beta, rho, H, D, R, t_end, dt, save_dt):
    def f(H, D):
        lam = Wn @ D
        inf = beta * H * lam
        rec = rho * D
        return -inf, inf - rec, rec

    n_steps = int(round(t_end / dt))
    every = int(round(save_dt / dt))
    t_s, D_s, err = [0.0], [D.astype(np.float32)], 0.0
    for k in range(n_steps):
        k1 = f(H, D)
        k2 = f(H + 0.5 * dt * k1[0], D + 0.5 * dt * k1[1])
        k3 = f(H + 0.5 * dt * k2[0], D + 0.5 * dt * k2[1])
        k4 = f(H + dt * k3[0], D + dt * k3[1])
        H = H + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        D = D + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        R = R + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if (k + 1) % every == 0:
            err = max(err, mc.check_simplex(H, D, R, where="slideseq_r3"))
            t_s.append((k + 1) * dt)
            D_s.append(D.astype(np.float32))
    return np.array(t_s), np.vstack(D_s).T, H, D, R, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=os.path.join(ROOT, "data", "slideseq_cerebellum.h5ad"))
    ap.add_argument("--tissue", default="cerebellum")
    ap.add_argument("--t_end", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--save_dt", type=float, default=0.5)
    a_ = ap.parse_args()

    import scanpy as sc
    out = os.path.join(ROOT, "results", "r3", "slideseq", a_.tissue)
    os.makedirs(out, exist_ok=True)

    a = sc.read_h5ad(a_.h5ad)
    a.var_names_make_unique()
    if "counts" in a.layers:
        a.X = a.layers["counts"]
    sc.pp.filter_genes(a, min_cells=3)
    print(f"[slideseq_r3] {a_.tissue}: {a.n_obs} beads x {a.n_vars} genes", flush=True)

    macro, used = annotate(a)
    xy = np.asarray(a.obsm["spatial"], dtype=float)
    zone = zones(xy)
    W, n_comp = knn_graph(xy)
    Wn = sp.csr_matrix(sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W)
    beta = np.array([BETA[m] for m in macro])
    rho = np.array([RHO[m] for m in macro])
    print(pd.Series(macro).value_counts().to_string(), flush=True)
    print(f"graph: {W.nnz // 2} edges, {n_comp} pieces joined", flush=True)

    D0 = np.where(zone == "surface", SEED_D, 0.0)
    t, D_t, H, D, R, err = rk4(Wn, beta, rho, 1.0 - D0, D0, np.zeros(a.n_obs),
                               a_.t_end, a_.dt, a_.save_dt)
    print(f"integration done; max simplex error {err:.1e}; max p^R {R.max():.3f}",
          flush=True)

    reached = D_t.max(1) >= THR
    t_inv = np.where(reached, t[np.argmax(D_t >= THR, axis=1)], np.nan)
    peak = D_t.max(1)

    rows = []
    for m in MACROS:
        for z in ZONES:
            s = (macro == m) & (zone == z)
            if not s.any():
                continue
            rows.append(dict(macro=m, zone=z, n=int(s.sum()), R_diff=BETA[m] / RHO[m],
                             peak_pD=float(peak[s].mean()),
                             frac_reached=float(reached[s].mean()),
                             t_cross=float(np.nanmean(t_inv[s])) if reached[s].any() else np.nan,
                             pR_fin=float(R[s].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "slideseq_by_macro_zone.csv"), index=False)

    tab = []
    for m in MACROS:
        s = macro == m
        if not s.any():
            continue
        sub = df[df.macro == m].set_index("zone")
        tab.append(dict(macro=m, n=int(s.sum()), R_diff=BETA[m] / RHO[m],
                        peak_pD=float(peak[s].mean()),
                        t_cross_surface=sub.t_cross.get("surface", np.nan),
                        t_cross_middle=sub.t_cross.get("middle", np.nan),
                        t_cross_deep=sub.t_cross.get("deep", np.nan),
                        frac_reached=float(reached[s].mean())))
    tab = pd.DataFrame(tab)
    tab.to_csv(os.path.join(out, "table16_slideseq.csv"), index=False)
    print(tab.round(3).to_string(index=False), flush=True)

    zone_t = {z: float(np.nanmean(t_inv[zone == z])) if reached[zone == z].any() else None
              for z in ZONES}
    json.dump(dict(tissue=a_.tissue, h5ad=os.path.basename(a_.h5ad), n_beads=int(a.n_obs),
                   markers_used=used, n_pieces=int(n_comp), t_end=a_.t_end, dt=a_.dt,
                   max_simplex_error=err, max_pR=float(R.max()),
                   frac_reached=float(reached.mean()), zone_mean_t_cross=zone_t,
                   macro_counts=pd.Series(macro).value_counts().to_dict()),
              open(os.path.join(out, "summary.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
    for k, (v, cm, ttl, vmin, vmax) in enumerate([
            (peak, "magma", "peak $p^D$", 0, np.quantile(peak, 0.99)),
            (np.where(reached, t_inv, a_.t_end), "viridis_r",
             f"activation time (not reached = {a_.t_end:g})", 0, a_.t_end),
            (beta / rho, "coolwarm", r"$R_\mathrm{diff}=\beta/\rho$", 0, 5)]):
        s_ = ax[k].scatter(xy[:, 0], xy[:, 1], c=v, s=1, cmap=cm, vmin=vmin, vmax=vmax,
                           linewidths=0)
        plt.colorbar(s_, ax=ax[k], fraction=0.046)
        ax[k].set_title(ttl)
    for i, m in enumerate(MACROS):
        s = macro == m
        ax[3].scatter(xy[s, 0], xy[s, 1], s=1, color=plt.get_cmap("tab10")(i), label=m,
                      linewidths=0)
    ax[3].legend(fontsize=7, markerscale=6, frameon=False)
    ax[3].set_title("macro-type")
    for x in ax:
        x.set_aspect("equal"); x.invert_yaxis(); x.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slideseq_spatial_maps.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, m in enumerate(MACROS):
        s = macro == m
        if not s.any():
            continue
        mu, sd = D_t[s].mean(0), D_t[s].std(0)
        c = plt.get_cmap("tab10")(i)
        ax.plot(t, mu, color=c, lw=2, label=f"{m} (n = {s.sum()})")
        ax.fill_between(t, np.clip(mu - sd, 0, 1), mu + sd, color=c, alpha=0.2)
    ax.axhline(THR, ls="--", c="grey", lw=1)
    ax.set_xlabel("simulated time (model units)")
    ax.set_ylabel("mean $p^D$ (±1 SD across beads)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slideseq_trajectories.png"), dpi=200)
    plt.close(fig)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
