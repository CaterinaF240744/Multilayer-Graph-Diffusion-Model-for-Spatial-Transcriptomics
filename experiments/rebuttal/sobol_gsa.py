"""
sobol_gsa.py
============
Global sensitivity analysis (Sobol) of the exposure-driven compartment model
(manuscript Section "Global sensitivity analysis"; response letter R2.4).

Standalone implementation: reproduces the methods described in the
manuscript and writes its results in the sobol_results.json schema
(problem, indices, rank_stability_mean/sd, baseline_D_fin,
baseline_comps, N, n_eval).

Design
------
- Saltelli sampling, N = 256, calc_second_order=False
 -> N * (D + 2) = 256 * 11 = 2,816 model evaluations

- 9 kinetic/structural parameters, +-40% around the Table 1 values:
  beta_PT, beta_DCT, beta_TAL, rho_PT, rho_DCT, rho_TAL, D_H, D_D, kappa
- Model: MULTILAYER exposure-driven ODE
  (sir_multilayer.simulate_SIR_multilayer, BDF, rtol 1e-6,
  atol 1e-9, t_end = 15, seed p^D_0 = 0.10 in boundary compartments) on the
  compartment graph built by the repository pipeline
  (compartments.build_compartment2, kNN k=6, anatomical zoning with radial
  fallback) over ALL macro types present (PT/DCT/TAL/vascular/other ->
  18 compartments).
  Implementation notes:
  * coarse_grain_graph zeroes the diagonal; the within-compartment
    self-weight is restored on Wc2 (without it the compartment wave
    cannot sustain itself on the real graph);
  * kappa enters through the multilayer inter-layer coupling
    W_inter = kappa * flux * sqrt(D_D,i * D_D,j) (Eq. 11), extended to
    all macro pairs;
  * vascular/other layers keep the Table 1 base parameters (only the 9
    Sobol parameters vary; D_S = D_H is global).
- Outputs per parameter vector:
  t_inv : first time the tissue-mean p^D crosses 0.1
  peak_D_mean : max over time of the tissue-mean p^D
  auc_D_mean : integrated burden, integral of tissue-mean p^D
  t_peak_PT : time of peak mean p^D over the PT compartments
  plus the final state D_fin (per compartment) used for the
  ranking-stability check: Spearman correlation between each sample's
  compartment ranking and the baseline ranking (mean +- sd over all
  evaluations).

Repository placement
---------------------
Lives in experiments/rebuttal/, alongside the other rebuttal scripts; imports
the model code from src/ (two levels up), or from MODEL_SRC_DIR if set.

Usage
-----
 python sobol_gsa.py [--out-dir OUT] [--data-dir DATA] [--n-jobs 8] [--N 256] [--quick]
 (--data-dir defaults to the SOBOL_DATA_DIR env var, else ./data/visium)

Outputs (in --out-dir, default ./gsa_outputs):
 sobol_results.json (schema as documented above)
"""

    import anndata as ad
    from compartments import build_compartment2
    from preprocessing import MACRO_MAP
    from sir_multilayer import build_multilayer_network

    M = mmread(os.path.join(data_dir, "v1_kidney_raw_counts.mtx")).tocsr()
    genes = pd.read_csv(os.path.join(data_dir, "v1_kidney_genes.csv"),
                        header=None)[0].astype(str).values
    bc = pd.read_csv(os.path.join(data_dir, "v1_kidney_barcodes.csv"),
                     header=None)[0].astype(str).values
    meta = pd.read_csv(os.path.join(data_dir, "v1_kidney_meta.csv"),
                       index_col="barcode")
    coords = pd.read_csv(os.path.join(data_dir, "v1_kidney_spatial_coords.csv"),
                         index_col="barcode")

    keep = [b for b in meta.index if b in set(bc)]
    bpos = {b: i for i, b in enumerate(bc)}
    sel = [bpos[b] for b in keep]
    X = (M[:, sel].T.tocsr()).astype(np.float32)          # spots x genes

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_identity": meta.loc[keep, "macro_type"].values},
                         index=keep),
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["spatial"] = coords.loc[keep].values.astype(np.float32)

    # MACRO_MAP lacks the lowercase keys used by the released meta
    macro_map = {**MACRO_MAP, "vascular": "vascular", "immune": "immune"}

    g = build_compartment2(adata, macro_map=macro_map, k_knn=6,
                           use_anatomical_zones=True, fallback_radial=True)
    Wsp, Wc2 = g["Wsp"], g["Wc2"]
    comps, inv2 = g["comps2"], g["inv2"]
    macro_of, region_of = g["macro_of"], g["region_of"]
    print(f"[build_compartment2] {len(comps)} compartments: {list(comps)}")

    # ── minimal patch: restore the within-compartment self-weight ──────────
    # coarse_grain_graph('avg') zeroes the diagonal; the removed self term
    # under the same normalisation is Wc_full[i,i] / N_i^2 with
    # Wc_full = S^T Wsp S. Add it back so every compartment keeps its
    # self-exposure (without it the compartment wave cannot sustain itself).
    N_spot = Wsp.shape[0]
    C = len(comps)
    S_mat = sp.csr_matrix((np.ones(N_spot), (np.arange(N_spot), inv2)),
                          shape=(N_spot, C))
    Wc_full = (S_mat.T @ Wsp @ S_mat).tocsr()
    Nc = np.asarray(S_mat.sum(axis=0)).reshape(-1)
    self_w = np.asarray(Wc_full.diagonal()) / np.maximum(Nc ** 2, 1.0)
    Wc2 = (Wc2 + sp.diags(self_w)).tocsr()

    base_params = build_default_params_SIR(comps, macro_of, region_of)
    seed_mask = np.isin(region_of, ["outer_medulla", "boundary"])

    macros = tuple(pd.unique(macro_of))
    net = build_multilayer_network(Wc2, comps, macro_of, region_of,
                                   macros=macros)
    print(f"[multilayer] {net['n_total']} nodes over layers {macros}")
    return dict(Wsp=Wsp, inv2=inv2, net=net, comps=comps,
                base_params=base_params, seed_mask=seed_mask,
                macro_of=macro_of)


def _init_worker(ctx):
    _CTX.update(ctx)


def evaluate(theta):
    """
    Run the MULTILAYER exposure-driven ODE for one parameter vector and
    return the scalar outputs + final state D_fin (per layer compartment).
    theta = [beta_PT, beta_DCT, beta_TAL, rho_PT, rho_DCT, rho_TAL,
             D_H, D_D, kappa]
    """
    from sir_multilayer import (
        build_interlayer_coupling, simulate_SIR_multilayer,
    )
    ctx = _CTX
    net, comps = ctx["net"], ctx["comps"]
    C = len(comps)

    params = dict(
        beta=ctx["base_params"]["beta"].copy(),
        gamma=ctx["base_params"]["gamma"].copy(),
        D_S=np.full(C, theta[6]),
        D_I=ctx["base_params"]["D_I"].copy(),
    )
    for i, m in enumerate(ctx["macro_of"]):
        if m == "PT":
            params["beta"][i] = theta[0]
            params["gamma"][i] = theta[3]
        elif m == "DCT":
            params["beta"][i] = theta[1]
            params["gamma"][i] = theta[4]
        elif m == "TAL":
            params["beta"][i] = theta[2]
            params["gamma"][i] = theta[5]
        if m in ("PT", "DCT", "TAL"):
            params["D_I"][i] = theta[7]
    kappa = theta[8]

    # inter-layer coupling over ALL macro pairs present in the net
    lay = list(net["layers"])
    topology = [(lay[i], lay[j]) for i in range(len(lay))
                for j in range(i + 1, len(lay))]
    W_inter = build_interlayer_coupling(
        net, ctx["Wsp"], ctx["inv2"], params,
        topology=topology, D_inter=kappa)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_fin, I_fin, R_fin, sol, info = simulate_SIR_multilayer(
            net, W_inter, params, ctx["seed_mask"],
            seed_I=SEED_I, t_end=T_END, n_steps=N_STEPS)

    n = net["n_total"]
    gi = net["global_index"]
    D_t = np.clip(sol.y[n:2 * n, :], 0.0, 1.0)          # (n, n_steps)
    D_mean = D_t.mean(axis=0)

    t_inv = next((float(sol.t[i]) for i in range(len(sol.t))
                  if D_mean[i] >= ACT_THRESHOLD), np.nan)
    peak_D = float(D_mean.max())
    auc_D = float(trapezoid(D_mean, sol.t))
    pt_rows = np.where(net["layer_of_global"] == "PT")[0]
    t_peak_PT = float(sol.t[np.argmax(D_t[pt_rows, :].mean(axis=0))])

    D_fin = np.array([D_t[i, -1] for i in range(n)])    # layer compartments
    return [t_inv, peak_D, auc_D, t_peak_PT], D_fin


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("SOBOL_DATA_DIR", os.path.join("data", "visium")),
        help="Directory with v1_kidney_raw_counts.mtx, v1_kidney_genes.csv, "
             "v1_kidney_barcodes.csv, v1_kidney_meta.csv, "
             "v1_kidney_spatial_coords.csv (default: env SOBOL_DATA_DIR, "
             "else ./data/visium).")
    ap.add_argument("--out-dir", default="gsa_outputs")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--quick", action="store_true", help="N=16 smoke test")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    N = 16 if a.quick else a.N

    print("Building Visium compartment graph...")
    ctx = build_model_context(a.data_dir)
    print(f"  {len(ctx['comps'])} compartments:", list(ctx["comps"]))

    X = saltelli.sample(PROBLEM, N, calc_second_order=False)
    n_eval = X.shape[0]
    print(f"Sobol: {n_eval} model evaluations")

    # baseline (Table 1 centres) — also fixes the layer-compartment count
    _init_worker(ctx)  # make the context available in the main process too
    base_theta = [0.40, 0.28, 0.25, 0.08, 0.10, 0.12, 0.05, 0.20, 1.0]
    base_out, base_D_fin = evaluate(base_theta)
    layer_comps = [str(ctx["comps"][i]) for i in ctx["net"]["global_index"]]

    t0 = time.time()
    Y = np.zeros((n_eval, len(OUTPUTS)))
    D_fin_all = np.zeros((n_eval, len(base_D_fin)))
    with ProcessPoolExecutor(max_workers=a.n_jobs,
                             initializer=_init_worker,
                             initargs=(ctx,)) as ex:
        for i, (out, dfin) in enumerate(ex.map(evaluate, list(X),
                                               chunksize=8)):
            Y[i] = out
            D_fin_all[i] = dfin
            if (i + 1) % 500 == 0 or i + 1 == n_eval:
                print(f"  eval {i+1}/{n_eval}  ({time.time()-t0:.0f}s)")

    # NaN handling: SALib needs finite Y — report and impute with column mean
    n_nan = int(np.isnan(Y).sum())
    if n_nan:
        print(f"  [warning] {n_nan} non-finite output values imputed "
              f"(t_inv = NaN when the threshold is never reached)")
    col_mean = np.nanmean(Y, axis=0)
    Y_imp = np.where(np.isnan(Y), col_mean[None, :], Y)

    indices = {}
    for j, name in enumerate(OUTPUTS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            si = sobol_analyze.analyze(PROBLEM, Y_imp[:, j],
                                       calc_second_order=False)
        indices[name] = dict(
            S1=si["S1"].tolist(), S1_conf=si["S1_conf"].tolist(),
            ST=si["ST"].tolist(), ST_conf=si["ST_conf"].tolist(),
        )

    # ranking stability: Spearman of D_fin ranking vs baseline, all evals
    rhos = [spearmanr(base_D_fin, D_fin_all[i]).statistic
            for i in range(n_eval)]
    rhos = np.nan_to_num(np.array(rhos), nan=0.0)

    result = dict(
        problem=PROBLEM,
        indices=indices,
        rank_stability_mean=float(rhos.mean()),
        rank_stability_sd=float(rhos.std()),
        baseline_D_fin=base_D_fin.tolist(),
        baseline_comps=layer_comps,
        N=N,
        n_eval=n_eval,
    )
    out_path = os.path.join(a.out_dir, "sobol_results.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"Sobol analysis complete -> {out_path}  ({time.time()-t0:.0f}s)")

    st = {n: indices[n]["ST"] for n in OUTPUTS}
    print("ST beta_PT per output:",
          {n: round(st[n][PROBLEM["names"].index("beta_PT")], 2)
           for n in OUTPUTS})


if __name__ == "__main__":
    main()
