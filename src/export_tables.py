"""
export_tables.py -- Salva in un'unica cartella tutti i valori che compaiono
nelle tabelle e nel testo del caso studio (Sezioni 6-7), presi dagli oggetti
calcolati da main.py nella stessa run.

Uso: alla FINE di src/main.py (dopo lo STEP 13) aggiungere

    from export_tables import export_all
    export_all(globals(), out_dir="../results/manuscript_tables")

e salvare il risultato dello STEP 13 in una variabile:
    sec7 = run_section7_analysis(save_dir="./results/figures", dpi=200)

Ogni blocco e' indipendente: se un oggetto manca o ha una struttura diversa,
il blocco viene saltato con un avviso e gli altri vengono salvati comunque.
"""
import json
import os
import platform
import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------- utilita'
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()
                if not hasattr(v, "savefig") and not hasattr(v, "axes")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 5000 else f"<array {o.shape}>"
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return f"<{type(o).__name__}>"


def _dump_any(obj, path_base, log):
    """DataFrame -> CSV; dict -> CSV per ogni DataFrame interno + JSON del resto."""
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path_base + ".csv", index=False)
        log.append(path_base + ".csv")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, pd.DataFrame):
                v.to_csv(f"{path_base}__{k}.csv", index=False)
                log.append(f"{path_base}__{k}.csv")
        json.dump(_jsonable(obj), open(path_base + ".json", "w"), indent=1)
        log.append(path_base + ".json")
        return
    json.dump(_jsonable(obj), open(path_base + ".json", "w"), indent=1)
    log.append(path_base + ".json")


def _block(name, fn, log):
    try:
        fn()
        print(f"  [export] {name}: ok")
    except Exception as e:  # noqa: BLE001
        print(f"  [export] {name}: SALTATO ({type(e).__name__}: {e})")
        log.append(f"SKIPPED {name}: {e}")


# ---------------------------------------------------------------- export
def export_all(g, out_dir="../results/manuscript_tables", threshold=0.10):
    os.makedirs(out_dir, exist_ok=True)
    log = []
    p = lambda name: os.path.join(out_dir, name)
    comp = g["comp"]
    comps2 = np.asarray(comp["comps2"])
    macro_of = np.asarray(comp["macro_of"])
    region_of = np.asarray(comp["region_of"])
    seeds = np.asarray(comp["boundary_mask"]).astype(bool)
    n_spots = pd.Series(np.asarray(g["adata_st2"].obs["compartment2"])).value_counts()

    # Table 2 + heatmap: metriche per compartimento (single-layer, sol_diff)
    def t_single():
        sol, prm = g["sol_diff"], g["diff_params"]
        C = len(comps2)
        t = np.asarray(sol.t)
        D, R = sol.y[C:2 * C], sol.y[2 * C:3 * C]
        cross = np.where((D > threshold).any(1), t[np.argmax(D > threshold, axis=1)], np.nan)
        rdiff = prm["beta"] / np.where(prm["gamma"] > 0, prm["gamma"], 1e-12)
        df = pd.DataFrame(dict(
            compartment=comps2, macro=macro_of, zone=region_of,
            n_spots=[int(n_spots.get(c, 0)) for c in comps2], seeded=seeds,
            beta=prm["beta"], rho=prm["gamma"], R_diff=rdiff,
            peak_D=D.max(1), t_peak=t[D.argmax(1)],
            AUC_D=_trapz(D, t, axis=1), t_cross=cross, pR_fin=R[:, -1],
        ))
        df.round(4).to_csv(p("table2_single_layer_per_compartment.csv"), index=False)
        z = df.groupby("zone").agg(n_compartments=("compartment", "size"),
                                   n_spots=("n_spots", "sum"),
                                   peak_D=("peak_D", "mean"), t_peak=("t_peak", "mean"),
                                   AUC_D=("AUC_D", "mean"), t_cross=("t_cross", "mean"),
                                   pR_fin=("pR_fin", "mean"))
        z.round(4).to_csv(p("table2_single_layer_by_zone.csv"))
        json.dump(dict(t_end=float(t[-1]), n_timepoints=int(t.size), threshold=threshold,
                       seeded_compartments=list(comps2[seeds]),
                       global_peak_D_mean=float(D.mean(0).max()),
                       global_t_cross_mean=float(np.nanmean(cross)),
                       pD_mean_at_end=float(D.mean(0)[-1])),
                  open(p("single_layer_global.json"), "w"), indent=1)
        log.extend([p("table2_single_layer_per_compartment.csv"), p("table2_single_layer_by_zone.csv")])
    _block("Table 2 / invasion heatmap (single-layer)", t_single, log)

    _block("invasion analysis (df_invasion)",
           lambda: _dump_any(g["df_invasion"], p("invasion_analysis"), log), log)

    # Table 4: multilayer per layer
    def t_ml():
        m = g["ml_result"]["metrics"]
        bl = pd.DataFrame(m["by_layer"]).T
        bl.index.name = "layer"
        bl.round(4).to_csv(p("table4_multilayer_by_layer.csv"))
        rest = {k: v for k, v in m.items() if k != "by_layer"}
        json.dump(_jsonable(rest), open(p("table4_multilayer_global.json"), "w"), indent=1)
        log.append(p("table4_multilayer_by_layer.csv"))
    _block("Table 4 (multilayer)", t_ml, log)

    # Table 5: sweep del coupling
    def t_kappa():
        rows = []
        for D, bl in g["results_D_inter"].items():
            for layer, v in bl.items():
                rows.append(dict(kappa=D, layer=layer, peak_D=v.get("peak_I"),
                                 t_peak=v.get("t_peak"), t_inv=v.get("invasion_time")))
        pd.DataFrame(rows).round(4).to_csv(p("table5_kappa_sweep.csv"), index=False)
    _block("Table 5 (kappa sweep)", t_kappa, log)

    # Table 6 + entropia/flussi: trans-compartimentali
    def t_tc():
        tc = g["tc_results"]
        tc["K_dyn"].round(5).to_csv(p("table6_K_dyn.csv"))
        _dump_any({k: v for k, v in tc.items() if k != "K_dyn"}, p("transcompartment_other"), log)
    _block("Table 6 (K_dyn) e metriche trans-compartimentali", t_tc, log)

    # Tabelle farmacologiche
    _block("farmacologia (drug_result)",
           lambda: _dump_any(g["drug_result"], p("pharmacology"), log), log)

    # Sensibilita': OAT, Monte Carlo, Spearman, ranking stability
    def t_sens():
        s = g["sensitivity_results"]
        for k in ["df_oat", "df_morris", "df_mc", "df_spearman"]:
            if k in s and isinstance(s[k], pd.DataFrame):
                s[k].to_csv(p(f"sensitivity_{k}.csv"), index=False)
        if isinstance(s.get("df_mc"), pd.DataFrame):
            num = s["df_mc"].select_dtypes("number")
            summ = pd.DataFrame(dict(mean=num.mean(), p05=num.quantile(0.05),
                                     p95=num.quantile(0.95), sd=num.std()))
            summ.round(4).to_csv(p("table8_monte_carlo_summary.csv"))
        rob = s.get("robustness", {})
        out = {}
        for k in ["corr_inv_series", "corr_att_series"]:
            if k in rob:
                v = np.asarray(rob[k], float)
                v = v[~np.isnan(v)]
                out[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                              p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                              n=int(v.size))
        json.dump(out, open(p("table10_ranking_stability.json"), "w"), indent=1)
    _block("sensibilita' (Tables 7-10)", t_sens, log)

    _block("sweep 2D", lambda: _dump_any(g["sweep2d_result"], p("sweep2d"), log), log)

    # Sezione 7: edge importance, ablazione, seeding
    if "sec7" in g:
        for k, v in g["sec7"].items():
            _block(f"Sezione 7 - {k}", lambda k=k, v=v: _dump_any(v, p(f"section7_{k}"), log), log)
    else:
        print("  [export] Sezione 7: assegna il risultato a 'sec7' in main.py per esportarla")

    # Ambiente
    def t_env():
        import scanpy, scipy, anndata
        json.dump(dict(python=platform.python_version(), numpy=np.__version__,
                       pandas=pd.__version__, scipy=scipy.__version__,
                       scanpy=scanpy.__version__, anndata=anndata.__version__,
                       zoning_no_slc12a3=os.environ.get("ZONING_NO_SLC12A3") == "1"),
                  open(p("environment.json"), "w"), indent=1)
    _block("ambiente", t_env, log)

    open(p("_export_log.txt"), "w").write("\n".join(map(str, log)))
    print(f"\n  [export] completato: {out_dir}")
