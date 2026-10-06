"""
polonsky_seqfish_test.py -- Test predizione del modello vs danno osservato
(seqFISH, rene di topo, 3 controlli + 3 IRI; Polonsky et al. 2024,
Dryad doi:10.5061/dryad.bnzs7h4hj).

PRE-SPECIFICATO (data: ____________), prima di calcolare qualsiasi risultato.

Dati e annotazione
  * Annotazione degli autori (colonna Celltype), mappata sui macro-tipi (MACRO).
    DCT-CNT -> DCT (gli autori non separano DCT e tubulo connettore).
    Injured PT -> PT.
  * Grafo: kNN spaziale sulle cellule (coordinate in um), come nel caso V1.
  * Zonazione: marker canonici (ZONE_MARKERS) calcolati sull'espressione
    log-normalizzata MEDIATA sulle cellule entro R um (a singola cellula i
    marker indicano il tipo cellulare, non la regione; la media locale
    ricostruisce il segnale regionale, come fa uno spot Visium).
    R principale = 100 um (spaziatura Visium); sensibilita' 50 e 200 um.
  * Compartimenti: macro-tipo x zona con >= MIN_CELLS cellule.

Predizione (solo campioni di CONTROLLO)
  Modello multilayer canonico (parametri Table 1, seed canonico, t_end 60,
  soglia 0.10). Per compartimento: picco della frazione malata (peak_I) e
  tempo di attivazione. Media sui controlli; compartimenti presenti in >= 2.

Osservazione (campioni IRI vs controlli)
  * dHavcr1 = media Havcr1_Norm nei campioni IRI (IRI2, IRI3; IRI1 non misurato)
    meno la media nei controlli, per compartimento.
  * frac_injured = frazione di cellule 'Injured PT' tra le cellule PT del
    compartimento nei campioni IRI (solo compartimenti PT).

Test (Spearman, p per permutazione a due code, 10.000 permutazioni, seed 0)
  P1 (primario): peak_I vs dHavcr1, tutti i compartimenti PT/DCT/TAL.
  P2 (primario): peak_I vs frac_injured, solo compartimenti PT (confronto
     entro macro-tipo: indipendente dal fatto che beta_PT sia il piu' alto).
  Bonferroni per 2 test primari.
  Secondari: tempo di attivazione (atteso negativo); dVcam1; esclusione dei
  compartimenti seme; raggi 50 e 200 um. Tutti riportati comunque vadano.

Uso (cartella python_files, ambiente stpipeline):
  python experiments/rebuttal/polonsky_seqfish_test.py
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata
from sklearn.neighbors import radius_neighbors_graph
import scanpy.plotting as _scpl
_noop = lambda *args, **kwargs: None
sc.pl.spatial = _noop
_scpl.spatial = _noop

sys.path.insert(0, "src")
from compartments import build_compartment2, ZONE_MARKERS
from sir_compartments import build_default_params_SIR
from sir_multilayer import run_SIR_multilayer
from sir_transcompartment_metrics import _extract_SIR

DATA = "data/polonsky"
OUT = "results/polonsky_seqfish"
RADII = [100, 50, 200]          # il primo e' il principale
MIN_CELLS = 30
SEED_I, D_INTER, T_END, N_STEPS, THR = 0.05, 0.05, 60.0, 800, 0.10
N_PERM = 10000
MACRO = {
    "PTS1": "PT", "PTS2": "PT", "PTS3": "PT", "Injured PT": "PT",
    "TAL_1": "TAL", "TAL_2": "TAL", "TAL_3": "TAL",
    "DCT-CNT": "DCT",
    "Vasc_1": "vascular", "Vasc_2": "vascular", "Vasc_3": "vascular",
    "Macroph": "immune", "T": "immune", "DC": "immune",
    "Fib": "other", "PC": "other", "IC": "other", "LOH-TL-C": "other",
    "LOH-TL-JM": "other", "Podo": "other", "Uro": "other", "Per": "other",
}
os.makedirs(OUT, exist_ok=True)


def perm_test(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return dict(n=int(len(x)), rho=None, p=None)
    rx, ry = rankdata(x), rankdata(y)
    r = np.corrcoef(rx, ry)[0, 1]
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(N_PERM)])
    return dict(n=int(len(x)), rho=round(float(r), 3),
                p=round(float((np.abs(null) >= abs(r) - 1e-12).mean()), 4))


# ---------------------------------------------------------------- load
t0 = time.time()
meta = pd.read_csv(f"{DATA}/metadata.csv").set_index("CellID")
coords = pd.read_csv(f"{DATA}/coordinates_raw.csv").set_index("CellID")
unmapped = set(meta["Celltype"].unique()) - set(MACRO)
if unmapped:
    raise ValueError(f"Celltype senza mappatura: {sorted(unmapped)}")
genes_all = pd.read_csv(f"{DATA}/Counts_raw.csv", nrows=0).columns[2:]
zone_genes = sorted({g for m in ZONE_MARKERS.values() for g in m})
lower = {g.lower(): g for g in genes_all}
zone_cols = [lower[g.lower()] for g in zone_genes if g.lower() in lower]
print(f"Marker di zona nel pannello: {zone_cols}", flush=True)

print("Lettura Counts_raw.csv (qualche minuto)...", flush=True)
parts = []
for ch in pd.read_csv(f"{DATA}/Counts_raw.csv", chunksize=20000):
    ch = ch.set_index(ch.columns[0])
    part = ch[zone_cols].astype(np.float32)
    part["__lib"] = ch[genes_all].sum(axis=1).astype(np.float32)
    part["SampleID"] = ch[ch.columns[0]].values
    parts.append(part)
zc = pd.concat(parts)
del parts
print(f"Caricato in {time.time()-t0:.0f}s", flush=True)


def build_sample(sid, radius):
    """AnnData di un campione con espressione di zona mediata entro radius um."""
    d = zc[zc["SampleID"] == sid]
    ids = d.index.intersection(coords.index).intersection(meta.index)
    d = d.loc[ids]
    X = d[zone_cols].values / np.maximum(d["__lib"].values[:, None], 1) * 1e4
    X = np.log1p(X).astype(np.float32)
    xy = coords.loc[ids, ["x_um", "y_um"]].values.astype(float)
    A = radius_neighbors_graph(xy, radius, include_self=True, mode="connectivity")
    A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    Xs = np.asarray(A @ X, dtype=np.float32)
    a = sc.AnnData(Xs, obs=pd.DataFrame(index=ids))
    a.var_names = [g.lower() for g in zone_cols]
    a.obsm["spatial"] = xy
    a.obs["cell_identity"] = meta.loc[ids, "Celltype"].astype(str).values
    for c in ["Havcr1_Norm", "Vcam1_Norm"]:
        a.obs[c] = meta.loc[ids, c].values
    return a


def run_radius(radius):
    rng = np.random.default_rng(0)
    samples = meta["SampleID"].unique()
    per_sample, obs_rows, pred_rows = {}, [], []
    for sid in sorted(samples):
        grp = meta.loc[meta["SampleID"] == sid, "Group"].iloc[0]
        a = build_sample(sid, radius)
        comp = build_compartment2(a, macro_map=MACRO, use_anatomical_zones=True,
                                  fallback_radial=True)
        comps2 = np.asarray(comp["comps2"])
        a.obs["compartment"] = comps2[np.asarray(comp["inv2"])]
        a.obs["macro"] = a.obs["cell_identity"].map(MACRO)
        seeds = set(comps2[np.asarray(comp["boundary_mask"])])
        per_sample[sid] = dict(group=grp, n_cells=int(a.n_obs), zone_method=comp["zone_method"],
                               cells_per_macro=a.obs["macro"].value_counts().to_dict(),
                               seed_compartments=sorted(seeds))
        g = a.obs.groupby("compartment")
        o = pd.DataFrame({"n": g.size(), "havcr1": g["Havcr1_Norm"].mean(),
                          "vcam1": g["Vcam1_Norm"].mean()})
        pt = a.obs[a.obs["macro"] == "PT"]
        o["frac_injured"] = pt.groupby("compartment")["cell_identity"].apply(
            lambda s: (s == "Injured PT").mean())
        o["sample"], o["group"] = sid, grp
        o["seed"] = o.index.isin(seeds)
        obs_rows.append(o.reset_index())
        if grp == "Ctrl":
            params = build_default_params_SIR(comps2, comp["macro_of"], comp["region_of"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ml = run_SIR_multilayer(comp=comp, sir_params=params, adata_st=a,
                                        seed_I=SEED_I, D_inter=D_INTER, t_end=T_END,
                                        n_steps=N_STEPS, threshold_I=THR, plot=False)
            S, I, t = _extract_SIR(ml["sol"])
            names = comps2[np.asarray(ml["net"]["global_index"])]
            act = np.where((I > THR).any(0), t[np.argmax(I > THR, axis=0)], np.nan)
            pred_rows.append(pd.DataFrame({"compartment": names, "peak_I": I.max(0),
                                           "t_act": act, "sample": sid}))
            per_sample[sid]["most_vulnerable"] = ml["metrics"].get("most_vulnerable")
        print(f"  [{radius} um] {sid} ({grp}): {a.n_obs} cellule, "
              f"{len(comps2)} compartimenti, zone={comp['zone_method']}", flush=True)

    obs = pd.concat(obs_rows)
    obs = obs[obs["n"] >= MIN_CELLS]
    obs = obs[obs["compartment"].str.split("_").str[0].isin(["PT", "DCT", "TAL"])]
    pred = pd.concat(pred_rows).merge(
        obs[obs["group"] == "Ctrl"][["compartment", "sample"]], on=["compartment", "sample"])
    pr = pred.groupby("compartment").agg(peak_I=("peak_I", "mean"), t_act=("t_act", "mean"),
                                         n_ctrl=("sample", "nunique"))
    pr = pr[pr["n_ctrl"] >= 2]
    ctrl = obs[obs["group"] == "Ctrl"].groupby("compartment")[["havcr1", "vcam1"]].mean()
    iri = obs[obs["group"] == "IRI"].groupby("compartment")[["havcr1", "vcam1", "frac_injured"]].mean()
    seed_c = set(obs.loc[obs["seed"], "compartment"])
    tab = pr.join(iri, how="inner").join(ctrl, rsuffix="_ctrl", how="left")
    tab["d_havcr1"] = tab["havcr1"] - tab["havcr1_ctrl"]
    tab["d_vcam1"] = tab["vcam1"] - tab["vcam1_ctrl"]
    tab["macro"] = tab.index.str.split("_").str[0]
    tab["seed"] = tab.index.isin(seed_c)
    ptab, nos = tab[tab["macro"] == "PT"], tab[~tab["seed"]]
    tests = dict(
        P1_peak_vs_dHavcr1_all=perm_test(tab["peak_I"], tab["d_havcr1"], rng),
        P2_peak_vs_fracInjured_PT=perm_test(ptab["peak_I"], ptab["frac_injured"], rng),
        S_tact_vs_dHavcr1_all=perm_test(tab["t_act"], tab["d_havcr1"], rng),
        S_peak_vs_dVcam1_all=perm_test(tab["peak_I"], tab["d_vcam1"], rng),
        S_peak_vs_dHavcr1_PT=perm_test(ptab["peak_I"], ptab["d_havcr1"], rng),
        S_peak_vs_dHavcr1_noSeed=perm_test(nos["peak_I"], nos["d_havcr1"], rng),
        S_peak_vs_fracInjured_PT_noSeed=perm_test(nos.loc[nos["macro"] == "PT", "peak_I"],
                                                  nos.loc[nos["macro"] == "PT", "frac_injured"], rng),
    )
    for k in ["P1_peak_vs_dHavcr1_all", "P2_peak_vs_fracInjured_PT"]:
        p = tests[k]["p"]
        tests[k]["p_bonferroni"] = None if p is None else round(min(1.0, 2 * p), 4)
    tab.round(5).to_csv(f"{OUT}/compartment_table_R{radius}.csv")
    return dict(radius_um=radius, samples=per_sample, n_compartments=int(len(tab)), tests=tests)


results = {}
for R in RADII:
    print(f"\n===== raggio {R} um =====", flush=True)
    results[f"R{R}"] = run_radius(R)
    print(json.dumps(results[f"R{R}"]["tests"], indent=1), flush=True)

json.dump(dict(settings=dict(radii=RADII, min_cells=MIN_CELLS, seed_I=SEED_I, D_inter=D_INTER,
                             t_end=T_END, threshold=THR, n_perm=N_PERM, macro_map=MACRO),
               results=results), open(f"{OUT}/polonsky_summary.json", "w"), indent=1, default=str)
print(f"\nFatto in {time.time()-t0:.0f}s. Risultati in {OUT}/")
