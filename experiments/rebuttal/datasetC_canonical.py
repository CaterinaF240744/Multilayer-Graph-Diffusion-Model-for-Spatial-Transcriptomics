"""
datasetC_canonical.py -- Exploratory application to human kidney transplant
rejection Visium (GSE304669) using the SAME canonical pipeline as the
V1 Mouse Kidney case study.

Differences from the previous datasetC_pipeline.py (all intentional):
  * label transfer with preprocessing.align_sc_st (sc.tl.ingest), as in V1;
  * compartments with compartments.build_compartment2 (marker-based zoning
    with the same marker list -- human orthologs share the symbols -- and
    radial fallback only for spots without marker signal), as in V1;
  * canonical seeding: outer-medulla / boundary compartments (boundary_mask),
    as in V1 -- NOT all cortical compartments;
  * canonical multilayer model (sir_multilayer.run_SIR_multilayer) with the
    Table 1 parameters (build_default_params_SIR), as in V1;
  * explicit, reviewed cell-type -> macro-type mapping (CSV), no substring rules.

PRE-SPECIFIED ANALYSIS (fill in the date before the full run: ____________)
  Evaluability: a sample is analysed only if PT, DCT and TAL each have
    >= MIN_SPOTS spots and at least one seed compartment exists.
  Primary (structural, as for GSE269622): most vulnerable layer and
    activation order among PT/DCT/TAL (layer invasion times).
  Annotation check: each marker is highest in its assigned macro-type
    (LRP2 -> PT; SLC12A3 -> DCT; UMOD, SLC12A1 -> TAL).
  Injury programme: mean log-normalised HAVCR1 + LCN2 + VCAM1 per sample
    and per macro-type, rejection samples vs control (descriptive).
  Interpretation: PT activating first is expected from beta_PT (uniform-
    parameter control); only the annotation, evaluability and injury-
    programme results carry information independent of the parameters.

Usage (repository root, environment with scanpy):
  Step A -- list reference labels and write a DRAFT mapping to review:
    python datasetC_canonical.py --list-labels
  Step B -- after saving the reviewed mapping as data/human_ref/human_macro_map.csv:
    python datasetC_canonical.py --dataset GSE304669
    python datasetC_canonical.py --dataset GSE183456   (reference nephrectomies, Lake et al. 2023)
  With RCTD annotation (after export_rctd_human.py and run_rctd_human.R):
    python datasetC_canonical.py --dataset GSE183456 --annotation rctd
"""
import argparse, gzip, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio

DATASETS = {
    # GSE304669: FFPE CytAssist needle biopsies (GEO per-sample files)
    "GSE304669": dict(data_dir="data/GSE304669", loader="geo_files", samples={
        "control":            "GSM9155022_control_58055",
        "active_AMR":         "GSM9155023_active_AMR_58056",
        "acute_TCMR":         "GSM9155024_acute_TCMR_58057",
        "chronic_active_AMR": "GSM9155025_chronic_active_AMR_58058"}),
    # GSE183456 (Lake et al. 2023): fresh-frozen reference nephrectomies (Space Ranger outs)
    "GSE183456": dict(data_dir="data/GSE183456", loader="spaceranger", samples={
        "IU-F59": "GSM6047774_V19S25-016_XY01_18-0006",
        "IU-F52": "GSM6047775_V19S25-019_XY04-F52",
        "IU-M61": "GSM6047776_V19S25-019_XY03-M61",
        "IU-M32": "GSM6047777_V19S25-019_XY02-M32",
        "21-015": "GSM6047778_V10S15-102_XY03_IU-21-015-2",
        "21-019": "GSM6047779_V10S15-102_XY02_IU-21-019-5"}),
}
MACROS = ("PT", "DCT", "TAL")
VALID_MACROS = {"PT", "DCT", "TAL", "vascular", "immune", "other"}
MIN_SPOTS = 10
N_REF_CELLS = 20000
SEED_I, D_INTER, T_END, N_STEPS, THR = 0.05, 0.05, 60.0, 800, 0.10   # as V1
ANNOT_MARKERS = {"PT": ["LRP2"], "DCT": ["SLC12A3"], "TAL": ["UMOD", "SLC12A1"]}
INJURY = ["HAVCR1", "LCN2", "VCAM1"]


# ---------------------------------------------------------------- loading
def load_sample(prefix):
    X = sio.mmread(f"{prefix}_matrix.mtx.gz").T.tocsr()          # spots x genes
    with gzip.open(f"{prefix}_barcodes.tsv.gz", "rt") as f:
        barcodes = [l.strip() for l in f if l.strip()]
    with gzip.open(f"{prefix}_features.tsv.gz", "rt") as f:
        genes = pd.read_csv(f, sep="\t", header=None).iloc[:, 1].astype(str).values
    with gzip.open(f"{prefix}_tissue_positions.csv.gz", "rt") as f:
        pos = pd.read_csv(f)
    if pos.columns[0] != "barcode":                               # headerless (old format)
        pos = pd.read_csv(gzip.open(f"{prefix}_tissue_positions.csv.gz", "rt"), header=None,
                          names=["barcode", "in_tissue", "array_row", "array_col",
                                 "pxl_row_in_fullres", "pxl_col_in_fullres"])
    pos = pos.set_index("barcode")
    pos = pos[pos["in_tissue"] == 1]
    keep = [i for i, b in enumerate(barcodes) if b in pos.index]
    bc = [barcodes[i] for i in keep]
    a = sc.AnnData(X[keep], obs=pd.DataFrame(index=bc), var=pd.DataFrame(index=genes))
    a.obsm["spatial"] = pos.loc[bc, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
    a.var_names_make_unique()
    return a


def load_spaceranger(sample_dir):
    """Load a Space Ranger output folder (searches for filtered_feature_bc_matrix.h5)."""
    import glob
    hits = glob.glob(os.path.join(sample_dir, "**", "filtered_feature_bc_matrix.h5"), recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError(f"{sample_dir}: found {len(hits)} filtered_feature_bc_matrix.h5")
    outs = os.path.dirname(hits[0])
    a = sc.read_visium(outs, count_file="filtered_feature_bc_matrix.h5")
    a.var_names_make_unique()
    a.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=float)
    return a


def load_reference(ref_dir, keep_genes=None):
    counts = sio.mmread(os.path.join(ref_dir, "sc_counts.mtx")).T.tocsr()
    genes = [l.strip() for l in open(os.path.join(ref_dir, "sc_genes.txt"))]
    barcodes = [l.strip() for l in open(os.path.join(ref_dir, "sc_barcodes.txt"))]
    meta = pd.read_csv(os.path.join(ref_dir, "sc_meta.txt"), sep="\t")
    meta = meta.set_index(meta.columns[0]).loc[barcodes]
    ct_col = "Cluster_Idents" if "Cluster_Idents" in meta.columns else meta.columns[-1]
    a = sc.AnnData(counts, obs=pd.DataFrame({"cell_identity": meta[ct_col].astype(str).values},
                                            index=barcodes), var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    if a.n_obs > N_REF_CELLS:
        rng = np.random.default_rng(0)
        a = a[np.sort(rng.choice(a.n_obs, N_REF_CELLS, replace=False))].copy()
    if keep_genes is not None:                                    # memory: genes measured in ST
        a = a[:, [g for g in a.var_names if g in keep_genes]].copy()
    sc.pp.filter_genes(a, min_counts=1)
    return a


def preprocess_st(a):
    """Same steps as preprocessing.load_and_preprocess_visium (V1)."""
    sc.pp.filter_genes(a, min_counts=1)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    full = a.copy()                                               # all genes, for markers
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    a = a[:, a.var["highly_variable"]].copy()
    return a, full


def mean_by(full, genes, groups):
    out = {}
    for g in genes:
        if g in full.var_names:
            v = full[:, g].X
            v = np.asarray(v.todense()).ravel() if hasattr(v, "todense") else np.asarray(v).ravel()
            out[g] = pd.Series(v).groupby(groups).mean().round(4).to_dict()
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-src", default="src")
    ap.add_argument("--dataset", default="GSE304669", choices=sorted(DATASETS))
    ap.add_argument("--ref-dir", default="data/human_ref")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--list-labels", action="store_true")
    ap.add_argument("--annotation", default="ingest", choices=["ingest", "rctd"],
                    help="ingest = label transfer as in V1; rctd = dominant cell type from RCTD "
                         "(full mode), read from data/rctd_human/<dataset>/<sample>/labels_rctd.csv")
    args = ap.parse_args()
    sys.path.insert(0, args.repo_src)
    cfg = DATASETS[args.dataset]
    if args.out_dir is None:
        args.out_dir = f"results/datasetC_canonical/{args.dataset}_{args.annotation}"
    os.makedirs(args.out_dir, exist_ok=True)
    map_path = os.path.join(args.ref_dir, "human_macro_map.csv")

    # ---- Step A: labels and draft mapping ----
    if args.list_labels:
        ref = load_reference(args.ref_dir)
        counts = ref.obs["cell_identity"].value_counts()
        draft = pd.DataFrame({"cell_identity": counts.index, "n_cells_subsample": counts.values,
                              "macro_type": ""})
        path = os.path.join(args.ref_dir, "human_macro_map_DRAFT.csv")
        draft.to_csv(path, index=False)
        print(counts.to_string())
        print(f"\nDraft written to {path}. Fill 'macro_type' with one of {sorted(VALID_MACROS)} "
              f"and save as {map_path}.")
        return

    # ---- Step B: full run ----
    mp = pd.read_csv(map_path)
    bad = set(mp["macro_type"].dropna()) - VALID_MACROS
    if bad or mp["macro_type"].isna().any():
        raise ValueError(f"Mapping incomplete or invalid values: {bad}")
    macro_map = dict(zip(mp["cell_identity"].astype(str), mp["macro_type"]))

    from preprocessing import preprocess_sc_for_integration, align_sc_st
    from compartments import build_compartment2
    from sir_compartments import build_default_params_SIR
    from sir_multilayer import run_SIR_multilayer

    t0 = time.time()
    loader = load_sample if cfg["loader"] == "geo_files" else load_spaceranger
    st = {n: loader(os.path.join(cfg["data_dir"], p)) for n, p in cfg["samples"].items()}
    for n, a in st.items():
        print(f"{n}: {a.n_obs} spots x {a.n_vars} genes", flush=True)
    st_genes = set().union(*[set(a.var_names) for a in st.values()])
    ref = load_reference(args.ref_dir, keep_genes=st_genes)
    unmapped = set(ref.obs["cell_identity"].unique()) - set(macro_map)
    if unmapped:
        raise ValueError(f"Reference labels missing from mapping: {sorted(unmapped)}")
    print(f"Reference: {ref.shape}; preprocessing...", flush=True)
    ref = preprocess_sc_for_integration(ref)
    print(f"Reference ready ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for name, a in st.items():
        t1 = time.time()
        print(f"\n===== {name} ({a.n_obs} spots) =====", flush=True)
        a_hvg, full = preprocess_st(a)
        a2 = align_sc_st(ref, a_hvg, obs_key="cell_identity")
        if args.annotation == "rctd":
            lab = pd.read_csv(os.path.join("data/rctd_human", args.dataset, name, "labels_rctd.csv"))
            lab = lab.set_index("barcode")
            keep = [b for b in a2.obs_names if b in lab.index]
            print(f"  RCTD labels for {len(keep)}/{a2.n_obs} spots", flush=True)
            a2 = a2[keep].copy()
            full = full[keep].copy()
            a2.obs["cell_identity"] = lab.loc[keep, "rctd_label"].astype(str).values
            unmapped = set(a2.obs["cell_identity"]) - set(macro_map)
            if unmapped:
                raise ValueError(f"RCTD labels missing from mapping: {sorted(unmapped)}")
        comp = build_compartment2(a2, macro_map=macro_map,
                                  use_anatomical_zones=True, fallback_radial=True)
        macro = a2.obs["macro_type"].astype(str)
        zone = a2.obs["compartment2"].astype(str).str.split("_", n=1).str[1]
        n_macro = macro.value_counts().to_dict()
        n_seed = int(np.asarray(comp["boundary_mask"]).sum())
        res = dict(n_spots=int(a2.n_obs), n_genes_transfer=int(a2.n_vars),
                   spots_per_macro=n_macro, spots_per_zone=zone.value_counts().to_dict(),
                   zone_method=comp["zone_method"], compartments=list(comp["comps2"]),
                   n_seed_compartments=n_seed)
        full_groups = macro.reindex(full.obs_names).values
        res["annotation_markers"] = mean_by(full, sum(ANNOT_MARKERS.values(), []), full_groups)
        res["annotation_check"] = {
            g: (max(res["annotation_markers"][g], key=res["annotation_markers"][g].get) == m)
            for m, gs in ANNOT_MARKERS.items() for g in gs if g in res["annotation_markers"]}
        res["injury_by_macro"] = mean_by(full, INJURY, full_groups)
        res["injury_sample_mean"] = {g: round(float(np.asarray(full[:, g].X.mean())), 4)
                                     for g in INJURY if g in full.var_names}

        evaluable = all(n_macro.get(m, 0) >= MIN_SPOTS for m in MACROS) and n_seed > 0
        res["evaluable"] = bool(evaluable)
        if evaluable:
            params = build_default_params_SIR(np.asarray(comp["comps2"]),
                                              comp["macro_of"], comp["region_of"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = run_SIR_multilayer(comp=comp, sir_params=params, adata_st=a2,
                                         seed_I=SEED_I, D_inter=D_INTER, t_end=T_END,
                                         n_steps=N_STEPS, threshold_I=THR, plot=False)
            m = sim["metrics"]
            bl = m.get("by_layer", {})
            times = {k: v.get("invasion_time") for k, v in bl.items()}
            res.update(most_vulnerable=m.get("most_vulnerable"),
                       global_inv_time=m.get("global_inv_time"),
                       layer_invasion_time=times,
                       activation_order=[k for k, v in sorted(
                           ((k, v) for k, v in times.items() if v is not None),
                           key=lambda kv: kv[1])],
                       by_layer=bl)
        else:
            res["reason_not_evaluable"] = (f"spots per macro {n_macro}, seed compartments {n_seed}; "
                                           f"minimum {MIN_SPOTS} spots for each of {MACROS}")
        res["seconds"] = round(time.time() - t1)
        summary[name] = res
        print(json.dumps({k: res[k] for k in ["spots_per_macro", "spots_per_zone", "zone_method",
                                              "evaluable", "annotation_check"]}, default=str), flush=True)
        if evaluable:
            print("  order:", res["activation_order"], "| most vulnerable:", res["most_vulnerable"], flush=True)

    out = os.path.join(args.out_dir, "datasetC_canonical_summary.json")
    json.dump(dict(dataset=args.dataset, annotation=args.annotation, settings=dict(min_spots=MIN_SPOTS, n_ref_cells=N_REF_CELLS, seed_I=SEED_I,
                                 D_inter=D_INTER, t_end=T_END, threshold=THR),
                   samples=summary), open(out, "w"), indent=1, default=str)
    print(f"\nSaved {out} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
