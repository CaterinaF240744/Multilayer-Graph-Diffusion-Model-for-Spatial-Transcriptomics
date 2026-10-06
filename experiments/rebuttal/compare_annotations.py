"""
compare_annotations.py -- Robustezza all'annotazione (Sezione 8.3, Table tab:annotation).

Passo 1 (dopo NN, RCTD e cell2location):
    python experiments/rebuttal/compare_annotations.py prepare
  - concordanza con NN a livello di macro-tipo e di tipo del riferimento,
    ARI e NMI, confidenza per spot (peso RCTD, entropia cell2location);
  - consenso a maggioranza sul macro-tipo (parita' -> NN);
  - scrive results/annotation/labels_for_sim_{rctd,c2l,consensus}.csv
    (indice barcode, colonna cell_identity) per le simulazioni.

Passo 2 (dopo le quattro simulazioni rapide di main.py):
    python experiments/rebuttal/compare_annotations.py summarize
  - scrive results/annotation/table_annotation.csv con concordanza e
    tempi di invasione per NN, RCTD, cell2location e consenso.

Il macro-tipo dominante di cell2location e' calcolato, come per RCTD, dalla
somma delle abbondanze per macro-tipo (results/annotation/c2l_abundance.csv).
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

A, RUNS = "results/annotation", "results/annotation_runs"
mp = pd.read_csv("data/rctd_io/macro_map.csv").set_index("cell_identity")["macro_type"].to_dict()


def load_labels():
    nn = pd.read_csv(f"{RUNS}/NN/labels.csv", index_col=0)
    nn = pd.DataFrame({"id": nn["cell_identity"].astype(str), "macro": nn["macro_type"].astype(str)})
    r = pd.read_csv(f"{A}/labels_rctd.csv").set_index("barcode")
    rc = pd.DataFrame({"id": r["rctd_label"].astype(str), "macro": r["rctd_macro"].astype(str),
                       "conf": r["rctd_macro_weight"]})
    ab = pd.read_csv(f"{A}/c2l_abundance.csv", index_col=0)
    ab.columns = [c.replace("q05cell_abundance_w_sf_", "") for c in ab.columns]
    macros = pd.Series({c: mp.get(c, "other") for c in ab.columns})
    msum = ab.T.groupby(macros).sum().T
    win = msum.idxmax(axis=1)
    cid = [ab.loc[b, macros.index[macros == win[b]]].idxmax() for b in ab.index]
    p = ab.div(ab.sum(axis=1), axis=0)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    c2 = pd.DataFrame({"id": cid, "macro": win.values, "conf": ent.values}, index=ab.index)
    common = nn.index.intersection(rc.index).intersection(c2.index)
    return nn.loc[common], rc.loc[common], c2.loc[common]


def agreement(a, b):
    return dict(macro_concordance=round(float((a["macro"] == b["macro"]).mean()), 3),
                type_concordance=round(float((a["id"] == b["id"]).mean()), 3),
                ARI=round(adjusted_rand_score(a["macro"], b["macro"]), 3),
                NMI=round(normalized_mutual_info_score(a["macro"], b["macro"]), 3))


def prepare():
    nn, rc, c2 = load_labels()
    n = len(nn)
    votes = pd.DataFrame({"NN": nn["macro"], "RCTD": rc["macro"], "c2l": c2["macro"]})

    def vote(row):
        cnt = row.value_counts()
        return cnt.index[0] if cnt.iloc[0] >= 2 else row["NN"]
    cons_macro = votes.apply(vote, axis=1)
    cons_id = []
    for b in nn.index:
        m = cons_macro[b]
        for src in (nn, rc, c2):
            if src.loc[b, "macro"] == m:
                cons_id.append(src.loc[b, "id"]); break
    cons = pd.DataFrame({"id": cons_id, "macro": cons_macro.values}, index=nn.index)
    all3 = (votes.nunique(axis=1) == 1)
    out = dict(n_common_spots=int(n),
               NN_vs_RCTD=agreement(nn, rc), NN_vs_c2l=agreement(nn, c2),
               RCTD_vs_c2l=agreement(rc, c2), consensus_vs_NN=agreement(cons, nn),
               all_three_agree=round(float(all3.mean()), 3),
               rctd_weight_median=dict(concordant=round(float(rc.loc[all3, "conf"].median()), 3),
                                       discordant=round(float(rc.loc[~all3, "conf"].median()), 3)),
               c2l_entropy=dict(median=round(float(c2["conf"].median()), 3),
                                IQR=[round(float(c2["conf"].quantile(q)), 3) for q in (0.25, 0.75)],
                                concordant=round(float(c2.loc[all3, "conf"].median()), 3),
                                discordant=round(float(c2.loc[~all3, "conf"].median()), 3)),
               macro_counts={k: v["macro"].value_counts().to_dict()
                             for k, v in dict(NN=nn, RCTD=rc, c2l=c2, consensus=cons).items()})
    json.dump(out, open(f"{A}/agreement.json", "w"), indent=1)
    for name, df in dict(rctd=rc, c2l=c2, consensus=cons).items():
        unk = set(df["id"]) - set(mp)
        if unk:
            raise ValueError(f"{name}: etichette fuori da MACRO_MAP: {unk}")
        df[["id"]].rename(columns={"id": "cell_identity"}).to_csv(f"{A}/labels_for_sim_{name}.csv")
    print(json.dumps(out, indent=1))


def summarize():
    ag = json.load(open(f"{A}/agreement.json"))
    conc = {"NN": None, "RCTD": ag["NN_vs_RCTD"], "c2l": ag["NN_vs_c2l"], "consensus": ag["consensus_vs_NN"]}
    rows = []
    for tag in ["NN", "RCTD", "c2l", "consensus"]:
        s = json.load(open(f"{RUNS}/{tag}/summary.json"))
        bl = s["by_layer"]
        rows.append(dict(annotation=tag,
                         macro_concordance_vs_NN=None if conc[tag] is None else conc[tag]["macro_concordance"],
                         ARI_vs_NN=None if conc[tag] is None else conc[tag]["ARI"],
                         t_inv_global=round(s["global_inv_time"], 2),
                         t_inv_PT=round(bl["PT"]["invasion_time"], 2) if bl.get("PT", {}).get("invasion_time") else None,
                         t_inv_DCT=round(bl["DCT"]["invasion_time"], 2) if bl.get("DCT", {}).get("invasion_time") else None,
                         t_inv_TAL=round(bl["TAL"]["invasion_time"], 2) if bl.get("TAL", {}).get("invasion_time") else None,
                         most_vulnerable=s["most_vulnerable"],
                         spots_PT=s["spots_per_macro"].get("PT", 0),
                         spots_DCT=s["spots_per_macro"].get("DCT", 0),
                         spots_TAL=s["spots_per_macro"].get("TAL", 0)))
    t = pd.DataFrame(rows)
    t.to_csv(f"{A}/table_annotation.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    {"prepare": prepare, "summarize": summarize}[sys.argv[1]]()
