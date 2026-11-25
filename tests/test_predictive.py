"""R3 predictive test: pure functions behave as pre-specified."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "r3"))
import predictive_test_gse269622 as P  # noqa: E402


def test_perm_test_detects_and_rejects():
    rng = np.random.default_rng(1)
    x = np.arange(14.0)
    assert P.perm_test(x, x + rng.normal(0, 1, 14), n_perm=2000)["p"] < 0.01
    assert P.perm_test(x, -x, n_perm=2000)["p"] > 0.9


def test_within_macro_removes_group_effect():
    # y depends only on macro-type: overall rho high, within-macro rho absent
    macro = np.repeat(["PT", "DCT", "TAL"], 5)
    x = np.r_[np.arange(5) + 20, np.arange(5) + 10, np.arange(5)][::1].astype(float)
    y = np.repeat([3.0, 2.0, 1.0], 5) + np.random.default_rng(2).normal(0, .01, 15)[::-1]
    assert P.perm_test(x, y, n_perm=2000)["p"] < 0.01
    w = P.perm_test(x, y, strata=macro, n_perm=2000)
    assert w["p"] is None or w["p"] > 0.05


def test_holm():
    np.testing.assert_allclose(P.holm(np.array([0.01, 0.04])), [0.02, 0.04])


def test_activation_times_and_scores():
    rng = np.random.default_rng(0); n = 8
    W = rng.uniform(size=(n, n)); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    comp = dict(Wc2=W, boundary_mask=np.arange(n) < 2)
    t, auc = P.activation_times(comp, dict(beta=np.full(n, .4), gamma=np.full(n, .08)))
    assert t[:2].max() < t[2:].min() + 1e-9 and np.all(auc > 0)
    s, c = P.compartment_scores(np.ones((6, 4)) * np.arange(6)[:, None],
                                np.array([0, 0, 1, 1, 2, 2]), 3)
    np.testing.assert_allclose(s, [0.5, 2.5, 4.5]); np.testing.assert_array_equal(c, [2, 2, 2])


def test_stage_test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUT", str(tmp_path))
    rng = np.random.default_rng(3)
    comps = [f"{m}_{z}" for m in ["PT", "DCT", "TAL"] for z in ["cortex", "outer_medulla", "inner_medulla", "core"]]
    base = pd.DataFrame(dict(compartment=comps, macro=[c.split("_")[0] for c in comps],
                             zone=[c.split("_", 1)[1] for c in comps], n_spots=20, seeded=False,
                             beta=.3, rho=.1, R_diff=rng.uniform(2, 5, 12),
                             t_act=rng.uniform(3, 15, 12), auc_D=1.0, injury_score=1.0))
    base.to_csv(tmp_path / "sham_compartments.csv", index=False)
    for s in ["hour4", "hour12", "day2"]:
        d = base.copy(); d["injury_score"] = 1 + (20 - d.t_act) * .1 + rng.normal(0, .05, 12)
        d.loc[0, "n_spots"] = 2                                   # must be excluded
        d.to_csv(tmp_path / f"{s}_compartments.csv", index=False)
    P.N_PERM = 2000
    P.stage_test()
    import json
    s = json.load(open(tmp_path / "summary.json"))
    assert s["results"]["hour12"]["n"] == 11 and s["H1_significant"]
    assert "PT_cortex" in s["results"]["hour12"]["excluded"]
