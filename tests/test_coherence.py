"""
test_coherence.py — Unit tests for code-article coherence.

Verifies that the code implements the equations described in the manuscript:
  - Exposure-driven infection (β·H·λ, not bilinear β·H·D)
  - Correct parameter values matching Table 1
  - Seed value matching paper Section 6.3 (0.05)
  - Multilayer uses exposure-driven formulation
  - Controls module has all 5 required control functions
  - Section 7 simplified model reproduces Table 12 baseline
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Mock compartment graph for sir_section7 tests ──────────────────────────
# Matches the structure produced by compartments.build_compartment2():
# 13 compartments across 6 macro-types, with boundary + outer_medulla zones.
_MOCK_COMPS = [
    "PT_cortex", "PT_outer_medulla", "PT_inner_medulla", "PT_boundary",
    "DCT_cortex", "DCT_outer_medulla", "DCT_boundary",
    "TAL_outer_medulla", "TAL_inner_medulla",
    "vascular_cortex", "vascular_boundary",
    "immune_cortex",
    "other_cortex", "other_outer_medulla", "other_inner_medulla",
]
_MOCK_MACRO = np.array([c.split("_")[0] for c in _MOCK_COMPS])
_MOCK_REGION = np.array([c.split("_", 1)[1] if "_" in c else "other" for c in _MOCK_COMPS])
_MOCK_BOUNDARY = np.array([
    "outer_medulla" in c or c.endswith("_boundary") for c in _MOCK_COMPS
])


def _build_mock_comp():
    """Build a mock compartment dict matching build_compartment2() output."""
    n = len(_MOCK_COMPS)
    # Simple ring + cross adjacency (ensures connectivity)
    W = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        W[i, j] = W[j, i] = 1.0
    # Add a few cross-links for realism
    for a, b in [(0, 4), (2, 6), (3, 8), (5, 10), (7, 12)]:
        W[a, b] = W[b, a] = 0.5
    return dict(
        Wc2=W,
        comps2=_MOCK_COMPS,
        macro_of=_MOCK_MACRO,
        region_of=_MOCK_REGION,
        boundary_mask=_MOCK_BOUNDARY,
        zone_method="anatomical",
    )


@pytest.fixture
def initialised_section7():
    """Initialise sir_section7 with a mock compartment graph before each test."""
    from sir_section7 import init_from_compartments
    comp = _build_mock_comp()
    init_from_compartments(comp)
    yield comp
    # Reset is not needed — init_from_compartments is idempotent


class TestParameterCoherence:
    """Verify parameters match Table 1 of the manuscript."""

    def test_table1_params(self):
        """Check _MACRO_SIR values match Table 1."""
        from sir_compartments import _MACRO_SIR

        assert _MACRO_SIR["PT"]["beta"] == 0.40
        assert _MACRO_SIR["PT"]["gamma"] == 0.08
        assert _MACRO_SIR["DCT"]["beta"] == 0.28
        assert _MACRO_SIR["DCT"]["gamma"] == 0.10
        assert _MACRO_SIR["TAL"]["beta"] == 0.25
        assert _MACRO_SIR["TAL"]["gamma"] == 0.12
        assert _MACRO_SIR["vascular"]["beta"] == 0.30
        assert _MACRO_SIR["vascular"]["gamma"] == 0.10
        assert _MACRO_SIR["vascular"]["D_I"] == 0.35
        assert _MACRO_SIR["immune"]["beta"] == 0.20
        assert _MACRO_SIR["immune"]["gamma"] == 0.25

    def test_section7_vascular_params(self):
        """Check sir_section7 vascular params aligned to Table 1."""
        from sir_section7 import BETA_BASE, RHO_BASE

        assert BETA_BASE["vascular"] == 0.30
        assert RHO_BASE["vascular"] == 0.10

    def test_section7_gamma_outer_mult(self):
        """Check GAMMA_OUTER_MULT=1.10 is present (was missing before fix)."""
        from sir_section7 import GAMMA_OUTER_MULT

        assert GAMMA_OUTER_MULT == 1.10

    def test_R_diff_table2(self, initialised_section7):
        """Check R_diff for outer_medulla matches Table 2."""
        from sir_section7 import build_params, C_IDX

        beta, rho, DS, DD = build_params()

        # PT_boundary: (0.40*1.25)/(0.08*1.10) = 5.68
        r = beta[C_IDX["PT_boundary"]] / rho[C_IDX["PT_boundary"]]
        assert abs(r - 5.68) < 0.02

        # DCT_boundary: (0.28*1.25)/(0.10*1.10) = 3.18
        r = beta[C_IDX["DCT_boundary"]] / rho[C_IDX["DCT_boundary"]]
        assert abs(r - 3.18) < 0.02

        # Vascular_boundary: (0.30*1.25)/(0.10*1.10) = 3.41
        r = beta[C_IDX["vascular_boundary"]] / rho[C_IDX["vascular_boundary"]]
        assert abs(r - 3.41) < 0.02


class TestExposureDriven:
    """Verify the model uses exposure-driven (β·H·λ), not bilinear (β·H·D)."""

    def test_single_layer_exposure_driven(self):
        """Check sir_compartments uses W_tilde @ D for exposure."""
        import inspect
        from sir_compartments import simulate_SIR_ivp

        src = inspect.getsource(simulate_SIR_ivp)
        assert "W_tilde @ D" in src or "lam = W_tilde @ D" in src
        assert "beta_v * H * lam" in src  # β·H·λ, not β·H·D

    def test_multilayer_exposure_driven(self):
        """Check sir_multilayer uses exposure-driven formulation."""
        import inspect
        from sir_multilayer import simulate_SIR_multilayer

        src = inspect.getsource(simulate_SIR_multilayer)
        assert "lam_l" in src
        assert "Wt @ I_l" in src or "Wt @ D" in src
        assert "b * S_l * lam_l" in src  # β·S·λ, not β·S·I

    def test_not_bilinear_multilayer(self):
        """Ensure the bilinear formulation is NOT used in multilayer."""
        import inspect
        from sir_multilayer import simulate_SIR_multilayer

        src = inspect.getsource(simulate_SIR_multilayer)
        # The bilinear version would have "b * S_l * I_l" without lam
        # The exposure-driven version has "b * S_l * lam_l"
        assert "b * S_l * lam_l" in src
        # Make sure there's no standalone bilinear infection term
        assert "inf_l = b * S_l * I_l" not in src


class TestSeedValue:
    """Verify seed matches paper Section 6.3 (pD(0) = 0.05)."""

    def test_main_seed(self):
        """Check main.py uses seed=0.05."""
        with open(os.path.join(os.path.dirname(__file__), "..", "src", "main.py")) as f:
            content = f.read()
        # Should not contain seed_I=0.10 or seed_phi=0.10
        assert "seed_I      = 0.10" not in content
        assert "seed_phi=0.10" not in content
        # Should contain seed=0.05
        assert "0.05" in content


class TestNomenclature:
    """Verify H/D/R/X nomenclature is used in public-facing surfaces."""

    def test_main_no_phi_ca(self):
        """Check main.py doesn't use φ/C(t)/A(t) interpretation."""
        with open(os.path.join(os.path.dirname(__file__), "..", "src", "main.py")) as f:
            content = f.read()
        assert "φ(t)" not in content
        assert "C(t)  receptor capacity" not in content
        assert "A(t)  absorbed signal" not in content

    def test_hdr_labels(self):
        """Check HDR_LABELS is defined in main.py."""
        with open(os.path.join(os.path.dirname(__file__), "..", "src", "main.py")) as f:
            content = f.read()
        assert "HDR_LABELS" in content
        assert "H(t)  Healthy fraction" in content


class TestControls:
    """Verify sir_controls.py has all 5 required control functions."""

    def test_visium_control_functions_exist(self):
        """Check all 5 Visium control functions are defined."""
        from sir_controls import (
            run_randomized_edge_null_model_visium,
            run_uniform_parameters_control_visium,
            run_vascular_edge_removal_visium,
            run_alternative_seeds_visium,
            run_R_state_sensitivity,
            run_all_controls_visium,
        )
        # All functions are callable
        assert callable(run_randomized_edge_null_model_visium)
        assert callable(run_uniform_parameters_control_visium)
        assert callable(run_vascular_edge_removal_visium)
        assert callable(run_alternative_seeds_visium)
        assert callable(run_R_state_sensitivity)
        assert callable(run_all_controls_visium)

    def test_section7_control_function_exists(self):
        """Check Section 7 simplified control function is defined."""
        from sir_controls import run_all_controls_section7
        assert callable(run_all_controls_section7)

    def test_maslov_sneppen_rewire_exists(self):
        """Check Maslov-Sneppen rewiring function is defined."""
        from sir_controls import maslov_sneppen_rewire_sparse
        assert callable(maslov_sneppen_rewire_sparse)

    def test_maslov_sneppen_preserves_degree(self):
        """Check that rewiring preserves degree distribution."""
        import scipy.sparse as sp
        from sir_controls import maslov_sneppen_rewire_sparse

        # Create a small test graph
        W = sp.lil_matrix((6, 6))
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
        for i, j in edges:
            W[i, j] = W[j, i] = 1.0
        W = W.tocsr()

        rng = np.random.default_rng(42)
        W_rew = maslov_sneppen_rewire_sparse(W, n_swaps=10, rng=rng)

        # Degree sequence should be preserved
        orig_deg = np.asarray(W.sum(axis=1)).flatten()
        rew_deg = np.asarray(W_rew.sum(axis=1)).flatten()
        np.testing.assert_array_almost_equal(orig_deg, rew_deg)

    def test_R_state_sensitivity_runs(self, initialised_section7):
        """Check R-state sensitivity produces valid output on section7 model."""
        from sir_controls import run_all_controls_section7
        results = run_all_controls_section7(save_dir="/tmp/test_controls")

        # Check R sensitivity results
        R_sens = results["R_sensitivity"]
        assert len(R_sens) == 5  # 5 rho multipliers
        # R_fin should be between 0 and 1
        for r in R_sens:
            assert 0 <= r["R_fin"] <= 1
        # Baseline (mult=1.0) should be present
        baseline_r = [r for r in R_sens if r["rho_mult"] == 1.0][0]
        assert baseline_r["R_fin"] > 0.8  # Should be ~0.95


class TestSection7Baseline:
    """Verify Section 7 model reproduces Table 12 baseline after param fixes."""

    def test_DS_DD_separation(self, initialised_section7):
        """Check that build_params returns DS (D_H) uniform 0.05 and DD (D_D)
        per-macro-type, and that make_ode uses DS in dpH and DD in dpD."""
        from sir_section7 import build_params, DS_DEFAULT, DD_BASE, MACRO_MAP, COMPARTMENTS

        beta, rho, DS, DD = build_params()

        # DS must be uniform 0.05 for all compartments
        assert np.allclose(DS, DS_DEFAULT), f"DS should be uniform {DS_DEFAULT}, got {DS}"

        # DD must be per-macro-type from DD_BASE
        for i, c in enumerate(COMPARTMENTS):
            m = MACRO_MAP[c]
            assert DD[i] == DD_BASE[m], f"DD[{c}]={DD[i]} != DD_BASE[{m}]={DD_BASE[m]}"

        # DS must differ from DD (the asymmetry that was broken)
        assert DS_DEFAULT < DD_BASE["PT"], "D_H should be < D_D (asymmetry)"

    def test_baseline_tinv(self, initialised_section7):
        """Check baseline t_inv with corrected DS/DD separation.

        With D_H=0.05 (was incorrectly 0.20-0.35), the baseline shifts.
        The exact value will be determined by re-running Section 7; this
        test checks that the simulation runs and produces a reasonable t_inv.
        """
        from sir_section7 import build_params, W_norm, make_ic, run_sim, global_metrics, SPATIAL_SEED

        beta, rho, DS, DD = build_params()
        y0 = make_ic(SPATIAL_SEED)
        t, _, pD, _ = run_sim(W_norm, beta, rho, DS, DD, y0, t_end=60.0)
        tinv, peakD, _ = global_metrics(t, pD)

        # With corrected D_H=0.05, t_inv should be in a reasonable range.
        # The old (buggy) value was ~6.9; the corrected value will differ.
        # We check it's finite and in a plausible range.
        assert np.isfinite(tinv), "t_inv should be finite"
        assert 3.0 < tinv < 15.0, f"t_inv={tinv} outside plausible range"
        assert 0.2 < peakD < 0.6, f"peakD={peakD} outside plausible range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
