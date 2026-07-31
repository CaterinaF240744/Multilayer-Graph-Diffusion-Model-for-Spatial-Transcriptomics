"""
compartments.py
===============
Construction of the compartmental graph.

Main functions
--------------
coarse_grain_graph      : coarse-grained graph between compartments
build_compartment2      : compartment = macro_type × anatomical_zone
                          Zone is determined by marker genes (cortex /
                          outer_medulla / inner_medulla) instead of geometric
                          radial tertiles.

Murine renal anatomical zoning
--------------------------------
The V1_Mouse_Kidney Visium section covers the cortico-papillary axis.
The three zones are identified by validated marker genes:

  cortex       : slc34a1 (PT-S1/S2), nphs1/nphs2 (glomerular podocytes),
                 pecam1 (cortical endothelium)
  outer_medulla: umod (TAL — expressed exclusively in OSOM/ISOM),
                 slc12a1 (NKCC2 — TAL)
  inner_medulla: aqp2 (papillary principal CD), slc12a3 (DCT, but
                 when co-localised with aqp2 indicates a transition area),
                 rare in this dataset; assigned to boundary by default

Each spot receives the zone of the marker with the highest normalised
expression (or the cell type transferred via ingest if markers are absent).
A conservative radial tertile fallback is used when no markers are available.

NOTE ON BROADCASTING
---------------------
SIR dynamics are simulated over C compartments.
Results are then expanded to Visium spots via inv2:
    value_spot = value_compartment[inv2]
Spatial maps are therefore uniform within each compartment.
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as sp_laplacian

from preprocessing import MACRO_MAP
from graph_utils import build_knn_graph


# ── Marker genes per anatomical zone ────────────────────────────────────────
# Source: Lake et al. Nature 2023; Gerhardt & Bhatt CurrOpNephrol 2023;
#        Lei et al. Nat Comm 2025 (Visium topo murino).
# Names are lowercased per corrispondere alle var_names di adata_st2
# after align_sc_st (che porta tutto in lowercase).

ZONE_MARKERS: dict[str, list[str]] = {
    "cortex": [
        "slc34a1",   # PT-S1/S2 (cotrasportatore Na-Pi — esclusivo cortex)
        "lrp2",      # megalin — PT apicale
        "cubn",      # cubilina — PT
        "nphs1",     # nefrina — podociti glomerulari
        "nphs2",     # podocina
        "pecam1",    # PECAM-1 / CD31 — endotelio, glomerulo + vasa recta cortex
    ],
    "outer_medulla": [
        "umod",      # uromodulina — ESCLUSIVAMENTE TAL (OSOM + ISOM)
        "slc12a1",   # NKCC2 — TAL ascending limb
        "cldn16",    # claudina-16 — TAL tight junctions
        "slc12a3",   # NCC — DCT (transizione ISOM→cortex)
    ],
    "inner_medulla": [
        "aqp2",      # aquaporina-2 — CD principale, papilla
        "aqp3",      # aquaporina-3 — CD basolaterale
        "avpr2",     # recettore ADH — CD principale
        "atp6v1b1",  # ATPasi — CD intercalato
    ],
}

# Priority order: if a spot expresses markers for multiple zones, the zone
# with the highest normalised score is preferred.
ZONE_ORDER = ["inner_medulla", "outer_medulla", "cortex"]

# Zone → region token mapping usato nel nome del compartimento
# (maintains backward compatibility with downstream code che usa region_of)
ZONE_TO_REGION: dict[str, str] = {
    "cortex":        "cortex",
    "outer_medulla": "outer_medulla",
    "inner_medulla": "inner_medulla",
    # fallback geometrico (usato solo se i marker non sono disponibili)
    "core":          "core",
    "mid":           "mid",
    "boundary":      "boundary",
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def coarse_grain_graph(W, labels, normalize: str = "avg"):
    """
    Costruisce il grafo coarse-grained tra compartments.

    Parameter *normalize*:
    - ``"none"``  → Wc = Sᵀ W S  (weight sum)
    - ``"avg"``   → mean per (Nᵢ · Nⱼ) — preserves orders of magnitude
    - ``"rw"``    → lumping of the transition matrix P = D⁻¹W

    Parameters
    ----------
    W      : sparse (N, N) — grafo nodi
    labels : array-like (N,) — etichetta compartimento per ogni nodo
    normalize : str

    Returns
    -------
    Wc    : sparse (C, C) — grafo compartimentale senza auto-loop
    comps : ndarray (C,)  — nomi dei compartments (ordinati)
    inv   : ndarray (N,)  — indice compartimento per ogni nodo
    """
    if not sp.issparse(W):
        W = sp.csr_matrix(W)
    W = W.tocsr()

    labels = np.asarray(labels)
    comps, inv = np.unique(labels, return_inverse=True)
    C = len(comps)
    N = W.shape[0]

    S = sp.csr_matrix(
        (np.ones(N, dtype=np.float64), (np.arange(N), inv)),
        shape=(N, C),
    )

    if normalize == "none":
        Wc = (S.T @ W @ S).tocsr()

    elif normalize == "avg":
        Wc  = (S.T @ W @ S).tocsr()
        Nc  = np.asarray(S.sum(axis=0)).reshape(-1)
        den = np.maximum(Nc[:, None] * Nc[None, :], 1.0)
        Wc  = Wc.multiply(1.0 / den)

    elif normalize == "rw":
        deg     = np.asarray(W.sum(axis=1)).reshape(-1)
        inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
        P_rw    = (sp.diags(inv_deg) @ W).tocsr()
        Wc      = (S.T @ P_rw @ S).tocsr()

    else:
        raise ValueError("normalize deve essere: none | avg | rw")

    Wc.setdiag(0)
    Wc.eliminate_zeros()
    return Wc, comps, inv


def split_into_connected_components(Wsp, inv2_base, comps2_base,
                                    min_size: int = 5):
    """
    Divide ogni compartimento nelle sue componenti connesse spaziali.

    A compartment like "PT_cortex" may consist of separate islands.
    Questa funzione le separa in PT_cortex_cc0, PT_cortex_cc1, ecc.

    Parameters
    ----------
    Wsp         : sparse (N,N) — grafo kNN spot-level
    inv2_base   : ndarray int (N,) — assegnazione compartimento base
    comps2_base : ndarray (C,) — nomi compartments base
    min_size    : int — cc con meno di min_size spot vengono fuse
                  nella componente principale del compartimento

    Returns
    -------
    inv2_cc   : ndarray int (N,)
    comps2_cc : ndarray
    cc_map    : dict {nuovo_nome: nome_base}
    """
    from scipy.sparse.csgraph import connected_components

    N = Wsp.shape[0]
    new_labels = np.empty(N, dtype=object)

    for ci, cname in enumerate(comps2_base):
        node_mask = inv2_base == ci
        node_idx  = np.where(node_mask)[0]
        if len(node_idx) == 0:
            continue

        Wsub = sp.csr_matrix(Wsp)[np.ix_(node_idx, node_idx)]
        n_cc, cc_labels = connected_components(Wsub, directed=False)

        if n_cc == 1:
            new_labels[node_idx] = cname
            continue

        cc_sizes = np.bincount(cc_labels)
        main_cc  = int(np.argmax(cc_sizes))
        for k in range(n_cc):
            members = node_idx[cc_labels == k]
            if cc_sizes[k] < min_size:
                new_labels[members] = cname
            else:
                cc_id = k if k != main_cc else 0
                new_labels[members] = f"{cname}_cc{cc_id}"

    comps2_cc, inv2_cc = np.unique(new_labels, return_inverse=True)
    cc_map = {c: "_".join(c.split("_")[:-1]) if "_cc" in c else c
              for c in comps2_cc}

    return inv2_cc, comps2_cc, cc_map


# ═══════════════════════════════════════════════════════════════════════════
# ANATOMICAL ZONING
# ═══════════════════════════════════════════════════════════════════════════

def assign_anatomical_zone(
    adata_st2,
    zone_markers: dict[str, list[str]] | None = None,
    fallback_radial: bool = True,
) -> np.ndarray:
    """
    Assegna ogni spot Visium a una zona anatomica renale.

    Strategia
    ----------
    1. Per ogni zona in ZONE_MARKERS, calcola il punteggio medio
       normalizzato dei marker disponibili nell'adata (var_names lowercase).
    2. Assegna ogni spot alla zona con punteggio massimo.
    3. If no marker is available (or fallback_radial=True and the score
       is zero), use the radial tertile partition as fallback.

    Parameters
    ----------
    adata_st2      : AnnData — spots × geni, var_names in lowercase
    zone_markers   : dict {zona: [marker, ...]} — se None usa ZONE_MARKERS
    fallback_radial: bool — usa terzili radiali se i marker non bastano

    Returns
    -------
    zone_labels : ndarray (N,) di stringhe — zona per ogni spot
    """
    if zone_markers is None:
        zone_markers = ZONE_MARKERS

    import scipy.sparse as sp_local

    N = adata_st2.n_obs
    var_lower = np.array([v.lower() for v in adata_st2.var_names])

    # Matrice X densa (solo se serve — usiamo slicing sparso)
    X = adata_st2.X
    if sp_local.issparse(X):
        X = X.tocsr()

    # ── Compute zone scores ────────────────────────────────────────
    zone_scores = {}
    for zone, markers in zone_markers.items():
        markers_lower = [m.lower() for m in markers]
        present = [m for m in markers_lower if m in var_lower]
        if not present:
            zone_scores[zone] = np.zeros(N, dtype=np.float32)
            continue

        idx = np.array([np.where(var_lower == m)[0][0] for m in present])
        if sp_local.issparse(X):
            sub = np.asarray(X[:, idx].todense(), dtype=np.float32)
        else:
            sub = np.asarray(X[:, idx], dtype=np.float32)

        # Normalise each gene to [0, 1] per evitare dominanza di singoli geni
        gene_max = sub.max(axis=0)
        gene_max[gene_max == 0] = 1.0
        sub = sub / gene_max

        zone_scores[zone] = sub.mean(axis=1)

    # ── Assignment ────────────────────────────────────────────────────
    score_matrix = np.stack(
        [zone_scores[z] for z in ZONE_ORDER], axis=1
    )  # (N, n_zones)
    max_score    = score_matrix.max(axis=1)  # (N,)
    best_zone_idx = np.argmax(score_matrix, axis=1)  # (N,)

    zones = np.array(ZONE_ORDER)
    zone_labels = zones[best_zone_idx]

    # ── Radial fallback for spots with zero score ─────────────────────
    zero_mask = max_score <= 0.0
    n_zero    = int(zero_mask.sum())

    if n_zero > 0:
        if fallback_radial:
            coords = adata_st2.obsm["spatial"].astype(np.float32)
            center = coords.mean(axis=0)
            dist_c = np.linalg.norm(coords - center, axis=1)
            q1, q2 = np.quantile(dist_c, [0.33, 0.66])
            radial = np.where(dist_c <= q1, "core",
                              np.where(dist_c <= q2, "mid", "boundary"))
            zone_labels[zero_mask] = radial[zero_mask]
            warnings.warn(
                f"[assign_anatomical_zone] {n_zero} spot senza marker "
                f"espresso — assegnati via terzili radiali (fallback).",
                UserWarning, stacklevel=2,
            )
        else:
            # Assign "cortex" by default (most common zone in murine kidney)
            zone_labels[zero_mask] = "cortex"
            warnings.warn(
                f"[assign_anatomical_zone] {n_zero} spot senza marker "
                f"espresso — assegnati a 'cortex' (default).",
                UserWarning, stacklevel=2,
            )

    n_zones = {z: int((zone_labels == z).sum()) for z in np.unique(zone_labels)}
    print(f"  [Anatomical zoning] spot distribution: {n_zones}")

    # How many markers found per zone
    for zone in ZONE_ORDER:
        present = [m.lower() for m in zone_markers[zone]
                   if m.lower() in var_lower]
        print(f"    {zone}: {len(present)}/{len(zone_markers[zone])} marker "
              f"found ({', '.join(present) if present else '—'})")

    return zone_labels


# ═══════════════════════════════════════════════════════════════════════════
# COMPARTMENT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_compartment2(
    adata_st2,
    macro_map:                  dict  | None = None,
    k_knn:                      int   = 6,
    use_anatomical_zones:       bool  = True,
    zone_markers:               dict  | None = None,
    fallback_radial:            bool  = True,
    split_connected_components: bool  = False,
    cc_min_size:                int   = 5,
) -> dict:
    """
    Costruisce il grafo compartimentale a grana fine.

    compartment2 = macro_type × zona
    dove zona ∈ {cortex, outer_medulla, inner_medulla} se use_anatomical_zones=True
    oppure ∈ {core, mid, boundary} (terzili radiali) se False.

    Zonazione anatomica
    -------------------
    Con use_anatomical_zones=True, ogni spot viene assegnato alla zona
    tramite espressione di marker genici (vedi assign_anatomical_zone).
    Questo produce compartments biologicamente significativi:
      PT_cortex, TAL_outer_medulla, CD_inner_medulla, ecc.

    Invece della partizione radiale precedente (terzili della distanza
    dal centroide geometrico) che non aveva corrispondenza anatomica.

    Backward compatibility
    ----------------------
    If use_anatomical_zones=False the behaviour is identical to
    versione precedente (terzili radiali → core/mid/boundary).

    NOTA SUL BROADCASTING
    ----------------------
    La dinamica SIR viene simulata sui C compartments.
    I risultati vengono poi espansi agli spot Visium tramite inv2:
        valore_spot = valore_compartimento[inv2]
    Le mappe spaziali sono quindi uniformi dentro ogni compartimento.

    Parameters
    ----------
    adata_st2              : AnnData Visium con obsm['spatial'] e cell_identity
    macro_map              : dizionario cell_identity → macro_type
    k_knn                  : numero di vicini per il grafo kNN spot-level
    use_anatomical_zones   : se True usa marker genici per la zona (consigliato)
    zone_markers           : override per ZONE_MARKERS (default None = usa ZONE_MARKERS)
    fallback_radial        : se True, spot senza marker ricevono zona radiale
    split_connected_components : se True, divide compartments in cc spaziali
    cc_min_size            : soglia dimensione per fusione cc piccole

    Returns
    -------
    dict con chiavi:
        Wsp, Wc2, Lc2    — grafi e Laplaciano
        comps2           — nomi compartments (es. "PT_cortex")
        inv2             — indici compartimento per spot
        macro_of         — macro_type per compartimento
        region_of        — zona per compartimento (cortex/outer_medulla/...)
        boundary_mask    — compartments di bordo (outer_medulla o boundary)
        zone_method      — "anatomical" o "radial"
        cc_map           — (solo se split_cc=True) {nome_cc: nome_base}
    """
    coords = adata_st2.obsm["spatial"].astype(np.float32)
    Wsp    = build_knn_graph(coords, k=k_knn)

    # ── Macro_type ──────────────────────────────────────────────────────────
    if macro_map is None:
        macro_map = MACRO_MAP
    adata_st2.obs["macro_type"] = (
        adata_st2.obs["cell_identity"].astype(str)
        .map(macro_map).fillna("other")
    )

    # ── Zoning ───────────────────────────────────────────────────────────
    if use_anatomical_zones:
        print("  [build_compartment2] Anatomical zoning from marker genes...")
        zone_labels = assign_anatomical_zone(
            adata_st2,
            zone_markers=zone_markers,
            fallback_radial=fallback_radial,
        )
        zone_method = "anatomical"
    else:
        print("  [build_compartment2] Radial zoning (distance tertiles)...")
        center = coords.mean(axis=0)
        dist_c = np.linalg.norm(coords - center, axis=1)
        q1, q2 = np.quantile(dist_c, [0.33, 0.66])
        zone_labels = np.where(dist_c <= q1, "core",
                               np.where(dist_c <= q2, "mid", "boundary"))
        zone_method = "radial"

    adata_st2.obs["region"]      = zone_labels
    adata_st2.obs["compartment2"] = (
        adata_st2.obs["macro_type"].astype(str)
        + "_"
        + adata_st2.obs["region"].astype(str)
    )

    # ── Base compartmental graph ──────────────────────────────────────────
    Wc2_base, comps2_base, inv2_base = coarse_grain_graph(
        Wsp, adata_st2.obs["compartment2"].values, normalize="avg"
    )

    # ── Option: split connected components ────────────────────────────────
    cc_map = None
    if split_connected_components:
        inv2, comps2, cc_map = split_into_connected_components(
            Wsp, inv2_base, comps2_base, min_size=cc_min_size
        )
        Wc2, _, _ = coarse_grain_graph(
            Wsp, comps2[inv2], normalize="avg"
        )
        print(f"  [CC split] {len(comps2_base)} → {len(comps2)} compartments")
    else:
        inv2   = inv2_base
        comps2 = comps2_base
        Wc2    = Wc2_base

    Lc2 = sp_laplacian(Wc2, normed=True).tocsr()

    # ── Per-compartment attributes ─────────────────────────────────────────
    # macro_of: first token of the name (es. "PT" da "PT_cortex")
    macro_of = np.array([c.split("_")[0] for c in comps2])

    # region_of: everything after the first "_"
    # handles names like "PT_outer_medulla" (due underscore) correttamente
    def _region(name: str) -> str:
        parts = name.split("_", 1)
        return parts[1] if len(parts) > 1 else "other"

    region_of = np.array([_region(c) for c in comps2])

    # boundary_mask: anatomical or radial boundary compartments
    # With anatomical zoning: outer_medulla is the cortico-medullary boundary
    # With radial zoning: boundary as before
    if use_anatomical_zones:
        boundary_mask = np.array([
            "outer_medulla" in c or c.endswith("_boundary")
            for c in comps2
        ])
    else:
        boundary_mask = np.array([c.endswith("_boundary") for c in comps2])

    print(f"  [build_compartment2] {len(comps2)} compartments  "
          f"({boundary_mask.sum()} boundary/outer_medulla)  "
          f"metodo={zone_method}")

    # Diagnostics: zone distribution per macro_type
    # Useful to verify that TAL has spots in outer_medulla (seed zone)
    print("  [build_compartment2] zones per macro_type:")
    for mt in np.unique(macro_of):
        mask_mt   = macro_of == mt
        zones_mt  = region_of[mask_mt]
        zone_dist = {z: int((zones_mt == z).sum()) for z in np.unique(zones_mt)}
        has_seed  = any(boundary_mask[mask_mt])
        seed_tag  = " [SEED OK]" if has_seed else " [NO SEED — solo diffusione]"
        print(f"    {mt:12s}: {zone_dist}{seed_tag}")

    result = dict(
        Wsp=Wsp, Wc2=Wc2, Lc2=Lc2,
        comps2=comps2, inv2=inv2,
        macro_of=macro_of, region_of=region_of,
        boundary_mask=boundary_mask,
        zone_method=zone_method,
    )
    if cc_map is not None:
        result["cc_map"] = cc_map
    return result

