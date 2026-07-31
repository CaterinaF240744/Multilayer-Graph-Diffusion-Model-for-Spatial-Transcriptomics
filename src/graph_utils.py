"""
graph_utils.py
==============
Graph construction and propagation utilities for spatial transcriptomics data.

Exported functions
------------------
build_spatial_graph   : spatial kNN graph + boundary detection + random-walk matrix
build_knn_graph       : Gaussian-weighted kNN graph from coordinates
random_walk           : T-step random-walk diffusion on a graph
profile_by_distance   : mean signal profile binned by boundary distance
find_optimal_T        : find the random-walk step T that best matches stationary dist.
crossing_time         : first time each node crosses a threshold in an ODE trajectory
graph_geodesic_dist   : geodesic (hop) distance from a set of source nodes
invasion_metrics      : area, radius and velocity of a spreading front

Note: torch / torch-geometric GNN classes (NodeFields, GNNFields, EdgeTransport,
BetaParam, GNN_PCA_Predictor) have been removed from this module.
If needed, reintroduce them from version history — they are not required by the
current analysis pipeline.
"""

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian, shortest_path, dijkstra
from sklearn.neighbors import NearestNeighbors


# ═══════════════════════════════════════════════════════════════════════════
# SPATIAL GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def build_spatial_graph(adata, boundary_quantile: float = 0.10) -> dict:
    """
    Build a spatial connectivity graph from a Visium AnnData object.

    Uses squidpy to compute spatial neighbours, then derives:
    - symmetric adjacency matrix A
    - graph Laplacian L
    - boundary nodes (low-degree spots at the tissue edge)
    - shortest-path distance of each spot from the boundary
    - row-normalised transition matrix P for random walks

    Parameters
    ----------
    adata             : AnnData with spatial coordinates in obsm['spatial']
    boundary_quantile : spots with degree <= this quantile are labelled boundary

    Returns
    -------
    dict with keys: A, L, deg, is_boundary, boundary_nodes, dist_boundary, P
    """
    import squidpy as sq
    sq.gr.spatial_neighbors(adata)

    A = adata.obsp["spatial_connectivities"].tocsr()
    A = (A + A.T) * 0.5

    L = laplacian(A, normed=False)

    deg = np.array(A.sum(axis=1)).flatten()
    thr = np.quantile(deg, boundary_quantile)
    is_boundary   = deg <= thr
    boundary_nodes = np.where(is_boundary)[0]

    A_bool = A.copy()
    A_bool.data[:] = 1.0
    A_bool.eliminate_zeros()
    dist_boundary = shortest_path(
        A_bool, directed=False, unweighted=True, indices=boundary_nodes
    ).min(axis=0)

    adata.obs["is_boundary"]   = is_boundary
    adata.obs["dist_boundary"] = dist_boundary

    deg_safe = deg.copy()
    deg_safe[deg_safe == 0] = 1.0
    P = sp.csr_matrix(A.multiply(1.0 / deg_safe[:, None]))

    return dict(
        A=A, L=L, deg=deg,
        is_boundary=is_boundary,
        boundary_nodes=boundary_nodes,
        dist_boundary=dist_boundary,
        P=P,
    )


def build_knn_graph(coords: np.ndarray, k: int = 6,
                    sigma: float = None) -> sp.csr_matrix:
    """
    Build a Gaussian-weighted symmetric kNN graph from 2-D coordinates.

    Parameters
    ----------
    coords : (N, 2) array of spatial coordinates
    k      : number of nearest neighbours (excluding self)
    sigma  : bandwidth for Gaussian weights; defaults to median pairwise distance

    Returns
    -------
    W : symmetric (N, N) sparse weight matrix
    """
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dist_nn, ind_nn = nn.kneighbors(coords)
    dist_nn = dist_nn[:, 1:]
    ind_nn  = ind_nn[:, 1:]

    if sigma is None:
        sigma = float(np.median(dist_nn))

    w    = np.exp(-(dist_nn ** 2) / (2 * sigma ** 2)).astype(np.float32)
    rows = np.repeat(np.arange(n), k)
    W    = sp.csr_matrix((w.reshape(-1), (rows, ind_nn.reshape(-1))), shape=(n, n))
    return W.maximum(W.T)


# ═══════════════════════════════════════════════════════════════════════════
# RANDOM-WALK DIFFUSION
# ═══════════════════════════════════════════════════════════════════════════

def random_walk(x0: np.ndarray, P: sp.spmatrix, T: int) -> np.ndarray:
    """
    Apply T steps of a random walk to an initial signal x0.

    Parameters
    ----------
    x0 : (N,) initial signal
    P  : (N, N) row-normalised transition matrix
    T  : number of steps

    Returns
    -------
    x : (N,) diffused signal after T steps
    """
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(T):
        x = P @ x
    return x


def profile_by_distance(x: np.ndarray, dist_boundary: np.ndarray) -> pd.DataFrame:
    """
    Compute the mean signal value at each integer boundary-distance bin.

    Returns a DataFrame with columns ['d', 'x'].
    """
    return (
        pd.DataFrame({"d": dist_boundary.astype(int), "x": x})
        .groupby("d")["x"].mean()
        .reset_index()
    )


def find_optimal_T(x0: np.ndarray, P: sp.spmatrix,
                   dist_boundary: np.ndarray,
                   T_range=range(0, 101, 5)) -> tuple:
    """
    Find the random-walk step count T that minimises the MSE between the
    diffused signal and the (column-sum) stationary distribution of P.

    Returns
    -------
    T_vals  : list of T values tested
    errors  : corresponding MSE values (T=0 is forced to inf to avoid trivial minimum)
    """
    col_sum = np.asarray(P.sum(axis=0)).flatten()
    stat    = col_sum / (col_sum.max() + 1e-12)

    T_vals, errors = [], []
    for T in T_range:
        xT      = random_walk(x0, P, T)
        xT_norm = xT / (xT.max() + 1e-12)
        errors.append(float(np.mean((xT_norm - stat) ** 2)))
        T_vals.append(T)

    # Force T=0 to infinity so the trivial solution is never selected
    if int(np.argmin(errors)) == 0:
        errors[0] = float("inf")

    return T_vals, errors


# ═══════════════════════════════════════════════════════════════════════════
# PROPAGATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def crossing_time(sol, thr: float) -> np.ndarray:
    """
    Return the first time each node's signal crosses threshold *thr*.

    Nodes that never cross the threshold receive NaN.

    Parameters
    ----------
    sol : ODE solution object with attributes .y (N×T) and .t (T,)
    thr : crossing threshold

    Returns
    -------
    t_cross : (N,) array; NaN for nodes that never cross
    """
    X       = sol.y
    t       = sol.t
    crossed = X > thr
    first_idx      = np.argmax(crossed, axis=1)
    never          = ~crossed.any(axis=1)
    t_cross        = t[first_idx].astype(float)
    t_cross[never] = np.nan
    return t_cross


def graph_geodesic_dist(Wsp: sp.spmatrix,
                        sources_mask: np.ndarray) -> np.ndarray:
    """
    Compute the minimum hop distance from any source node to every other node.

    Parameters
    ----------
    Wsp          : (N, N) sparse adjacency matrix (weights are ignored)
    sources_mask : (N,) boolean mask indicating source nodes

    Returns
    -------
    dist : (N,) minimum geodesic distance from the source set
    """
    A = Wsp.tocsr().copy()
    A.data = np.ones_like(A.data)
    sources = np.where(sources_mask)[0]
    dist = dijkstra(A, directed=False, indices=sources, unweighted=True)
    return np.min(dist, axis=0)


def invasion_metrics(sol, coords: np.ndarray, x0=None,
                     threshold: float = 0.5, q: float = 0.9,
                     tissue_mask=None) -> dict:
    """
    Compute spatial invasion front metrics from an ODE trajectory.

    Metrics
    -------
    area          : fraction of tissue nodes above threshold at each time step
    radius        : q-th quantile distance of invaded nodes from the seed centre
    radius_smooth : moving-average smoothed radius (window = min(5, T//10))
    velocita      : non-negative gradient of radius_smooth (front velocity)

    Parameters
    ----------
    sol          : ODE solution (.y shape N×T, .t shape T)
    coords       : (N, 2) spatial coordinates
    x0           : (N,) initial condition; if None uses sol.y[:, 0]
    threshold    : invasion threshold
    q            : quantile used to define the invasion radius
    tissue_mask  : (N,) boolean; if None all nodes are included

    Returns
    -------
    dict with keys: time, area, radius, radius_smooth, velocita, center, seed_center
    """
    X = sol.y
    n_nodes, T = X.shape

    if tissue_mask is None:
        tissue_mask = np.ones(n_nodes, dtype=bool)
    else:
        tissue_mask = np.asarray(tissue_mask, dtype=bool)

    if x0 is None:
        x0 = X[:, 0]
    x0 = np.asarray(x0)

    # Geometric centre of the tissue (fixed, independent of seed)
    center = coords[tissue_mask].mean(axis=0)

    # Centre of the seed from which to measure the invasion radius
    seed_mask = (x0 > threshold) & tissue_mask
    if seed_mask.sum() == 0:
        seed_mask = tissue_mask & (x0 >= x0[tissue_mask].max() * 0.8)
    seed_center    = coords[seed_mask].mean(axis=0)
    dist_from_seed = np.linalg.norm(coords - seed_center, axis=1)

    area   = np.zeros(T)
    radius = np.zeros(T)
    for t in range(T):
        invaded   = (X[:, t] > threshold) & tissue_mask
        area[t]   = invaded.sum() / tissue_mask.sum()
        if invaded.sum() > 0:
            radius[t] = np.quantile(dist_from_seed[invaded], q)

    # Enforce monotonic radius (invasion front never shrinks)
    radius = np.maximum.accumulate(radius)

    # Smooth radius with a moving average
    window = min(5, T // 10)
    if window > 1:
        kernel        = np.ones(window) / window
        radius_smooth = np.convolve(radius, kernel, mode="same")
        radius_smooth[:window // 2]  = radius[:window // 2]
        radius_smooth[-window // 2:] = radius[-window // 2:]
    else:
        radius_smooth = radius

    # Front velocity — clipped to non-negative values
    velocita = np.gradient(radius_smooth, sol.t)
    velocita = np.clip(velocita, 0.0, None)

    return dict(time=sol.t, area=area, radius=radius,
                radius_smooth=radius_smooth, velocita=velocita,
                center=center, seed_center=seed_center)
