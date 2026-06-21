"""
src/methods/pca_isomap.py

Wrappers for PCA and Isomap with a consistent interface.

Every function returns:
    X_emb : np.ndarray (n_samples, 2)  — 2D embedding
    meta  : dict                        — method-specific diagnostics

Usage:
    from src.methods.pca_isomap import run_pca, run_isomap, run_all_baselines
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import time


def run_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, dict]:
    """
    Apply PCA and return 2D embedding + diagnostics.

    Parameters
    ----------
    X            : (n, d) input array
    n_components : target dimensionality (default 2)

    Returns
    -------
    X_emb : (n, 2)
    meta  : {
        'explained_variance_ratio': array of per-component ratios,
        'total_variance_explained': float,
        'components':               (2, d) principal directions,
        'fit_time_s':               float,
    }
    """
    t0  = time.perf_counter()
    pca = PCA(n_components=n_components)
    X_emb = pca.fit_transform(X)
    elapsed = time.perf_counter() - t0

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "total_variance_explained": pca.explained_variance_ratio_.sum(),
        "components":               pca.components_,
        "fit_time_s":               elapsed,
    }
    return X_emb, meta


def run_isomap(X: np.ndarray, n_neighbors: int = 10,
               n_components: int = 2) -> tuple[np.ndarray, dict]:
    """
    Apply Isomap and return 2D embedding + diagnostics.

    Parameters
    ----------
    X            : (n, d) input array
    n_neighbors  : k for the kNN graph (typical range 5–50)
    n_components : target dimensionality

    Returns
    -------
    X_emb : (n, 2)
    meta  : {
        'n_neighbors':       int,
        'reconstruction_err': float  (residual variance),
        'fit_time_s':         float,
        'graph_n_components': int    (should be 1 for connected graph),
    }
    """
    t0 = time.perf_counter()
    iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_emb = iso.fit_transform(X)
    elapsed = time.perf_counter() - t0

    # Check graph connectivity
    from scipy.sparse.csgraph import connected_components
    n_cc, _ = connected_components(iso.dist_matrix_)

    meta = {
        "n_neighbors":        n_neighbors,
        "reconstruction_err": iso.reconstruction_error(),
        "fit_time_s":         elapsed,
        "graph_n_components": int(n_cc),
    }
    return X_emb, meta


def pca_variance_sweep(X: np.ndarray, max_components: int = 10) -> dict:
    """
    Run PCA up to max_components and return cumulative explained variance.
    Useful for a scree plot.

    Returns
    -------
    {
      'n_components': list[int],
      'cumulative_variance': list[float],
      'per_component': list[float],
    }
    """
    pca = PCA(n_components=min(max_components, X.shape[1], X.shape[0]))
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    return {
        "n_components":       list(range(1, len(evr) + 1)),
        "cumulative_variance": list(np.cumsum(evr)),
        "per_component":       list(evr),
    }


def isomap_neighbor_sweep(X: np.ndarray,
                          neighbor_values: list[int] | None = None
                          ) -> list[dict]:
    """
    Run Isomap across multiple n_neighbors values.
    Returns a list of dicts with keys: n_neighbors, X_emb, meta.

    Default sweep: [5, 10, 15, 20, 30, 50]
    """
    if neighbor_values is None:
        neighbor_values = [5, 10, 15, 20, 30, 50]

    results = []
    for k in neighbor_values:
        try:
            X_emb, meta = run_isomap(X, n_neighbors=k)
            results.append({"n_neighbors": k, "X_emb": X_emb, "meta": meta})
            status = "ok" if meta["graph_n_components"] == 1 else f"DISCONNECTED ({meta['graph_n_components']} components)"
            print(f"  k={k:>3}  recon_err={meta['reconstruction_err']:.4f}  "
                  f"time={meta['fit_time_s']:.2f}s  graph={status}")
        except Exception as e:
            print(f"  k={k:>3}  FAILED: {e}")
    return results


def run_all_baselines(datasets: dict,
                      isomap_k: int = 10) -> dict:
    """
    Run PCA + Isomap on every dataset in the dict.

    Parameters
    ----------
    datasets : dict from load_all_datasets()
    isomap_k : n_neighbors for Isomap

    Returns
    -------
    {
      dataset_name: {
        'pca':    (X_emb, meta),
        'isomap': (X_emb, meta),
      },
      ...
    }
    """
    results = {}
    for name, (X, t) in datasets.items():
        print(f"\n{name}")
        pca_emb, pca_meta     = run_pca(X)
        iso_emb, iso_meta     = run_isomap(X, n_neighbors=isomap_k)
        results[name] = {
            "pca":    (pca_emb, pca_meta),
            "isomap": (iso_emb, iso_meta),
        }
        print(f"  PCA    variance explained: {pca_meta['total_variance_explained']:.3f}  "
              f"time={pca_meta['fit_time_s']:.3f}s")
        print(f"  Isomap recon error:        {iso_meta['reconstruction_err']:.4f}  "
              f"time={iso_meta['fit_time_s']:.2f}s  "
              f"graph_cc={iso_meta['graph_n_components']}")
    return results