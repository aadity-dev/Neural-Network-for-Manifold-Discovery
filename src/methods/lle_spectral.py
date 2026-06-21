"""
src/methods/lle_spectral.py

Wrappers for LLE variants and Spectral Embedding with a consistent interface.

Every function returns:
    X_emb : np.ndarray (n_samples, 2)
    meta  : dict  — method-specific diagnostics

Methods
-------
run_lle(X, method, n_neighbors)   — standard / modified / hessian / ltsa
run_spectral(X, n_neighbors)      — Laplacian Eigenmaps
run_all_lle(datasets)             — convenience: all methods x all datasets
lle_neighbor_sweep(X)             — stability across k values
"""


import time
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding


VALID_METHODS = ("standard", "modified", "hessian", "ltsa")


def run_lle(X: np.ndarray,
            method: str = "standard",
            n_neighbors: int = 10,
            n_components: int = 2,
            random_state: int = 42) -> tuple[np.ndarray, dict]:
    """
    Apply a LocallyLinearEmbedding variant.

    Parameters
    ----------
    X            : (n, d) input array
    method       : one of 'standard' | 'modified' | 'hessian' | 'ltsa'
    n_neighbors  : k for the kNN graph
    n_components : target dimensionality

    Hessian LLE constraint:
        n_neighbors >= n_components * (n_components + 3) / 2 + 1
        For n_components=2 this means n_neighbors >= 6

    Returns
    -------
    X_emb : (n, 2)
    meta  : {
        'method', 'n_neighbors', 'reconstruction_err',
        'fit_time_s', 'error_msg'
    }
    """
    assert method in VALID_METHODS, f"method must be one of {VALID_METHODS}"

    # Hessian LLE hard constraint
    min_k = int(n_components * (n_components + 3) / 2) + 1
    if method == "hessian" and n_neighbors < min_k:
        n_neighbors = min_k
        print(f"    [hessian] n_neighbors raised to {min_k} (minimum for n_components={n_components})")

    meta = {"method": method, "n_neighbors": n_neighbors,
            "reconstruction_err": None, "fit_time_s": None, "error_msg": None}

    try:
        t0  = time.perf_counter()
        lle = LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=n_components,
            method=method,
            random_state=random_state,
        )
        X_emb = lle.fit_transform(X)
        meta["fit_time_s"]         = time.perf_counter() - t0
        meta["reconstruction_err"] = float(lle.reconstruction_error_)
    except Exception as e:
        meta["error_msg"] = str(e)
        X_emb = np.zeros((X.shape[0], n_components))
        print(f"    [{method}] FAILED: {e}")

    return X_emb, meta


def run_spectral(X: np.ndarray,
                 n_neighbors: int = 10,
                 n_components: int = 2,
                 random_state: int = 42) -> tuple[np.ndarray, dict]:
    """
    Spectral Embedding (Laplacian Eigenmaps).

    Constructs a similarity graph from kNN, then embeds using the
    eigenvectors of the graph Laplacian.

    Returns
    -------
    X_emb : (n, 2)
    meta  : {'n_neighbors', 'fit_time_s'}
    """
    t0  = time.perf_counter()
    se  = SpectralEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    X_emb = se.fit_transform(X)
    meta  = {
        "n_neighbors": n_neighbors,
        "fit_time_s":  time.perf_counter() - t0,
    }
    return X_emb, meta


def lle_neighbor_sweep(X: np.ndarray,
                       method: str = "standard",
                       neighbor_values: list[int] | None = None,
                       n_components: int = 2) -> list[dict]:
    """
    Run one LLE variant across multiple n_neighbors values.

    Returns list of dicts: {n_neighbors, X_emb, meta}
    """
    if neighbor_values is None:
        neighbor_values = [5, 10, 15, 20, 30, 50]

    results = []
    for k in neighbor_values:
        X_emb, meta = run_lle(X, method=method, n_neighbors=k,
                               n_components=n_components)
        err = meta["reconstruction_err"]
        t   = meta["fit_time_s"]
        ok  = meta["error_msg"] is None
        err_str = f"{err:.4f}" if err is not None else "N/A"
        t_str   = f"{t:.2f}" if t is not None else "0.00"
        status  = "ok" if ok else "FAILED"
        print(f"  k={k:>3}  recon_err={err_str:>8}  time={t_str}s  {status}")
        results.append({"n_neighbors": k, "X_emb": X_emb, "meta": meta})
    return results


def run_all_lle(datasets: dict,
                n_neighbors: int = 10) -> dict:
    """
    Run standard LLE, Modified LLE, Hessian LLE, LTSA, and Spectral Embedding
    on every dataset.

    Returns
    -------
    {
      dataset_name: {
        'lle_standard':  (X_emb, meta),
        'lle_modified':  (X_emb, meta),
        'lle_hessian':   (X_emb, meta),
        'lle_ltsa':      (X_emb, meta),
        'spectral':      (X_emb, meta),
      },
      ...
    }
    """
    results = {}
    for name, (X, t) in datasets.items():
        print(f"\n{name}")
        entry = {}
        for method in VALID_METHODS:
            label = f"lle_{method}"
            print(f"  {label}")
            X_emb, meta = run_lle(X, method=method, n_neighbors=n_neighbors)
            entry[label] = (X_emb, meta)
            if meta["error_msg"] is None:
                print(f"    recon_err={meta['reconstruction_err']:.4f}  "
                      f"time={meta['fit_time_s']:.2f}s")

        print("  spectral")
        X_emb, meta = run_spectral(X, n_neighbors=n_neighbors)
        entry["spectral"] = (X_emb, meta)
        print(f"    time={meta['fit_time_s']:.2f}s")

        results[name] = entry
    return results