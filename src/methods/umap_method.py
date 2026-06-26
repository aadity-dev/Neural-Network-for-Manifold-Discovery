"""
src/methods/umap_method.py

UMAP wrapper with consistent (X_emb, meta) interface.

Functions
---------
run_umap(X, n_neighbors, min_dist, metric, n_components)
umap_param_sweep(X, neighbor_values, dist_values)   — 2D grid sweep
umap_metric_compare(X)                               — euclidean vs cosine
umap_speed_benchmark(datasets)                       — vs Isomap & LLE
run_all_umap(datasets)                               — all datasets, default params
"""

import time
import numpy as np
import umap


def run_umap(X: np.ndarray,
             n_neighbors: int  = 15,
             min_dist:    float = 0.1,
             metric:      str   = "euclidean",
             n_components: int  = 2,
             random_state: int  = 42) -> tuple[np.ndarray, dict]:
    """
    Apply UMAP and return embedding + diagnostics.

    Parameters
    ----------
    X            : (n, d) input array
    n_neighbors  : controls local vs global structure balance
                   small  → fine local detail
                   large  → broader global structure
    min_dist     : minimum distance between points in the embedding
                   0.0   → tight clumps
                   1.0   → evenly spread points
    metric       : distance metric ('euclidean', 'cosine', 'manhattan', ...)
    n_components : output dimensionality (default 2)
    random_state : seed for reproducibility — always set this!

    Returns
    -------
    X_emb : (n, 2)
    meta  : {
        'n_neighbors', 'min_dist', 'metric',
        'fit_time_s', 'error_msg'
    }
    """
    meta = {
        "n_neighbors": n_neighbors,
        "min_dist":    min_dist,
        "metric":      metric,
        "fit_time_s":  None,
        "error_msg":   None,
    }
    try:
        t0      = time.perf_counter()
        reducer = umap.UMAP(
            n_neighbors  = n_neighbors,
            min_dist     = min_dist,
            metric       = metric,
            n_components = n_components,
            random_state = random_state,
        )
        X_emb = reducer.fit_transform(X)
        meta["fit_time_s"] = time.perf_counter() - t0
    except Exception as e:
        meta["error_msg"] = str(e)
        X_emb = np.zeros((X.shape[0], n_components))
        print(f"  UMAP FAILED: {e}")

    return X_emb, meta


def umap_param_sweep(X: np.ndarray,
                     t: np.ndarray,
                     neighbor_values: list[int]   = None,
                     dist_values:     list[float] = None,
                     random_state: int = 42) -> dict:
    """
    Full 2D grid sweep over n_neighbors × min_dist.

    Returns
    -------
    {
      (n_neighbors, min_dist): {'X_emb': ..., 'meta': ...},
      ...
    }
    """
    if neighbor_values is None:
        neighbor_values = [5, 15, 30, 50]
    if dist_values is None:
        dist_values = [0.0, 0.1, 0.5, 1.0]

    results = {}
    total   = len(neighbor_values) * len(dist_values)
    done    = 0

    for k in neighbor_values:
        for d in dist_values:
            X_emb, meta = run_umap(X, n_neighbors=k, min_dist=d,
                                   random_state=random_state)
            results[(k, d)] = {"X_emb": X_emb, "meta": meta}
            done += 1
            t_s = meta["fit_time_s"] or 0
            print(f"  [{done:>2}/{total}] k={k:>3}  min_dist={d}  "
                  f"time={t_s:.2f}s  "
                  f"{'ok' if not meta['error_msg'] else 'FAILED'}")

    return results


def umap_metric_compare(X: np.ndarray,
                        metrics: list[str] = None,
                        n_neighbors: int   = 15,
                        min_dist:    float = 0.1,
                        random_state: int  = 42) -> dict:
    """
    Compare UMAP embeddings under different distance metrics.

    Returns { metric_name: (X_emb, meta) }
    """
    if metrics is None:
        metrics = ["euclidean", "cosine", "manhattan", "chebyshev"]

    results = {}
    for m in metrics:
        print(f"  metric={m}")
        X_emb, meta = run_umap(X, n_neighbors=n_neighbors,
                                min_dist=min_dist, metric=m,
                                random_state=random_state)
        results[m] = (X_emb, meta)
        if meta["fit_time_s"]:
            print(f"    time={meta['fit_time_s']:.2f}s")
    return results


def umap_speed_benchmark(datasets: dict,
                         random_state: int = 42) -> list[dict]:
    """
    Compare UMAP fit time vs Isomap and LLE on every dataset.

    Returns list of dicts suitable for a DataFrame.
    """
    from sklearn.manifold import Isomap, LocallyLinearEmbedding

    rows = []
    for name, (X, _) in datasets.items():
        print(f"\n  {name}  (n={X.shape[0]})")
        row = {"Dataset": name, "n_samples": X.shape[0]}

        # UMAP
        _, meta  = run_umap(X, random_state=random_state)
        row["UMAP (s)"] = round(meta["fit_time_s"] or 0, 3)
        print(f"    UMAP    {row['UMAP (s)']:.3f}s")

        # Isomap
        t0 = time.perf_counter()
        Isomap(n_neighbors=15).fit_transform(X)
        row["Isomap (s)"] = round(time.perf_counter() - t0, 3)
        print(f"    Isomap  {row['Isomap (s)']:.3f}s")

        # LLE
        t0 = time.perf_counter()
        LocallyLinearEmbedding(n_neighbors=15).fit_transform(X)
        row["LLE (s)"] = round(time.perf_counter() - t0, 3)
        print(f"    LLE     {row['LLE (s)']:.3f}s")

        rows.append(row)
    return rows


def run_all_umap(datasets: dict,
                 n_neighbors: int  = 15,
                 min_dist:    float = 0.1,
                 random_state: int  = 42) -> dict:
    """
    Run UMAP on every dataset with default params.

    Returns { dataset_name: (X_emb, meta) }
    """
    results = {}
    for name, (X, _) in datasets.items():
        print(f"  {name}")
        X_emb, meta = run_umap(X, n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                random_state=random_state)
        results[name] = (X_emb, meta)
        if meta["fit_time_s"]:
            print(f"    time={meta['fit_time_s']:.2f}s")
    return results