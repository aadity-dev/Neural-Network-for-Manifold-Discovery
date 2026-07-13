"""
src/evaluation/metrics.py

Unified evaluation metrics for manifold learning methods.

Functions
---------
compute_trustworthiness(X, X_emb, k)   — neighbourhood preservation (high = good)
compute_continuity(X, X_emb, k)        — inverse of trustworthiness
compute_spearman(X, X_emb)             — pairwise distance rank correlation
compute_reconstruction_mse(X, X_hat)   — autoencoder reconstruction error
compute_all_metrics(X, X_emb, k, X_hat) — run all metrics at once
build_comparison_table(datasets, all_results) — full methods x datasets DataFrame
"""

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


def compute_trustworthiness(X: np.ndarray,
                             X_emb: np.ndarray,
                             k: int = 10) -> float:
    """
    Trustworthiness: fraction of k nearest neighbours in the embedding
    that were also nearest neighbours in the original space.

    Score = 1.0 → perfect neighbourhood preservation
    Score < 0.9 → embedding is introducing false neighbours
    """
    if np.std(X_emb) < 1e-6:
        return float("nan")
    return float(sklearn_trustworthiness(X, X_emb, n_neighbors=k))


def compute_continuity(X: np.ndarray,
                        X_emb: np.ndarray,
                        k: int = 10) -> float:
    """
    Continuity: fraction of k nearest neighbours in the original space
    that are also nearest neighbours in the embedding.

    Complement to trustworthiness:
      - Trustworthiness catches false neighbours (tears)
      - Continuity catches missing neighbours (compressions)

    Score = 1.0 → no neighbours are lost in the embedding
    """
    if np.std(X_emb) < 1e-6:
        return float("nan")

    n = X.shape[0]

    # k-NN in original space
    nbrs_orig = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind_orig = nbrs_orig.kneighbors(X)
    ind_orig = ind_orig[:, 1:]  # exclude self

    # k-NN in embedding space
    nbrs_emb = NearestNeighbors(n_neighbors=k + 1).fit(X_emb)
    _, ind_emb = nbrs_emb.kneighbors(X_emb)
    ind_emb = ind_emb[:, 1:]

    # Rank of each original neighbour in the embedding ordering
    cont = 0.0
    for i in range(n):
        orig_set = set(ind_orig[i])
        for rank_emb, j in enumerate(ind_emb[i]):
            if j not in orig_set:
                # Rank in original space
                orig_list = list(ind_orig[i])
                try:
                    r = orig_list.index(j) + 1
                except ValueError:
                    r = n
                cont += r - k

    cont = 1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * cont
    return float(np.clip(cont, 0.0, 1.0))


def compute_spearman(X: np.ndarray,
                      X_emb: np.ndarray,
                      subsample: int = 500) -> float:
    """
    Spearman rank correlation between pairwise distances in the
    original space and the embedding.

    Score = 1.0 → distances perfectly rank-preserved
    Score ~ 0   → no distance relationship preserved

    Uses subsampling for speed (O(n²) otherwise).
    """
    if np.std(X_emb) < 1e-6:
        return float("nan")

    n = min(subsample, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=n, replace=False)

    X_sub   = X[idx]
    emb_sub = X_emb[idx]

    # Pairwise distances — upper triangle only
    from scipy.spatial.distance import pdist
    d_orig = pdist(X_sub, metric="euclidean")
    d_emb  = pdist(emb_sub, metric="euclidean")

    corr, _ = spearmanr(d_orig, d_emb)
    return float(corr)


def compute_reconstruction_mse(X: np.ndarray,
                                 X_hat: np.ndarray) -> float:
    """
    Mean squared error between original and reconstructed points.
    Only meaningful for the autoencoder (other methods don't reconstruct).
    """
    return float(np.mean((X - X_hat) ** 2))


def compute_all_metrics(X:     np.ndarray,
                         X_emb: np.ndarray,
                         k:     int = 10,
                         X_hat: np.ndarray = None) -> dict:
    """
    Run all metrics for one (method, dataset) pair.

    Returns
    -------
    {
      'trustworthiness': float,
      'continuity':      float,
      'spearman':        float,
      'recon_mse':       float | None,
    }
    """
    return {
        "trustworthiness": compute_trustworthiness(X, X_emb, k),
        "continuity":      compute_continuity(X, X_emb, k),
        "spearman":        compute_spearman(X, X_emb),
        "recon_mse":       compute_reconstruction_mse(X, X_hat) if X_hat is not None else None,
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a tidy DataFrame from pre-computed metric dicts.

    Parameters
    ----------
    results : {
        dataset_name: {
            method_name: {
                'trustworthiness': float,
                'continuity':      float,
                'spearman':        float,
                'recon_mse':       float | None,
                'fit_time_s':      float | None,
            }
        }
    }

    Returns
    -------
    DataFrame with MultiIndex (Dataset, Method) and metric columns
    """
    rows = []
    for dataset, methods in results.items():
        for method, m in methods.items():
            rows.append({
                "Dataset":          dataset,
                "Method":           method,
                "Trustworthiness":  round(m.get("trustworthiness", float("nan")), 4),
                "Continuity":       round(m.get("continuity",      float("nan")), 4),
                "Spearman r":       round(m.get("spearman",        float("nan")), 4),
                "Recon MSE":        round(m["recon_mse"], 4) if m.get("recon_mse") else None,
                "Fit time (s)":     round(m["fit_time_s"], 3) if m.get("fit_time_s") else None,
            })

    df = pd.DataFrame(rows).set_index(["Dataset", "Method"])
    return df
