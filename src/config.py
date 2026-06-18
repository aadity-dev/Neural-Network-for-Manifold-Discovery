"""
Global project configuration.
Import anywhere with:  from src.config import CFG
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CFG = {
    "root":         ROOT,
    "data_dir":     ROOT / "data",
    "figures_dir":  ROOT / "results" / "figures",
    "metrics_dir":  ROOT / "results" / "metrics",
    "models_dir":   ROOT / "results" / "models",
    "random_seed":  42,
    "n_samples":    2000,
    "noise_levels": [0.0, 0.1, 0.3],
    "figsize":      (8, 6),
    "dpi":          300,
    "device":       "cpu",   # updated at runtime in notebook 05
}
