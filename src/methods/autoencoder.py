"""
src/methods/autoencoder.py

PyTorch Autoencoder for manifold learning.

Architecture
------------
Encoder : 3 → 64 → 32 → 16 → latent_dim   (ReLU activations)
Decoder : latent_dim → 16 → 32 → 64 → 3   (ReLU hidden, linear output)

Classes
-------
Encoder(input_dim, hidden_dims, latent_dim)
Decoder(latent_dim, hidden_dims, output_dim)
Autoencoder(input_dim, hidden_dims, latent_dim)

Functions
---------
train_autoencoder(X, latent_dim, epochs, lr, batch_size, device, seed)
encode_dataset(model, X, device)
run_all_autoencoders(datasets, ...)
"""

import os
# Limit multithreading to 1 to prevent OpenMP collisions causing segmentation faults on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim: int = 3,
                 hidden_dims: list[int] = None,
                 latent_dim: int = 2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))   # no activation on bottleneck

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 2,
                 hidden_dims: list[int] = None,
                 output_dim: int = 3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 64]   # mirror of encoder

        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))   # linear output for regression

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    """
    Full autoencoder: Encoder + Decoder.

    Parameters
    ----------
    input_dim   : ambient dimension of input (3 for our datasets)
    hidden_dims : list of hidden layer sizes (encoder order)
    latent_dim  : bottleneck size (2 for 2D embedding)

    Usage
    -----
    model = Autoencoder(input_dim=3, latent_dim=2)
    z     = model.encode(x)     # (n, 2)
    x_hat = model(x)            # (n, 3) reconstructed
    """
    def __init__(self, input_dim: int = 3,
                 hidden_dims: list[int] = None,
                 latent_dim: int = 2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Note: MPS (Metal Performance Shaders) backend is disabled by default here because
    # PyTorch 2.2.0 has known instability (segmentation faults) on macOS with some MPS drivers,
    # and CPU is significantly faster for this tiny model (5.7k parameters) due to GPU dispatch overhead.
    # To force MPS, you can pass device="mps" explicitly.
    return "cpu"


def train_autoencoder(
    X:           np.ndarray,
    latent_dim:  int   = 2,
    hidden_dims: list  = None,
    epochs:      int   = 200,
    lr:          float = 1e-3,
    batch_size:  int   = 256,
    device:      str   = None,
    seed:        int   = 42,
    verbose:     bool  = True,
    log_every:   int   = 20,
) -> tuple:
    """
    Train an Autoencoder on array X and return model + training history.

    Parameters
    ----------
    X           : (n, d) numpy array — input point cloud
    latent_dim  : bottleneck size (2 for 2D visualisation)
    hidden_dims : encoder hidden layer sizes (default [64, 32, 16])
    epochs      : number of full passes through the data
    lr          : Adam learning rate
    batch_size  : mini-batch size
    device      : 'cpu' | 'cuda' | 'mps' | None (auto-detect)
    seed        : random seed for reproducibility
    verbose     : print loss every log_every epochs
    log_every   : epoch interval for progress printing

    Returns
    -------
    model   : trained Autoencoder (on CPU)
    history : {'train_loss': [float per epoch], 'fit_time_s': float}
    """
    if hidden_dims is None:
        hidden_dims = [64, 32, 16]
    if device is None:
        device = get_device()

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalise input to zero mean, unit std per feature
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # DataLoader
    tensor  = torch.FloatTensor(X_norm).to(device)
    dataset = TensorDataset(tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model + optimiser
    model     = Autoencoder(input_dim=X.shape[1],
                             hidden_dims=hidden_dims,
                             latent_dim=latent_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()

    if verbose:
        print(f"  Device     : {device}")
        print(f"  Parameters : {model.count_params():,}")
        print(f"  Arch       : 3 → {hidden_dims} → {latent_dim} → "
              f"{list(reversed(hidden_dims))} → 3")
        print(f"  Training for {epochs} epochs ...")

    history   = {"train_loss": [], "fit_time_s": 0.0}
    t0        = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(batch)

        epoch_loss /= len(X)
        history["train_loss"].append(epoch_loss)
        scheduler.step()

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4}/{epochs}  loss={epoch_loss:.6f}")

    history["fit_time_s"] = time.perf_counter() - t0
    if verbose:
        print(f"  Done — {history['fit_time_s']:.1f}s  "
              f"final loss={history['train_loss'][-1]:.6f}")

    # Store normalisation stats on model for inverse transform later
    model.X_mean = X_mean
    model.X_std  = X_std
    model.cpu()
    return model, history


def encode_dataset(model: Autoencoder,
                   X:     np.ndarray,
                   device: str = "cpu") -> np.ndarray:
    """
    Pass X through the trained encoder and return 2D latent coordinates.

    Parameters
    ----------
    model  : trained Autoencoder (output of train_autoencoder)
    X      : (n, d) numpy array
    device : inference device

    Returns
    -------
    Z : (n, latent_dim) numpy array
    """
    model.eval().to(device)
    X_norm = (X - model.X_mean) / model.X_std
    tensor = torch.FloatTensor(X_norm).to(device)
    with torch.no_grad():
        Z = model.encode(tensor).cpu().numpy()
    return Z


def run_all_autoencoders(
    datasets:    dict,
    latent_dim:  int   = 2,
    hidden_dims: list  = None,
    epochs:      int   = 200,
    lr:          float = 1e-3,
    batch_size:  int   = 256,
    seed:        int   = 42,
    save_dir           = None,
    device:      str   = None,
) -> dict:
    """
    Train one autoencoder per dataset and return embeddings.

    Returns
    -------
    {
      dataset_name: {
        'X_emb':   np.ndarray (n, 2),
        'model':   Autoencoder,
        'history': dict,
      },
      ...
    }
    """
    device  = device or get_device()
    results = {}

    for name, (X, _) in datasets.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        model, history = train_autoencoder(
            X, latent_dim=latent_dim, hidden_dims=hidden_dims,
            epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, seed=seed,
        )
        X_emb = encode_dataset(model, X)

        # Optionally save checkpoint
        if save_dir is not None:
            import pathlib
            save_dir = pathlib.Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname    = name.lower().replace(" ", "_") + "_ae.pt"
            torch.save({
                "model_state": model.state_dict(),
                "X_mean":      model.X_mean,
                "X_std":       model.X_std,
                "history":     history,
                "latent_dim":  latent_dim,
                "hidden_dims": hidden_dims or [64, 32, 16],
            }, save_dir / fname)
            print(f"  Checkpoint saved → {fname}")

        results[name] = {"X_emb": X_emb, "model": model, "history": history}

    return results