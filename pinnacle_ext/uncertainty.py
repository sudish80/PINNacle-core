"""pinnacle_ext.uncertainty -- predictive uncertainty for PINNs.

Two complementary mechanisms sharing one API:

  * MC-DROPOUT  : a single ``DropoutMLP`` model is evaluated ``n_samples``
                  times with dropout running (Gal & Ghahramani).  Cheap,
                  one-model uncertainty; quality correlates with fit quality.
  * DEEP ENSEMBLE : k independently-seeded nets (any architecture) average
                  their predictions; std -> epistemic (weight diversity)
                  uncertainty.  The gold standard, kx the cost.

Both return (mean, std) on a regular evaluation grid plus the grid itself.
The primitive ``predict_std`` works on ANY net that returns tensors.
"""

import numpy as np
import torch

import pinn_solver as P
from pinnacle_ext import arch

DEVICE = P.DEVICE


def eval_grid(prob, grid_pts=31):
    """Regular space x time tensor grid -> (X, T, z).

    ``z`` carries requires_grad=True leaf leaves (same contract as
    ``pinn_solver.sample_points``) so derivative engines can consume it.
    """
    Xs = np.meshgrid(*[np.linspace(prob.a[k], prob.b[k], grid_pts)
                       for k in range(prob.d)],
                     np.linspace(prob.t0, max(prob.t1, prob.t0 + 1e-12),
                                 grid_pts))
    Xn = np.stack([m.ravel() for m in Xs[:prob.d]], axis=1)
    Tn = Xs[prob.d].ravel().reshape(-1, 1)
    z = torch.tensor(np.hstack([Xn, Tn]), dtype=torch.get_default_dtype(),
                     device=DEVICE).requires_grad_(True)
    return Xn, Tn, z


def predict_std(net, prob, grid_pts=31, n_samples=20, hard=False):
    """MC-dropout/ensemble-style stochastic predictions -> (mean, std, X, T).

    ``net`` is expected to be a DropoutMLP with ``mc_enable()`` (a plain net
    yields a constant width-0 band -- which is itself informative).
    """
    from pinnacle_ext.engine import is_core_problem
    Xn, Tn, z = eval_grid(prob, grid_pts)
    U = []
    for _ in range(n_samples):
        with torch.no_grad():
            U.append(net(z).detach().cpu().numpy())
    U = np.stack(U)                       # (n_samples, N, n_out)
    mean = U.mean(axis=0)
    std = U.std(axis=0)
    return mean, std, Xn, Tn


def mc_dropout(net, prob, grid_pts=31, n_samples=30, hard=False, drop=None):
    """MC-dropout uncertainty from a DropoutMLP (enables dropout internally)."""
    if isinstance(net, arch.DropoutMLP):
        net.mc_enable(p=drop)
    else:
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
    try:
        return predict_std(net, prob, grid_pts=grid_pts, n_samples=n_samples,
                           hard=hard)
    finally:
        net.eval()


def deep_ensemble(nets, prob, grid_pts=31, hard=False):
    """Mean + std over k independently-trained nets."""
    Xn, Tn, z = eval_grid(prob, grid_pts)
    U = []
    for net in nets:
        with torch.no_grad():
            U.append(net(z).detach().cpu().numpy())
    U = np.stack(U)
    return U.mean(axis=0), U.std(axis=0), Xn, Tn


def calibration_curve(mean, std, truth, n_bins=8):
    """(Mean,PINN-face) distance calibration vs truth.

    Bins the predictions by uncertainty std; returns dict with per-bin
    (count, mean_std, rmse_to_truth) so we can SEE whether std tracks error.
    """
    err = np.abs(mean - truth)
    s = std[..., 0] if std.ndim > err.ndim else std
    e = err[..., 0] if err.ndim > s.ndim else err
    idx = np.argsort(s.ravel())
    n = idx.size
    out = []
    lim = np.linspace(0, n, n_bins + 1).astype(int)
    for i in range(n_bins):
        sel = idx[lim[i]: max(lim[i + 1], lim[i] + 1)]
        out.append({"count": int(sel.size),
                    "mean_std": float(s.ravel()[sel].mean()),
                    "rmse": float(np.sqrt(np.mean(e.ravel()[sel] ** 2)))})
    return out