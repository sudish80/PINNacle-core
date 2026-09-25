#!/usr/bin/env python3
"""
Physics-Informed Neural Networks (PINNs): Production Implementation
====================================================================

Solves the 1D viscous Burgers' equation

        u_t + u u_x = nu * u_xx,
        x in [-1, 1],  t in [0, 1],
        u(x, 0)   = -sin(pi x),
        u(-1, t)  = u(1, t) = 0,

using a physics-informed neural network with two-stage optimization
(Adam -> L-BFGS with Strong-Wolfe line search).

Reference solution
------------------
The exact solution follows from the Cole-Hopf transformation:

    u = -2*nu * (d/dx) ln(phi),   phi_t = nu*phi_xx,

with  phi(x,0) = exp( (1 - cos(pi x)) / (2*nu*pi) ).  Working with the
*reciprocal*  w = 1/phi  (which keeps all series coefficients O(1) --
critical for floating-point conditioning), expanding on the Neumann basis
cos(n*pi*x) over [-1, 1] and using the Bessel integral identity

    int_0^pi e^{a cos(theta)} cos(n theta) d(theta) = pi * I_n(a)

gives the closed form used below (evaluated in extended precision via
mpmath, with a disk cache so repeated runs are fast):

    w(x, t) = b_0 + 2 * sum_{n>=1} b_n e^{-nu n^2 pi^2 t} cos(n pi x),
    b_n     = e^{-A} I_n(A),   A = 1 / (2 nu pi),
    u(x, t) = -4 nu pi * sum_n n b_n e^{-nu n^2 pi^2 t} sin(n pi x) / w.

Author: PINN Research Group
"""

import argparse
import hashlib
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc

import mpmath as mp

# --------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------- #
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pinn_reference_cache")


# --------------------------------------------------------------------- #
# Exact reference solution (Cole-Hopf, mpmath, disk-cached)
# --------------------------------------------------------------------- #
def _cache_key(x_grid, t_grid, nu, n_terms, dps):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x_grid).tobytes())
    h.update(np.ascontiguousarray(t_grid).tobytes())
    h.update(repr(float(nu)).encode())
    h.update(bytes([n_terms & 0xFF, dps & 0xFF]))
    return h.hexdigest()[:24]


def _bessel_coeffs(nu, n_terms, dps):
    """Return (b0, bn) series coefficients for the reciprocal w-series."""
    with mp.workdps(dps):
        A = mp.mpf(1.0) / (2.0 * mp.mpf(nu) * mp.pi)
        b0 = mp.e ** (-A) * mp.besseli(0, A)
        bn = [mp.e ** (-A) * mp.besseli(k, A) for k in range(1, n_terms + 1)]
        nu_mp = mp.mpf(nu)
        ck = [nu_mp * mp.pi ** 2 * (k * k) for k in range(1, n_terms + 1)]
    return b0, bn, ck


def _series_u(x, t, nu, b0, bn, ck, dps):
    """Evaluate u(x,t) for ONE scalar (x, t) at extended precision."""
    try:
        return _series_u_at_dps(x, t, nu, b0, bn, ck, dps)
    except (ZeroDivisionError, ValueError, OverflowError):
        # Cancellation at degenerate points (e.g. x=+/-1, t=0) can need
        # ~1/(2 nu pi) + O(10) significant digits. Retry with headroom.
        return _series_u_at_dps(x, t, nu, b0, bn, ck, dps + 60)


def _series_u_at_dps(x, t, nu, b0, bn, ck, dps):
    with mp.workdps(dps):
        xt, tt = mp.mpf(x), mp.mpf(t)
        sw = mp.mpf(0)
        swx = mp.mpf(0)
        for k in range(1, len(bn) + 1):
            e = mp.exp(-ck[k - 1] * tt)
            w = xt * k
            b = bn[k - 1]
            sw += b * e * mp.cospi(w)
            swx += k * b * e * mp.sinpi(w)
        wx = -2 * mp.pi * swx
        return 2 * mp.mpf(nu) * wx / (b0 + 2 * sw)


def exact_burgers_grid(x_grid, t_grid, nu, n_terms=180, dps=60, use_cache=True):
    """Return u_exact with shape (len(t_grid), len(x_grid))."""
    x_grid = np.atleast_1d(np.asarray(x_grid, dtype=np.float64))
    t_grid = np.atleast_1d(np.asarray(t_grid, dtype=np.float64))
    os.makedirs(CACHE_DIR, exist_ok=True)

    if use_cache:
        key = _cache_key(x_grid, t_grid, nu, n_terms, dps)
        path = os.path.join(CACHE_DIR, f"burgers_ref_{key}.npy")
        if os.path.exists(path):
            return np.load(path)

    b0, bn, ck = _bessel_coeffs(nu, n_terms, dps)
    print(f"[exact] computing Cole-Hopf reference: "
          f"{len(t_grid) * len(x_grid)} points x {n_terms} terms "
          f"(mpmath dps={dps}) -- cached for future runs ...")
    out = np.empty((len(t_grid), len(x_grid)))
    t0 = time.time()
    for i, t in enumerate(t_grid):
        for j, x in enumerate(x_grid):
            out[i, j] = float(_series_u(x, t, nu, b0, bn, ck, dps))
    print(f"[exact] done in {time.time() - t0:.1f}s.")

    if use_cache:
        np.save(path, out)
    return out


# --------------------------------------------------------------------- #
# Network architecture
# --------------------------------------------------------------------- #
def _activation_factory(name, w0=30.0):
    def act(z):
        if name == "tanh":
            return torch.tanh(z)
        if name == "gelu":
            return torch.nn.functional.gelu(z)
        if name == "siren":
            return torch.sin(w0 * z)
        if name == "relu":
            return torch.relu(z)
        raise ValueError(f"unknown activation: {name}")
    return act


class PINN(nn.Module):
    """Fully-connected MLP with standard PINN initialisation.

    Glorot (Xavier) uniform init is the safe default for smooth,
    derivative-sensitive activations (tanh / GELU / sine).  For 'siren'
    we apply the Sitzmann et al. init (first layer U(-1/w0, 1/w0),
    inner layers scaled by 6/fan_in) which preserves weight statistics
    through the nonlinearity.
    """

    def __init__(self, layers=(2, 40, 40, 40, 40, 1),
                 activation="tanh", w0=30.0):
        super().__init__()
        self.activation = activation
        self.w0 = w0
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            fc = nn.Linear(layers[i], layers[i + 1])
            if activation == "siren":
                if i == 0:
                    nn.init.uniform_(fc.weight, -1.0 / w0, 1.0 / w0)
                else:
                    with torch.no_grad():
                        fc.weight *= np.sqrt(6.0 / layers[i])
            else:
                nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fcs.append(fc)

    def forward(self, x, t):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            t = t.unsqueeze(-1)
        z = torch.cat([x, t], dim=-1)
        act = _activation_factory(self.activation, self.w0)
        for fc in self.fcs[:-1]:
            z = act(fc(z))
        return self.fcs[-1](z)


class HardConstrainedPINN(nn.Module):
    """Wraps a PINN and enforces IC + Dirichlet BC exactly.

    Ansatz (see report section 4):

        u_hat(x, t) = g(x, t) + A(x, t) * NN(x, t)
                    = -sin(pi x) + (1 - x^2) * t * NN(x, t),

    with A(x) = (1 - x^2) == 0 on the boundaries x = +/-1 and an
    explicit factor 't' to satisfy the initial condition at t = 0.
    The network then only has to satisfy the PDE residual, so the
    optimizer sees a single objective (L_pde).
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t):
        u0 = -torch.sin(np.pi * x)
        distance = 1.0 - x * x          # zero on x = -1 and x = 1
        return u0 + distance * t * self.net(x, t)


# --------------------------------------------------------------------- #
# Physics residual via automatic differentiation
# --------------------------------------------------------------------- #
def compute_pde_residual(net, x, t, nu):
    """Residual of u_t + u u_x - nu u_xx = 0 (fish-tail chain rule)."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = net(x, t)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True)[0]

    return u_t + u * u_x - nu * u_xx, u


def compute_total_loss(net, x_r, t_r, x_bc, t_bc, x_ic, nu, weights,
                       hard_constraint=False):
    """Composite weighted loss (section 1 of report)."""
    r, _ = compute_pde_residual(net, x_r, t_r, nu)
    loss_pde = torch.mean(r ** 2)

    if hard_constraint:
        # IC and BC are enforced by construction -> no penalty needed.
        loss_bc = torch.zeros((), dtype=x_r.dtype)
        loss_ic = torch.zeros((), dtype=x_r.dtype)
        loss_data = torch.zeros((), dtype=x_r.dtype)
    else:
        u_bc = net(x_bc, t_bc)
        loss_bc = torch.mean((u_bc - 0.0) ** 2)           # g(x,t) = 0
        u_ic = net(x_ic, torch.zeros_like(x_ic))
        loss_ic = torch.mean((u_ic - (-torch.sin(np.pi * x_ic))) ** 2)
        loss_data = torch.zeros((), dtype=x_r.dtype)

    total = (weights["pde"] * loss_pde
             + weights["bc"] * loss_bc
             + weights["ic"] * loss_ic
             + weights["data"] * loss_data)
    return total, {"pde": loss_pde, "bc": loss_bc, "ic": loss_ic,
                   "data": loss_data}


# --------------------------------------------------------------------- #
# Gradient-norm annealing (Wang, Yu & Perdikaris 2021)
# --------------------------------------------------------------------- #
def _mean_grad_magnitude(model, term_losses):
    """Mean |dL/dtheta| per loss term without stepping the optimizer."""
    mags = {}
    for key, term in term_losses.items():
        if not term.requires_grad:
            mags[key] = 0.0
            continue
        grads = torch.autograd.grad(
            term, model.parameters(), retain_graph=True,
            create_graph=False, allow_unused=True)
        flat = torch.cat([g.flatten() for g in grads if g is not None])
        mags[key] = float(flat.abs().mean()) if flat.numel() else 0.0
    return mags


def anneal_weights(model, term_losses, weights, ema=0.9, low=0.01, high=100.0):
    """Up-weight under-represented loss terms by gradient statistics."""
    mags = _mean_grad_magnitude(model, term_losses)
    active = {k: m for k, m in mags.items() if m > 0.0 and weights.get(k, 0.0) > 0.0}
    m_max = max(active.values()) if active else 1.0
    for key in active:
        target = m_max / mags[key]
        target = min(max(target, low), high)
        weights[key] = (1.0 - ema) * weights[key] + ema * target
    return weights, mags


# --------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------- #
def sample_points(n_f, n_bc, n_ic, use_lhs=True):
    """Collocation / boundary / initial points on x in [-1,1], t in [0,1]."""
    if use_lhs:
        sampler = qmc.LatinHypercube(d=2, seed=0)
        s = sampler.random(n=n_f)
        x_r = 2.0 * s[:, 0] - 1.0
        t_r = s[:, 1]
    else:
        x_r = np.random.uniform(-1.0, 1.0, n_f)
        t_r = np.random.uniform(0.0, 1.0, n_f)

    x_bc = np.concatenate([np.full(n_bc // 2, -1.0),
                           np.full(n_bc - n_bc // 2, 1.0)])
    t_bc = np.random.uniform(0.0, 1.0, n_bc)
    x_ic = np.random.uniform(-1.0, 1.0, n_ic)

    to = lambda a: torch.tensor(a, dtype=torch.get_default_dtype(),
                                requires_grad=True, device=DEVICE)
    return (to(x_r), to(t_r), to(x_bc), to(t_bc), to(x_ic))


# --------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------- #
def make_loss_closure(net, data, nu, weights, hard_constraint):
    x_r, t_r, x_bc, t_bc, x_ic = data
    return lambda: compute_total_loss(
        net, x_r, t_r, x_bc, t_bc, x_ic, nu, weights, hard_constraint)


def train(net, data, nu, weights=None, hard_constraint=False,
          adam_iters=5000, lbfgs_max_iter=2000, anneal=False, seed=0):
    weights = weights or {"pde": 1.0, "bc": 1.0, "ic": 1.0, "data": 0.0}
    torch.manual_seed(seed)

    # ---- Phase 1: Adam -- global exploration -------------------------
    print("\n===== Phase 1: Adam =====")
    opt = optim.Adam(net.parameters(), lr=1e-3)
    close = make_loss_closure(net, data, nu, weights, hard_constraint)
    for it in range(1, adam_iters + 1):
        opt.zero_grad()
        loss, terms = close()
        if anneal and (it % 50 == 0 or it == adam_iters):
            weights, mags = anneal_weights(net, terms, weights)
        loss.backward()
        opt.step()
        if it % 1000 == 0 or it == adam_iters:
            print(f"iter {it:5d}  loss={loss.item():.3e}  "
                  f"[pde {terms['pde'].item():.2e}] weights="
                  f"{tuple(round(w,3) for w in weights.values())}")

    # ---- Phase 2: L-BFGS -- local second-order refinement -------------
    print("\n===== Phase 2: L-BFGS (Strong-Wolfe line search) =====")
    lbfgs = optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs_max_iter,
                        max_eval=lbfgs_max_iter * 2, history_size=50,
                        line_search_fn="strong_wolfe")
    t0 = time.time()
    last = [0.0]

    def closure():
        lbfgs.zero_grad()
        loss, _ = close()
        last[0] = loss.item()
        loss.backward()
        return loss

    lbfgs.step(closure)
    print(f"L-BFGS completed in {time.time() - t0:.1f}s, "
          f"final loss={last[0]:.3e}")
    return weights


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #
def evaluate(net, nu, t_grid, x_grid, exact, hard_constraint=False):
    X, T = np.meshgrid(x_grid, t_grid)              # (nt, nx) each
    u_exact = exact

    xt = torch.tensor(X.ravel(), dtype=torch.get_default_dtype(),
                      requires_grad=True, device=DEVICE)
    tt = torch.tensor(T.ravel(), dtype=torch.get_default_dtype(),
                      requires_grad=True, device=DEVICE)
    with torch.no_grad():
        if hard_constraint:
            u_hat = net(xt.unsqueeze(-1), tt.unsqueeze(-1))
        else:
            u_hat = net(xt.unsqueeze(-1), tt.unsqueeze(-1))
    u_hat = u_hat.detach().cpu().numpy().reshape(X.shape)

    err = u_hat - u_exact
    rel_l2 = np.linalg.norm(err) / (np.linalg.norm(u_exact) + 1e-14)
    linf = np.max(np.abs(err))

    # PDE residual diagnostic on the same grid
    r, _ = compute_pde_residual(net, xt.unsqueeze(-1), tt.unsqueeze(-1), nu)
    res = r.detach().cpu().numpy().reshape(X.shape)
    return {"rel_l2": rel_l2, "linf": linf,
            "max_res": float(np.max(np.abs(res))),
            "u_hat": u_hat, "exact": u_exact,
            "X": X, "T": T, "residual": res}


def main():
    ap = argparse.ArgumentParser(description="PINN for Burgers' equation")
    ap.add_argument("--nu", type=float, default=0.01 / np.pi,
                    help="viscosity (classic benchmark default 0.01/pi)")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 40, 40, 40, 40, 1])
    ap.add_argument("--activation", default="tanh",
                    choices=["tanh", "gelu", "siren", "relu"])
    ap.add_argument("--N_f", type=int, default=3000, help="collocation points")
    ap.add_argument("--N_bc", type=int, default=200)
    ap.add_argument("--N_ic", type=int, default=300)
    ap.add_argument("--adam_iters", type=int, default=5000)
    ap.add_argument("--lbfgs_max_iter", type=int, default=2000)
    ap.add_argument("--anneal", action="store_true",
                    help="gradient-norm weight annealing (Wang et al.)")
    ap.add_argument("--hard", action="store_true",
                    help="hard-enforce IC + Dirichlet BC via ansatz")
    ap.add_argument("--lhs", action="store_true", help="LHS collocation")
    ap.add_argument("--eval_pts", type=int, default=41,
                    help="points per axis in the report grid")
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    print(f"device={DEVICE}  nu={args.nu:.6f}  activation={args.activation} "
          f"loss-annealing={args.anneal} constraint={'hard' if args.hard else 'soft'}")

    net = PINN(layers=tuple(args.layers), activation=args.activation).to(DEVICE)
    if args.hard:
        net = HardConstrainedPINN(net).to(DEVICE)

    data = sample_points(args.N_f, args.N_bc, args.N_ic, use_lhs=args.lhs)
    train(net, data, args.nu, hard_constraint=args.hard,
          adam_iters=args.adam_iters, lbfgs_max_iter=args.lbfgs_max_iter,
          anneal=args.anneal)

    x_grid = np.linspace(-1.0, 1.0, args.eval_pts)
    t_grid = np.linspace(0.0, 1.0, args.eval_pts)
    exact = exact_burgers_grid(x_grid, t_grid, args.nu,
                               use_cache=not args.no_cache)
    met = evaluate(net, args.nu, t_grid, x_grid, exact,
                   hard_constraint=args.hard)

    print("\n===== Evaluation on %d x %d grid =====" %
          (args.eval_pts, args.eval_pts))
    print(f"relative L2 error : {met['rel_l2']:.6e}")
    print(f"L-infinity error  : {met['linf']:.6e}")
    print(f"max |PDE residual|: {met['max_res']:.6e}")

    # python burgers_pinn.py --eval_pts 41 --save_plot ... (optional hook)


if __name__ == "__main__":
    main()