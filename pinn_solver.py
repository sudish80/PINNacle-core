#!/usr/bin/env python3
"""
pinn_solver.py -- General-Purpose Physics-Informed Neural Network PDE Solver
=============================================================================

Solves an ARBITRARY initial-boundary-value problem of the form

        N[u; params](x, t) = f(x, t),          x in box,  t in (t0, t1]

where N is any (nonlinear) differential operator the user expresses with
first/second derivatives of u supplied by the framework:

        u       : network output          (N, 1)
        u_t     : first time derivative   (N, 1)
        u_tt    : second time derivative  (N, 1)   [time_orders = 2]
        u_x     : list of d spatial 1st partials  (N, 1)
        u_xx    : list of d spatial 2nd pure partials

Example problems are registered below (subset):

  1. burgers     -- 1D viscous Burgers' equation       (Cole-Hopf exact ref)
  2. heat        -- 1D heat / diffusion equation         (analytic exact ref)
  3. wave        -- 1D wave equation, second-order in t (analytic exact ref)
  4. forced_heat -- arbitrary manufactured PDE:
                    "solve any operator you can write, against a
                     manufactured solution" -> validation against a known
                     closed-form field (this IS the "arbitrary PDE" proof).

Shared tooling: two-stage Adam->L-BFGS(Strong-Wolfe) training, Latin
Hypercube sampling, generic box hard-constraints (exact IC/BC ansatz),
gradient-norm annealing, LHS collocation, and rel.-L2 / L-inf / residual
metrics.

Author: PINN Research Group
"""

import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc

import mpmath as mp

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pinn_reference_cache")


# ===================================================================== #
# 1. NETWORK (generic input dimension d+1)
# ===================================================================== #
class PINN(nn.Module):
    """Fully-connected MLP on concatenated inputs z = [x_1..x_d, t]."""

    def __init__(self, in_dim, layers=(4, 40, 40, 40, 40), activation="tanh",
                 w0=30.0):
        super().__init__()
        self.activation, self.w0 = activation, w0
        dims = [in_dim] + list(layers) + [1]
        self.fcs = nn.ModuleList()
        for i in range(len(dims) - 1):
            fc = nn.Linear(dims[i], dims[i + 1])
            if activation == "siren":
                if i == 0:
                    nn.init.uniform_(fc.weight, -1.0 / w0, 1.0 / w0)
                else:
                    with torch.no_grad():
                        fc.weight *= np.sqrt(6.0 / dims[i])
            else:
                nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fcs.append(fc)

    def _act(self, z):
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "gelu":
            return torch.nn.functional.gelu(z)
        if self.activation == "relu":
            return torch.relu(z)
        if self.activation == "siren":
            return torch.sin(self.w0 * z)
        raise ValueError(self.activation)

    def forward(self, z):
        for fc in self.fcs[:-1]:
            z = self._act(fc(z))
        return self.fcs[-1](z)


# ===================================================================== #
# 2. DIFFERENTIAL OPERATOR / DERIVATIVE ENGINE
# ===================================================================== #
def _grad(u, inp, create=True, retain=True):
    return torch.autograd.grad(
        u, inp, grad_outputs=torch.ones_like(u),
        create_graph=create, retain_graph=retain)[0]


def derivative_engine(net, xs, t, time_orders=1, space_orders=2):
    """Compute u and its requested derivatives in one differentiable pass.

    Parameters
    ----------
    net   : PINN (accepts torch.cat([*xs, t], dim=1))
    xs    : list of d leaf tensors, each (N, 1), requires_grad=True
    t     : leaf tensor (N, 1), requires_grad=True

    Returns
    -------
    (u, der) where der contains 'u_t' (and 'u_tt'), 'u_x' (list), 'u_xx' (list).
    """
    z = torch.cat([*xs, t], dim=1)
    u = net(z)
    der: Dict[str, object] = {}

    der["u_t"] = _grad(u, t)
    if time_orders >= 2:
        der["u_tt"] = _grad(der["u_t"], t)

    if space_orders >= 1:
        der["u_x"] = [_grad(u, x) for x in xs]
    if space_orders >= 2:
        der["u_xx"] = [_grad(der["u_x"][k], xs[k]) for k in range(len(xs))]
    return u, der


# generic box domains
def box_annihilator(x, a, b):
    """A(x) = prod_k (x_k - a_k)(b_k - x_k) / ((b_k-a_k)/2)^2 ; 0 only on faces."""
    A = torch.ones_like(x[:, :1])
    for k in range(x.shape[1]):
        A = A * (x[:, k:k + 1] - a[k]) * (b[k] - x[:, k:k + 1]) \
            / (((b[k] - a[k]) / 2.0) ** 2)
    return A


class HardConstrainedPINN(nn.Module):
    """Satisfies IC (and IC_t) plus Dirichlet BCs exactly on a box.

        u_hat = u0(x) + A(x) * (t - t0)^p * NN(x, t)

    with p = ic_orders, so that time derivatives up to order p-1 of the free
    term vanish at t = t0 and the distance field A kills the correction on the
    boundary. Requires the IC and BC to be *compatible* on the corners.
    """

    def __init__(self, net, ic_fn, a, b, t0, ic_orders=1):
        super().__init__()
        self.net, self.ic_fn, self.a, self.b, self.t0, self.p = (
            net, ic_fn, a, b, t0, ic_orders)

    def forward(self, z):
        xs = [z[:, k:k + 1] for k in range(z.shape[1] - 1)]
        t = z[:, -1:]
        u0 = self.ic_fn(torch.cat(xs, dim=1))
        A = box_annihilator(torch.cat(xs, dim=1), self.a, self.b)
        return u0 + A * (t - self.t0) ** self.p * self.net(z)


# ===================================================================== #
# 3. PROBLEM DEFINITION REGISTRY
# ===================================================================== #
@dataclass
class Problem:
    name: str
    d: int
    a: List[float]
    b: List[float]
    t0: float
    t1: float
    residual: Callable                      # (u, der, x, t) -> r  (N,1)
    ic_fn: Callable                         # (x)  -> u0            (N,1)
    bc_val: Callable                        # (x, t) -> g           (N,1)
    time_orders: int = 1
    ic_t_fn: Optional[Callable] = None      # (x) -> du/dt|t0        (N,1)
    ic_orders: int = 1                      # hard-constraint exponent p
    force_fn: Optional[Callable] = None     # RHS f(x,t) (else 0)
    exact_fn: Optional[Callable] = None     # reference / manufactured
    bc_slope_fn: Optional[Callable] = None  # (x,t) -> target du/dx at bnd (N,d)
    params: Dict[str, float] = field(default_factory=dict)
    default_layers: List[int] = field(default_factory=lambda: [40, 40, 40, 40])


# --- 1D viscous Burgers' equation ------------------------------------- #
_burgers_coeff_cache: Dict[float, tuple] = {}


def _burgers_coeffs(nu):
    """Cole-Hopf reciprocal-series coefficients, memoised per nu."""
    if nu not in _burgers_coeff_cache:
        N, dps, A = 180, 60, mp.mpf(1.0) / (2.0 * mp.mpf(nu) * mp.pi)
        with mp.workdps(dps):
            b0 = mp.e ** (-A) * mp.besseli(0, A)
            bn = [mp.e ** (-A) * mp.besseli(k, A) for k in range(1, N + 1)]
            ck = [mp.mpf(nu) * mp.pi ** 2 * (k * k) for k in range(1, N + 1)]
        _burgers_coeff_cache[nu] = (b0, bn, ck, dps)
    return _burgers_coeff_cache[nu]


def _burgers_u_point(x, t, nu):
    b0, bn, ck, dps = _burgers_coeffs(nu)
    with mp.workdps(dps):
        xt, tt = mp.mpf(x), mp.mpf(t)
        sw = swx = mp.mpf(0)
        for k in range(1, len(bn) + 1):
            e = mp.exp(-ck[k - 1] * tt)
            w = xt * k
            sw += bn[k - 1] * e * mp.cospi(w)
            swx += k * bn[k - 1] * e * mp.sinpi(w)
        return float(2 * mp.mpf(nu) * (-2 * mp.pi * swx) / (b0 + 2 * sw))


def _burgers_exact(x, t, nu):
    """Exact u for arbitrary 1-D arrays x, t of equal length (pointwise)."""
    x = np.asarray(x, float).ravel()
    t = np.asarray(t, float).ravel()
    return np.array([_burgers_u_point(xi, ti, nu) for xi, ti in zip(x, t)])


def problem_burgers(nu=0.01 / np.pi):
    def residual(u, der, x, t):
        ux = der["u_x"][0]
        return der["u_t"] + u * ux - nu * der["u_xx"][0]

    def ic(x):
        return -torch.sin(np.pi * x[:, :1])

    def bc(x, t):
        return torch.zeros_like(x[:, :1])

    return Problem(
        name="burgers", d=1, a=[-1.0], b=[1.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=ic, bc_val=bc,
        exact_fn=lambda x1, t, nu=nu: _burgers_exact_burgers_helper(x1, t, nu),
        params={"nu": nu},
        default_layers=[40, 40, 40, 40],
    )


def _burgers_exact_burgers_helper(x, t, nu):
    """Pointwise exact solution for the raveled (x, t) evaluation arrays."""
    return _burgers_exact(x, t, nu)


# --- 1D heat / diffusion ---------------------------------------------- #
def problem_heat(D=0.01):
    def residual(u, der, x, t):
        return der["u_t"] - D * der["u_xx"][0]

    def ic(x):
        return torch.sin(np.pi * x[:, :1])

    def exact(x, t):
        return np.sin(np.pi * x) * np.exp(-D * np.pi ** 2 * t)

    return Problem(
        name="heat", d=1, a=[-1.0], b=[1.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=ic, bc_val=lambda x, t: torch.zeros_like(x[:, :1]),
        exact_fn=exact, params={"D": D},
        default_layers=[40, 40, 40],
    )


# --- 1D wave equation (second-order in time) ---------------------------- #
def problem_wave(c=1.0):
    def residual(u, der, x, t):
        return der["u_tt"] - c ** 2 * der["u_xx"][0]

    def ic(x):
        return torch.sin(np.pi * x[:, :1])

    def ic_t(x):
        return torch.zeros_like(x[:, :1])

    return Problem(
        name="wave", d=1, a=[0.0], b=[1.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=ic, bc_val=lambda x, t: torch.zeros_like(x[:, :1]),
        time_orders=2, ic_t_fn=ic_t, ic_orders=2,
        exact_fn=lambda x, t, c=c: np.sin(np.pi * x) * np.cos(np.pi * c * t),
        params={"c": c}, default_layers=[40, 40, 40],
    )


# --- manufactured forced heat: SOLVE ARBITRARY OPERATOR ------------------ #
def problem_forced_heat(D=0.05, r=1.0):
    """Manufactured solution test.

    We pick an arbitrary smooth field  u*(x,t) = e^{-t} cos(pi x / 2),
    define the PDE operator  N[u] = u_t - D u_xx - r u,
    and set the forcing  f = N[u*]  so that u* is the exact solution.
    This proves the framework trivially adapts to any user-supplied
    operator + RHS (this is the "arbitrary PDE" demonstration).
    """
    def exact(x, t):
        return np.exp(-t) * np.cos(np.pi * x / 2.0)

    def residual(u, der, x, t):
        return der["u_t"] - D * der["u_xx"][0] - r * u

    def force(x, t):
        e = np.exp(-t); q = np.cos(np.pi * x / 2.0)
        return -e * q - D * (-(np.pi / 2.0) ** 2) * e * q - r * e * q

    return Problem(
        name="forced_heat", d=1, a=[-1.0], b=[1.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=lambda x: torch.cos(np.pi * x[:, :1] / 2.0),
        bc_val=lambda x, t: torch.exp(-t * torch.ones_like(x[:, :1])) * \
                            torch.cos(np.pi * x[:, :1] / 2.0),
        force_fn=force, exact_fn=exact, params={"D": D, "r": r},
        default_layers=[40, 40, 40],
    )


REGISTRY = {
    "burgers": problem_burgers,
    "heat": problem_heat,
    "wave": problem_wave,
    "forced_heat": problem_forced_heat,
}


# ===================================================================== #
# 4. SAMPLING (LHS over the box, generic boundary faces)
# ===================================================================== #
def sample_points(prob, n_f, n_bc, n_ic, use_lhs=True, seed=0):
    d = prob.d
    rng = np.random.default_rng(seed)

    def to(arr_2d):
        return torch.tensor(arr_2d, dtype=torch.get_default_dtype(),
                            requires_grad=True, device=DEVICE)

    if use_lhs:
        s = qmc.LatinHypercube(d=d + 1, seed=seed).random(n=n_f)
    else:
        s = rng.random((n_f, d + 1))
    box = np.array([prob.a + [prob.t0], prob.b + [prob.t1]]).T   # (d+1, 2)
    col = np.array([lo + (hi - lo) * s[:, k] for k, (lo, hi) in enumerate(box)]).T
    xs_r = [to(col[:, k:k + 1]) for k in range(d)]    # per-coord leaf (N_f,1)
    t_r = to(col[:, d:d + 1])                         # (N_f,1) leaf

    # boundary: 2*d faces, random points on each face
    faces = []
    for k in range(d):
        for lo_hi, val in ((0, prob.a[k]), (1, prob.b[k])):
            pt = rng.uniform(np.array(prob.a), np.array(prob.b),
                             size=(n_bc, d))
            pt[:, k] = val
            faces.append(pt)
    x_bc = to(np.concatenate(faces))
    t_bc = to(rng.uniform(prob.t0, prob.t1, size=(len(faces) * n_bc, 1)))

    xs_ic = [to(rng.uniform(prob.a[k], prob.b[k], size=(n_ic, 1)))
             for k in range(d)]
    t_ic = to(np.full((n_ic, 1), prob.t0))
    return xs_r, t_r, x_bc, t_bc, xs_ic, t_ic


# ===================================================================== #
# 5. LOSS / ANNEALING
# ===================================================================== #
def compute_losses(net, prob, data, weights, hard=False):
    xs_r, t_r, x_bc, t_bc, xs_ic, t_ic = data

    u, der = derivative_engine(net, xs_r, t_r, time_orders=prob.time_orders)
    x_all = torch.cat(xs_r, dim=1)
    r = prob.residual(u, der, x_all, t_r)
    if prob.force_fn is not None:
        f = torch.tensor(prob.force_fn(x_all.detach().cpu().numpy(),
                                       t_r.detach().cpu().numpy()),
                         dtype=torch.get_default_dtype(), device=DEVICE)
        r = r - f
    loss_pde = torch.mean(r ** 2)

    if hard:
        loss_ic = loss_ic_t = loss_bc = loss_data = \
            torch.zeros((), device=DEVICE)
    else:
        u_ic = net(torch.cat([*xs_ic, t_ic], dim=1))
        loss_ic = torch.mean((u_ic - prob.ic_fn(torch.cat(xs_ic, dim=1))) ** 2)
        if prob.ic_t_fn is not None:
            u_ic2, der_ic = derivative_engine(net, xs_ic, t_ic,
                                              time_orders=1)
            loss_ic_t = torch.mean(
                (der_ic["u_t"] - prob.ic_t_fn(torch.cat(xs_ic, dim=1))) ** 2)
        else:
            loss_ic_t = torch.zeros((), device=DEVICE)
        u_bc = net(torch.cat([x_bc, t_bc], dim=1))
        loss_bc = torch.mean((u_bc - prob.bc_val(x_bc, t_bc)) ** 2)
        loss_data = torch.zeros((), device=DEVICE)

    total = (weights["pde"] * loss_pde
             + weights["ic"] * loss_ic
             + weights.get("ic_t", 1.0) * loss_ic_t
             + weights["bc"] * loss_bc
             + weights["data"] * loss_data)
    terms = {"pde": loss_pde, "ic": loss_ic, "ic_t": loss_ic_t,
             "bc": loss_bc, "data": loss_data}
    return total, terms


def mean_grad_magnitudes(model, terms):
    mags = {}
    for k, term in terms.items():
        if not term.requires_grad:
            mags[k] = 0.0
            continue
        g = torch.autograd.grad(term, model.parameters(), retain_graph=True,
                                create_graph=False, allow_unused=True)
        flat = torch.cat([v.flatten() for v in g if v is not None])
        mags[k] = float(flat.abs().mean()) if flat.numel() else 0.0
    return mags


def anneal_weights(model, terms, weights, ema=0.9, lo=1e-2, hi=1e2):
    mags = mean_grad_magnitudes(model, terms)
    active = {k: m for k, m in mags.items()
              if m > 0.0 and weights.get(k, 0.0) > 0.0}
    m_max = max(active.values()) if active else 1.0
    for k in active:
        target = min(max(m_max / mags[k], lo), hi)
        weights[k] = (1 - ema) * weights[k] + ema * target
    return weights, mags


# ===================================================================== #
# 6. TRAINING (Adam -> L-BFGS)
# ===================================================================== #
def train(net, prob, data, weights=None, hard=False, adam_iters=5000,
          lbfgs_max_iter=2000, anneal=False, seed=0):
    weights = weights or {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0,
                          "data": 0.0}
    torch.manual_seed(seed)

    print("===== Phase 1: Adam =====")
    opt = optim.Adam(net.parameters(), lr=1e-3)
    for it in range(1, adam_iters + 1):
        opt.zero_grad()
        loss, terms = compute_losses(net, prob, data, weights, hard)
        if anneal and (it % 50 == 0 or it == adam_iters):
            weights, _ = anneal_weights(net, terms, weights)
        loss.backward()
        opt.step()
        if it % 1000 == 0 or it == adam_iters:
            print(f"iter {it:5d} loss={loss.item():.3e} w="
                  f"{tuple(round(weights[k],3) for k in ('pde','ic','bc'))}")

    print("===== Phase 2: L-BFGS (Strong-Wolfe) =====")
    lbfgs = optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs_max_iter,
                        max_eval=lbfgs_max_iter * 2, history_size=50,
                        line_search_fn="strong_wolfe")
    t0, last = time.time(), [0.0]

    def closure():
        lbfgs.zero_grad()
        loss, _ = compute_losses(net, prob, data, weights, hard)
        last[0] = loss.item()
        loss.backward()
        return loss

    lbfgs.step(closure)
    print(f"L-BFGS done in {time.time()-t0:.1f}s, final loss={last[0]:.3e}")
    return weights


# ===================================================================== #
# 7. EVALUATION (rel-L2, L-inf, residual map)   -- fully general  ----
# ===================================================================== #
def evaluate(net, prob, grid_pts=31, hard=False):
    d = prob.d
    axes = [np.linspace(prob.a[k], prob.b[k], grid_pts) for k in range(d)] \
           + [np.linspace(prob.t0, prob.t1, grid_pts)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)     # (M, d+1)
    X = pts[:, :d]; T = pts[:, d:d + 1]

    xs = [torch.tensor(X[:, k:k+1], dtype=torch.get_default_dtype(),
                       requires_grad=True, device=DEVICE) for k in range(d)]
    t = torch.tensor(T, dtype=torch.get_default_dtype(),
                     requires_grad=True, device=DEVICE)
    u, der = derivative_engine(net, xs, t, time_orders=prob.time_orders)
    u_hat = u.detach().cpu().numpy()

    if prob.exact_fn is not None:
        u_exact = prob.exact_fn(X, T)
        if isinstance(u_exact, torch.Tensor):
            u_exact = u_exact.detach().cpu().numpy()
        err = u_hat - u_exact
        rel_l2 = np.linalg.norm(err) / (np.linalg.norm(u_exact) + 1e-14)
        linf = np.max(np.abs(err))
    else:
        rel_l2 = linf = np.nan

    # PDE residual diagnostic (uses the same differentiable pass)
    r = prob.residual(u, der, torch.cat(xs, dim=1), t)
    if prob.force_fn is not None:
        fval = prob.force_fn(X, T)
        r = r - torch.tensor(fval, dtype=r.dtype, device=DEVICE)
    max_res = float(r.abs().max().detach().cpu().numpy())

    return {"rel_l2": rel_l2, "linf": linf, "max_res": max_res,
            "u_hat": u_hat, "exact": u_exact, "X": X, "T": T,
            "residual": r.detach().cpu().numpy()}


# ===================================================================== #
# 8. CLI
# ===================================================================== #
def main():
    ap = argparse.ArgumentParser(description="General-purpose PINN PDE solver")
    ap.add_argument("--problem", choices=list(REGISTRY), default="heat")
    ap.add_argument("--D", type=float, default=0.01, help="diffusivity")
    ap.add_argument("--nu", type=float, default=0.01 / np.pi,
                    help="viscosity (burgers)")
    ap.add_argument("--c", type=float, default=1.0, help="wave speed")
    ap.add_argument("--r", type=float, default=1.0, help="reaction rate")
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--activation", default="tanh",
                    choices=["tanh", "gelu", "siren", "relu"])
    ap.add_argument("--N_f", type=int, default=3000)
    ap.add_argument("--N_bc", type=int, default=200)
    ap.add_argument("--N_ic", type=int, default=300)
    ap.add_argument("--adam_iters", type=int, default=5000)
    ap.add_argument("--lbfgs_max_iter", type=int, default=2000)
    ap.add_argument("--anneal", action="store_true")
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--lhs", action="store_true")
    ap.add_argument("--eval_pts", type=int, default=41)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.problem == "burgers":
        prob = problem_burgers(nu=args.nu)
    elif args.problem == "heat":
        prob = problem_heat(D=args.D)
    elif args.problem == "wave":
        prob = problem_wave(c=args.c)
    elif args.problem == "forced_heat":
        prob = problem_forced_heat(D=args.D, r=args.r)

    in_dim = prob.d + 1
    layers = args.layers if args.layers else prob.default_layers
    net = PINN(in_dim, layers=layers, activation=args.activation).to(DEVICE)
    if args.hard:
        net = HardConstrainedPINN(net, prob.ic_fn, prob.a, prob.b, prob.t0,
                                  ic_orders=prob.ic_orders).to(DEVICE)

    data = sample_points(prob, args.N_f, args.N_bc, args.N_ic,
                         use_lhs=args.lhs, seed=args.seed)
    print(f"problem={prob.name}  d={prob.d}  hard={args.hard}  "
          f"activation={args.activation}  device={DEVICE}")
    train(net, prob, data, hard=args.hard,
          adam_iters=args.adam_iters, lbfgs_max_iter=args.lbfgs_max_iter,
          anneal=args.anneal, seed=args.seed)

    met = evaluate(net, prob, grid_pts=args.eval_pts, hard=args.hard)
    print("\n===== Evaluation (%s) =====" % prob.name)
    print(f"relative L2 error : {met['rel_l2']:.6e}")
    print(f"L-infinity error  : {met['linf']:.6e}")
    print(f"max |PDE residual|: {met['max_res']:.6e}")


if __name__ == "__main__":
    main()