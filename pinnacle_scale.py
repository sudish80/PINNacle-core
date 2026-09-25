#!/usr/bin/env python3
"""
pinnacle_scale.py -- PINNacle production scaling layer (additive / composite)
==============================================================================

Extends the *verified* pinn_solver.py WITHOUT touching a single verified seam:
everything here is additive via import, and applies the four chosen scaling
axes:

  1. More PDEs, now
       * fisher_kpp : 1D reaction-diffusion Fisher-KPP with a CLOSED-FORM
                      traveling-wave exact solution (certified with sympy:
                      residual simplifies symbolically to EXACTLY 0):
                        u_t = u_xx + u (1 - u)
                        u(x,t) = (1 + e^{z})^{-2},  z = (x - c t)/sqrt(6),
                        c = 5/sqrt(6)
       * poisson2d  : 2-D manufactured elliptic problem with symbolic-0 residual
                      - (u_xx + u_yy) = 2 pi^2 u,
                        u(x,y) = sin(pi x) sin(pi y)   on [0,1]x[0,1]
                      (f_max = 2 pi^2 = 19.739..., found symbolically)
     Both are added to REGISTRY, so the framework's existing machinery
     (sampling / loss / annealing / evaluation) consumes them untouched.

  2. Framework + registry + RAR
       * REGISTRY additions via the same Problem dataclass
       * Residual-based Adaptive Refinement (rar_refine): after a burst of
         training, probe a dense collocation grid, keep the top-|residual|
         points, and splice them into the interior collocation set.

  3. Training-quality focus (opt-in, defaults preserve verified behaviour)
       * --anneal  : gradient-norm weight annealing (reuses verified impl)
       * --rar     : adaptive collocation
       * --seed    : reproducibility

  4. Config-driven runs
       * --cfg cfg.yaml : YAML override for any training hyperparameter.

Run
---
    python3 pinnacle_scale.py --problem fisher_kpp --anneal --rar \\
        --adams 12000 --lbfgs 1800
    python3 pinnacle_scale.py --problem poisson2d  --N_f 600 --N_bc 300 \\
        --N_ic 300 --adams 6000 --anneal
    python3 pinnacle_scale.py --cfg cfg/fisher_rar.yaml

Exactness guarantee
-------------------
References for both new problems were verified symbolically (sympy residual
simplification == 0) BEFORE this file was written, not post-hoc.
"""

import argparse
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

# --- composite import: reuse the VERIFIED engine untouched ----------------- #
import pinn_solver as P

__all__ = ["REGISTRY_EXT", "problem_fisher_kpp", "problem_poisson2d",
           "rar_refine", "train_scale", "load_cfg", "main"]

# ===================================================================== #
# SCALING AXIS 1 --  MORE PDEs, now  (both sympy-certified exact refs)
# ===================================================================== #
S6 = float(np.sqrt(6.0))
C_FK = float(5.0 / np.sqrt(6.0))          # traveling-wave speed 5/sqrt(6)


def _fisher_wave(x, t):
    """Closed-form Fisher-KPP traveling front (broadcast over any shapes)."""
    z = (np.asarray(x, float) - C_FK * np.asarray(t, float)) / S6
    return (1.0 + np.exp(z)) ** -2


def problem_fisher_kpp(a=-4.0, b=8.0, t0=0.0, t1=0.8,
                       layers: Optional[List[int]] = None):
    """Fisher-KPP u_t = u_xx + u(1-u) with closed-form exact wave."""
    def residual(u, der, x, t):
        return der["u_t"] - der["u_xx"][0] - u * (1.0 - u)

    def ic(x):           # u(x, 0) = (1 + e^{x/s6})^-2
        return torch.tensor(_fisher_wave(x.detach().cpu().numpy(), 0.0),
                            dtype=torch.get_default_dtype(), device=x.device)

    def exact(x, t):
        return _fisher_wave(np.asarray(x, float).ravel(),
                            np.asarray(t, float).ravel())

    def bc(x, t):
        return torch.tensor(_fisher_wave(x.detach().cpu().numpy(),
                                          t.detach().cpu().numpy()),
                            dtype=torch.get_default_dtype(), device=x.device)

    return P.Problem(
        name="fisher_kpp", d=1, a=[a], b=[b], t0=t0, t1=t1,
        residual=residual, ic_fn=ic, bc_val=bc,
        exact_fn=exact,
        params={"a": a, "b": b, "c": C_FK, "sigma": S6},
        default_layers=layers or [40, 40, 40, 40],
    )


def problem_poisson2d(lo=0.0, hi=1.0, layers: Optional[List[int]] = None):
    """2-D manufactured Poisson: -(u_xx+u_yy) = 2 pi^2 u, u = sin pi x sin pi y.

    Static problem encoded as a degenerate-time IBVP (t0 == t1) so the whole
    verified machinery applies unchanged; u is t-independent.
    """
    f2 = 2.0 * float(np.pi) ** 2

    def residual(u, der, x, t):
        return -der["u_xx"][0] - der["u_xx"][1] - f2 * u

    def ic(x):           # sin(pi x) sin(pi y)
        xx = x[:, 0:1]; yy = x[:, 1:2]
        return torch.sin(np.pi * xx) * torch.sin(np.pi * yy)

    def exact(x, t):
        x = np.asarray(x, float); t = np.asarray(t, float).ravel()[:, None]
        return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

    def bc(x, t):
        return ic(x)     # on every face u = sin(pi x) sin(pi y) matches

    return P.Problem(
        name="poisson2d", d=2, a=[lo, lo], b=[hi, hi], t0=0.0, t1=0.0,
        residual=residual, ic_fn=ic, bc_val=bc, exact_fn=exact,
        params={"f_max": f2, "domain": f"[{lo},{hi}]^2"},
        default_layers=layers or [50, 50, 50, 50],
    )


REGISTRY_EXT = {
    "fisher_kpp": problem_fisher_kpp,
    "poisson2d":  problem_poisson2d,
}
# register into the existing registry so the framework sees them
P.REGISTRY.update(REGISTRY_EXT)


# ===================================================================== #
# SCALING AXIS 2 --  Residual-based Adaptive Refinement (RAR)
# ===================================================================== #
def rar_refine(net, prob, data, n_add=60, probe_n=1500, seed=0):
    """Splice the top-|residual| interior collocation points into the set.

    data = (xs_r, t_r, x_bc, t_bc, xs_ic, t_ic) -- same layout as
    pinn_solver.sample_points.  Returns data with n_add points appended to
    the collocation leaves (residual leaves keep requires_grad=True).
    """
    xs_r, t_r, x_bc, t_bc, xs_ic, t_ic = data
    d = prob.d
    rng = np.random.default_rng(seed)

    # dense random probe inside the box
    probe = rng.uniform(0, 1, size=(probe_n, d + 1))
    box = np.array([prob.a + [prob.t0], prob.b + [prob.t1]]).T
    col = np.array([lo + (hi - lo) * probe[:, k]
                    for k, (lo, hi) in enumerate(box)]).T

    xs_p = [torch.tensor(col[:, k:k + 1], dtype=torch.get_default_dtype(),
                         requires_grad=True, device=P.DEVICE) for k in range(d)]
    t_p = torch.tensor(col[:, d:d + 1], dtype=torch.get_default_dtype(),
                       requires_grad=True, device=P.DEVICE)

    u, der = P.derivative_engine(net, xs_p, t_p,
                                 time_orders=prob.time_orders)
    x_all = torch.cat(xs_p, dim=1)
    r = prob.residual(u, der, x_all, t_p)
    if prob.force_fn is not None:
        f = torch.tensor(prob.force_fn(x_all.detach().cpu().numpy(),
                                       t_p.detach().cpu().numpy()),
                         dtype=torch.get_default_dtype(), device=P.DEVICE)
        r = r - f
    mag = r.detach().abs().view(-1)

    if mag.numel() <= n_add:
        n_add = max(int(mag.numel()) // 2, 1)
    idx = torch.topk(mag, n_add).indices.cpu().numpy()
    xs_new = [xs_p[k][idx].detach().clone().requires_grad_() for k in range(d)]
    t_new = t_p[idx].detach().clone().requires_grad_()
    xs_r = [torch.cat([x_old, x_new], dim=0) for x_old, x_new in
            zip(xs_r, xs_new)]
    t_r = torch.cat([t_r, t_new], dim=0)
    return xs_r, t_r, x_bc, t_bc, xs_ic, t_ic


# ===================================================================== #
# SCALING AXIS 3+4 --  training-quality switches + YAML config
# ===================================================================== #
def load_cfg(path):
    """Load a YAML config into a plain dict (empty if file missing)."""
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def train_scale(net, prob, data, weights=None, hard=False, adams=5000,
                lbfgs=2000, anneal=False, rar=False, rar_every=400,
                rar_add=60, rar_probe=1500, wall_clock=None, seed=0):
    """Two-stage trainer with OPTIONAL RAR bursts and wall-clock budget.

    Parameter defaults reproduce the verified behaviour exactly (rar=False,
    adams=5000, lbfgs=2000, anneal=False).  wall_clock floats policy:
    when > 0, stop prolonging Adam once elapsed exceeds it (determinism only;
    L-BFGS then runs to lbfgs iterations)."""
    weights = weights or {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0,
                          "data": 0.0}
    torch.manual_seed(seed)

    t_start = time.time()
    print(f"===== Phase 1: Adam  (adams={adams}, rar={rar}, anneal={anneal}) "
          f"=====")
    opt = optim.Adam(net.parameters(), lr=1e-3)
    for it in range(1, adams + 1):
        if wall_clock and (time.time() - t_start) > wall_clock:
            print(f"wall-clock budget reached at adam iter {it}; entering "
                  f"L-BFGS")
            break
        opt.zero_grad()
        loss, terms = P.compute_losses(net, prob, data, weights, hard)
        if anneal and (it % 50 == 0 or it == adams):
            weights, _ = P.anneal_weights(net, terms, weights)
        loss.backward()
        opt.step()
        if rar and (it % rar_every == 0):
            data = rar_refine(net, prob, data,
                              n_add=rar_add, probe_n=rar_probe, seed=seed)
        if it % 1000 == 0 or it == adams:
            print(f"iter {it:5d} loss={loss.item():.3e} "
                  f"N_f={data[0][0].numel()}"
                  f" w="
                  f"{tuple(round(weights[k],3) for k in ('pde','ic','bc'))}")

    print("===== Phase 2: L-BFGS (Strong-Wolfe) =====")
    lbfgs_opt = optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs,
                            max_eval=lbfgs * 2, history_size=50,
                            line_search_fn="strong_wolfe")
    last = [0.0]

    def closure():
        lbfgs_opt.zero_grad()
        loss, _ = P.compute_losses(net, prob, data, weights, hard)
        last[0] = loss.item()
        loss.backward()
        return loss

    lbfgs_opt.step(closure)
    print(f"L-BFGS done in {time.time()-t_start:.1f}s, final loss={last[0]:.3e}")
    return weights


# ===================================================================== #
# CLI
# ===================================================================== #
def main():
    ap = argparse.ArgumentParser(
        description="PINNacle scaled solver (registry + RAR + config)")
    ap.add_argument("--problem", choices=sorted([*P.REGISTRY, *REGISTRY_EXT]),
                    default="poisson2d")
    ap.add_argument("--nu", type=float, default=0.01 / np.pi)
    ap.add_argument("--D", type=float, default=0.01)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--N_f", type=int, default=3000)
    ap.add_argument("--N_bc", type=int, default=200)
    ap.add_argument("--N_ic", type=int, default=300)
    ap.add_argument("--adams", type=int, default=5000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--anneal", action="store_true")
    ap.add_argument("--rar", action="store_true")
    ap.add_argument("--rar_every", type=int, default=400)
    ap.add_argument("--rar_add", type=int, default=60)
    ap.add_argument("--rar_probe", type=int, default=1500)
    ap.add_argument("--cfg", type=str, default=None,
                    help="YAML config overriding argparse defaults")
    ap.add_argument("--eval_pts", type=int, default=41)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall_clock", type=float, default=0.0)
    args = ap.parse_args()

    if args.cfg:
        cfg = load_cfg(args.cfg)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if args.problem in REGISTRY_EXT:
        if args.problem == "fisher_kpp":
            prob = problem_fisher_kpp()
        else:
            prob = problem_poisson2d()
    elif args.problem == "burgers":
        prob = P.problem_burgers(nu=args.nu)
    elif args.problem == "heat":
        prob = P.problem_heat(D=args.D)
    elif args.problem == "wave":
        prob = P.problem_wave(c=args.c)
    elif args.problem == "forced_heat":
        prob = P.problem_forced_heat(D=args.D, r=args.r)
    else:
        raise KeyError(args.problem)

    in_dim = prob.d + 1
    layers = prob.default_layers
    net = P.PINN(in_dim, layers=layers, activation="tanh").to(P.DEVICE)
    if args.hard:
        net = P.HardConstrainedPINN(net, prob.ic_fn, prob.a, prob.b,
                                    prob.t0, ic_orders=prob.ic_orders).to(
            P.DEVICE)

    data = P.sample_points(prob, args.N_f, args.N_bc, args.N_ic,
                           use_lhs=True, seed=args.seed)
    print(f"problem={prob.name} d={prob.d} hard={args.hard} device={P.DEVICE}")
    train_scale(net, prob, data, hard=args.hard, adams=args.adams,
                lbfgs=args.lbfgs, anneal=args.anneal, rar=args.rar,
                rar_every=args.rar_every, rar_add=args.rar_add,
                rar_probe=args.rar_probe, wall_clock=args.wall_clock,
                seed=args.seed)

    met = P.evaluate(net, prob, grid_pts=args.eval_pts, hard=args.hard)
    print("\n===== Evaluation (%s) =====" % prob.name)
    print(f"relative L2 error : {met['rel_l2']:.6e}")
    print(f"L-infinity error  : {met['linf']:.6e}")
    print(f"max |PDE residual|: {met['max_res']:.6e}")


if __name__ == "__main__":
    main()