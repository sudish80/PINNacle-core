"""pinnacle_ext.engine -- extended derivative engine + vector-aware train loop.

The core pinn_solver stops at second spatial derivatives and single outputs.
This module provides additive capability (THIRD and FOURTH spatial
derivatives for KdV / beam / Cahn-Hilliard, MULTI-COMPONENT outputs for
Navier-Stokes) while keeping the same contract:

    residual(u, der, x, t)   scalar:  u, der as in core
    residual(us, ds, x, t)   vector:  us   = list of per-component u (N,1)
                                      ds   = list of per-component der dicts

For scalar problems with space_orders<=2 everything routes back through the
verified core (compute_losses / evaluate) -- the ext loop is zero-cost.  The
vector / third-order paths use this module's engine and wrap the identical
loss maths so the published pipeline stays consistent.
"""

import time

import numpy as np
import torch
import torch.optim as optim

import pinn_solver as P


# --------------------------------------------------------------------- #
# derivative engine (space_orders <= 4, per-component)
# --------------------------------------------------------------------- #
def _grad_scalar(u, inp, create=True, retain=True):
    return torch.autograd.grad(u, inp, grad_outputs=torch.ones_like(u),
                               create_graph=create, retain_graph=retain)[0]


def component_engine(net, z, xs, t, time_orders, space_orders, c):
    """Derivatives of ONE output channel c: net(z) -> u_c (N,1), dict der."""
    u_c = net(z)[:, c:c + 1]
    der: dict = {}
    der["u_t"] = _grad_scalar(u_c, t)
    if time_orders >= 2:
        der["u_tt"] = _grad_scalar(der["u_t"], t)
    if space_orders >= 1:
        der["u_x"] = [_grad_scalar(u_c, x) for x in xs]
    if space_orders >= 2:
        der["u_xx"] = [P._grad(der["u_x"][k], xs[k]) for k in range(len(xs))]
    if space_orders >= 3:
        # chain through the verified _grad for consistency
        der["u_xxx"] = [P._grad(der["u_xx"][k], xs[k]) for k in range(len(xs))]
    if space_orders >= 4:
        # 4th-order (Euler-Bernoulli beam, Cahn-Hilliard...)
        der["u_xxxx"] = [P._grad(der["u_xxx"][k], xs[k])
                         for k in range(len(xs))]
    return u_c, der


def derivative_engine_ext(net, xs, t, time_orders=1, space_orders=2,
                          n_out=None):
    """Like core derivative_engine but supports 3rd spatial order and
    multi-output.  Returns (u, der) for n_out==1 (core contract) or
    (us, ds) lists for n_out>1."""
    z = torch.cat([*xs, t], dim=1)
    with torch.no_grad():
        n = int(net(z).shape[1]) if n_out is None else n_out
    if n == 1 and space_orders <= 2:
        # fast-path: defer to the fully verified core engine
        return P.derivative_engine(net, xs, t, time_orders=time_orders,
                                   space_orders=space_orders)
    us, ds = [], []
    for c in range(n):
        u_c, der = component_engine(net, z, xs, t, time_orders, space_orders, c)
        us.append(u_c); ds.append(der)
    if n == 1:
        return us[0], ds[0]
    return us, ds


# --------------------------------------------------------------------- #
# losses (vector-aware, mirrors core compute_losses math exactly)
# --------------------------------------------------------------------- #
def is_core_problem(prob):
    """True if the verified core can handle it (scalar, <=2nd sp. order)."""
    return getattr(prob, "n_out", 1) == 1 and getattr(prob, "space_orders", 2) <= 2


def compute_losses_ext(net, prob, data, weights, hard=False):
    """pinn_solver.compute_losses contract + scalar/space<=2 passthrough.

    For problems the core handles we simply call the verified function.
    Otherwise we replicate the exact loss math around the extended engine.
    """
    if is_core_problem(prob):
        return P.compute_losses(net, prob, data, weights, hard)

    xs_r, t_r, x_bc, t_bc, xs_ic, t_ic = data
    u, der = derivative_engine_ext(
        net, xs_r, t_r, time_orders=prob.time_orders,
        space_orders=prob.space_orders, n_out=prob.n_out)
    x_all = torch.cat(xs_r, dim=1)
    r = prob.residual(u, der, x_all, t_r)
    if prob.force_fn is not None:
        f = prob.force_fn(x_all.detach().cpu().numpy(),
                          t_r.detach().cpu().numpy())
        ft = torch.tensor(f, dtype=torch.get_default_dtype(), device=P.DEVICE)
        r = r - ft
    loss_pde = torch.mean(torch.mul(r, r))

    # boundary / initial conditions are scalar data regardless of n_out
    if hard:
        loss_ic = loss_ic_t = loss_bc = loss_bc_slope = loss_data = \
            torch.zeros((), device=P.DEVICE)
    else:
        ub = net(torch.cat([x_bc, t_bc], dim=1))
        bc = prob.bc_val(x_bc, t_bc)
        loss_bc = torch.mean(torch.mul(ub - bc, ub - bc))
        # optional slope (Neumann/clamped) BC: enforce du/dx at boundary faces.
        # target has shape (N, d); for 4th-order 1D problems d == 1 is the
        # clamped case (beam / Cahn-Hilliard).
        loss_bc_slope = torch.zeros((), device=P.DEVICE)
        if prob.bc_slope_fn is not None:
            if prob.n_out != 1:
                raise NotImplementedError("slope BC currently scalar-output only")
            gx = torch.autograd.grad(torch.sum(ub), x_bc, create_graph=True,
                                     retain_graph=True)[0]
            target = prob.bc_slope_fn(x_bc.detach().cpu().numpy(),
                                      t_bc.detach().cpu().numpy())
            tt = torch.tensor(target, dtype=torch.get_default_dtype(),
                              device=P.DEVICE)
            loss_bc_slope = torch.mean(torch.mul(gx - tt, gx - tt))
        ui = net(torch.cat([*xs_ic, t_ic], dim=1))
        i0 = prob.ic_fn(torch.cat(xs_ic, dim=1))
        loss_ic = torch.mean(torch.mul(ui - i0, ui - i0))
        if prob.ic_t_fn is not None:
            _, der_ic = derivative_engine_ext(net, xs_ic, t_ic,
                                              time_orders=1,
                                              space_orders=prob.space_orders,
                                              n_out=prob.n_out)
            vit = prob.ic_t_fn(torch.cat(xs_ic, dim=1))
            loss_ic_t = torch.mean(torch.mul(der_ic["u_t"] - vit,
                                             der_ic["u_t"] - vit))
        else:
            loss_ic_t = torch.zeros((), device=P.DEVICE)
        loss_data = torch.zeros((), device=P.DEVICE)

    total = (weights["pde"] * loss_pde + weights["ic"] * loss_ic
             + weights.get("ic_t", 1.0) * loss_ic_t + weights["bc"] * loss_bc
             + weights.get("bc_slope", 1.0) * loss_bc_slope
             + weights["data"] * loss_data)
    terms = {"pde": loss_pde, "ic": loss_ic, "ic_t": loss_ic_t,
             "bc": loss_bc, "bc_slope": loss_bc_slope, "data": loss_data}
    return total, terms


# --------------------------------------------------------------------- #
# evaluate (vector-aware)
# --------------------------------------------------------------------- #
def relative_l2(u_hat, u_exact, zero_eps=1e-6):
    """Relative L2 with two definitions.

    ``raw``      : ||e|| / ||u*|| over all grid points (core definition).
    ``active``   : same ratio on points where |u*| > zero_eps, so zero-valued
                   Dirichlet faces do not dominate the metric.
    Returns (raw, active, zero_frac).
    """
    err = np.asarray(u_hat, float).ravel() - np.asarray(u_exact, float).ravel()
    ue = np.asarray(u_exact, float).ravel()
    raw = np.linalg.norm(err) / (np.linalg.norm(ue) + 1e-14)
    m = np.abs(ue) > zero_eps
    active = np.linalg.norm(err[m]) / (np.linalg.norm(ue[m]) + 1e-14) if m.any() \
        else float("nan")
    return raw, active, 1.0 - m.mean()


def evaluate_ext(net, prob, grid_pts=31, hard=False):
    if is_core_problem(prob):
        met = P.evaluate(net, prob, grid_pts=grid_pts, hard=hard)
        if met.get("exact") is not None:
            raw, active, zf = relative_l2(met["u_hat"], met["exact"])
            met["rel_l2_active"], met["zero_frac"] = active, zf
            met["rel_l2_raw"] = raw
        return met
    d = prob.d
    axes = [np.linspace(prob.a[k], prob.b[k], grid_pts) for k in range(d)] \
           + [np.linspace(prob.t0, prob.t1, grid_pts)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    X = pts[:, :d]; T = pts[:, d:d + 1]
    xs = [torch.tensor(X[:, k:k + 1], dtype=torch.get_default_dtype(),
                       requires_grad=True, device=P.DEVICE) for k in range(d)]
    t = torch.tensor(T, dtype=torch.get_default_dtype(),
                     requires_grad=True, device=P.DEVICE)
    u, der = derivative_engine_ext(net, xs, t, time_orders=prob.time_orders,
                                   space_orders=prob.space_orders,
                                   n_out=prob.n_out)
    if isinstance(u, list):
        u_hat = np.concatenate([u_c.detach().cpu().numpy() for u_c in u], axis=1)
    else:
        u_hat = u.detach().cpu().numpy()
    if prob.exact_fn is not None:
        u_exact = prob.exact_fn(X, T)
        if isinstance(u_exact, torch.Tensor):
            u_exact = u_exact.detach().cpu().numpy()
    else:
        u_exact = None
    raw, active, zf = relative_l2(u_hat, u_exact) if u_exact is not None \
        else (np.nan, np.nan, np.nan)
    r = prob.residual(u, der, torch.cat(xs, dim=1), t)
    if prob.force_fn is not None:
        fv = prob.force_fn(X, T)
        r = r - torch.tensor(fv, dtype=r.dtype, device=P.DEVICE)
    linf = np.inf if u_exact is None else np.max(
        np.abs(u_hat - u_exact))
    return {"rel_l2": raw, "rel_l2_active": active, "zero_frac": zf,
            "linf": linf, "max_res": float(r.abs().max().detach().cpu().numpy()),
            "u_hat": u_hat, "exact": u_exact, "X": X, "T": T}


# --------------------------------------------------------------------- #
# unified trainer (delegates to verified train for core-able problems)
# --------------------------------------------------------------------- #
def train_ext(net, prob, data, weights=None, hard=False, adam_iters=5000,
              lbfgs_max_iter=2000, anneal=False, seed=0, adam_lr=1e-3,
              lbfgs_lr=1.0):
    weights = weights or {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0,
                          "data": 0.0}
    if is_core_problem(prob):
        return P.train(net, prob, data, weights=weights, hard=hard,
                       adam_iters=adam_iters, lbfgs_max_iter=lbfgs_max_iter,
                       anneal=anneal, seed=seed)
    torch.manual_seed(seed)
    print("===== Phase 1: Adam (ext engine) =====")
    opt = optim.Adam(net.parameters(), lr=adam_lr)
    for it in range(1, adam_iters + 1):
        opt.zero_grad()
        loss, terms = compute_losses_ext(net, prob, data, weights, hard)
        if anneal and (it % 50 == 0 or it == adam_iters):
            weights, _ = P.anneal_weights(net, terms, weights)
        loss.backward()
        opt.step()
        if it % 1000 == 0 or it == adam_iters:
            print(f"iter {it:5d} loss={loss.item():.3e} w="
                  f"{tuple(round(weights[k],3) for k in ('pde','ic','bc'))}")
    print("===== Phase 2: L-BFGS (ext engine) =====")
    lbfgs = optim.LBFGS(net.parameters(), lr=lbfgs_lr, max_iter=lbfgs_max_iter,
                        max_eval=lbfgs_max_iter * 2, history_size=50,
                        line_search_fn="strong_wolfe")
    t0, last = time.time(), [0.0]
    def closure():
        lbfgs.zero_grad()
        loss, _ = compute_losses_ext(net, prob, data, weights, hard)
        last[0] = loss.item()
        loss.backward()
        return loss
    lbfgs.step(closure)
    print(f"L-BFGS done in {time.time()-t0:.1f}s, final loss={last[0]:.3e}")
    return weights