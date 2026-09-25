"""pinnacle_ext.consistency -- conservation & physical-consistency diagnostics.

A PINN that fits well can still violate integral conservation (mass/energy)
or the divergence constraint away from the sampled collocation.  These checks
quantify that gap on FRESH grids and between train/eval regions, so a residual
"looks small on the training set but isn't physical" is caught explicitly.

All functions are pure diagnostics -- nothing trains anything.
"""

import numpy as np
import torch

import pinn_solver as P
from pinnacle_ext import engine as E

DEVICE = P.DEVICE


def _quad(x, u):
    """Trapezoid quadrature along the LAST axis."""
    return np.trapz(u, x, axis=-1)


def mass_history(net, prob, n_x=256, n_t=48, hard=False):
    """For d==1: M(t) = int_a^b u(x,t) dx as a function of t.

    Returns dict with t grid, M(t), the normalized drift
    (M(t) - M(0))/M(0), and the max |deriv of M| -- nonzero for non-conserved
    fields; should be ~0 for advection / KdV / NS-style problems.
    """
    if prob.d != 1:
        raise NotImplementedError("mass_history is 1-D (x-domain integral)")
    ts = np.linspace(prob.t0, max(prob.t1, prob.t0 + 1e-12), n_t)
    xs = np.linspace(prob.a[0], prob.b[0], n_x)
    M = np.zeros(n_t)
    for i, t in enumerate(ts):
        z = torch.tensor(np.hstack([xs.reshape(-1, 1),
                                    np.full((n_x, 1), t)]),
                         dtype=torch.get_default_dtype(), device=DEVICE)
        with torch.no_grad():
            u = net(z).detach().cpu().numpy().ravel()
        M[i] = _quad(xs, u)
    drift = (M - M[0]) / (abs(M[0]) + 1e-14)
    dM = np.gradient(M, ts)
    return {"t": ts, "M": M, "max_drift": float(np.max(np.abs(drift))),
            "max_dM_dt": float(np.max(np.abs(dM))),
            "M_range": float(np.ptp(M))}


def divergence_stats(net, prob, grid_pts=21, hard=False, u_hat=None):
    """Mean/max |div(u,v)| for incompressible flow (n_out>=2).

    Evaluate at each (x_i) with NO training:  du_x + dv_y.  The steady NS
    manufactured state has div==0 identically, so the PINN's violation is a
    pure discretization/optimization artifact.
    """
    n_out = getattr(prob, "n_out", 1)
    if n_out < 2:
        return {"mean_div": 0.0, "max_div": 0.0, "grid_pts": grid_pts,
                "note": "scalar problem; div is not a constraint"}
    grid = np.linspace(0.0, 1.0, grid_pts)
    X = np.meshgrid(*[grid for _ in range(prob.d)])
    Xn = np.stack([m.ravel() for m in X], axis=1)
    Tn = np.zeros((Xn.shape[0], 1))
    z = torch.tensor(np.hstack([Xn, Tn]), dtype=torch.get_default_dtype(),
                     device=DEVICE).requires_grad_(True)
    xs_p = [z[:, k:k + 1] for k in range(prob.d)]
    t_p = z[:, prob.d:prob.d + 1]
    us, ds = E.derivative_engine_ext(net, xs_p, t_p, time_orders=1,
                                     space_orders=2, n_out=n_out)
    u, v = us[0], us[1]
    div = ds[0]["u_x"][0] + ds[1]["u_x"][1]
    div = div.detach().cpu().numpy()
    return {"mean_div": float(np.mean(np.abs(div))),
            "max_div": float(np.max(np.abs(div))),
            "grid_pts": grid_pts}


def residual_gap(net, prob, n_train=400, grid_pts=21, seed=0, hard=False):
    """|residual| on the TRAINING collocation vs a FRESH uniform grid.

    A big train/eval gap = overfit to the sampled points / hidden stiffness.
    Returns dict: train_mean, eval_mean, eval_max, gap_ratio.
    """
    w = {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0, "bc_slope": 1.0,
         "data": 0.0}
    data = P.sample_points(prob, n_train, n_bc=100, n_ic=100, seed=seed)
    xs_r, t_r = data[0], data[1]
    us, ds = E.derivative_engine_ext(net, xs_r, t_r,
                                     time_orders=prob.time_orders,
                                     space_orders=getattr(prob, "space_orders", 2),
                                     n_out=getattr(prob, "n_out", 1))
    r = prob.residual(us, ds, torch.cat(xs_r, dim=1), t_r)
    if prob.force_fn is not None:
        f = torch.tensor(prob.force_fn(torch.cat(xs_r, dim=1).detach().cpu().numpy(),
                                       t_r.detach().cpu().numpy()),
                         dtype=r.dtype, device=DEVICE)
        r = r - f
    train_mean = float(r.detach().abs().mean())

    # fresh uniform evaluation grid (fine-grained, independent of collocation)
    from pinnacle_ext.uncertainty import eval_grid
    Xn, Tn, zp = eval_grid(prob, grid_pts)
    xs_e = [zp[:, k:k + 1] for k in range(prob.d)]
    t_e = zp[:, prob.d:prob.d + 1]
    us_e, ds_e = E.derivative_engine_ext(net, xs_e, t_e,
                                         time_orders=prob.time_orders,
                                         space_orders=getattr(prob, "space_orders", 2),
                                         n_out=getattr(prob, "n_out", 1))
    re = prob.residual(us_e, ds_e, torch.cat(xs_e, dim=1), t_e)
    if prob.force_fn is not None:
        fe = torch.tensor(prob.force_fn(zp[:, :prob.d].detach().cpu().numpy(),
                                        t_e.detach().cpu().numpy()),
                          dtype=re.dtype, device=DEVICE)
        re = re - fe
    eval_mean = float(re.detach().abs().mean())
    eval_max = float(re.detach().abs().max())
    return {"train_mean": train_mean, "eval_mean": eval_mean,
            "eval_max": eval_max,
            "gap_ratio": eval_mean / (train_mean + 1e-14),
            "grid_pts": grid_pts}


def conservation_report(net, prob, hard=False):
    """One-line physical consistency summary across the space of checks."""
    lines = []
    if prob.d == 1:
        m = mass_history(net, prob, hard=hard)
        lines.append(f"mass: max|drift|={m['max_drift']:.3e} "
                     f"dM/dt={m['max_dM_dt']:.3e}")
    if getattr(prob, "n_out", 1) >= 2:
        d = divergence_stats(net, prob, hard=hard)
        lines.append(f"div: mean={d['mean_div']:.3e} max={d['max_div']:.3e}")
    g = residual_gap(net, prob, hard=hard)
    lines.append(f"residual gap train/eval: {g['train_mean']:.2e} / "
                 f"{g['eval_mean']:.2e} (ratio {g['gap_ratio']:.1f}x)")
    return " | ".join(lines)