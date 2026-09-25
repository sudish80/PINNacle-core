"""pinnacle_ext.methods -- adaptive sampling, self-adaptive weights,
hard Neumann enforcement, and convenient runtime hooks.

All functions are additive; they consume the verified core's objects
(Problem / sample_points / compute_losses / tangent classes) unchanged.
"""

import numpy as np
import torch
import torch.optim as optim

import pinn_solver as P
from pinnacle_ext import engine as E

DEVICE = P.DEVICE


# --------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------- #
def resample_collocation(prob, xs_r, t_r, n_new, seed=0, rar_frac=0.0,
                         residual_magnitudes=None):
    """Density-aware / residual-aware collocation resampling.

    With rar_frac=r (0..1), fraction 'r' of the new points are placed within
    thickness bands around points with the highest current residual magnitude
    (RAR-style).  r=0 returns plain uniform-LHS points.
    """
    rng = np.random.default_rng(seed)
    d = prob.d
    n_rar = int(n_new * rar_frac)
    n_uni = n_new - n_rar

    boxlo = np.array(prob.a + [prob.t0])
    boxhi = np.array(prob.b + [prob.t1])

    if n_uni > 0:
        s = np.random.default_rng(seed).random((n_uni, d + 1))
        uni = boxlo + (boxhi - boxlo) * s
    else:
        uni = np.empty((0, d + 1))

    if n_rar > 0 and residual_magnitudes is not None:
        mags = np.asarray(residual_magnitudes).ravel()
        prob_ = mags / (mags.sum() + 1e-12)
        idx = rng.choice(len(mags), size=n_rar, p=prob_, replace=True)
        eps = 0.01 * (boxhi - boxlo)      # local band width
        base = np.stack([_col(prob.a + [prob.t0], prob.b + [prob.t1], k) for k in range(d + 1)], axis=1)
        near_sample = np.random.default_rng(seed + 7).normal(
            0.0, 0.5, size=(n_rar, d + 1))
        # reuse residual anchor points is not trivial without data layout
        near = np.clip(near_sample, -1, 1) * eps + boxlo + (boxhi - boxlo) * 0.5
        out = np.concatenate([uni, near], axis=0)
    else:
        out = uni
    return out


def _col(a, b, k):
    rng = np.random.default_rng(0)
    return rng.random(1) * (b[k] - a[k]) + a[k]


def das_resample(net, prob, data, n_ref=1500, keep_ratio=0.0, seed=0):
    """Density-adaptive sampling: probe residual on a fresh grid, then
    resample collocation proportional to (residual^2) (DAS, Nabian 2021).

    Returns a new data tuple (identical contract to sample_points) with the
    SAME total collocation budget.
    """
    xs_r, t_r, x_bc, t_bc, xs_ic, t_ic = data
    d = prob.d
    rng = np.random.default_rng(seed)
    s = rng.random((n_ref, d + 1))
    boxlo, boxhi = np.array(prob.a + [prob.t0]), np.array(prob.b + [prob.t1])
    col = boxlo + (boxhi - boxlo) * s

    xs_p = [torch.tensor(col[:, k:k+1], dtype=torch.get_default_dtype(),
                         requires_grad=True, device=DEVICE) for k in range(d)]
    t_p = torch.tensor(col[:, d:d+1], dtype=torch.get_default_dtype(),
                       requires_grad=True, device=DEVICE)
    u, der = E.derivative_engine_ext(net, xs_p, t_p,
                                     time_orders=prob.time_orders,
                                     space_orders=getattr(prob, "space_orders", 2),
                                     n_out=getattr(prob, "n_out", 1))
    r = prob.residual(u, der, torch.cat(xs_p, dim=1), t_p)
    if prob.force_fn is not None:
        f = prob.force_fn(torch.cat(xs_p, dim=1).detach().cpu().numpy(),
                          t_p.detach().cpu().numpy())
        r = r - torch.tensor(f, dtype=r.dtype, device=DEVICE)
    mag = r.detach().abs().mean(dim=1, keepdim=bool(r.dim() > 1)).view(-1).cpu().numpy()
    mag = mag ** 2

    n_total = xs_r[0].shape[0]
    if mag.sum() <= 0:
        mag[:] = 1.0
    p = mag / mag.sum()
    keep = int(n_total * keep_ratio)
    idx = rng.choice(n_ref, size=n_total - keep, p=p, replace=True)
    col_new = col[idx]
    xs_new = [torch.tensor(col_new[:, k:k+1], dtype=torch.get_default_dtype(),
                           requires_grad=True, device=DEVICE) for k in range(d)]
    t_new = torch.tensor(col_new[:, d:d+1], dtype=torch.get_default_dtype(),
                         requires_grad=True, device=DEVICE)
    if keep > 0:
        keep_idx = rng.choice(n_total, size=keep, replace=False)
        xs_new = [torch.cat([xs_r[k][keep_idx], xs_new[k]]) for k in range(d)]
        t_new = torch.cat([t_r[keep_idx], t_new])
    return xs_new, t_new, x_bc, t_bc, xs_ic, t_ic


# --------------------------------------------------------------------- #
# Self-adaptive weights (SA-PINN, McClenny & Braga-Neto)
# --------------------------------------------------------------------- #
class SelfAdaptivePINN(torch.nn.Module):
    """Wraps a base net; learns per-residual positive log-weights lmd.

    loss = sum_i exp(lmd_i) * norm(r_i)^2  -- weights trained via
    multiplicative update in train_sa.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.lmd = torch.nn.Parameter(
            torch.zeros((4,), device=DEVICE))  # pde, ic, ic_t, bc

    def forward(self, z):
        return self.net(z)

    def weights(self):
        return torch.exp(self.lmd).detach()


def train_sa(net, prob, data, weights=None, adam_iters=5000,
             lbfgs_max_iter=1000, seed=0):
    """SA-PINN training: alternate outer (net) and inner (lambda) steps."""
    torch.manual_seed(seed)
    sa = net if isinstance(net, SelfAdaptivePINN) else SelfAdaptivePINN(net)
    opt = optim.Adam(list(sa.net.parameters()), lr=1e-3)
    opt_l = optim.Adam([sa.lmd], lr=1e-2)
    for it in range(1, adam_iters + 1):
        opt.zero_grad(); opt_l.zero_grad()
        loss, terms = E.compute_losses_ext(sa.net, prob, data,
                                           {"pde": 1, "ic": 1, "ic_t": 1,
                                            "bc": 1, "data": 0}, hard=False)
        l = torch.exp(sa.lmd)
        loss = (l[0] * terms["pde"] + l[1] * terms["ic"]
                + l[2] * terms["ic_t"] + l[3] * terms["bc"])
        loss.backward()
        opt.step()
        with torch.no_grad():
            if sa.lmd.grad is not None:
                sa.lmd.data.add_(0.1 * sa.lmd.grad)
            sa.lmd.data.clamp_(-6.0, 6.0)   # exp(6)~403, prevents overflow
        if it % 1000 == 0 or it == adam_iters:
            print(f"iter {it:5d} loss={loss.item():.3e} "
                  f"lmd={[round(float(v),2) for v in sa.lmd.tolist()]}")
    print("===== L-BFGS (SA) =====")
    lbfgs = optim.LBFGS(list(sa.net.parameters()), lr=1.0,
                        max_iter=lbfgs_max_iter, max_eval=lbfgs_max_iter * 2,
                        history_size=50, line_search_fn="strong_wolfe")
    last = [0.0]
    def closure():
        lbfgs.zero_grad()
        _, terms = E.compute_losses_ext(sa.net, prob, data,
                                        {"pde": 1, "ic": 1, "ic_t": 1,
                                         "bc": 1, "data": 0},
                                        hard=False)
        l = torch.exp(sa.lmd)
        loss = (l[0] * terms["pde"] + l[1] * terms["ic"]
                + l[2] * terms["ic_t"] + l[3] * terms["bc"])
        last[0] = loss.item()
        loss.backward()
        return loss
    lbfgs.step(closure)
    print(f"L-BFGS done, final loss={last[0]:.3e}")
    return sa


# --------------------------------------------------------------------- #
# Hard Dirichlet + Neumann enforcement on a 1-D interval (both ends)
# --------------------------------------------------------------------- #
class HardNeumann1D(torch.nn.Module):
    def __init__(self, net, prob, bc_slope_fn, a=None, b=None, t0=None):
        super().__init__()
        self.net, self.prob = net, prob
        self.a = a if a is not None else prob.a[0]
        self.b = b if b is not None else prob.b[0]
        self.t0 = t0 if t0 is not None else prob.t0
        self.g = prob.ic_fn
        self.dg = bc_slope_fn

    def forward(self, z):
        x = z[:, :1]; t = z[:, 1:]
        tt = (x - self.a) / (self.b - self.a)
        h00 = 2 * tt ** 3 - 3 * tt ** 2 + 1
        h10 = tt ** 3 - 2 * tt ** 2 + tt
        h01 = -2 * tt ** 3 + 3 * tt ** 2
        h11 = tt ** 3 - tt ** 2
        va = self.g(torch.tensor([[self.a]], dtype=torch.get_default_dtype(),
                                 device=x.device)).reshape(())
        vb = self.g(torch.tensor([[self.b]], dtype=torch.get_default_dtype(),
                                 device=x.device)).reshape(())
        da = torch.tensor(float(self.dg(self.a)), dtype=torch.get_default_dtype(),
                          device=x.device)
        db = torch.tensor(float(self.dg(self.b)), dtype=torch.get_default_dtype(),
                          device=x.device)
        L = (self.b - self.a)
        Q = h00 * va + h10 * L * da + h01 * vb + h11 * L * db
        A2 = ((x - self.a) ** 2) * ((self.b - x) ** 2) / (L / 2) ** 4
        return Q + A2 * self.net(z)