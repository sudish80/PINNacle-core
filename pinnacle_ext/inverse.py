"""pinnacle_ext.inverse -- data-driven parameter identification + assimilation.

Two related capabilities on top of the verified core:

  * INVERSE  : unknown PDE coefficient(s) (D, nu, c, ...) are promoted to
               learnable nn.Parameter(s); the residual reads the LIVE value
               each forward pass, so gradients flow to both the net AND the
               coefficient.  Given sparse (noisy) observations of the true
               field, we recover theta approx true(theta).

  * ASSIMILATION : coefficients known; point observations are folded into the
               training loss so the network reconstructs the field from data.

Design notes
------------
- ``InverseProblem`` delegates whatever it does NOT override to the base
  problem via __getattr__ (domain, ic/bc, exact fn with TRUE params for
  reporting); it overrides ``residual`` to use theta.
- ``data_mse`` is our own observation term because core ``compute_losses``
  hardcodes loss_data=0 (we stay additive and do not touch pinn_solver).
- Core ``dataclasses.replace`` is the clean way to swap residuals; here we keep
  the wrapper simple and override ``residual`` explicitly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pinn_solver as P
from pinnacle_ext import engine as E

DEVICE = P.DEVICE


# --------------------------------------------------------------------- #
# Observations (synthetic data) ---------------------------------------- #
# --------------------------------------------------------------------- #
def sample_observations(prob, n_obs=60, noise=0.0, seed=0, t_like=None):
    """Draw synthetic noisy point data from the exact reference.

    Returns dict with X (N,d), T (N,1), U (N, n_out) and the true problem.
    ``noise`` is the absolute Gaussian sigma on u (in u-units).
    """
    rng = np.random.default_rng(seed)
    n_out = getattr(prob, "n_out", 1)

    X = np.column_stack([rng.uniform(prob.a[k], prob.b[k], n_obs)
                         for k in range(prob.d)])
    T = (np.full((n_obs, 1), prob.t0) if t_like is None or prob.t0 == prob.t1
         else rng.uniform(prob.t0, prob.t1, (n_obs, 1)))
    U = prob.exact_fn(X, T)
    U = np.asarray(U.detach().cpu().numpy() if torch.is_tensor(U) else U,
                   float)
    U = U.reshape(n_obs, n_out)
    if noise > 0.0:
        U = U + rng.normal(0.0, noise, U.shape)
    return {"X": X, "T": T, "U": U}


def data_mse(net, obs):
    """MSE between net output and (possibly noisy) observations."""
    X, T, U = obs["X"], obs["T"], obs["U"]
    z = torch.tensor(np.hstack([X, T]), dtype=torch.get_default_dtype(),
                     device=DEVICE)
    pred = net(z)
    target = torch.tensor(U, dtype=torch.get_default_dtype(), device=DEVICE)
    return torch.mean((pred - target) ** 2)


# --------------------------------------------------------------------- #
# Learnable-coefficient problem ---------------------------------------- #
# --------------------------------------------------------------------- #
class InverseProblem:
    """Base problem + learnable theta; delegates un-overridden attributes.

    ``learn``   : dict  name -> init value          (what we fit)
    ``true``    : dict  name -> true value          (for error reporting)
    theta       : dict  name -> nn.Parameter        (what we optimise)
    """

    def __init__(self, base_name, learn, true=None, **extra):
        if base_name not in P.REGISTRY:
            raise KeyError(f"no problem '{base_name}' in registry")
        self._base = P.REGISTRY[base_name](**(true or {}))
        self.name = f"{base_name}_inv"
        self._base_name = base_name
        self.true = dict(true or {})
        self.learn = dict(learn)
        self.theta = {k: nn.Parameter(torch.tensor(float(v), device=DEVICE))
                      for k, v in learn.items()}
        self.residual = self._make_residual(base_name)
        # honour optional extra overrides (e.g. n_obs weighting)
        for k, v in extra.items():
            setattr(self, k, v)

    # ---- residual closures per PDE (the ONLY place theta is consumed) ----
    def _make_residual(self, name):
        th = self.theta

        if name == "heat":
            return lambda u, der, x, t: der["u_t"] - th["D"] * der["u_xx"][0]
        if name == "advection":
            return lambda u, der, x, t: der["u_t"] + th["c"] * der["u_x"][0]
        if name == "convection_diffusion":
            return (lambda u, der, x, t:
                    der["u_t"] + th["c"] * der["u_x"][0]
                    - th["D"] * der["u_xx"][0])
        if name == "burgers":
            return (lambda u, der, x, t:
                    der["u_t"] + u * der["u_x"][0] - th["nu"] * der["u_xx"][0])
        if name == "allen_cahn":
            return (lambda u, der, x, t:
                    der["u_t"] - th["D"] * der["u_xx"][0] - u + u ** 3)
        raise KeyError(f"no learnable-coefficient residual for '{name}'")

    # ---- delegation to the base problem --------------------------------- #
    def __getattr__(self, key):
        return getattr(self.__dict__["_base"], key)

    def __repr__(self):
        fit = {k: f"{float(v):.4g}" for k, v in self.theta.items()}
        return f"InverseProblem({self._base_name}, theta={fit}, true={self.true})"

    def param_error(self):
        """dict name -> |estimated - true| / |true| (0 if unknown)."""
        out = {}
        for k in self.learn:
            est = float(self.theta[k].detach())
            t = self.true.get(k, float("nan"))
            out[k] = abs(est - t) / (abs(t) + 1e-300) if t == t else float("nan")
        return out


# --------------------------------------------------------------------- #
# Joint training: net + coefficients against PDE + data + BC/IC --------- #
# --------------------------------------------------------------------- #
def train_with_data(net, prob, obs, data=None, weights=None, adam_iters=800,
                    lbfgs_max_iter=60, seed=0, verbose=True, w_obs=1.0,
                    data_informed=False):
    """Joint Adam + L-BFGS over net (+ prob.theta for InverseProblem).

    Loss = pde + bc + ic + ic_t + w_obs * data_mse.  Returns (net, history)
    where history records pde/data loss + current coefficient estimates.

    ``data_informed`` folds the observation points into the COLLOCATION set so
    the PDE residual and the data term are evaluated on the same locations --
    the coupling that makes coefficient identification well-posed (the
    residual penalises u_t - theta*u_xx *at the data points*).
    """
    weights = dict(weights or {"pde": 1.0, "ic": 1.0, "ic_t": 1.0,
                               "bc": 1.0, "data": 0.0})
    if data is None:
        data = P.sample_points(prob, 300, 150, 150, use_lhs=True, seed=seed)
    if data_informed and obs is not None:
        xs_r, t_r, x_bc, t_bc, xs_ic, t_ic = data
        X, T = obs["X"], obs["T"]
        z = np.hstack([X, T])
        xs_r = [torch.cat([xs_r[k],
                           torch.tensor(X[:, k:k+1], dtype=torch.get_default_dtype(),
                                        requires_grad=True, device=DEVICE)]) for k in range(prob.d)]
        t_r = torch.cat([t_r, torch.tensor(T, dtype=torch.get_default_dtype(),
                                           requires_grad=True, device=DEVICE)])
        data = (xs_r, t_r, x_bc, t_bc, xs_ic, t_ic)
    theta = list(getattr(prob, "theta", {}).values())
    params = list(net.parameters()) + theta

    def loss_fn():
        total, terms = E.compute_losses_ext(net, prob, data, weights)
        if obs is not None:
            total = total + w_obs * data_mse(net, obs)
        return total, terms

    torch.manual_seed(seed); np.random.seed(seed)
    opt = optim.Adam(params, lr=1e-3)
    history = []
    for it in range(1, adam_iters + 1):
        opt.zero_grad()
        loss, terms = loss_fn()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == adam_iters:
            hist = {"it": it, "loss": float(loss.item()),
                    "pde": float(terms["pde"].item())}
            if obs is not None:
                hist["data"] = float(data_mse(net, obs).item())
            if theta:
                hist["theta"] = {k: float(v.detach()) for k, v in
                                 getattr(prob, "theta", {}).items()}
            history.append(hist)
            if verbose:
                print(f"  adam {it:>5d} loss={hist['loss']:.3e} "
                      + (f"theta={hist.get('theta')} " if "theta" in hist else ""),
                      flush=True)

    # newton polish on the joint objective
    lbfgs = optim.LBFGS(params, lr=0.4, max_iter=lbfgs_max_iter,
                        max_eval=lbfgs_max_iter * 2, history_size=20,
                        line_search_fn="strong_wolfe")
    finals = [0.0]

    def closure():
        lbfgs.zero_grad()
        loss, terms = loss_fn()
        finals[0] = float(loss.item())
        loss.backward()
        return loss

    lbfgs.step(closure)
    if verbose:
        print(f"  L-BFGS done, final loss={finals[0]:.3e}", flush=True)
        if theta:
            print(f"  estimated theta = {[round(float(v.detach()),6) for v in theta]}")
    return net, history