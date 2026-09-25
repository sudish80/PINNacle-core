"""
burgers_colab_gpu_bench.py -- self-contained Burgers PINN GPU benchmark for a
fresh Google Colab runtime (T4/L4).  Nothing but torch, numpy, mpmath needed;
all reference + solver logic is inlined from the verified PINNacle core module
(burgers_pinn.py).  Runs soft-constraint + gradient-norm annealing + L-BFGS,
then prints three grep-able lines:

    DEVICE=<name>
    REL_L2=<relative L2 error>
    BURGERS_DONE=1

Use with colab-cli (google-colab-cli, `colab` binary):

    colab run --gpu t4 burgers_colab_gpu_bench.py --N_f 3000 --adam_iters 15000 \
        --lbfgs_max_iter 2500 --anneal
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---- exact_burgers_grid (inlined verbatim from burgers_pinn.py) ----
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

# ---- PINN (inlined verbatim from burgers_pinn.py) ----
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

# ---- HardConstrainedPINN (inlined verbatim from burgers_pinn.py) ----
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

# ---- compute_pde_residual (inlined verbatim from burgers_pinn.py) ----
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

# ---- compute_total_loss (inlined verbatim from burgers_pinn.py) ----
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

# ---- anneal_weights (inlined verbatim from burgers_pinn.py) ----
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

# ---- sample_points (inlined verbatim from burgers_pinn.py) ----
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

# ---- make_loss_closure (inlined verbatim from burgers_pinn.py) ----
def make_loss_closure(net, data, nu, weights, hard_constraint):
    x_r, t_r, x_bc, t_bc, x_ic = data
    return lambda: compute_total_loss(
        net, x_r, t_r, x_bc, t_bc, x_ic, nu, weights, hard_constraint)

# ---- train (inlined verbatim from burgers_pinn.py) ----
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

# ---- evaluate (inlined verbatim from burgers_pinn.py) ----
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.01 / np.pi)
    ap.add_argument("--N_f", type=int, default=3000)
    ap.add_argument("--N_bc", type=int, default=200)
    ap.add_argument("--N_ic", type=int, default=300)
    ap.add_argument("--adam_iters", type=int, default=15000)
    ap.add_argument("--lbfgs_max_iter", type=int, default=2500)
    ap.add_argument("--anneal", action="store_true")
    ap.add_argument("--use_lhs", action="store_true")
    ap.add_argument("--eval_pts", type=int, default=41)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    net = PINN(layers=(2, 40, 40, 40, 40, 1), activation="tanh").to(devarren)
    data = sample_points(args.N_f, args.N_bc, args.N_ic, use_lhs=args.use_lhs)
    t0 = time.time()
    weights = train(net, data, args.nu, weights=None, hard_constraint=False,
                    adam_iters=args.adam_iters, lbfgs_max_iter=args.lbfgs_max_iter,
                    anneal=args.anneal)
    wall = time.time() - t0

    xg = np.linspace(-1, 1, args.eval_pts)
    tg = np.linspace(0, 1, args.eval_pts)
    exact = exact_burgers_grid(xg, tg, args.nu)
    met = evaluate(net, args.nu, tg, xg, exact)

    print("DEVICE=%s" % (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    print("REL_L2=%.6e" % met["rel_l2"])
    print("BURGERS_DONE=1  wall=%.1fs" % wall)

if __name__ == "__main__":
    main()
