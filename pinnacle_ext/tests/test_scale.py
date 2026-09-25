"""Verification suite for the scaling layer.

Run:  python3 -m pytest pinnacle_ext/tests/ -q
All tests must pass; certification errors are FAIL by design.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import pytest

import pinn_solver as P
import pinnacle_ext.problems as probs
import pinnacle_ext.certified as certified
import pinnacle_ext.engine as E
import pinnacle_ext.arch as arch
import pinnacle_ext.methods as methods
from pinnacle_ext.config import train_one, DEFAULTS


# --------------------------------------------------------------------- #
# 1. Certification gate (the non-negotiable one)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(certified.CERTIFIED))
def test_certified(name):
    expr = certified.CERTIFIED[name]
    vec = expr if isinstance(expr, tuple) else (expr,)
    for e in vec:
        assert sp_simplify(e) == 0, f"{name} residual not symbolically zero"


def sp_simplify(expr):
    import sympy as sp
    return sp.simplify(expr)


def test_all_new_problems_have_certified_templates():
    new = set(probs.REGISTRY) - {"burgers", "heat", "wave", "forced_heat"}
    assert new <= set(certified.CERTIFIED), new - set(certified.CERTIFIED)


# --------------------------------------------------------------------- #
# 2. Registry / problem wiring
# --------------------------------------------------------------------- #
def test_registry_complete():
    for name in probs.REGISTRY:
        assert name in P.REGISTRY, f"{name} not in core registry"


@pytest.mark.parametrize("name", sorted(probs.REGISTRY))
def test_problem_constructs(name):
    prob = probs.REGISTRY[name]()
    assert prob.name == name
    assert prob.d >= 1
    assert prob.residual is not None
    for k in range(prob.d):
        assert np.isfinite(prob.a[k]) and np.isfinite(prob.b[k])
        assert prob.a[k] < prob.b[k]


# --------------------------------------------------------------------- #
# 3. Extended derivative engine correctness
# --------------------------------------------------------------------- #
def test_engine_scalar_matches_core():
    net = P.PINN(3, layers=(3, 8, 8), activation="tanh")
    xs = [torch.randn(16, 1, requires_grad=True), torch.randn(16, 1, requires_grad=True)]
    t = torch.randn(16, 1, requires_grad=True)
    u1, d1 = P.derivative_engine(net, xs, t)
    u2, d2 = E.derivative_engine_ext(net, xs, t)
    assert torch.allclose(u1, u2)
    for k in range(2):
        assert torch.allclose(d1["u_xx"][k], d2["u_xx"][k], atol=1e-6)


def test_engine_space3():
    net = P.PINN(3, layers=(3, 8, 8), activation="tanh")
    xs = [torch.randn(8, 1, requires_grad=True), torch.randn(8, 1, requires_grad=True)]
    t = torch.randn(8, 1, requires_grad=True)
    u, der = E.derivative_engine_ext(net, xs, t, space_orders=3)
    # compare u_xxx with finite difference on a scalar input
    assert der["u_xxx"][0].shape == (8, 1)
    # sanity: third derivative of u wrt x0 equals grad of u_xx
    uxx = der["u_xx"][0]
    g = torch.autograd.grad(uxx, xs[0], torch.ones_like(uxx), create_graph=True)[0]
    assert torch.allclose(der["u_xxx"][0], g, atol=1e-5)


def test_engine_vector():
    net = arch.VectorPINN(3, layers=(3, 8, 8), n_out=3)
    xs = [torch.randn(8, 1, requires_grad=True), torch.randn(8, 1, requires_grad=True)]
    t = torch.randn(8, 1, requires_grad=True)
    us, ds = E.derivative_engine_ext(net, xs, t, n_out=3)
    assert len(us) == 3
    for u_c in us:
        assert u_c.shape == (8, 1)


# --------------------------------------------------------------------- #
# 4. Loss plumbing for the new problems
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["kdv", "advection", "heat2d", "poisson3d",
                                  "helmholtz", "sine_gordon", "wave2d",
                                  "convection_diffusion", "allen_cahn"])
def test_losses_small(name):
    prob = probs.REGISTRY[name]()
    net = arch.build_net(prob.d + 1, prob.default_layers, n_out=prob.n_out)
    data = P.sample_points(prob, 24, 6, 6, seed=3)
    w = {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0, "data": 0.0}
    loss, terms = E.compute_losses_ext(net, prob, data, w, hard=False)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # gradient flows
    loss.backward()
    nz = [p.grad for p in net.parameters() if p.grad is not None]
    assert nz


def test_losses_ns():
    prob = probs.problem_kovasznay_ns()
    net = arch.build_net(3, prob.default_layers, n_out=3)
    data = P.sample_points(prob, 24, 6, 6, seed=3)
    w = {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0, "data": 0.0}
    loss, terms = E.compute_losses_ext(net, prob, data, w, hard=False)
    assert torch.isfinite(loss)
    assert loss.shape == torch.zeros(()).shape


# --------------------------------------------------------------------- #
# 5. Architectures
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("archname", ["mlp", "fourier", "siren", "residual",
                                      "multiscale", "adaptive", "weight_normed",
                                      "dropout"])
def test_arch(archname):
    net = arch.build_net(2, [8, 8], arch=archname)
    z = torch.randn(4, 2, requires_grad=True)
    y = net(z)
    assert y.shape == (4, 1)
    y.sum().backward()


def test_vector_out_shape():
    net = arch.build_net(3, [8, 8], n_out=3)
    assert net(torch.randn(4, 3)).shape == (4, 3)


# --------------------------------------------------------------------- #
# 6. Methods
# --------------------------------------------------------------------- #
def test_das_resample_contract():
    prob = probs.problem_heat2d()
    net = arch.build_net(3, [8, 8])
    data = P.sample_points(prob, 30, 6, 6, seed=1)
    data2 = methods.das_resample(net, prob, data, n_ref=60, keep_ratio=0.5, seed=2)
    assert data2[0][0].shape[0] == 30  # budget preserved
    assert all(all(c.shape[0] == 30 for c in xs) for xs in [data2[0]])


def test_sa_train_small():
    prob = P.problem_heat()
    net = arch.build_net(2, [8, 8])
    data = P.sample_points(prob, 20, 5, 5, seed=0)
    sa = methods.SelfAdaptivePINN(net)
    l = torch.exp(sa.lmd).sum()
    assert l.shape == torch.zeros(()).shape  # lmd properly initialized


def test_hard_neumann_enforces_bc():
    a, b = 0.0, 1.0
    net = P.PINN(2, layers=(2, 8, 8))
    prob = P.problem_heat()

    def dg(x):
        # analytic slope of u0(x)=sin(pi x) at the ends
        return np.pi * np.cos(np.pi * float(x))

    hm = methods.HardNeumann1D(net, prob, dg, a=a, b=b, t0=0.0)
    # gather boundary points at a few t values
    for tval in (0.0, 0.25, 0.75):
        z = torch.tensor([[a, tval], [b, tval]], dtype=torch.get_default_dtype(),
                         requires_grad=True)
        out = hm(z)
        ua = prob.ic_fn(torch.tensor([[a]], dtype=torch.get_default_dtype()))[0, 0]
        ub = prob.ic_fn(torch.tensor([[b]], dtype=torch.get_default_dtype()))[0, 0]
        assert abs(out[0, 0].item() - ua.item()) < 1e-5
        assert abs(out[1, 0].item() - ub.item()) < 1e-5
    # slope at left end via autograd must match the Neuamnn data
    z = torch.tensor([[a, 0.0], [0.0003, 0.0], [b, 0.0]],
                     dtype=torch.get_default_dtype(), requires_grad=True)
    g = torch.autograd.grad(hm(z), z, torch.ones(3, 1))[0]
    slope_left = g[0, 0].item()
    slope_true = dg(a) / 1.0
    assert abs(slope_left - slope_true) < 1e-2


# --------------------------------------------------------------------- #
# 7. Config / orchestrator
# --------------------------------------------------------------------- #
def test_train_one_smoke():
    cfg = dict(DEFAULTS)
    cfg.update({"problem": "heat", "adams": 10, "lbfgs": 2, "N_f": 40,
                "N_bc": 8, "N_ic": 8, "eval_pts": 9})
    met = train_one(cfg)
    assert "rel_l2" in met
    assert met["mode"] == "core"
    assert np.isfinite(met["rel_l2"])


def test_train_one_ext_kdv_smoke():
    cfg = dict(DEFAULTS)
    cfg.update({"problem": "kdv", "adams": 5, "lbfgs": 1, "N_f": 20,
                "N_bc": 4, "N_ic": 4, "eval_pts": 7})
    met = train_one(cfg)
    assert met["mode"] == "ext"


# --------------------------------------------------------------------- #
# 8. Relative-L2 metric semantics (zero-Dirichlet faces)
# --------------------------------------------------------------------- #
def test_relative_l2_equal_fields():
    u = np.linspace(0, 1, 50)
    raw, active, _ = E.relative_l2(u, u)
    assert abs(raw) < 1e-12
    assert abs(active) < 1e-12


def test_relative_l2_separates_zero_face_noise():
    ue = np.concatenate([np.zeros(40), np.full(40, 0.5)])  # 50% zero face
    uh = ue.copy()
    uh[:40] = 0.5                        # gross 1.0 error confined to zero face
    raw, active, zf = E.relative_l2(uh, ue)
    assert zf == pytest.approx(0.5)
    assert raw > 0.9                     # naive metric dominated by face noise
    assert active < 1e-12                # active metric ignores the face


def test_eval_active_metric_present():
    prob = probs.problem_heat2d()
    net = arch.build_net(3, [8, 8], n_out=1)
    met = E.evaluate_ext(net, prob, grid_pts=5)
    assert "rel_l2_active" in met
    assert "zero_frac" in met
    assert met["zero_frac"] > 0.15        # heat2d has zero-Dirichlet faces


if __name__ == "__main__":
    pytest.main([__file__, "-v"])