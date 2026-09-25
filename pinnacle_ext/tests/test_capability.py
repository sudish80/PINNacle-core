"""Verification for the v2 capability batch:

  4th-order problems (beam / Cahn-Hilliard), slope-BC term, HardNeumann clamp,
  inverse / data-assimilation, checkpoint roundtrip, MC-dropout + ensemble
  uncertainty, conservation/consistency diagnostics.

Each test is intentionally tiny -- these are contract tests, not benchmarks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import pytest

import pinn_solver as P
import pinnacle_ext.problems as probs
import pinnacle_ext.engine as E
import pinnacle_ext.arch as arch
import pinnacle_ext.methods as methods
from pinnacle_ext.config import DEFAULTS, _build_problem, _build_net
from pinnacle_ext import inverse as INV
from pinnacle_ext import uncertainty as UNC
from pinnacle_ext import consistency as CONS
from pinnacle_ext import checkpoint as CK


# --------------------------------------------------------------------- #
# 1. 4th-order derivative engine + clamp BC term
# --------------------------------------------------------------------- #
def test_engine_space4():
    net = P.PINN(2, layers=(3, 8, 8), activation="tanh")
    xs = [torch.randn(8, 1, requires_grad=True)]
    t = torch.randn(8, 1, requires_grad=True)
    u, der = E.derivative_engine_ext(net, xs, t, space_orders=4)
    assert all(k in der for k in ("u_x", "u_xx", "u_xxx", "u_xxxx"))
    # u_xxxx == grad of u_xxx along x
    uxxx = der["u_xxx"][0]
    g = torch.autograd.grad(uxxx, xs[0], torch.ones_like(uxxx),
                            create_graph=True)[0]
    assert torch.allclose(der["u_xxxx"][0], g, atol=1e-5)
    der["u_xxxx"][0].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


@pytest.mark.parametrize("name", ["beam1d", "cahn_hilliard"])
def test_4th_order_loss_and_slope_term(name):
    prob = probs.REGISTRY[name]()
    assert prob.space_orders == 4
    assert prob.bc_slope_fn is not None
    net = arch.build_net(prob.d + 1, prob.default_layers, activation="tanh")
    data = P.sample_points(prob, 24, 6, 6, seed=3)
    w = {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0, "bc_slope": 1.0,
         "data": 0.0}
    loss, terms = E.compute_losses_ext(net, prob, data, w, hard=False)
    assert torch.isfinite(loss)
    assert terms["bc_slope"].dim() == 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_hard_neumann_clamped_beam_trains():
    prob = probs.problem_beam1d()
    inner = P.PINN(2, layers=(24, 24, 24), activation="tanh")
    net = methods.HardNeumann1D(inner, prob, prob.bc_slope_fn)
    data = P.sample_points(prob, 80, 20, 20, seed=3)
    w = {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0, "data": 0.0}
    opt = torch.optim.Adam(inner.parameters(), lr=1e-3)
    for _ in range(200):
        loss, _ = E.compute_losses_ext(net, prob, data, w, hard=False)
        opt.zero_grad(); loss.backward(); opt.step()
    met = E.evaluate_ext(net, prob, grid_pts=17)
    # the Hermite-clamped ansatz + cheap adam must already beat the trivial net
    assert float(met["rel_l2"]) < 1.0


@pytest.mark.parametrize("name", ["beam1d", "cahn_hilliard"])
def test_4th_order_problem_certified(name):
    assert name in __import__("pinnacle_ext.certified", fromlist=["CERTIFIED"]).CERTIFIED


# --------------------------------------------------------------------- #
# 2. Inverse / data-assimilation
# --------------------------------------------------------------------- #
def test_inverse_problem_delegation_and_theta():
    inv = INV.InverseProblem("heat", learn={"D": 0.3}, true={"D": 0.05})
    assert inv.a == [-1.0] and inv.t1 == 1.0            # delegated from base
    assert tuple(inv.theta.keys()) == ("D",)
    assert isinstance(inv.theta["D"], nn_param())
    # residual reads live theta: perturbing theta changes the residual
    xs = [torch.tensor([[0.2]], requires_grad=True)]
    t = torch.tensor([[0.3]], requires_grad=True)
    net = P.PINN(2, layers=(2, 4, 4))
    u, der = E.derivative_engine_ext(net, xs, t, time_orders=1, space_orders=2)
    r1 = inv.residual(u, der, torch.cat(xs, 1), t)
    with torch.no_grad():
        inv.theta["D"].mul_(2.0)
    u2, der2 = E.derivative_engine_ext(net, xs, t, time_orders=1, space_orders=2)
    r2 = inv.residual(u2, der2, torch.cat(xs, 1), t)
    assert not torch.allclose(r1, r2)


def nn_param():
    import torch.nn as nn
    return nn.Parameter


def test_observations_from_exact():
    inv = INV.InverseProblem("advection", learn={"c": 0.5}, true={"c": 1.0})
    obs = INV.sample_observations(inv, n_obs=30, seed=0)
    assert obs["X"].shape == (30, 1)
    assert obs["T"].shape == (30, 1)
    assert obs["U"].shape == (30, 1)
    assert np.isfinite(obs["U"]).all()


def test_inverse_heat_recovers_D():
    # dense-grid observations (Raissi-style) are the well-posed regime
    inv = INV.InverseProblem("heat", learn={"D": 0.3}, true={"D": 0.05})
    xs = np.linspace(-1.0, 1.0, 11)
    ts = np.linspace(0.0, 1.0, 7)
    Xg, Tg = np.meshgrid(xs, ts)
    obs = {"X": Xg.ravel().reshape(-1, 1), "T": Tg.ravel().reshape(-1, 1),
           "U": inv.exact_fn(Xg.ravel().reshape(-1, 1),
                            Tg.ravel().reshape(-1, 1))}
    net = _build_net(inv, {"arch": "mlp", "layers": [16, 16, 16],
                           "activation": "tanh", "seed": 0})
    INV.train_with_data(net, inv, obs, adam_iters=600, lbfgs_max_iter=60,
                        seed=0, w_obs=2.0, verbose=False)
    err = abs(float(inv.theta["D"].detach()) - 0.05) / 0.05
    assert err < 0.4, f"D recovered as {float(inv.theta['D'].detach()):.4f}"


def test_assimilation_data_term_helpful():
    # known coefficient, sparse noisy data -> data term should pull field down
    prob = probs.problem_advection(c=1.0)
    obs = INV.sample_observations(prob, n_obs=30, noise=0.0, seed=1)
    net = _build_net(prob, {"arch": "mlp", "layers": [16, 16, 16],
                            "activation": "tanh", "seed": 0})
    data = P.sample_points(prob, 100, 30, 30, seed=0)
    w = {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0, "data": 0.0}
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(300):
        loss, _ = E.compute_losses_ext(net, prob, data, w)
        loss = loss + INV.data_mse(net, obs)
        opt.zero_grad(); loss.backward(); opt.step()
    met = E.evaluate_ext(net, prob, grid_pts=13)
    assert float(met["rel_l2"]) < 1.0


# --------------------------------------------------------------------- #
# 3. Uncertainty
# --------------------------------------------------------------------- #
def test_dropout_arch_deterministic_at_zero():
    net = arch.build_net(2, [8, 8], arch="dropout", drop=0.0)
    z = torch.randn(4, 2)
    out1 = net(z).detach().numpy()
    net.eval()
    out2 = net(z).detach().numpy()
    assert np.allclose(out1, out2)


def test_mc_dropout_std_nonzero():
    prob = probs.problem_advection()
    net = arch.build_net(2, [12, 12], arch="dropout", drop=0.2, seed=1)
    mean, std, _, _ = UNC.mc_dropout(net, prob, grid_pts=7, n_samples=12)
    assert mean.shape == (49, 1)
    assert float(std.mean()) > 1e-6


def test_deep_ensemble_calibration():
    prob = probs.problem_advection()
    nets = [arch.build_net(2, [12, 12], seed=s) for s in range(2)]
    mean, std, Xn, Tn = UNC.deep_ensemble(nets, prob, grid_pts=5)
    assert mean.shape == (25, 1) and std.shape == (25, 1)
    truth = prob.exact_fn(Xn, Tn).reshape(mean.shape)
    cal = UNC.calibration_curve(mean, std, truth)
    assert cal and all(isinstance(c["rmse"], float) for c in cal)


# --------------------------------------------------------------------- #
# 4. Checkpoint
# --------------------------------------------------------------------- #
def test_checkpoint_roundtrip(tmp_path):
    path = str(tmp_path / "run.pt")
    prob = probs.problem_heat2d()
    cfg = dict(DEFAULTS); cfg.update({"problem": "heat2d"})
    net = _build_net(prob, cfg)
    met = {"rel_l2": 0.123}
    CK.save_checkpoint(path, net, cfg=cfg, metrics=met)
    net2, prob2, ck = CK.build_from_checkpoint(path)
    assert prob2.name == "heat2d"
    assert ck["metrics"]["rel_l2"] == 0.123
    # reloaded net reproduces *identical* outputs on a fixed input
    z = torch.randn(5, 3)
    assert torch.allclose(net(z), net2(z))


def test_checkpoint_wrapped_prefix_strip(tmp_path):
    path = str(tmp_path / "wrap.pt")
    prob = probs.problem_beam1d()
    inner = P.PINN(2, layers=(6, 6, 6))
    net = methods.HardNeumann1D(inner, prob, prob.bc_slope_fn)
    CK.save_checkpoint(path, net, cfg={"problem": "beam1d"})
    net2, prob2, _ = CK.build_from_checkpoint(path,
                                              cfg={"arch": "mlp",
                                                   "activation": "tanh",
                                                   "seed": 0,
                                                   "layers": [6, 6, 6]})
    z = torch.randn(5, 2)
    # checkpoint stores the WRAPPED net's inner weights under "net."
    assert torch.allclose(net.net(z), net2(z), atol=1e-8)


# --------------------------------------------------------------------- #
# 5. Conservation / consistency diagnostics
# --------------------------------------------------------------------- #
def test_mass_history_advection_conserved():
    prob = probs.problem_advection(c=1.0)
    net = _build_net(prob, dict(DEFAULTS, problem="advection"))
    m = CONS.mass_history(net, prob, n_x=128, n_t=24)
    # advection conserves int u dx; a random-init net still has tiny total
    # because tanh output is near zero -- assert the API shape + boundedness
    assert m["t"].shape == (24,)
    assert np.isfinite(m["M"]).all()
    assert 0.0 <= m["max_drift"] < 1e12


def test_divergence_stats_ns():
    prob = probs.problem_kovasznay_ns()
    net = _build_net(prob, dict(DEFAULTS, problem="kovasznay_ns"))
    d = CONS.divergence_stats(net, prob, grid_pts=5)
    assert "mean_div" in d and d["mean_div"] >= 0.0


def test_residual_gap_consistent():
    prob = probs.problem_heat2d()
    net = _build_net(prob, dict(DEFAULTS, problem="heat2d"))
    g = CONS.residual_gap(net, prob, n_train=50, grid_pts=7, seed=0)
    for k in ("train_mean", "eval_mean", "eval_max", "gap_ratio"):
        assert np.isfinite(g[k])
        assert g[k] >= 0.0


def test_conservation_report_string():
    prob = probs.problem_advection()
    net = _build_net(prob, dict(DEFAULTS, problem="advection"))
    rep = CONS.conservation_report(net, prob)
    assert isinstance(rep, str) and "residual gap" in rep


# --------------------------------------------------------------------- #
# 6. run pin / config wiring
# --------------------------------------------------------------------- #
def test_config_dropout_arch_and_beam_hard():
    cfg = dict(DEFAULTS)
    cfg.update({"problem": "beam1d", "adams": 5, "lbfgs": 1, "hard": True,
                "N_f": 24, "N_bc": 6, "N_ic": 6, "eval_pts": 7})
    met = train_one_local(cfg)
    assert met["mode"] == "ext"
    assert np.isfinite(met["rel_l2"])
    cfg2 = dict(DEFAULTS)
    cfg2.update({"problem": "heat", "arch": "dropout", "drop": 0.2,
                 "adams": 5, "lbfgs": 1, "N_f": 24, "N_bc": 6, "N_ic": 6,
                 "eval_pts": 7})
    met2 = train_one_local(cfg2)
    assert met2["mode"] == "core"


def train_one_local(cfg):
    from pinnacle_ext.config import train_one
    return train_one(cfg)


# --------------------------------------------------------------------- #
# 7. helmholtz remedy: Fourier knobs, explicit basis, proper-SIREN init
# --------------------------------------------------------------------- #
def test_fourier_basis_rows_and_outdim():
    basis = [[1.5, 0.0, 0.0], [0.0, 2.0, 0.0], [1.5, 2.0, 0.0]]
    net = arch.build_net(3, layers=(40, 40), arch="fourier", basis=basis,
                         n_freqs=0, seed=0)
    B = net.ff.B
    assert B.shape == (3, 3)
    torch.testing.assert_close(B, torch.tensor(basis, dtype=B.dtype))
    assert net.ff.out_dim == 6
    z = torch.randn(4, 3, dtype=B.dtype)
    y = net(z)
    assert y.shape == (4, 1) and torch.isfinite(y).all()


def test_fourier_hybrid_basis_random_append():
    basis = [[1.5, 0.0]]
    net = arch.build_net(2, layers=(12, 12), arch="fourier", basis=basis,
                         n_freqs=4, sigma=2.0, seed=0)
    assert net.ff.B.shape == (5, 2)
    assert torch.allclose(net.ff.B[0], torch.tensor([1.5, 0.0], dtype=net.ff.B.dtype))


def test_siren_proper_init_scales_hidden_by_w0():
    net = arch.build_net(2, layers=(20, 20), arch="siren", w0=30.0)
    fcs = list(net.fcs)
    for i, fc in enumerate(fcs[:-1]):       # hidden layers, per-layer bound
        bound = np.sqrt(6.0 / fc.in_features) / 30.0
        assert fc.weight.abs().max().item() <= bound + 1e-9
    head = fcs[-1]
    assert head.out_features == 1
    assert head.weight.abs().max().item() <= 1.0 / 30.0 + 1e-9


def test_config_sigma_nfreqs_w0_wired():
    prob = _build_problem(dict(DEFAULTS, problem="heat"))
    net = _build_net(prob, dict(DEFAULTS, arch="fourier", n_freqs=8,
                                sigma=3.0, seed=0))
    assert net.ff.B.shape[0] == 8
    assert net.ff.B.std().item() == pytest.approx(3.0, abs=0.6)
    prob = _build_problem(dict(DEFAULTS, problem="helmholtz"))
    net = _build_net(prob, dict(DEFAULTS, arch="siren", w0=12.0))
    assert net.w0 == 12.0


def test_sa_training_no_nan_and_weights_finite():
    cfg = dict(DEFAULTS, problem="heat", adams=30, lbfgs=3, N_f=40,
               N_bc=8, N_ic=8, eval_pts=9, sa=True)
    from pinnacle_ext.config import train_one
    met = train_one(cfg)
    assert np.isfinite(met["rel_l2"])
    assert met["mode"] == "SA"