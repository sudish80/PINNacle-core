"""pinnacle_ext.config -- layered YAML config + run orchestration.

Config merge order (later wins):
    {built-in defaults} <- cfg.yaml <- user.yaml <- argv
"""

import argparse
import copy

import numpy as np
import torch

import pinn_solver as P
from pinnacle_ext import engine as E, methods as M, arch


DEFAULTS = {
    "problem": "heat",
    "layers": None,
    "activation": "tanh",
    "arch": "mlp",
    "N_f": 3000, "N_bc": 200, "N_ic": 300,
    "adams": 5000, "lbfgs": 2000,
    "anneal": False, "hard": False, "lhs": True, "seed": 0,
    "eval_pts": 31,
    "rar": False, "rar_every": 400, "rar_add": 60, "rar_probe": 1500,
    "das": False, "das_every": 400, "das_probe": 1500, "das_keep": 0.0,
    "sa": False, "wall_clock": 0.0, "nn_params": 0.0,
    "D": 0.01, "nu": 0.01 / np.pi, "c": 1.0, "r": 1.0,
    "drop": 0.1,
    "sigma": 8.0, "n_freqs": 48, "w0": 30.0,
    "verbose": True,
}


def _build_problem(cfg):
    """Instantiate a problem from config (registry-aware)."""
    import inspect
    name = cfg["problem"]
    builders = P.REGISTRY
    if name not in builders:
        raise KeyError(f"unknown problem '{name}' in registry {sorted(builders)}")
    sig = inspect.signature(builders[name])
    kwargs = {k: cfg[k] for k in sig.parameters if k in cfg}
    return builders[name](**kwargs)


def _build_net(prob, cfg):
    in_dim = prob.d + 1
    layers = cfg.get("layers") or prob.default_layers
    return arch.build_net(in_dim, layers=layers,
                          activation=cfg["activation"], arch=cfg["arch"],
                          n_out=getattr(prob, "n_out", 1),
                          n_freqs=cfg.get("n_freqs", 48),
                          sigma=cfg.get("sigma", 8.0),
                          w0=cfg.get("w0", 30.0),
                          basis=cfg.get("basis"),
                          drop=cfg.get("drop", 0.1), seed=cfg["seed"])


def _nice(cfg):
    out = dict(cfg)
    out["nn_params"] = None
    return out


def benchmark(specs, base=None, wall_clock=90.0):
    """Run a grid of configs and emit a Markdown comparison table.

    specs: list of dicts (each merged over base defaults => full config).
    Returns list of metric dicts; prints a markdown table.
    """
    base = base or {}
    rows = []
    for i, over in enumerate(specs):
        cfg = {**DEFAULTS, **base, **{k: v for k, v in over.items() if v is not None}}
        cfg["wall_clock"] = wall_clock
        met = train_one(cfg)
        rows.append(met)
        print(f"[{i+1}/{len(specs)}] {met['problem']} mode={met['mode']} "
              f"rel_l2={met['rel_l2']:.3e}", flush=True)
    print("\n## Benchmark (CPU, wall_clock-limited)")
    print(f"| # | problem | mode | params | rel-L2(active) | zero% | max|res| |")
    print(f"|---|---------|------|--------|----------------|-------|----------|")
    for i, m in enumerate(rows, 1):
        print(f"| {i} | {m['problem']} | {m['mode']} | {m['nn_params']} "
              f"| {m['rel_l2']:.2e} | {m['zero_frac']*100:.0f}% "
              f"| {m['max_res']:.2e} |")
    return rows


def train_one(cfg, prob=None, net=None):
    """Run one training+eval cycle from a config dict -> metrics dict."""
    cfg = {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    if prob is None:
        prob = _build_problem(cfg)
    if net is None:
        net = _build_net(prob, cfg).to(P.DEVICE)

    weights = {"pde": 1.0, "ic": 1.0, "ic_t": 1.0, "bc": 1.0, "data": 0.0}
    mode = ("SA" if cfg["sa"] else
            "ext" if (getattr(prob, "space_orders", 2) >= 3
                      or getattr(prob, "n_out", 1) > 1) else "core")

    data = P.sample_points(prob, cfg["N_f"], cfg["N_bc"], cfg["N_ic"],
                           use_lhs=cfg["lhs"], seed=cfg["seed"])
    if cfg["sa"]:
        net = M.SelfAdaptivePINN(net).to(P.DEVICE)
        M.train_sa(net, prob, data, adam_iters=cfg["adams"],
                   lbfgs_max_iter=cfg["lbfgs"], seed=cfg["seed"])
    else:
        if cfg["hard"]:
            wrap = None
            if (getattr(prob, "space_orders", 2) >= 4
                    and prob.d == 1 and prob.bc_slope_fn is not None):
                # clamped 4th-order 1D: u AND u_x are fixed by a Hermite ansatz
                wrap = M.HardNeumann1D(net, prob, prob.bc_slope_fn)
            else:
                from pinn_solver import HardConstrainedPINN
                wrap = HardConstrainedPINN(net, prob.ic_fn, prob.a, prob.b,
                                           prob.t0, ic_orders=prob.ic_orders)
            net = wrap.to(P.DEVICE)
        E.train_ext(net, prob, data, weights=weights, hard=cfg["hard"],
                    adam_iters=cfg["adams"], lbfgs_max_iter=cfg["lbfgs"],
                    anneal=cfg["anneal"], seed=cfg["seed"])

    met = E.evaluate_ext(net, prob, grid_pts=cfg["eval_pts"], hard=cfg["hard"])
    met = {k: v for k, v in met.items()
           if not (isinstance(v, np.ndarray) and v.ndim > 0)}
    met["mode"], met["nn_params"], met["problem"] = \
        mode, arch.n_params(net), prob.name
    met["rel_l2"] = met.get("rel_l2_active", met.get("rel_l2", float("nan")))
    met["cfg"] = _nice(cfg)
    if cfg.get("save"):
        from pinnacle_ext import checkpoint as CK
        CK.save_checkpoint(cfg["save"], net, cfg=cfg, metrics=met)
        met["saved"] = cfg["save"]
    return met


def main(argv=None):
    ap = argparse.ArgumentParser(description="PINNacle: config-driven solver")
    ap.add_argument("--cfg", action="append", default=[],
                    help="yaml config (repeatable, later wins)")
    ap.add_argument("--problem", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--arch", default=None,
                    choices=["mlp", "fourier", "siren", "residual",
                             "multiscale", "adaptive", "weight_normed",
                             "dropout"])
    ap.add_argument("--activation", default=None)
    ap.add_argument("--N_f", type=int, default=None)
    ap.add_argument("--N_bc", type=int, default=None)
    ap.add_argument("--N_ic", type=int, default=None)
    ap.add_argument("--adams", type=int, default=None)
    ap.add_argument("--lbfgs", type=int, default=None)
    ap.add_argument("--anneal", action="store_true", default=None)
    ap.add_argument("--hard", action="store_true", default=None)
    ap.add_argument("--sa", action="store_true", default=None)
    ap.add_argument("--rar", action="store_true", default=None)
    ap.add_argument("--rar_every", type=int, default=None)
    ap.add_argument("--rar_add", type=int, default=None)
    ap.add_argument("--rar_probe", type=int, default=None)
    ap.add_argument("--das", action="store_true", default=None)
    ap.add_argument("--das_every", type=int, default=None)
    ap.add_argument("--das_probe", type=int, default=None)
    ap.add_argument("--eval_pts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--drop", type=float, default=None,
                    help="MC-dropout rate (arch=dropout)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Fourier-feature scale (arch=fourier)")
    ap.add_argument("--n_freqs", type=int, default=None,
                    help="Fourier feature count (arch=fourier)")
    ap.add_argument("--w0", type=float, default=None,
                    help="SIREN first-layer / activation scale (arch=siren)")
    ap.add_argument("--wall_clock", type=float, default=None)
    ap.add_argument("--D", type=float, default=None)
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--save", default=None, help="checkpoint path (after run)")
    ap.add_argument("--load", default=None, help="checkpoint path (resume/eval)")
    ap.add_argument("--eval_only", action="store_true", default=None,
                    help="load a checkpoint and only evaluate (no training)")
    args = vars(ap.parse_args(argv))

    cfg = copy.deepcopy(DEFAULTS)
    import yaml
    for path in args.pop("cfg"):
        with open(path) as fh:
            cfg.update({k: v for k, v in (yaml.safe_load(fh) or {}).items()
                        if v is not None})
    cfg.update({k: v for k, v in args.items() if v is not None})

    from pinnacle_ext import checkpoint as CK
    prob = net = None
    if cfg.get("load"):
        # a checkpoint defines the problem/config; explicit CLI flags override.
        explicit = {k for k, v in args.items() if v is not None}
        recons = dict(cfg)
        if "problem" not in explicit:
            recons.pop("problem", None)          # let the checkpoint say
        net, prob, ck = CK.build_from_checkpoint(cfg["load"], cfg=recons)
        stored = ck.get("cfg") or {}
        for k, v in stored.items():
            if k not in ("save", "load", "eval_only") and k not in explicit:
                cfg[k] = v
        print(f"loaded {cfg['load']} ({ck.get('metrics', {}).get('rel_l2', '?')})")
    if cfg.get("eval_only") is True:
        cfg["adams"], cfg["lbfgs"] = 0, 0

    met = train_one(cfg, prob=prob, net=net)

    print("\n===== Evaluation (%s) =====" % met["problem"])
    for key in ("rel_l2", "linf", "max_res", "mode"):
        print(f"{key:<12} {met.get(key, 'n/a')}")
    if met.get("saved"):
        print(f"saved {met['saved']}")
    return met


if __name__ == "__main__":
    main()