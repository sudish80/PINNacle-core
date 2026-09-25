"""pinnacle_ext.checkpoint -- save / load trained PINNs with full provenance.

Checkpoints are single torch files with four payloads:
    state_dict  : network weights (the wrapped/hard net if one was trained)
    cfg         : the merged config that produced the run (for `nn_params/..`)
    metrics     : evaluation metrics at save time
    prob        : the problem NAME (registry key), to rebuild on load

Everything is additive: no core module is touched.
"""

import json
import os

import torch

import pinn_solver as P
from pinnacle_ext import arch


def save_checkpoint(path, net, cfg=None, metrics=None, extra=None):
    """Persist a trained net (+ optional config/metrics) to `path`.

    Parent directories are created if missing; returns the path.
    """
    parent = os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else "."
    os.makedirs(parent, exist_ok=True)
    payload = {"state_dict": net.state_dict(),
               "cfg": cfg, "metrics": metrics, "extra": extra}
    torch.save(payload, path)
    return path


def load_into(net, path):
    """Load a checkpoint's state_dict into an existing net (in place)."""
    ck = torch.load(path, map_location=P.DEVICE, weights_only=False)
    sd = ck["state_dict"]
    # wrappers (HardConstrainedPINN / HardNeumann1D...) store keys under a
    # leading module prefix; drop it so a plain net can be restored.
    if sd and all(k.startswith("net.") for k in sd):
        sd = {k[4:]: v for k, v in sd.items()}
    net.load_state_dict(sd)
    return ck


def build_from_checkpoint(path, cfg=None):
    """Rebuild net + problem + metrics from a checkpoint.

    Uses the stored cfg to know the problem/arch (falling back to the passed
    cfg when the checkpoint predates config persistence).  Returns
    (net, prob, meta) ready for evaluation or continued training.
    """
    ck = torch.load(path, map_location=P.DEVICE, weights_only=False)
    stored_cfg = ck.get("cfg") or {}
    opt_cfg = cfg or {}
    merged = {**stored_cfg, **opt_cfg}
    if "problem" not in merged:
        raise ValueError("checkpoint carries no problem; pass --cfg with one")

    from pinnacle_ext.config import _build_problem, _build_net
    prob = _build_problem(merged)
    net = _build_net(prob, merged).to(P.DEVICE)
    sd = ck["state_dict"]
    if sd and all(k.startswith("net.") for k in sd):
        sd = {k[4:]: v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net, prob, ck


def describe(path):
    """Human-readable summary of a checkpoint (cfg + metrics, no weights)."""
    ck = torch.load(path, map_location=P.DEVICE, weights_only=False)
    cfg = ck.get("cfg") or {}
    met = ck.get("metrics") or {}
    return {"problem": cfg.get("problem"),
            "arch": cfg.get("arch"), "activation": cfg.get("activation"),
            "rel_l2": met.get("rel_l2"),
            "rel_l2_active": met.get("rel_l2_active"),
            "max_res": met.get("max_res"), "mode": met.get("mode"),
            "nn_params": met.get("nn_params"),
            "path": path}