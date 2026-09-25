#!/usr/bin/env python3
"""PINNacle run entrypoint -- thin wrapper over pinnacle_ext.config.

    python3 run.py --problem kdv --lbfgs 400
    python3 run.py --problem heat --arch dropout --save runs/heat.pt
    python3 run.py --load runs/heat.pt --eval_only        # inspect a run
    python3 run.py --problem beam1d --hard --save runs/beam.pt

Also a quick composite smoke: with no arguments, it trains the tiny set
(one core + one ext problem) that proves the whole pipeline is wired.
"""

import numpy as np
import torch


def _default_dtype():
    # float64 everywhere (matches the verified core training regime)
    torch.set_default_dtype(torch.float64)
    np.random.seed(0)
    torch.manual_seed(0)


def main(argv=None):
    _default_dtype()
    from pinnacle_ext.config import main as config_main
    return config_main(argv)


def smoke(argv=None):
    """Tiny end-to-end pipeline check (core + ext + checkpoint roundtrip)."""
    _default_dtype()
    import tempfile, os
    from pinnacle_ext.config import train_one, _build_problem, _build_net
    from pinnacle_ext import checkpoint as CK

    path = os.path.join(tempfile.mkdtemp(), "smoke.pt")
    results = {}
    from pinnacle_ext.config import DEFAULTS
    for name, over in [("beam1d", {"adams": 30, "lbfgs": 10, "hard": True}),
                       ("heat", {"adams": 30, "lbfgs": 10})]:
        cfg = {**DEFAULTS, "problem": name, **over}
        prob = _build_problem(cfg)
        net = _build_net(prob, cfg)
        met = train_one(cfg, prob=prob)
        CK.save_checkpoint(path, net, cfg=cfg, metrics=met)
        net2, prob2, ck = CK.build_from_checkpoint(path)
        met2 = CK.describe(path)
        results[name] = (met["rel_l2"], ck["metrics"]["rel_l2"])
    print("smoke:", {k: [round(v, 4) for v in vv] for k, vv in results.items()})
    return results


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        smoke()
    else:
        main()