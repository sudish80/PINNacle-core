"""PINNacle scaling layer package.

Composite, additive extension of the verified `pinn_solver`:

    pinnacle_ext/problems.py     +10 PDEs into the shared registry (sympy-certified)
    pinnacle_ext/certified.py    symbolic certification gate (residual == 0)
    pinnacle_ext/engine.py      3rd-order space + vector-output engine & looping
    pinnacle_ext/arch.py        advanced architectures (Fourier, SIREN, residual,
                                multiscale, adaptive slope, vector)
    pinnacle_ext/methods.py     DAS / SA-PINN / hard-Neumann / resampling
    pinnacle_ext/config.py      layered YAML config + runnable orchestration
"""

from pinnacle_ext import problems, engine, arch, methods  # noqa: F401

__version__ = "1.3.0"