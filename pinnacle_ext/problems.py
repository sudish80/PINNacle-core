"""pinnacle_ext.problems -- +10 PDEs registered into the shared registry.

Every problem carries an exact / manufactured reference.  Static problems are
encoded as degenerate-time IBVPs (t0 == t1) so the whole verified machinery
applies unchanged.  All references are sympy-certified by
pinnacle_ext.diagnostics.certified (see the certification suite), not merely
plausible.

Scalar problems keep the CORE residual contract  residual(u, der, x, t)
so they drop straight into the verified pinn_solver training loop.  The two
non-standard members set explicit capability flags:
    kdv          space_orders=3  (needs u_xxx -- ext engine)
    kovasznay_ns n_out=3         (vector u,v,p -- ext engine)
"""

import numpy as np
import torch

import pinn_solver as P


class ScaleProblem(P.Problem):
    """Problem + capability flags for the ext engine.

    Extra fields (all optional, safe defaults keep core compatibility):
      n_out        : number of output channels (u,v,p -> 3)
      space_orders : max spatial derivative order demanded by the residual
    """

    def __init__(self, *args, n_out=1, space_orders=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_out = n_out
        self.space_orders = space_orders


# --------------------------------------------------------------------- #
# 1. Advection (linear hyperbolic, exact traveling sinusoid)
# --------------------------------------------------------------------- #
def problem_advection(c=1.0):
    def residual(u, der, x, t):
        return der["u_t"] + c * der["u_x"][0]

    def ic(x):
        return torch.sin(np.pi * x[:, :1])

    def exact(x, t):
        return np.sin(np.pi * (np.asarray(x, float) - c * np.asarray(t, float)))

    # box [0,2]:  u(0,t) = -sin(pi c t) = u(2,t)  (same value), Dirichlet ok
    return ScaleProblem(
        name="advection", d=1, a=[0.0], b=[2.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=ic, bc_val=lambda x, t: ic(x),
        exact_fn=exact, params={"c": c}, default_layers=[40, 40, 40],
    )


# --------------------------------------------------------------------- #
# 2. Convection-diffusion (exact decaying traveling wave)
# --------------------------------------------------------------------- #
def problem_convection_diffusion(D=0.02, c=0.5):
    k = 1.0
    lam = D * (2.0 * np.pi * k) ** 2

    def residual(u, der, x, t):
        return der["u_t"] + c * der["u_x"][0] - D * der["u_xx"][0]

    def exact(x, t):
        xa, ta = np.asarray(x, float), np.asarray(t, float)
        return np.exp(-lam * ta) * np.sin(2.0 * np.pi * k * (xa - c * ta))

    def bc(x, t):
        # sin(0-ct) and sin(2π(1-ct)) coincide -> equal Dirichlet values
        return torch.tensor(exact(x[:, :1].detach().cpu().numpy(),
                                  t[:, :1].detach().cpu().numpy()),
                            dtype=torch.get_default_dtype(), device=x.device)

    return ScaleProblem(
        name="convection_diffusion", d=1, a=[0.0], b=[1.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=lambda x: torch.sin(2.0 * np.pi * x[:, :1]),
        bc_val=bc, exact_fn=exact,
        params={"D": D, "c": c, "lam": lam},
        default_layers=[40, 40, 40],
    )


# --------------------------------------------------------------------- #
# 3. Allen-Cahn (manufactured forcing, nonlinear bistable)
# --------------------------------------------------------------------- #
def problem_allen_cahn(D=0.001):
    """u_t - D u_xx - u + u^3 = f  with manufactured u* = cos(pi x) e^{-t}."""
    def exact(x, t):
        return np.cos(np.pi * np.asarray(x, float)) * np.exp(-np.asarray(t, float))

    def residual(u, der, x, t):
        return der["u_t"] - D * der["u_xx"][0] - u + u ** 3

    def force(x, t):
        e = np.exp(-np.asarray(t, float))
        c = np.cos(np.pi * np.asarray(x, float))
        return (D * np.pi ** 2 - 2.0) * c * e + c ** 3 * e ** 3

    return ScaleProblem(
        name="allen_cahn", d=1, a=[-1.0], b=[1.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=lambda x: torch.cos(np.pi * x[:, :1]),
        bc_val=lambda x, t: torch.cos(np.pi * x[:, :1]) *
                            torch.exp(-t * torch.ones_like(x[:, :1])),
        force_fn=force, exact_fn=exact,
        params={"D": D}, default_layers=[40, 40, 40],
    )


# --------------------------------------------------------------------- #
# 4. KdV (third order in space -- ext derivative engine)  exact 1-soliton
# --------------------------------------------------------------------- #
def problem_kdv(c=1.0, x0=-1.0):
    """u_t + 6 u u_x + u_xxx = 0 ;  u = (c/2) sech^2((sqrt(c)/2)(x-x0-ct))."""
    k = np.sqrt(c) / 2.0

    def exact(x, t):
        z = (np.asarray(x, float) - x0 - c * np.asarray(t, float)) * k
        return (c / 2.0) / np.cosh(z) ** 2

    def residual(u, der, x, t):
        return der["u_t"] + 6.0 * u * der["u_x"][0] + der["u_xxx"][0]

    def bc(x, t):
        return torch.tensor(exact(x[:, :1].detach().cpu().numpy(),
                                  t[:, :1].detach().cpu().numpy()),
                            dtype=torch.get_default_dtype(), device=x.device)

    def ic(x):
        return torch.tensor(exact(x[:, :1].detach().cpu().numpy(),
                                  np.zeros_like(x[:, :1].detach().cpu().numpy())),
                            dtype=torch.get_default_dtype(), device=x.device)

    return ScaleProblem(
        name="kdv", d=1, a=[-8.0], b=[8.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=ic, bc_val=bc, exact_fn=exact,
        params={"c": c, "x0": x0}, space_orders=3,
        default_layers=[50, 50, 50],
    )


# --------------------------------------------------------------------- #
# 5. Sine-Gordon (2nd order in time; exact topological kink)
# --------------------------------------------------------------------- #
def problem_sine_gordon(v=0.5):
    """u_tt - u_xx + sin(u) = 0 ;  u = 4 atan(exp((x - vt)/sqrt(1-v^2)))."""
    g = np.sqrt(1.0 - v ** 2)

    def exact(x, t):
        return 4.0 * np.arctan(np.exp((np.asarray(x, float) - v * np.asarray(t, float)) / g))

    def ic(x):
        return 4.0 * torch.atan(torch.exp(x[:, :1] / g))

    def ic_t(x):
        # d/dt u|0 = -(2 v / g) sech(x / g)
        return -(2.0 * v / g) / torch.cosh(x[:, :1] / g)

    def bc(x, t):
        return torch.tensor(exact(x[:, :1].detach().cpu().numpy(),
                                  t[:, :1].detach().cpu().numpy()),
                            dtype=torch.get_default_dtype(), device=x.device)

    return ScaleProblem(
        name="sine_gordon", d=1, a=[-6.0], b=[6.0], t0=0.0, t1=1.0,
        residual=lambda u, der, x, t: der["u_tt"] - der["u_xx"][0] + torch.sin(u),
        ic_fn=ic, bc_val=bc, ic_t_fn=ic_t, time_orders=2, ic_orders=2,
        exact_fn=exact, params={"v": v, "gamma": g},
        default_layers=[50, 50, 50],
    )


# --------------------------------------------------------------------- #
# 6. 2-D heat (separable exact)
# --------------------------------------------------------------------- #
def problem_heat2d(D=0.1):
    def residual(u, der, x, t):
        return der["u_t"] - D * (der["u_xx"][0] + der["u_xx"][1])

    def exact(x, t):
        return (np.sin(np.pi * np.asarray(x)[:, 0]) * np.sin(np.pi * np.asarray(x)[:, 1])
                * np.exp(-2.0 * np.pi ** 2 * D * np.asarray(t, float).ravel()))

    def ic(x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    return ScaleProblem(
        name="heat2d", d=2, a=[0.0, 0.0], b=[1.0, 1.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=ic, bc_val=lambda x, t: torch.zeros_like(x[:, :1]),
        exact_fn=exact, params={"D": D}, default_layers=[50, 50, 50],
    )


# --------------------------------------------------------------------- #
# 7. 2-D wave (2nd order in time; separable exact)
# --------------------------------------------------------------------- #
def problem_wave2d(c=1.0):
    w = np.sqrt(2.0) * np.pi * c

    def residual(u, der, x, t):
        return der["u_tt"] - c ** 2 * (der["u_xx"][0] + der["u_xx"][1])

    def exact(x, t):
        return (np.sin(np.pi * np.asarray(x)[:, 0]) * np.sin(np.pi * np.asarray(x)[:, 1])
                * np.cos(w * np.asarray(t, float).ravel()))

    def ic(x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    return ScaleProblem(
        name="wave2d", d=2, a=[0.0, 0.0], b=[1.0, 1.0], t0=0.0, t1=1.0,
        residual=residual, ic_fn=ic,
        ic_t_fn=lambda x: torch.zeros_like(x[:, :1]),
        bc_val=lambda x, t: torch.zeros_like(x[:, :1]),
        time_orders=2, ic_orders=2, exact_fn=exact, params={"c": c},
        default_layers=[50, 50, 50],
    )


# --------------------------------------------------------------------- #
# 8. 3-D Poisson (static; manufactured eigenfunction  sin sin sin)
# --------------------------------------------------------------------- #
def problem_poisson3d(lo=0.0, hi=1.0):
    """Manufactured 3-D Poisson with off-eigen λ and a forcing term.

    The homogeneous form -Δu = 3π²u admits the trivial solution u≡0, which a
    low-frequency PINN readily settles into.  Using λ=4π² (≠3π², no
    eigenfunction) and forcing f=(3π²-λ)·u* breaks that degeneracy.  The
    residual is normalized by λ for conditioning, keeping the certified
    identity operator - f ≡ 0 exactly.
    """
    lam = 4.0 * np.pi ** 2
    fcoef = 3.0 * np.pi ** 2 - lam          # == -π²
    force_coef = fcoef / lam

    def residual(u, der, x, t):
        return (-(der["u_xx"][0] + der["u_xx"][1] + der["u_xx"][2])
                - lam * u) / lam

    def force(x, t):
        xx = np.asarray(x, float)
        return force_coef * (np.sin(np.pi * xx[:, 0]) * np.sin(np.pi * xx[:, 1])
                             * np.sin(np.pi * xx[:, 2]))

    def ic(x):
        return (torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])
                * torch.sin(np.pi * x[:, 2:3]))

    def exact(x, t):
        return (np.sin(np.pi * np.asarray(x)[:, 0]) * np.sin(np.pi * np.asarray(x)[:, 1])
                * np.sin(np.pi * np.asarray(x)[:, 2]))

    return ScaleProblem(
        name="poisson3d", d=3, a=[lo] * 3, b=[hi] * 3, t0=0.0, t1=0.0,
        residual=residual, force_fn=force, ic_fn=ic,
        bc_val=lambda x, t: torch.zeros_like(x[:, :1]),
        exact_fn=exact, params={"f_max": 3.0 * np.pi ** 2, "lambda": lam},
        default_layers=[60, 60, 60],
    )


# --------------------------------------------------------------------- #
# 9. Helmholtz 2-D (static; exact eigenfunction, m=3 n=4)
# --------------------------------------------------------------------- #
def problem_helmholtz(m=3, n=4):
    """Manufactured Helmholtz: Δu + λu = f with a non-eigenfrequency λ.

    The naive choice λ=(m^2+n^2)π^2 turns u* = sin(mπx)sin(nπy) into an exact
    eigenfunction, i.e. the homogeneous PDE+Dirichlet admits the trivial
    solution u≡0 -- a trap the net happily finds.  Using λ=(m+n)^2 π^2 and a
    manufactured forcing f=(2mn)π^2 u* breaks that symmetry while keeping the
    same zero-Dirichlet boundary data.
    """
    kap2 = (m + n) ** 2 * np.pi ** 2
    fcoef = (kap2 - (m ** 2 + n ** 2) * np.pi ** 2)  # == 2mn π^2

    force_coef = fcoef / kap2                  # normalized forcing: f/k^2

    def residual(u, der, x, t):
        return (der["u_xx"][0] + der["u_xx"][1] + kap2 * u) / kap2

    def force(x, t):
        xx = np.asarray(x, float)
        return force_coef * (np.sin(m * np.pi * xx[:, 0])
                             * np.sin(n * np.pi * xx[:, 1]))

    def ic(x):
        return torch.sin(m * np.pi * x[:, 0:1]) * torch.sin(n * np.pi * x[:, 1:2])

    def exact(x, t):
        return (np.sin(m * np.pi * np.asarray(x)[:, 0])
                * np.sin(n * np.pi * np.asarray(x)[:, 1]))

    return ScaleProblem(
        name="helmholtz", d=2, a=[0.0, 0.0], b=[1.0, 1.0], t0=0.0, t1=0.0,
        residual=residual, force_fn=force, ic_fn=ic,
        bc_val=lambda x, t: torch.zeros_like(x[:, :1]),
        exact_fn=exact,
        params={"k2": kap2, "forcing": fcoef, "normalized": "residual/k^2"},
        default_layers=[50, 50, 50],
    )


# --------------------------------------------------------------------- #
# 10. Manufactured incompressible Navier-Stokes (steady, vector u,v,p)
# --------------------------------------------------------------------- #
def problem_kovasznay_ns(nu=0.02):
    """Steady incompressible NS on [0,1]^2 with analytic manufactured state.

        u = -sin(pi x) cos(pi y)
        v =  cos(pi x) sin(pi y)
        p =  0.5 sin(2pi x) cos(2pi y)

    div(u,v) = 0 identically; pressure balances the momentum operator and the
    (analytic, sympy-derived at import) forcing makes momentum exactly zero.
    Residual contract is VECTOR: residual(us, ds, x, t) -> r (N,1) or (N,3).
    """
    from pinnacle_ext.certified import kovasznay_force
    f = kovasznay_force(nu)

    def residual(us, ds, x, t):
        uu, vv, pp = us
        du, dv, dp = ds
        r_x = (uu * du["u_x"][0] + vv * du["u_x"][1] + dp["u_x"][0]
               - nu * (du["u_xx"][0] + du["u_xx"][1]))
        r_y = (uu * dv["u_x"][0] + vv * dv["u_x"][1] + dp["u_x"][1]
               - nu * (dv["u_xx"][0] + dv["u_xx"][1]))
        r_c = du["u_x"][0] + dv["u_x"][1]
        return torch.cat([r_x, r_y, r_c], dim=1)     # (N, 3)

    def force(x, t):
        # returns (N,3): [fx, fy, 0] so r (N,3) - f aligns with momentum+cont
        xa = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, float)
        a = xa[:, 0:1]; b_ = xa[:, 1:2]
        fab = f(a, b_)                            # (N,2)
        return np.c_[fab, np.zeros(a.shape[0])]   # (N,3)

    def bc(x, t):
        xa = x.detach()
        return torch.cat([-torch.sin(np.pi * xa[:, 0:1]) * torch.cos(np.pi * xa[:, 1:2]),
                          torch.cos(np.pi * xa[:, 0:1]) * torch.sin(np.pi * xa[:, 1:2]),
                          torch.zeros_like(xa[:, :1])], dim=1)

    def ic(x):
        return bc(x, None)

    def exact(x, t):
        xa = np.asarray(x, float)
        return np.stack([-np.sin(np.pi * xa[:, 0]) * np.cos(np.pi * xa[:, 1]),
                         np.cos(np.pi * xa[:, 0]) * np.sin(np.pi * xa[:, 1]),
                         0.5 * np.sin(2 * np.pi * xa[:, 0]) * np.cos(2 * np.pi * xa[:, 1])], axis=1)

    return ScaleProblem(
        name="kovasznay_ns", d=2, a=[0.0, 0.0], b=[1.0, 1.0], t0=0.0, t1=0.0,
        residual=residual, ic_fn=ic, bc_val=bc,
        force_fn=force, exact_fn=exact,
        params={"nu": nu, "Re": 1.0 / nu}, n_out=3,
        default_layers=[60, 60, 60],
    )


# --------------------------------------------------------------------- #
# 11. Euler-Bernoulli beam (static 4th order, clamped ends)
# --------------------------------------------------------------------- #
def problem_beam1d(lo=0.0, hi=1.0):
    """Considered: clamped-clamped Euler-Bernoulli beam with loading q.

    The 4th-order operator u_xxxx = q is special.  Two Dirichlet BCs alone
    leave the biharmonic problem under-determined (a continuum of solutions),
    so clamped ends must fix BOTH u and u_x.  The manufactured reference
        u* = 12 x^2 (1-x)^2 ,   q = u*_xxxx = 288
    satisfies u(0)=u(1)=0 and u_x(0)=u_x(1)=0, is nonzero on the interior
    (healthy 'active' error metric) and is enforced via the new optional
    ``bc_slope_fn`` term in compute_losses_ext.  Residual normalized by q.
    """
    q = 288.0

    def residual(u, der, x, t):
        return der["u_xxxx"][0] / q

    def force(x, t):
        n = np.asarray(x, float).shape[0]
        return np.ones(n)

    def ic(x):
        return 12.0 * x[:, 0:1] ** 2 * (1.0 - x[:, 0:1]) ** 2

    def exact(x, t):
        xx = np.asarray(x, float)[:, 0]
        return 12.0 * xx ** 2 * (1.0 - xx) ** 2

    def bc_slope(x, t=None):
        if np.isscalar(x):
            return 0.0
        n = np.asarray(x, float).shape[0]
        return np.zeros(n)

    return ScaleProblem(
        name="beam1d", d=1, a=[lo], b=[hi], t0=0.0, t1=0.0,
        residual=residual, force_fn=force, ic_fn=ic,
        bc_val=lambda x, t: torch.zeros_like(x[:, :1]),
        bc_slope_fn=bc_slope, exact_fn=exact, space_orders=4,
        params={"q": q, "normalized": "residual/q"}, default_layers=[50, 50, 50],
    )


# --------------------------------------------------------------------- #
# 12. Cahn-Hilliard (linear model-B / biharmonic flow, 4th order)
# --------------------------------------------------------------------- #
def problem_cahn_hilliard(eps=0.1):
    """Linearised Cahn-Hilliard / biharmonic flow:  u_t + eps^2 u_xxxx = 0.

    Manufactured clamped reference
        u* = sin^2(pi x) e^{-t}
    satisfies u(0)=u(1)=0, u_x(0)=u_x(1)=0 and the PDE with the exact forcing
         f = u_t* + eps^2 u_xxxx* = -(sin^2(pi x) + 8 pi^4 eps^2 cos(2pi x)) e^{-t}
    Residual is normalized by the 4th-order coefficient so the operator is
    O(1); bc_slope_fn enforces the clamped ends.  Sympy certifies f.
    """
    k = eps ** 2 * 8.0 * np.pi ** 4

    def residual(u, der, x, t):
        return (der["u_t"] + eps ** 2 * der["u_xxxx"][0]) / k

    def force(x, t):
        xa = np.asarray(x, float)[:, 0:1]
        ta = np.asarray(t, float)
        return (-np.sin(np.pi * xa) ** 2 * np.exp(-ta)
                - 8.0 * np.pi ** 4 * eps ** 2 * np.cos(2.0 * np.pi * xa)
                * np.exp(-ta)) / k

    def ic(x):
        return torch.sin(np.pi * x[:, 0:1]) ** 2

    def exact(x, t):
        return np.sin(np.pi * np.asarray(x, float)[:, 0:1]) ** 2 \
            * np.exp(-np.asarray(t, float))

    def bc_slope(x, t=None):
        if np.isscalar(x):
            return 0.0
        n = np.asarray(x, float).shape[0]
        return np.zeros(n)

    return ScaleProblem(
        name="cahn_hilliard", d=1, a=[0.0], b=[1.0], t0=0.0, t1=1.0,
        residual=residual, force_fn=force, ic_fn=ic,
        bc_val=lambda x, t: torch.zeros_like(x[:, :1]),
        bc_slope_fn=bc_slope, exact_fn=exact, space_orders=4,
        params={"eps": eps, "normalized": "residual/(8 pi^4 eps^2)"},
        default_layers=[50, 50, 50],
    )


REGISTRY = {name.replace("problem_", ""): fn for name, fn in globals().items()
            if name.startswith("problem_") and callable(fn)}

for name, builder in REGISTRY.items():
    P.REGISTRY[name] = builder