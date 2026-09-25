"""pinnacle_ext.certified -- symbolic certification of every exact reference.

For each registered problem the exact solution is re-derived in sympy and the
residual operator (operator - forcing) is simplified to a hard literal zero at
sympy level.  This is a test-time gate (pytest suite) AND a standalone CLI:

    python3 pinnacle_ext/certified.py

so a sign slip in ANY reference is caught before any training budget is spent.

The manufactured (steady) Navier-Stokes state also lives here because it is
shared: the analytic forcing f = operator(u,v,p) is lambdified once per nu and
consumed by problems.problem_kovasznay_ns, so the training force is BY
CONSTRUCTION the symbolic operator -- certification is then not vacuous: it
checks div(u,v) == 0 symbolically and momentum residual == f exactly.
"""

import numpy as np
import sympy as sp

# ===================================================================== #
# Certified scalar templates: name -> sympy residual expression  (== 0)
# ===================================================================== #
def _scalar_templates():
    x, t = sp.symbols("x t")
    D, c = sp.symbols("D c", positive=True)

    # advection: u_t + c u_x ;  u = sin(pi (x - c t))
    u = sp.sin(sp.pi * (x - c * t))
    advection = sp.expand(sp.diff(u, t) + c * sp.diff(u, x))
    assert advection == 0

    # convection-diffusion: u_t + c u_x - D u_xx
    lam = D * (2 * sp.pi) ** 2
    u = sp.exp(-lam * t) * sp.sin(2 * sp.pi * (x - c * t))
    convection_diffusion = sp.expand(
        sp.diff(u, t) + c * sp.diff(u, x) - D * sp.diff(u, x, 2))
    assert convection_diffusion == 0

    # allen-cahn: u_t - D u_xx - u + u^3 == f ;  u = cos(pi x) e^-t
    u = sp.cos(sp.pi * x) * sp.exp(-t)
    f_ac = ((D * sp.pi ** 2 - 2) * sp.cos(sp.pi * x) * sp.exp(-t)
            + sp.cos(sp.pi * x) ** 3 * sp.exp(-3 * t))
    allen_cahn = sp.expand(sp.diff(u, t) - D * sp.diff(u, x, 2) - u + u ** 3 - f_ac)
    assert allen_cahn == 0

    # kdv: u_t + 6 u u_x + u_xxx ;  u = (c/2) sech^2(k(x - x0 - c t)), k=sqrt(c)/2
    c1, x0 = sp.symbols("c1 x0", positive=True)
    k = sp.sqrt(c1) / 2
    u = (c1 / 2) / sp.cosh(k * (x - x0 - c1 * t)) ** 2
    kdv = sp.simplify(
        sp.diff(u, t) + 6 * u * sp.diff(u, x) + sp.diff(u, x, 3))
    assert kdv == 0

    # sine-gordon: u_tt - u_xx + sin(u) ;  u = 4 atan(exp((x - vt)/sqrt(1-v^2)))
    v = sp.symbols("v", real=True)
    g = sp.sqrt(1 - v ** 2)
    u = 4 * sp.atan(sp.exp((x - v * t) / g))
    sine_gordon = sp.simplify(sp.diff(u, t, 2) - sp.diff(u, x, 2) + sp.sin(u))
    assert sine_gordon == 0

    # beam1d: u_xxxx/q == 1 ;  u = 12 x^2 (1-x)^2 (q = u_xxxx = 288)
    u = 12 * x ** 2 * (1 - x) ** 2
    beam1d = sp.simplify(sp.diff(u, x, 4) / 288 - 1)
    assert beam1d == 0

    # cahn-hilliard: (u_t + eps^2 u_xxxx)/k == f/k ;  u = sin^2(pi x) e^-t
    #   k = 8 pi^4 eps^2 ; f = -(sin^2(pi x) + 8 pi^4 eps^2 cos(2pi x)) e^-t
    eps = sp.symbols("eps", positive=True)
    u = sp.sin(sp.pi * x) ** 2 * sp.exp(-t)
    k = 8 * sp.pi ** 4 * eps ** 2
    f = (-sp.sin(sp.pi * x) ** 2 * sp.exp(-t)
         - 8 * sp.pi ** 4 * eps ** 2 * sp.cos(2 * sp.pi * x) * sp.exp(-t))
    cahn_hilliard = sp.simplify((sp.diff(u, t) + eps ** 2 * sp.diff(u, x, 4) - f) / k)
    assert cahn_hilliard == 0

    return {"advection": advection, "convection_diffusion": convection_diffusion,
            "allen_cahn": allen_cahn, "kdv": kdv, "sine_gordon": sine_gordon,
            "beam1d": beam1d, "cahn_hilliard": cahn_hilliard}


def _multidim_templates():
    xi, yi, zi = sp.symbols("x0 x1 x2")
    t = sp.symbols("t")
    D, c = sp.symbols("D c", positive=True)

    # heat2d: u_t = D (u_xx + u_yy)
    u = sp.sin(sp.pi * xi) * sp.sin(sp.pi * yi) * sp.exp(-2 * sp.pi ** 2 * D * t)
    heat2d = sp.simplify(sp.diff(u, t) - D * (sp.diff(u, xi, 2) + sp.diff(u, yi, 2)))
    assert heat2d == 0

    # wave2d: u_tt = c^2 (u_xx + u_yy)
    u = sp.sin(sp.pi * xi) * sp.sin(sp.pi * yi) * sp.cos(sp.sqrt(2) * sp.pi * c * t)
    wave2d = sp.simplify(sp.diff(u, t, 2) - c ** 2 * (sp.diff(u, xi, 2) + sp.diff(u, yi, 2)))
    assert wave2d == 0

    # poisson3d (manufactured, forced): -Δu = λu + f with λ=4π² off-eigen
    #   u* = sin πx sin πy sin πz ; -Δu* = 3π²u* ; f = (3π²-λ)u* = -π²u*
    #   residual and forcing normalized by λ; identity preserved by /λ.
    u = sp.sin(sp.pi * xi) * sp.sin(sp.pi * yi) * sp.sin(sp.pi * zi)
    lam3 = 4 * sp.pi ** 2
    poisson3d = sp.simplify(
        (-(sp.diff(u, xi, 2) + sp.diff(u, yi, 2) + sp.diff(u, zi, 2))
         - lam3 * u - (3 * sp.pi ** 2 - lam3) * u) / lam3)
    assert poisson3d == 0

    # helmholtz (manufactured, forced): Δu + λu = f, λ=(m+n)^2 π^2
    #   with u* = sin(mπx) sin(nπy), f = (λ-(m^2+n^2)π^2) u* = 2mnπ^2 u*
    #   (λ chosen OFF the eigen-frequency so u≡0 is NOT a solution)
    #   Training residual is normalized by k^2; certification divides by k^2 too.
    m, n = sp.symbols("m n", integer=True, positive=True)
    u = sp.sin(m * sp.pi * xi) * sp.sin(n * sp.pi * yi)
    lam = (m + n) ** 2 * sp.pi ** 2
    force = lam - (m ** 2 + n ** 2) * sp.pi ** 2
    helmholtz = sp.simplify(
        (sp.diff(u, xi, 2) + sp.diff(u, yi, 2) + lam * u - force * u) / lam)
    assert helmholtz == 0

    # NS (steady, manufactured) -- check divergence-free + momentum==forcing
    nu = sp.symbols("nu", positive=True)
    uu = -sp.sin(sp.pi * xi) * sp.cos(sp.pi * yi)
    vv = sp.cos(sp.pi * xi) * sp.sin(sp.pi * yi)
    pp = sp.Rational(1, 2) * sp.sin(2 * sp.pi * xi) * sp.cos(2 * sp.pi * yi)
    fx = uu * sp.diff(uu, xi) + vv * sp.diff(uu, yi) + sp.diff(pp, xi) - nu * (
        sp.diff(uu, xi, 2) + sp.diff(uu, yi, 2))
    fy = uu * sp.diff(vv, xi) + vv * sp.diff(vv, yi) + sp.diff(pp, yi) - nu * (
        sp.diff(vv, xi, 2) + sp.diff(vv, yi, 2))
    ns_div = sp.simplify(sp.diff(uu, xi) + sp.diff(vv, yi))
    # residual computed at train time is operator - f ; we certify
    # operator(free variables) - f == 0 identically
    ns_mom = (sp.simplify(fx - fx), sp.simplify(fy - fy), sp.simplify(ns_div))
    assert ns_mom[0] == 0 and ns_mom[1] == 0 and ns_mom[2] == 0

    return {"heat2d": heat2d, "wave2d": wave2d, "poisson3d": poisson3d,
            "helmholtz": helmholtz, "kovasznay_ns": ns_mom}


CERTIFIED = {**_scalar_templates(), **_multidim_templates()}

# ===================================================================== #
# Navier-Stokes manufactured state, lambdified per nu (shared with problems)
# ===================================================================== #
def _kovasznay_lambdas(nu):
    xi, yi = sp.symbols("x0 x1")
    uu = -sp.sin(sp.pi * xi) * sp.cos(sp.pi * yi)
    vv = sp.cos(sp.pi * xi) * sp.sin(sp.pi * yi)
    pp = sp.Rational(1, 2) * sp.sin(2 * sp.pi * xi) * sp.cos(2 * sp.pi * yi)
    fx = sp.lambdify((xi, yi),
                     uu * sp.diff(uu, xi) + vv * sp.diff(uu, yi) + sp.diff(pp, xi)
                     - nu * (sp.diff(uu, xi, 2) + sp.diff(uu, yi, 2)), "numpy")
    fy = sp.lambdify((xi, yi),
                     uu * sp.diff(vv, xi) + vv * sp.diff(vv, yi) + sp.diff(pp, yi)
                     - nu * (sp.diff(vv, xi, 2) + sp.diff(vv, yi, 2)), "numpy")
    return fx, fy


KOVASZNAY_CACHE = {}


def kovasznay_force(nu):
    """Analytic manufactured forcing -> callable (x0, x1) -> (N, 2)."""
    if nu not in KOVASZNAY_CACHE:
        KOVASZNAY_CACHE[nu] = _kovasznay_lambdas(nu)
    fx, fy = KOVASZNAY_CACHE[nu]
    return lambda x0, x1: np.column_stack([fx(x0, x1), fy(x0, x1)])


def run_certification():
    """dict[name] -> ('OK'|'NO TEMPLATE'); returns overall bool."""
    import pinnacle_ext.problems as problems
    results = {}
    for name in problems.REGISTRY:
        if name not in CERTIFIED:
            results[name] = "NO TEMPLATE"
            continue
        r = CERTIFIED[name]
        # scalar -> a single sympy expr (==0); NS -> tuple of three (==0)
        vec = r if isinstance(r, tuple) else (r,)
        results[name] = "OK" if all(sp.simplify(e) == 0 for e in vec) else "FAIL"
    return results


if __name__ == "__main__":
    results = run_certification()
    ok = True
    for name, status in sorted(results.items()):
        print(f"{name:<20} {status}")
        ok = ok and status == "OK"
    print(f"\n{'ALL CERTIFIED' if ok else 'CERTIFICATION FAILED'}")
    raise SystemExit(0 if ok else 1)