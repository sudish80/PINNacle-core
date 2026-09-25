"""pinnacle_ext.arch -- advanced architectures.

All extend the core PINN contract: forward(z) with z = cat([x0..x_{d-1}, t]).
n_out=1 keeps exact core compatibility; n_out>1 powers vector problems (NS).
"""

import math

import numpy as np
import torch
import torch.nn as nn

import pinn_solver as P


# --------------------------------------------------------------------- #
# Fourier feature embedding (Tancik et al.)
# --------------------------------------------------------------------- #
class FourierFeatures(nn.Module):
    """gamma(v) = [cos(2pi B v), sin(2pi B v)].

    ``basis=None`` -> random B ~ N(0, sigma^2) rows (Tancik et al.).
    ``basis=[...]`` -> explicit rows (one wavenumber per feature); the exact
    target frequencies of a manufactured problem can be locked in so random
    sampling cannot miss them (the eigenmode fix used for helmholtz).
    """

    def __init__(self, in_dim, n_freqs=64, sigma=8.0, seed=0, basis=None):
        super().__init__()
        if basis is not None:
            rows = [np.asarray(b, float).reshape(1, -1) for b in basis]
            if n_freqs:
                g = torch.Generator().manual_seed(seed)
                rand = torch.randn(n_freqs, len(rows[0].ravel()), generator=g) * sigma
                self.B = torch.cat([torch.tensor(np.vstack(rows),
                                                 dtype=torch.get_default_dtype()),
                                    rand], dim=0)
            else:
                self.B = torch.tensor(np.vstack(rows),
                                      dtype=torch.get_default_dtype())
        else:
            g = torch.Generator().manual_seed(seed)
            self.B = torch.randn(n_freqs, in_dim, generator=g) * sigma
        self.out_dim = 2 * self.B.shape[0]

    def forward(self, z):
        proj = z @ self.B.t()
        return torch.cat([torch.cos(2 * math.pi * proj),
                          torch.sin(2 * math.pi * proj)], dim=1)


class FourierMLP(P.PINN):
    """Core PINN + Fourier input encoding (feature-space shortest path)."""

    def __init__(self, in_dim, layers=(4, 40, 40, 40), activation="tanh",
                 n_freqs=48, sigma=8.0, seed=0, n_out=1, basis=None):
        ff = FourierFeatures(in_dim, n_freqs=n_freqs, sigma=sigma,
                             seed=seed, basis=basis)
        super().__init__(ff.out_dim, layers=layers, activation=activation)
        self.ff = ff
        self.fcs[-1] = nn.Linear(layers[-1], n_out)

    def forward(self, z):
        return super().forward(self.ff(z))


# --------------------------------------------------------------------- #
# Multi-output version of the core PINN (vector PDEs)
# --------------------------------------------------------------------- #
class VectorPINN(P.PINN):
    def __init__(self, in_dim, layers=(4, 40, 40, 40, 40), activation="tanh",
                 n_out=1, w0=30.0):
        super().__init__(in_dim, layers=layers, activation=activation, w0=w0)
        self.fcs[-1] = nn.Linear(layers[-1], n_out)

    def n_outputs(self):
        return self.fcs[-1].out_features


def build_net(in_dim, layers, activation="tanh", arch="mlp", n_out=1,
              n_freqs=48, sigma=8.0, seed=0, drop=0.1, w0=30.0, basis=None):
    """Factory: 'mlp' | 'fourier' | 'siren' | 'residual' | 'multiscale'
               | 'adaptive' | 'weight_normed' | 'dropout'.

    Returns an nn.Module (optionally wrapped in AdaptiveSlope / WeightNorm).
    """
    if arch == "fourier":
        net = FourierMLP(in_dim, layers=layers, activation=activation,
                         n_freqs=n_freqs, sigma=sigma, seed=seed, n_out=n_out,
                         basis=basis)
    elif arch == "siren":
        net = SIREN(in_dim, layers=layers, n_out=n_out, w0=w0)
    elif arch == "adaptive":
        net = AdaptiveSlope(in_dim, layers=layers, n_out=n_out)
    elif arch == "weight_normed":
        base = VectorPINN(in_dim, layers=layers, activation=activation,
                          n_out=n_out)
        net = WeightNormedPINN(base)
    elif arch == "residual":
        base = VectorPINN(in_dim, layers=layers, activation=activation, n_out=n_out)
        net = ResidualMLP(base, in_dim, n_out)
    elif arch == "multiscale":
        k = max(1, (len(layers) // 2) or 1)
        sub = [VectorPINN(in_dim, layers=layers, activation=activation, n_out=n_out)
               for _ in range(k)]
        net = MultiScalePINN(sub)
    elif arch == "dropout":
        net = DropoutMLP(in_dim, layers=layers, activation=activation,
                         n_out=n_out, drop=drop)
    else:
        net = VectorPINN(in_dim, layers=layers, activation=activation, n_out=n_out)
    return net


# --------------------------------------------------------------------- #
# Residual / skip connections
# --------------------------------------------------------------------- #
class ResidualMLP(nn.Module):
    """Skip-connected wrapper: out = NN(z) + W_skip z (identity shortcut)."""

    def __init__(self, base, in_dim, n_out=1):
        super().__init__()
        self.base = base
        self.skip = nn.Linear(in_dim, n_out, bias=False)
        nn.init.zeros_(self.skip.weight)

    def forward(self, z):
        return self.base(z) + self.skip(z)


# --------------------------------------------------------------------- #
# Multi-scale ensemble: sum of sub-networks (each scale its own init)
# --------------------------------------------------------------------- #
class MultiScalePINN(nn.Module):
    def __init__(self, subnets):
        super().__init__()
        self.nets = nn.ModuleList(subnets)

    def forward(self, z):
        return sum(n(z) for n in self.nets)


# --------------------------------------------------------------------- #
# Learnable activation slope (a per-layer scaling before tanh)
# --------------------------------------------------------------------- #
class AdaptiveSlopeMLP(nn.Module):
    """MLP where each hidden layer is tanh(a_i z); slopes are learnable.

    Optimize over slopes separately, or let the base optimizer reach them via
    a shared parameter holder (use with a single Adam on net.parameters()).
    """

    def __init__(self, in_dim, layers=(4, 40, 40, 40, 40), n_out=1,
                 init_a=1.0):
        super().__init__()
        dims = [in_dim] + list(layers) + [n_out]
        self.fcs = nn.ModuleList()
        self.slopes = nn.ParameterList()
        for i in range(len(dims) - 2):
            fc = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fcs.append(fc)
            self.slopes.append(nn.Parameter(torch.tensor(float(init_a))))
        self.head = nn.Linear(layers[-1], n_out)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z):
        for fc, a in zip(self.fcs, self.slopes):
            z = torch.tanh(a * fc(z))
        return self.head(z)


# --------------------------------------------------------------------- #
# Dropout MLP (MC-dropout uncertainty: enables stochastic forward passes)
# --------------------------------------------------------------------- #
class DropoutMLP(nn.Module):
    """Core-style MLP with a dropout layer after every hidden activation.

    At inference the dropout modules are turned back on (``mc_enable``) so a
    single model yields an ensemble of stochastic predictions (Gal & Ghahramani
    MC-dropout).  ``drop=0`` reproduces a plain VectorPINN exactly.
    """

    def __init__(self, in_dim, layers=(4, 40, 40, 40, 40), activation="tanh",
                 n_out=1, drop=0.1, w0=30.0):
        super().__init__()
        self.activation = activation
        self.w0 = w0
        dims = [in_dim] + list(layers) + [n_out]
        self.fcs = nn.ModuleList()
        self.drops = nn.ModuleList()
        for i in range(len(dims) - 2):
            fc = nn.Linear(dims[i], dims[i + 1])
            if activation == "siren":
                if i == 0:
                    nn.init.uniform_(fc.weight, -1.0 / w0, 1.0 / w0)
                else:
                    with torch.no_grad():
                        fc.weight *= np.sqrt(6.0 / dims[i])
            else:
                nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fcs.append(fc)
            self.drops.append(nn.Dropout(drop))
        self.head = nn.Linear(layers[-1], n_out)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _act(self, z):
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "gelu":
            return torch.nn.functional.gelu(z)
        if self.activation == "relu":
            return torch.relu(z)
        if self.activation == "siren":
            return torch.sin(self.w0 * z)
        raise ValueError(self.activation)

    def forward(self, z):
        for fc, dr in zip(self.fcs, self.drops):
            z = dr(self._act(fc(z)))
        return self.head(z)

    def mc_enable(self, p=None):
        """Turn dropout back on (stochastic forward passes) while freezing BN."""
        self.eval()
        if p is not None:
            for dr in self.drops:
                dr.p = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        return self


class AdaptiveSlope(nn.Module):
    """Backward-compatible wrapper name (delegates to AdaptiveSlopeMLP)."""

    def __init__(self, in_dim, layers=(4, 40, 40, 40, 40), n_out=1, init=1.0,
                 lr=1e-3, ramp=1000):
        super().__init__()
        self.mlp = AdaptiveSlopeMLP(in_dim, layers=layers, n_out=n_out,
                                    init_a=init)
        self.lr, self.ramp = lr, ramp

    def forward(self, z):
        return self.mlp(z)


# --------------------------------------------------------------------- #
# Weight normalization wrapper (spectral-norm-ish cheap proxy)
# --------------------------------------------------------------------- #
class WeightNormedPINN(nn.Module):
    """Div2-style normalization: divide each weight by its 2-norm."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        with torch.no_grad():
            for m in base.modules():
                if isinstance(m, nn.Linear):
                    m.weight.div_(m.weight.norm(dim=1, keepdim=True)
                                  .clamp_min(1e-8))

    def forward(self, z):
        return self.base(z)


# --------------------------------------------------------------------- #
# SIREN standalone (full first-layer + recurrent omega0 scheme)
# --------------------------------------------------------------------- #
class SIREN(P.PINN):
    def __init__(self, in_dim, layers=(4, 40, 40, 40, 40), w0=30.0, n_out=1,
                 proper_init=True):
        super().__init__(in_dim, layers=layers, activation="siren", w0=w0)
        self.fcs[-1] = nn.Linear(layers[-1], n_out)
        # The core PINN init omits the /w0 factor on the hidden weights (it is
        # tailored for tanh); Sitzmann et al.'s scheme needs it so the pre-
        # activations stay O(1) despite sin(w0 x). additive fix in arch only.
        if proper_init:
            for i, fc in enumerate(self.fcs):
                if i == len(self.fcs) - 1:
                    nn.init.uniform_(fc.weight, -1.0 / w0, 1.0 / w0)
                else:
                    bound = np.sqrt(6.0 / fc.in_features) / w0
                    with torch.no_grad():
                        fc.weight.uniform_(-bound, bound)
                nn.init.zeros_(fc.bias)


# number-of-parameters helper for sweep tables
def n_params(net):
    return sum(p.numel() for p in net.parameters())