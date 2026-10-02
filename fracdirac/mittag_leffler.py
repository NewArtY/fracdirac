"""One-parameter Mittag-Leffler function E_alpha(z) for complex z (reference values).

The power series sum_k z^k / Gamma(alpha k + 1) is summed with mpmath in extended
precision, which is sufficient (and exact to double precision) for the moderate
arguments |z| <~ 40 used in the constant-generator benchmark.
"""
import mpmath as mp
import numpy as np


def ml(alpha, z, dps=None):
    z = complex(z)
    if dps is None:
        # leading growth exp(|z|^(1/alpha)); keep 30 digits beyond the cancellation
        dps = int(30 + abs(z) ** (1.0 / alpha) / 2.302585 + 10)
    with mp.workdps(dps):
        zz = mp.mpc(z.real, z.imag)
        a = mp.mpf(alpha)
        term_tol = mp.mpf(10) ** (-(dps - 5))
        total = mp.mpc(0)
        k = 0
        zk = mp.mpc(1)
        while True:
            term = zk / mp.gamma(a * k + 1)
            total += term
            if k > 5 and abs(term) < term_tol * max(1, abs(total)):
                break
            zk *= zz
            k += 1
            if k > 20000:
                raise RuntimeError('Mittag-Leffler series did not converge')
        return complex(total)


def ml_matrix_solution(alpha, kappa, M, b, s0, t):
    """Solution of  D^alpha s = kappa (M s + b),  s(0) = s0  for constant M (3x3), b (3,):

        s(t) = s* + V diag(E_alpha(kappa lambda_i t^alpha)) V^-1 (s0 - s*),  s* = -M^-1 b.
    """
    lam, V = np.linalg.eig(M)
    sstar = -np.linalg.solve(M, b)
    coef = np.linalg.solve(V, (s0 - sstar).astype(complex))
    out = []
    for ti in np.atleast_1d(t):
        E = np.array([ml(alpha, kappa * l * ti ** alpha) for l in lam])
        out.append((sstar + (V @ (E * coef))).real)
    return np.array(out)
