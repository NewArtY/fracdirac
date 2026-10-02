"""Dimensionless laser-driven massive Dirac model.

Units and symbols (see Table 1 of the manuscript)
-------------------------------------------------
x      = omega (t - t_c)                 dimensionless time
q(x)   = v_F pi_k(t) / Delta = sinh(theta)  normalized kinetic momentum
kappa  = hbar v_F k / Delta              normalized crystal momentum
a_D    = e v_F A_0 / Delta               strong-field parameter
s_p    = omega tau_p                     Gaussian pulse width
g      = Delta / (hbar omega)            gap ratio
T1, T2, tau_m                            given in units of 1/omega

Fixed-basis Bloch equation (zero temperature, w_eq = -1):

    tau_m^(alpha-1) D^alpha s = M(x) s + b(x),
    M = [Omega x] - n n^T / T1 - (1 - n n^T) / T2,   b = -n / T1,
    Omega = 2 g (q, 0, 1),   n = (q, 0, 1) / sqrt(1 + q^2).

All functions are vectorized over a set of P parameter points: the
parameters a_D, phi_cep, kappa and g may be scalars or arrays of shape (P,).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    a_D: object = 2.0        # strong-field parameter (scalar or array (P,))
    s_p: float = 3.0         # pulse width omega*tau_p
    phi_cep: object = 0.0    # carrier-envelope phase (scalar or array (P,))
    kappa: object = 0.0      # normalized crystal momentum (scalar or array (P,))
    g: object = 1.5          # gap ratio Delta/(hbar omega) (scalar or array (P,))
    T1: float = 40.0         # population relaxation time (units 1/omega)
    T2: float = 20.0         # dephasing time (units 1/omega)
    tau_m: float = 1.0       # memory time (units 1/omega)

    def arrays(self):
        a, p, k, _ = np.broadcast_arrays(np.atleast_1d(np.asarray(self.a_D, float)),
                                         np.atleast_1d(np.asarray(self.phi_cep, float)),
                                         np.atleast_1d(np.asarray(self.kappa, float)),
                                         np.atleast_1d(np.asarray(self.g, float)))
        return a, p, k

    @property
    def P(self):
        return self.arrays()[0].size


def q_of_x(x, prm):
    """Normalized kinetic momentum q = kappa + a_D exp(-x^2/2 s_p^2) cos(x + phi)."""
    a, p, k = prm.arrays()
    return k + a * np.exp(-x * x / (2 * prm.s_p ** 2)) * np.cos(x + p)


def dq_of_x(x, prm):
    """dq/dx (equals minus the normalized electric field)."""
    a, p, _ = prm.arrays()
    env = np.exp(-x * x / (2 * prm.s_p ** 2))
    return a * env * (-(x / prm.s_p ** 2) * np.cos(x + p) - np.sin(x + p))


def chi_of_x(x, prm):
    """Mixing angle chi = arctan(sinh theta) = arctan(q)."""
    return np.arctan(q_of_x(x, prm))


def dchi_of_x(x, prm):
    """d chi / dx = q' / (1 + q^2)."""
    q = q_of_x(x, prm)
    return dq_of_x(x, prm) / (1.0 + q * q)


def n_of_x(x, prm):
    """Instantaneous band direction n = (tanh theta, 0, sech theta); shape (P, 3)."""
    q = q_of_x(x, prm)
    den = np.sqrt(1.0 + q * q)
    return np.stack([q / den, np.zeros_like(q), 1.0 / den], axis=-1)


def cross_matrix(w):
    """Matrices [w x] for vectors w of shape (P, 3); returns (P, 3, 3)."""
    P = w.shape[0]
    C = np.zeros((P, 3, 3))
    C[:, 0, 1] = -w[:, 2]; C[:, 0, 2] = w[:, 1]
    C[:, 1, 0] = w[:, 2];  C[:, 1, 2] = -w[:, 0]
    C[:, 2, 0] = -w[:, 1]; C[:, 2, 1] = w[:, 0]
    return C


def gen_fixed(x, prm):
    """Generator of the fixed-basis Bloch equation: returns M (P,3,3), b (P,3)."""
    q = q_of_x(x, prm)
    n = n_of_x(x, prm)
    g = np.broadcast_to(np.asarray(prm.g, float), q.shape)[:, None]
    Om = 2.0 * g * np.stack([q, np.zeros_like(q), np.ones_like(q)], axis=-1)
    nn = n[:, :, None] * n[:, None, :]
    M = cross_matrix(Om) - nn / prm.T1 - (np.eye(3)[None] - nn) / prm.T2
    return M, -n / prm.T1


def gen_inst(x, prm, connection=True):
    """Generator in the instantaneous basis.

    connection=True : local (Markovian / "naive") form with the precession vector
                      (0, -chi', 2 g cosh(theta)).
    connection=False: transformed generator without the connection, precession
                      vector (0, 0, 2 g cosh(theta)).
    """
    q = q_of_x(x, prm)
    eps = np.sqrt(1.0 + q * q)
    wy = -dchi_of_x(x, prm) if connection else np.zeros_like(q)
    g = np.broadcast_to(np.asarray(prm.g, float), q.shape)
    Om = np.stack([np.zeros_like(q), wy, 2.0 * g * eps], axis=-1)
    D = np.diag([1.0 / prm.T2, 1.0 / prm.T2, 1.0 / prm.T1])[None]
    M = cross_matrix(Om) - D
    b = np.zeros((q.size, 3)); b[:, 2] = -1.0 / prm.T1
    return M, b


def rot_y(phi):
    """Rotation matrices about the y axis by angles phi (P,); returns (P,3,3)."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.zeros(phi.shape + (3, 3))
    R[..., 0, 0] = c; R[..., 0, 2] = s
    R[..., 1, 1] = 1.0
    R[..., 2, 0] = -s; R[..., 2, 2] = c
    return R


def to_inst(x, s, prm):
    """Fixed-basis Bloch vector(s) -> instantaneous basis: s~ = O_y(-chi) s."""
    chi = chi_of_x(x, prm)
    return np.einsum('pij,pj->pi', rot_y(-chi), s)


def to_fixed(x, st, prm):
    """Instantaneous-basis Bloch vector(s) -> fixed basis: s = O_y(chi) s~."""
    chi = chi_of_x(x, prm)
    return np.einsum('pij,pj->pi', rot_y(chi), st)


def observables_inst(st):
    """Population and coherence from instantaneous-basis Bloch vectors (..., 3)."""
    n_c = 0.5 * (1.0 + st[..., 2])
    p_cv = 0.5 * (st[..., 0] - 1j * st[..., 1])
    return n_c, p_cv
