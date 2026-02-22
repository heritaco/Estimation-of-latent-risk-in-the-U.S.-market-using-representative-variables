# -*- coding: utf-8 -*-
'''
Core spectral solver for Guerrero (2007) penalized trend.

Model:
  t̂(λ, m) = (I + λ K'K)^(-1) (Z + λ m K'1)
with fixed point for m = mean(K t̂).
'''
from __future__ import annotations

from dataclasses import dataclass
from math import comb as _comb
import numpy as np


def difference_matrix(N: int, d: int) -> np.ndarray:
    '''
    Construct K: (N-d) x N implementing the d-th forward difference.
    For d=0 returns I_N.
    '''
    N = int(N)
    d = int(d)
    if d < 0:
        raise ValueError("d must be >= 0")
    if d == 0:
        return np.eye(N, dtype=float)
    if d >= N:
        raise ValueError(f"d={d} must be < N={N} for forward differences.")
    K = np.zeros((N - d, N), dtype=float)
    coeffs = np.array([(-1) ** (d - k) * _comb(d, k) for k in range(d + 1)], dtype=float)
    for r in range(N - d):
        K[r, r : r + d + 1] = coeffs
    return K


@dataclass(frozen=True)
class SolverFit:
    t_hat: np.ndarray
    m_hat: float
    sigma2_hat: float
    diag_Ainv: np.ndarray
    s_unit_real: float


class GuerreroSpectralSolver:
    '''
    Spectral implementation of Guerrero (2007) penalized trend for fixed (N, d).

    Precomputes K, B=K'K, eig(B)=QΛQ', and K'1.

    Smoothness index:
      S_raw(λ) = 1 - (1/N) tr[(I + λ K'K)^(-1)]
      s_unit = S_raw / (1 - d/N)
    '''

    def __init__(self, N: int, d: int):
        self.N = int(N)
        self.d = int(d)
        self.K = difference_matrix(self.N, self.d)
        self.KT = self.K.T
        self.B = self.KT @ self.K
        eigvals, Q = np.linalg.eigh(self.B)
        self.eigvals = eigvals
        self.Q = Q
        self.K1 = self.KT @ np.ones(self.N - self.d, dtype=float)
        self.S_max = 1.0 - self.d / self.N

    def _S_raw(self, lam: float) -> float:
        denom = 1.0 + lam * self.eigvals
        trAinv = np.sum(1.0 / denom)
        return 1.0 - trAinv / self.N

    def lambda_from_s(self, s_unit: float, tol: float = 1e-11, maxit: int = 80) -> float:
        '''
        Map s_unit in (0,1) to λ by solving S_raw(λ) = s_unit * S_max.
        Uses bisection over λ >= 0.
        '''
        s_unit = float(s_unit)
        if s_unit <= 0.0:
            return 0.0
        if s_unit >= 1.0:
            s_unit = 0.999999

        if self.d == 0:
            return s_unit / (1.0 - s_unit)

        target = s_unit * self.S_max
        lo, hi = 0.0, 1.0
        while self._S_raw(hi) < target and hi < 1e16:
            hi *= 10.0

        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            Smid = self._S_raw(mid)
            if abs(Smid - target) < tol:
                return mid
            if Smid < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def fit_for_lambda(
        self,
        Z: np.ndarray,
        lam: float,
        m_tol: float = 1e-10,
        max_m_iter: int = 120,
    ) -> SolverFit:
        '''
        Fit trend for a given λ using spectral decomposition.
        '''
        Z = np.asarray(Z, dtype=float).ravel()
        if Z.size != self.N:
            raise ValueError(f"Z length {Z.size} != N {self.N} in solver.")

        Q = self.Q
        eigvals = self.eigvals
        K = self.K

        m = float(np.mean(K @ Z))
        for _ in range(max_m_iter):
            b = Z + lam * m * self.K1
            y = Q.T @ b
            denom = 1.0 + lam * eigvals
            t_hat = Q @ (y / denom)
            m_new = float(np.mean(K @ t_hat))
            if abs(m_new - m) < m_tol:
                m = m_new
                break
            m = m_new

        alpha = 1.0 / (1.0 + lam * eigvals)
        diag_Ainv = (Q ** 2) @ alpha
        resid = Z - t_hat
        pen = (K @ t_hat) - m
        dof = max(1, self.N - self.d - 1)
        sigma2_hat = float((resid.T @ resid + lam * (pen.T @ pen)) / dof)

        S_raw = 1.0 - np.sum(alpha) / self.N
        s_unit_real = S_raw / self.S_max if self.S_max > 0 else 0.0

        return SolverFit(
            t_hat=np.asarray(t_hat, dtype=float),
            m_hat=float(m),
            sigma2_hat=sigma2_hat,
            diag_Ainv=np.asarray(diag_Ainv, dtype=float),
            s_unit_real=float(s_unit_real),
        )

    def fit_for_s(
        self,
        Z: np.ndarray,
        s_unit: float,
        m_tol: float = 1e-10,
        max_m_iter: int = 120,
    ) -> tuple[SolverFit, float]:
        '''
        Fit using smoothness s (maps s -> λ, then fits).
        Returns (fit, lam).
        '''
        lam = self.lambda_from_s(s_unit)
        fit = self.fit_for_lambda(Z, lam, m_tol=m_tol, max_m_iter=max_m_iter)
        return fit, lam
