"""Load losses from joint power categories (plan rev. 5, section 2.1, Phase C11).

No power flow is run. The wind record is reduced once to a table of
categories; category ``k`` carries its hours per year ``h_k`` and the
SIMULTANEOUS output ``S_{t,k} = P_{t,k} + j Q_{t,k}`` of every turbine,
averaged over the same hours (user decision K2, 2026-09-24: turbine-specific
but time-step-matched values, category width a setting). The engines consume
one number per turbine subset ``X``:

    ``W(X) = sum_k h_k |sum_{t in X} S_{t,k}|^2``        [MVA^2 h per year]

A three-phase system of resistance ``R`` (ohm per phase) carrying ``X`` at
line voltage ``U`` (kV) then dissipates ``R W(X) / U^2`` MWh per year, so the
number of categories never enters the optimisation (plan section 2.1).
``W`` is a quadratic form in the subset indicator, ``W(X) = 1_X^T M 1_X``
with ``M = sum_k h_k (P_k P_k^T + Q_k Q_k^T)``, which is how tables store it.

The mean-only category table errs LOW: ``W_cat(X) <= W_exact(X)`` by the
variance decomposition. A table that also stores the within-category
covariance reproduces the hourly ``W`` exactly (``W_var``); on the Runkel
record the table is already exact at speed width 0 because the DWD record
is quantised (0.1 m/s, 10 deg). :meth:`LossCategories.loss_weight` chooses.

Valuation: losses are lost SALES (user decision P1). The capitalised value
of one MWh per year of loss energy is ``Lambda = price * AF(H, r, g)``,
with the price per MWh of LOSS energy (P^2-weighted, not P-weighted).
:func:`loss_price_eur_per_mwh` states the two conventions the plan and its
implementation have used; see the implementation log section 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "LossCategories",
    "annuity_factor",
    "category_square_sums",
    "loss_price_eur_per_mwh",
    "loss_coef",
]


def category_square_sums(hours: np.ndarray, P: np.ndarray,
                         Q: np.ndarray | None = None) -> np.ndarray:
    """``W(X)`` for every turbine bitmask ``X`` (index 0 is ``W(empty) = 0``).

    Parameters:
        hours: ``(K,)`` hours per year of each category (or of each hour,
            with ``8760 / N`` each, for the exact hourly sum).
        P: ``(K, n)`` simultaneous active output per turbine, MW.
        Q: ``(K, n)`` reactive output, Mvar; zero when omitted.

    Returns:
        ``(2**n,)`` float64, MVA^2 h per year.
    """
    h = np.asarray(hours, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != h.shape[0]:
        raise ValueError(f"P must be (K, n) with K = {h.shape[0]}")
    Q = np.zeros_like(P) if Q is None else np.asarray(Q, dtype=np.float64)
    if Q.shape != P.shape:
        raise ValueError("Q must have the shape of P")
    M = (P * h[:, None]).T @ P + (Q * h[:, None]).T @ Q
    return _w_from_matrix(M)


def _w_from_matrix(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    masks = np.arange(1 << n)
    member = ((masks[:, None] >> np.arange(n)) & 1).astype(np.float64)
    return np.einsum("si,ij,sj->s", member, M, member)


def annuity_factor(years: float, rate: float, growth: float = 0.0) -> float:
    """Present value of 1 per year for ``years`` years, growing at ``growth``.

    ``AF = sum_{t=1..H} ((1+g)/(1+r))^t``; ``g = r`` gives ``H``. The plan's
    headline is H = 25 a, r = 4 % real, g = 0 %: AF = 15.622.
    """
    if years <= 0:
        raise ValueError("years must be > 0")
    q = (1.0 + growth) / (1.0 + rate)
    if abs(q - 1.0) < 1e-15:
        return float(years)
    return float(q * (1.0 - q ** years) / (1.0 - q))


def loss_price_eur_per_mwh(*, price: float, nu: float = 0.0,
                           convention: str = "plan") -> float:
    """Value of one MWh of loss energy (lost sales).

    ``convention="plan"`` is plan revision 5 as written: ``price * (1 - nu)``,
    ``price`` the loss-weighted (P^2-weighted) market value and ``nu`` the
    share of loss energy in negative-price hours, valued at zero because
    new plants curtail there.

    ``convention="clipped"``: ``price`` is ALREADY the P^2-weighted mean of
    ``max(p, 0)``, so ``nu`` must not be applied again. The C11 check of
    2026-09-24 found that ``price * (1 - nu)`` counts the negative hours
    twice when ``price`` is the all-hours P^2-weighted mean, which already
    includes their negative prices (2023 national: 61.86 against the
    consistent 67.37 EUR/MWh). Implementation log section 4.
    """
    if not 0.0 <= nu < 1.0:
        raise ValueError("nu must lie in [0, 1)")
    if convention == "plan":
        return float(price) * (1.0 - nu)
    if convention == "clipped":
        return float(price)
    raise ValueError("convention must be 'plan' or 'clipped'")


def loss_coef(*, price_eur_per_mwh: float, u_kv: float, years: float = 25.0,
              rate: float = 0.04, growth: float = 0.0) -> float:
    """``CollectorModel.loss_coef``: capitalised EUR per (ohm MVA^2 h / year).

    A system of resistance ``r`` ohm/m per phase carrying ``W`` MVA^2 h per
    year at ``u_kv`` loses ``r W / U^2`` MWh per year per metre; times the
    price and the annuity factor that is ``loss_coef * r * W``.
    """
    if u_kv <= 0:
        raise ValueError("u_kv must be > 0")
    return (float(price_eur_per_mwh) * annuity_factor(years, rate, growth)
            / (u_kv * u_kv))


@dataclass
class LossCategories:
    """A category table written by
    ``case_studies/runkel_free_siting/scripts/build_loss_categories.py``.

    Attributes:
        hours: ``(K,)`` hours per year per category.
        P, Q: ``(K, n)`` time-matched mean output per turbine (MW, Mvar).
        M_cat, M_within, M_exact: ``(n, n)`` quadratic forms of the mean-only
            table, of the within-category covariance, and of the hourly
            record.
        meta: The builder's JSON metadata (curve, level, width, DWD hash...).
        sha256: Hash of the file the table was read from.
    """
    hours: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    M_cat: np.ndarray
    M_within: np.ndarray
    M_exact: np.ndarray
    meta: dict = field(default_factory=dict)
    sha256: str | None = None

    @property
    def n(self) -> int:
        """Number of turbines."""
        return int(self.P.shape[1])

    @classmethod
    def load(cls, path) -> LossCategories:
        """Read a table and check it is internally consistent."""
        raw = Path(path).read_bytes()
        with np.load(Path(path), allow_pickle=False) as z:
            tab = cls(hours=np.asarray(z["h"], dtype=np.float64),
                      P=np.asarray(z["P"], dtype=np.float64),
                      Q=np.asarray(z["Q"], dtype=np.float64),
                      M_cat=np.asarray(z["M_cat"], dtype=np.float64),
                      M_within=np.asarray(z["M_within"], dtype=np.float64),
                      M_exact=np.asarray(z["M_exact"], dtype=np.float64),
                      meta=json.loads(str(z["meta"])),
                      sha256=hashlib.sha256(raw).hexdigest())
        tab.check()
        return tab

    def check(self, rtol: float = 1e-9) -> None:
        """Hours sum to a year; M_cat matches the category means; the
        mean-only form never exceeds the exact one on the diagonal."""
        if abs(self.hours.sum() - 8760.0) > 1e-6 * 8760.0:
            raise ValueError(f"category hours sum to {self.hours.sum()}")
        m = ((self.P * self.hours[:, None]).T @ self.P
             + (self.Q * self.hours[:, None]).T @ self.Q)
        if not np.allclose(m, self.M_cat, rtol=rtol, atol=1e-9):
            raise ValueError("M_cat does not match the category table")
        if np.any(np.diag(self.M_cat) > np.diag(self.M_exact) * (1 + rtol)):
            raise ValueError("a mean-only W exceeds the exact W")

    def loss_weight(self, kind: str = "exact") -> tuple[float, ...]:
        """``W(X)`` for every bitmask, for ``CollectorModel.loss_weight``.

        ``kind``: ``"exact"`` (the hourly record), ``"var"`` (categories plus
        within-category covariance, equal to exact), or ``"cat"`` (the
        mean-only table, a lower bound).
        """
        if kind == "exact":
            M = self.M_exact
        elif kind == "var":
            M = self.M_cat + self.M_within
        elif kind == "cat":
            M = self.M_cat
        else:
            raise ValueError("kind must be exact, var or cat")
        return tuple(float(x) for x in _w_from_matrix(M))

    def theta_hours(self, rated_mw_total: float, kind: str = "exact") -> float:
        """The coherent single-theta comparison: ``W(all) / P_farm^2``."""
        return self.loss_weight(kind)[-1] / (rated_mw_total ** 2)

    def describe(self) -> dict:
        """Provenance fields for the A1 record."""
        return {"sha256": self.sha256, "n": self.n,
                "categories": int(self.hours.shape[0]),
                "curve": self.meta.get("curve"),
                "width": self.meta.get("width"),
                "dwd_sha256": (self.meta.get("dwd") or {}).get("sha256")}
