"""Refractive-index (Sellmeier) models for common QPM crystals.

Two model families are implemented:

``TemperatureSellmeier``
    The Jundt / Gayer form used for lithium niobate, where the Sellmeier
    coefficients themselves depend on temperature through
    ``f(T) = (T - 24.5)(T + 570.82)``.

``SellmeierLinearDnDt``
    A room-temperature Sellmeier equation plus a wavelength-dependent
    ``dn/dT`` polynomial, as published for KTP by Kato & Takaoka (2002).

All wavelengths passed to the public API are in nanometres; the Sellmeier
coefficients internally use micrometres, which is the convention in the
literature.

References
----------
* D. H. Jundt, "Temperature-dependent Sellmeier equation for the index of
  refraction, n_e, in congruent lithium niobate", Opt. Lett. 22, 1553 (1997).
* O. Gayer et al., "Temperature and wavelength dependent refractive index
  equations for MgO-doped congruent and stoichiometric LiNbO3",
  Appl. Phys. B 91, 343 (2008).
* K. Kato and E. Takaoka, "Sellmeier and thermo-optic dispersion formulas for
  KTP", Appl. Opt. 41, 5040 (2002).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Sequence, Tuple

from .units import nm_to_um

Axis = str


class MaterialError(ValueError):
    """Raised for unknown materials, axes, or out-of-range inputs."""


@dataclass(frozen=True)
class WavelengthRange:
    min_nm: float
    max_nm: float

    def contains(self, lam_nm: float) -> bool:
        return self.min_nm <= lam_nm <= self.max_nm


@dataclass(frozen=True)
class TemperatureSellmeier:
    """Jundt/Gayer-style temperature-dependent Sellmeier equation.

    n^2 = a1 + b1 f + (a2 + b2 f) / (lam^2 - (a3 + b3 f)^2)
          + (a4 + b4 f) / (lam^2 - a5^2) - a6 lam^2

    with ``f = (T - 24.5) (T + 570.82)`` and ``lam`` in micrometres.
    """

    a: Tuple[float, float, float, float, float, float]
    b: Tuple[float, float, float, float]

    @staticmethod
    def thermal_factor(temperature_c: float) -> float:
        return (temperature_c - 24.5) * (temperature_c + 570.82)

    def n_squared(self, lam_um: float, temperature_c: float) -> float:
        a1, a2, a3, a4, a5, a6 = self.a
        b1, b2, b3, b4 = self.b
        f = self.thermal_factor(temperature_c)
        lam2 = lam_um * lam_um
        term1 = a1 + b1 * f
        term2 = (a2 + b2 * f) / (lam2 - (a3 + b3 * f) ** 2)
        term3 = (a4 + b4 * f) / (lam2 - a5 * a5)
        term4 = a6 * lam2
        return term1 + term2 + term3 - term4

    def __call__(self, lam_um: float, temperature_c: float) -> float:
        n2 = self.n_squared(lam_um, temperature_c)
        if n2 <= 0.0:
            raise MaterialError(f"Sellmeier equation gave n^2={n2:.4g} <= 0 at {lam_um} um")
        return math.sqrt(n2)


@dataclass(frozen=True)
class SellmeierLinearDnDt:
    """Room-temperature Sellmeier equation with a linear thermo-optic term.

    n(lam, T) = n0(lam) + dn/dT(lam) * (T - T_ref)

    n0^2 = A + B / (lam^2 - C) + D / (lam^2 - E)
    dn/dT = (p3 / lam^3 + p2 / lam^2 + p1 / lam + p0) * 1e-5   [1/degC]
    """

    sellmeier: Tuple[float, float, float, float, float]
    dndt_poly: Tuple[float, float, float, float]
    reference_temperature_c: float = 20.0

    def n0(self, lam_um: float) -> float:
        A, B, C, D, E = self.sellmeier
        lam2 = lam_um * lam_um
        n2 = A + B / (lam2 - C) + D / (lam2 - E)
        if n2 <= 0.0:
            raise MaterialError(f"Sellmeier equation gave n^2={n2:.4g} <= 0 at {lam_um} um")
        return math.sqrt(n2)

    def dn_dt(self, lam_um: float) -> float:
        p3, p2, p1, p0 = self.dndt_poly
        return (p3 / lam_um**3 + p2 / lam_um**2 + p1 / lam_um + p0) * 1e-5

    def __call__(self, lam_um: float, temperature_c: float) -> float:
        return self.n0(lam_um) + self.dn_dt(lam_um) * (temperature_c - self.reference_temperature_c)


IndexModel = Callable[[float, float], float]


@dataclass(frozen=True)
class Material:
    """A nonlinear crystal with one index model per polarisation axis."""

    name: str
    description: str
    axes: Mapping[Axis, IndexModel]
    default_axis: Axis
    valid_range: WavelengthRange
    d_eff_pm_per_v: float
    """Effective nonlinear coefficient for the default QPM interaction [pm/V]."""
    thermal_expansion: Tuple[float, float] = field(default=(0.0, 0.0))
    """(alpha, beta) in dL/L = alpha (T-25) + beta (T-25)^2 [1/degC, 1/degC^2]."""

    def axis_names(self) -> Sequence[Axis]:
        return tuple(self.axes.keys())

    def _model(self, axis: Axis | None) -> IndexModel:
        key = self.default_axis if axis is None else axis
        try:
            return self.axes[key]
        except KeyError:
            raise MaterialError(
                f"{self.name} has no axis {key!r}; available: {list(self.axes)}"
            ) from None

    def refractive_index(
        self,
        lam_nm: float,
        temperature_c: float = 25.0,
        axis: Axis | None = None,
        strict: bool = False,
    ) -> float:
        """Refractive index at ``lam_nm`` and ``temperature_c``.

        With ``strict=True`` wavelengths outside the published validity range
        raise ``MaterialError``; otherwise the formula is simply extrapolated.
        """
        if lam_nm <= 0.0:
            raise MaterialError("wavelength must be positive")
        if strict and not self.valid_range.contains(lam_nm):
            raise MaterialError(
                f"{lam_nm} nm is outside the valid range "
                f"[{self.valid_range.min_nm}, {self.valid_range.max_nm}] nm for {self.name}"
            )
        return self._model(axis)(nm_to_um(lam_nm), temperature_c)

    def dn_dlambda(
        self,
        lam_nm: float,
        temperature_c: float = 25.0,
        axis: Axis | None = None,
        step_nm: float = 0.1,
    ) -> float:
        """Numerical dn/dlambda in 1/nm using a central difference."""
        n_plus = self.refractive_index(lam_nm + step_nm, temperature_c, axis)
        n_minus = self.refractive_index(lam_nm - step_nm, temperature_c, axis)
        return (n_plus - n_minus) / (2.0 * step_nm)

    def group_index(
        self,
        lam_nm: float,
        temperature_c: float = 25.0,
        axis: Axis | None = None,
    ) -> float:
        """Group index n_g = n - lambda dn/dlambda."""
        n = self.refractive_index(lam_nm, temperature_c, axis)
        return n - lam_nm * self.dn_dlambda(lam_nm, temperature_c, axis)

    def dn_dt(
        self,
        lam_nm: float,
        temperature_c: float = 25.0,
        axis: Axis | None = None,
        step_c: float = 0.5,
    ) -> float:
        """Numerical thermo-optic coefficient dn/dT in 1/degC."""
        n_plus = self.refractive_index(lam_nm, temperature_c + step_c, axis)
        n_minus = self.refractive_index(lam_nm, temperature_c - step_c, axis)
        return (n_plus - n_minus) / (2.0 * step_c)


# --------------------------------------------------------------------------
# Lithium niobate
# --------------------------------------------------------------------------

_MGO_CLN_E = TemperatureSellmeier(
    a=(5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2),
    b=(2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4),
)
_MGO_CLN_O = TemperatureSellmeier(
    a=(5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97e-2),
    b=(2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4),
)
_CLN_E = TemperatureSellmeier(
    a=(5.35583, 0.100473, 0.20692, 100.0, 11.34927, 1.5334e-2),
    b=(4.629e-7, 3.862e-8, -0.89e-8, 2.657e-5),
)

MGO_LN = Material(
    name="MgO:LN",
    description="5 mol% MgO-doped congruent LiNbO3 (Gayer 2008)",
    axes={"e": _MGO_CLN_E, "o": _MGO_CLN_O},
    default_axis="e",
    valid_range=WavelengthRange(500.0, 4000.0),
    d_eff_pm_per_v=14.0,
    thermal_expansion=(1.54e-5, 5.3e-9),
)

CLN = Material(
    name="CLN",
    description="Undoped congruent LiNbO3, extraordinary axis only (Jundt 1997)",
    axes={"e": _CLN_E},
    default_axis="e",
    valid_range=WavelengthRange(400.0, 5000.0),
    d_eff_pm_per_v=14.0,
    thermal_expansion=(1.54e-5, 5.3e-9),
)

# --------------------------------------------------------------------------
# KTP
# --------------------------------------------------------------------------

_KTP_X = SellmeierLinearDnDt(
    sellmeier=(3.29100, 0.04140, 0.03978, 9.35522, 31.45571),
    dndt_poly=(0.1717, -0.5353, 0.8416, 0.1627),
)
_KTP_Y = SellmeierLinearDnDt(
    sellmeier=(3.45018, 0.04341, 0.04597, 16.98825, 39.43799),
    dndt_poly=(0.1997, -0.4063, 0.5154, 0.5425),
)
_KTP_Z = SellmeierLinearDnDt(
    sellmeier=(4.59423, 0.06206, 0.04763, 110.80672, 86.12171),
    dndt_poly=(0.9221, -2.9220, 3.6677, -0.1897),
)

KTP = Material(
    name="KTP",
    description="Flux-grown KTiOPO4 (Kato & Takaoka 2002)",
    axes={"x": _KTP_X, "y": _KTP_Y, "z": _KTP_Z},
    default_axis="z",
    valid_range=WavelengthRange(430.0, 3540.0),
    d_eff_pm_per_v=9.5,
    thermal_expansion=(6.7e-6, 0.0),
)

_REGISTRY: Dict[str, Material] = {
    "mgo:ln": MGO_LN,
    "mgoln": MGO_LN,
    "ppln": MGO_LN,
    "cln": CLN,
    "ln": CLN,
    "ktp": KTP,
    "ppktp": KTP,
}


def list_materials() -> Sequence[str]:
    """Canonical material names, in a stable order."""
    seen = []
    for m in _REGISTRY.values():
        if m.name not in seen:
            seen.append(m.name)
    return tuple(seen)


def get_material(name: str | Material) -> Material:
    """Look up a material by (case-insensitive) name or alias."""
    if isinstance(name, Material):
        return name
    key = name.strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise MaterialError(
            f"unknown material {name!r}; known: {', '.join(list_materials())}"
        ) from None
