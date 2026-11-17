"""Quasi-phase-matching (QPM) design and tuning-curve calculations.

Sign conventions
----------------
For a three-wave process with ``1/lam_3 = 1/lam_1 + 1/lam_2`` (so the
highest-frequency wave is "3"), the wavevector mismatch of a bulk crystal is

    dk = k_3 - k_1 - k_2 = 2*pi*( n_3/lam_3 - n_1/lam_1 - n_2/lam_2 )

and a first-order QPM grating of period ``Lambda`` adds ``-2*pi*m/Lambda``:

    dk_qpm = dk - 2*pi*m / Lambda

Perfect phase matching is ``dk_qpm == 0``, i.e. ``Lambda = 2*pi*m / dk``.

All wavelengths are in nm, lengths in um unless the argument name says
otherwise. Wavevectors are returned in rad/um.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .materials import Material, get_material
from .units import nm_to_um


class PhaseMatchingError(ValueError):
    pass


# --------------------------------------------------------------------------
# Wavelength bookkeeping
# --------------------------------------------------------------------------

def shg_wavelength(lam_fund_nm: float) -> float:
    """Second-harmonic wavelength of a fundamental at ``lam_fund_nm``."""
    if lam_fund_nm <= 0.0:
        raise PhaseMatchingError("fundamental wavelength must be positive")
    return lam_fund_nm / 2.0


def sfg_wavelength(lam_1_nm: float, lam_2_nm: float) -> float:
    """Sum-frequency wavelength: 1/lam_3 = 1/lam_1 + 1/lam_2."""
    if lam_1_nm <= 0.0 or lam_2_nm <= 0.0:
        raise PhaseMatchingError("input wavelengths must be positive")
    return 1.0 / (1.0 / lam_1_nm + 1.0 / lam_2_nm)


def dfg_wavelength(lam_pump_nm: float, lam_signal_nm: float) -> float:
    """Difference-frequency (idler) wavelength: 1/lam_i = 1/lam_p - 1/lam_s.

    The pump must be the shorter wavelength (higher frequency).
    """
    if lam_pump_nm <= 0.0 or lam_signal_nm <= 0.0:
        raise PhaseMatchingError("input wavelengths must be positive")
    inv = 1.0 / lam_pump_nm - 1.0 / lam_signal_nm
    if inv <= 0.0:
        raise PhaseMatchingError(
            "DFG requires the pump wavelength to be shorter than the signal wavelength"
        )
    return 1.0 / inv


def cascaded_idler_wavelength(lam_pump_nm: float, lam_signal_nm: float) -> float:
    """Idler of a cascaded SHG+DFG process: 1/lam_i = 2/lam_p - 1/lam_s.

    The pump is first frequency doubled, then mixed with the signal.
    """
    inv = 2.0 / lam_pump_nm - 1.0 / lam_signal_nm
    if inv <= 0.0:
        raise PhaseMatchingError("signal wavelength too short for cascaded SHG+DFG with this pump")
    return 1.0 / inv


# --------------------------------------------------------------------------
# Wavevector mismatch
# --------------------------------------------------------------------------

def wavevector_rad_per_um(lam_nm: float, n: float) -> float:
    return 2.0 * math.pi * n / nm_to_um(lam_nm)


@dataclass(frozen=True)
class Interaction:
    """Three-wave interaction specified by its two long-wavelength inputs.

    ``lam_1_nm`` and ``lam_2_nm`` are the two lower-frequency waves. For SHG
    they are equal. ``axes`` gives the polarisation axis used for each of
    (wave 1, wave 2, wave 3); ``None`` means the material default (type-0
    "eee" interaction in PPLN).
    """

    material: Material
    lam_1_nm: float
    lam_2_nm: float
    axes: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)

    @classmethod
    def shg(cls, material: str | Material, lam_fund_nm: float, axis: Optional[str] = None) -> "Interaction":
        return cls(get_material(material), lam_fund_nm, lam_fund_nm, (axis, axis, axis))

    @classmethod
    def sfg(cls, material: str | Material, lam_1_nm: float, lam_2_nm: float) -> "Interaction":
        return cls(get_material(material), lam_1_nm, lam_2_nm)

    @classmethod
    def dfg(cls, material: str | Material, lam_pump_nm: float, lam_signal_nm: float) -> "Interaction":
        """DFG of pump and signal; the output idler and the signal are the
        two low-frequency waves while the pump is wave 3."""
        lam_idler = dfg_wavelength(lam_pump_nm, lam_signal_nm)
        return cls(get_material(material), lam_signal_nm, lam_idler)

    @property
    def lam_3_nm(self) -> float:
        return sfg_wavelength(self.lam_1_nm, self.lam_2_nm)

    def wavelengths_nm(self) -> Tuple[float, float, float]:
        return (self.lam_1_nm, self.lam_2_nm, self.lam_3_nm)

    def indices(self, temperature_c: float = 25.0) -> Tuple[float, float, float]:
        a1, a2, a3 = self.axes
        m = self.material
        return (
            m.refractive_index(self.lam_1_nm, temperature_c, a1),
            m.refractive_index(self.lam_2_nm, temperature_c, a2),
            m.refractive_index(self.lam_3_nm, temperature_c, a3),
        )

    def bulk_mismatch(self, temperature_c: float = 25.0) -> float:
        """dk = k_3 - k_1 - k_2 in rad/um (no grating)."""
        n1, n2, n3 = self.indices(temperature_c)
        return (
            wavevector_rad_per_um(self.lam_3_nm, n3)
            - wavevector_rad_per_um(self.lam_1_nm, n1)
            - wavevector_rad_per_um(self.lam_2_nm, n2)
        )

    def with_wavelengths(self, lam_1_nm: float, lam_2_nm: float) -> "Interaction":
        return Interaction(self.material, lam_1_nm, lam_2_nm, self.axes)


# --------------------------------------------------------------------------
# QPM period and coherence length
# --------------------------------------------------------------------------

def coherence_length_um(dk_rad_per_um: float) -> float:
    """Coherence length L_c = pi / |dk|."""
    if dk_rad_per_um == 0.0:
        return math.inf
    return math.pi / abs(dk_rad_per_um)


def qpm_period_um(interaction: Interaction, temperature_c: float = 25.0, order: int = 1) -> float:
    """Grating period that phase matches ``interaction`` at ``temperature_c``.

    ``Lambda = 2*pi*m / dk`` for QPM order ``m``. Only odd orders have a
    non-zero Fourier coefficient for a 50% duty-cycle grating, but any
    positive integer is accepted.
    """
    if order < 1:
        raise PhaseMatchingError("QPM order must be a positive integer")
    dk = interaction.bulk_mismatch(temperature_c)
    if dk <= 0.0:
        raise PhaseMatchingError("bulk mismatch is non-positive; QPM period undefined")
    return 2.0 * math.pi * order / dk


def qpm_mismatch(
    interaction: Interaction,
    period_um: float,
    temperature_c: float = 25.0,
    order: int = 1,
) -> float:
    """Residual mismatch dk - 2*pi*m/Lambda in rad/um."""
    if period_um <= 0.0:
        raise PhaseMatchingError("grating period must be positive")
    return interaction.bulk_mismatch(temperature_c) - 2.0 * math.pi * order / period_um


def grating_fourier_amplitude(order: int, duty_cycle: float = 0.5) -> float:
    """Normalised amplitude of the m-th Fourier component of a square grating.

    ``G_m = (2 / (m*pi)) * sin(m*pi*D)`` for duty cycle ``D``. At ``D=0.5``
    the first-order amplitude is ``2/pi``; even orders vanish.
    """
    if order < 1:
        raise PhaseMatchingError("QPM order must be a positive integer")
    if not 0.0 < duty_cycle < 1.0:
        raise PhaseMatchingError("duty cycle must lie strictly between 0 and 1")
    return abs(2.0 / (order * math.pi) * math.sin(order * math.pi * duty_cycle))


# --------------------------------------------------------------------------
# Tuning curves
# --------------------------------------------------------------------------

def sinc_squared(x):
    """Normalised sinc^2 with the physics convention: (sin x / x)^2, value 1 at 0."""
    x = np.asarray(x, dtype=float)
    out = np.sinc(x) ** 2
    return out if out.ndim else float(out)


def phase_matching_efficiency(dk_rad_per_um, length_mm: float):
    """Relative conversion efficiency sinc^2(dk * L / 2) for a uniform grating."""
    if length_mm <= 0.0:
        raise PhaseMatchingError("crystal length must be positive")
    length_um = length_mm * 1e3
    return sinc_squared(np.asarray(dk_rad_per_um, dtype=float) * length_um / 2.0)


def temperature_tuning_curve(
    interaction: Interaction,
    period_um: float,
    length_mm: float,
    temperatures_c: Sequence[float],
    order: int = 1,
) -> np.ndarray:
    """Relative efficiency versus temperature for a fixed grating period."""
    temps = np.asarray(temperatures_c, dtype=float)
    dk = np.array([qpm_mismatch(interaction, period_um, float(t), order) for t in temps])
    return phase_matching_efficiency(dk, length_mm)


def wavelength_tuning_curve(
    interaction: Interaction,
    period_um: float,
    length_mm: float,
    wavelengths_1_nm: Sequence[float],
    temperature_c: float = 25.0,
    order: int = 1,
    fixed_wave: str = "2",
) -> np.ndarray:
    """Relative efficiency versus the wavelength of wave 1.

    ``fixed_wave`` selects what is held constant while wave 1 is scanned:

    * ``"2"``  -- wave 2 stays fixed (e.g. DFG with a fixed pump and
      scanned signal, or SFG with one fixed input),
    * ``"shg"`` -- wave 2 tracks wave 1 (second-harmonic generation).
    """
    lams = np.asarray(wavelengths_1_nm, dtype=float)
    dk = np.empty_like(lams)
    for i, lam in enumerate(lams):
        lam2 = float(lam) if fixed_wave == "shg" else interaction.lam_2_nm
        inter = interaction.with_wavelengths(float(lam), lam2)
        dk[i] = qpm_mismatch(inter, period_um, temperature_c, order)
    return phase_matching_efficiency(dk, length_mm)


def _fwhm_of_curve(x: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of a single-peaked sampled curve."""
    if x.size < 3:
        raise PhaseMatchingError("need at least three samples to estimate a FWHM")
    peak = int(np.argmax(y))
    half = y[peak] / 2.0
    left = peak
    while left > 0 and y[left] > half:
        left -= 1
    right = peak
    while right < y.size - 1 and y[right] > half:
        right += 1
    if y[left] > half or y[right] > half:
        raise PhaseMatchingError("curve does not fall to half maximum within the sampled range")

    def interp(i0: int, i1: int) -> float:
        y0, y1 = y[i0], y[i1]
        if y1 == y0:
            return float(x[i0])
        return float(x[i0] + (half - y0) * (x[i1] - x[i0]) / (y1 - y0))

    x_left = interp(left, left + 1) if left < peak else float(x[left])
    x_right = interp(right - 1, right) if right > peak else float(x[right])
    return x_right - x_left


def temperature_bandwidth_c(
    interaction: Interaction,
    length_mm: float,
    temperature_c: float = 25.0,
    order: int = 1,
    span_c: float = 40.0,
    samples: int = 4001,
) -> float:
    """FWHM temperature acceptance bandwidth [degC] of a crystal phase matched
    at ``temperature_c``. The grating period is chosen to phase match exactly
    at that temperature."""
    period = qpm_period_um(interaction, temperature_c, order)
    temps = np.linspace(temperature_c - span_c, temperature_c + span_c, samples)
    eff = temperature_tuning_curve(interaction, period, length_mm, temps, order)
    return _fwhm_of_curve(temps, eff)


def wavelength_bandwidth_nm(
    interaction: Interaction,
    length_mm: float,
    temperature_c: float = 25.0,
    order: int = 1,
    span_nm: float = 10.0,
    samples: int = 4001,
    fixed_wave: str = "2",
) -> float:
    """FWHM acceptance bandwidth [nm] for wave 1 of an interaction that is
    exactly phase matched at its nominal wavelengths."""
    period = qpm_period_um(interaction, temperature_c, order)
    lams = np.linspace(interaction.lam_1_nm - span_nm, interaction.lam_1_nm + span_nm, samples)
    eff = wavelength_tuning_curve(interaction, period, length_mm, lams, temperature_c, order, fixed_wave)
    return _fwhm_of_curve(lams, eff)


def analytic_temperature_bandwidth_c(
    interaction: Interaction,
    length_mm: float,
    temperature_c: float = 25.0,
    step_c: float = 0.1,
) -> float:
    """First-order analytic estimate of the temperature FWHM.

    For sinc^2(dk L / 2) the half-maximum points lie at ``dk L / 2 = +-1.391557``,
    so ``dT_FWHM = 2 * 1.391557 * 2 / (L * |d(dk)/dT|) = 5.566 / (L |d(dk)/dT|)``.
    """
    length_um = length_mm * 1e3
    dk_plus = interaction.bulk_mismatch(temperature_c + step_c)
    dk_minus = interaction.bulk_mismatch(temperature_c - step_c)
    ddk_dt = (dk_plus - dk_minus) / (2.0 * step_c)
    if ddk_dt == 0.0:
        return math.inf
    return 2.0 * 1.3915573 * 2.0 / (length_um * abs(ddk_dt))


SINC2_HALF_MAX_ARG = 1.3915573
"""Positive x where (sin x / x)^2 = 0.5."""
