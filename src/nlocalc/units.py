"""Unit conversions shared across the package.

Conventions used throughout ``nlocalc``:

* wavelengths are in nanometres (``*_nm``) unless the name says otherwise,
* optical power is in Watts (``*_W``) or dBm (``*_dBm``),
* losses and extinction ratios are positive numbers in dB,
* temperatures are in degrees Celsius.
"""

from __future__ import annotations

import math

C_M_PER_S = 299_792_458.0
"""Speed of light in vacuum [m/s]."""

NEG_INF_DBM = -math.inf


def dbm_to_w(p_dbm: float) -> float:
    """Convert a power in dBm to Watts."""
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


def w_to_dbm(p_w: float) -> float:
    """Convert a power in Watts to dBm. Non-positive powers map to ``-inf``."""
    if p_w <= 0.0:
        return NEG_INF_DBM
    return 10.0 * math.log10(p_w) + 30.0


def db_to_linear(loss_db: float) -> float:
    """Return the power transmission factor for a loss of ``loss_db`` dB.

    A loss of 3 dB gives roughly 0.5; a loss of 0 dB gives exactly 1.0.
    Negative values represent gain.
    """
    return 10.0 ** (-loss_db / 10.0)


def linear_to_db(transmission: float) -> float:
    """Return the loss in dB corresponding to a power transmission factor."""
    if transmission <= 0.0:
        return math.inf
    return -10.0 * math.log10(transmission)


def nm_to_um(lam_nm: float) -> float:
    return lam_nm * 1e-3


def um_to_nm(lam_um: float) -> float:
    return lam_um * 1e3


def nm_to_thz(lam_nm: float) -> float:
    """Vacuum wavelength [nm] to optical frequency [THz]."""
    if lam_nm <= 0.0:
        raise ValueError("wavelength must be positive")
    return C_M_PER_S / (lam_nm * 1e-9) / 1e12


def thz_to_nm(f_thz: float) -> float:
    """Optical frequency [THz] to vacuum wavelength [nm]."""
    if f_thz <= 0.0:
        raise ValueError("frequency must be positive")
    return C_M_PER_S / (f_thz * 1e12) * 1e9


def nm_to_angular_frequency(lam_nm: float) -> float:
    """Vacuum wavelength [nm] to angular frequency [rad/s]."""
    return 2.0 * math.pi * nm_to_thz(lam_nm) * 1e12


def wavenumber_rad_per_um(lam_nm: float, n: float = 1.0) -> float:
    """Propagation constant k = 2*pi*n/lambda in rad/um."""
    return 2.0 * math.pi * n / nm_to_um(lam_nm)


def celsius_to_kelvin(t_c: float) -> float:
    return t_c + 273.15


def kelvin_to_celsius(t_k: float) -> float:
    return t_k - 273.15


def add_powers_dbm(*levels_dbm: float) -> float:
    """Incoherently sum several power levels given in dBm.

    The sum is performed in the linear (Watt) domain, so adding two equal
    levels yields a result 3.01 dB higher than either input.
    """
    total_w = sum(dbm_to_w(p) for p in levels_dbm if math.isfinite(p))
    return w_to_dbm(total_w)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the closed interval ``[low, high]``."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))
