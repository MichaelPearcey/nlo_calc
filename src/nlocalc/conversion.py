"""Power conversion models for SHG, DFG/SFG and cascaded chi(2):chi(2) devices.

Model summary
-------------
Normalised efficiency ``eta`` is quoted the way device vendors do, in
``%/(W cm^2)``: for an undepleted fundamental of on-chip power ``P_w`` in a
device of length ``L`` the second harmonic is

    P_2w = (eta/100) * L^2 * P_w^2 * sinc^2(dk L / 2)

With pump depletion the standard plane-wave result is used instead:

    P_2w = P_w * tanh^2( sqrt( (eta/100) * L^2 * P_w ) )

which reduces to the undepleted expression for small arguments and never
exceeds the input power.

The cascaded converter (``cascaded_shg_dfg``) follows the signal path of a
PPLN waveguide wavelength converter: the pump is frequency doubled, then the
second harmonic mixes with the signal to create an idler at
``1/lam_i = 2/lam_p - 1/lam_s``. Losses are applied in this order:

1. input coupling loss (pump, signal),
2. nonlinear interaction using the on-chip input powers,
3. propagation loss over the full length for every wave,
4. output coupling loss,
5. optional filter extinction on pump, signal and SH residuals.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, Mapping

from . import units
from .materials import Material, get_material
from .phasematching import cascaded_idler_wavelength, dfg_wavelength, sinc_squared

EPS0 = 8.8541878128e-12
"""Vacuum permittivity [F/m]."""


class ConversionError(ValueError):
    pass


# --------------------------------------------------------------------------
# Normalised efficiency
# --------------------------------------------------------------------------

def normalized_efficiency_from_deff(
    material: str | Material,
    lam_fund_nm: float,
    mode_area_um2: float,
    temperature_c: float = 25.0,
    qpm_order: int = 1,
    duty_cycle: float = 0.5,
) -> float:
    """Normalised SHG efficiency in %/(W cm^2) from the material d_eff.

    Uses the plane-wave expression with the first-order QPM reduction
    ``d_Q = (2/pi) d_eff`` (scaled for other orders / duty cycles):

        eta = 8 pi^2 d_Q^2 / (eps0 c n_w^2 n_2w lam_w^2 A_eff)
    """
    from .phasematching import grating_fourier_amplitude

    mat = get_material(material)
    if mode_area_um2 <= 0.0:
        raise ConversionError("mode area must be positive")
    n_w = mat.refractive_index(lam_fund_nm, temperature_c)
    n_2w = mat.refractive_index(lam_fund_nm / 2.0, temperature_c)
    d_q = mat.d_eff_pm_per_v * 1e-12 * grating_fourier_amplitude(qpm_order, duty_cycle)
    lam_m = lam_fund_nm * 1e-9
    area_m2 = mode_area_um2 * 1e-12
    eta_per_w_m2 = 8.0 * math.pi**2 * d_q**2 / (EPS0 * units.C_M_PER_S * n_w**2 * n_2w * lam_m**2 * area_m2)
    return eta_per_w_m2 * 1e-4 * 100.0  # 1/(W m^2) -> %/(W cm^2)


def pct_per_w_cm2_to_si(eta_pct: float) -> float:
    """%/(W cm^2) -> 1/(W m^2)."""
    return eta_pct / 100.0 * 1e4


# --------------------------------------------------------------------------
# Single-stage processes
# --------------------------------------------------------------------------

def shg_power_w(
    pump_w: float,
    eta_pct_per_w_cm2: float,
    length_cm: float,
    dk_l_half: float = 0.0,
    depleted: bool = True,
) -> float:
    """Second-harmonic power for an on-chip fundamental of ``pump_w`` Watts.

    ``dk_l_half`` is ``dk*L/2`` (dimensionless); zero means perfectly phase
    matched.
    """
    if pump_w < 0.0 or eta_pct_per_w_cm2 < 0.0 or length_cm < 0.0:
        raise ConversionError("powers, efficiencies and lengths must be non-negative")
    eta = eta_pct_per_w_cm2 / 100.0
    pm = float(sinc_squared(dk_l_half))
    if not depleted:
        return eta * length_cm**2 * pump_w**2 * pm
    arg = math.sqrt(eta * length_cm**2 * pump_w * pm)
    return pump_w * math.tanh(arg) ** 2


def shg_conversion_efficiency(pump_w: float, eta_pct_per_w_cm2: float, length_cm: float, depleted: bool = True) -> float:
    """Fractional conversion P_2w / P_w (0..1)."""
    if pump_w == 0.0:
        return 0.0
    return shg_power_w(pump_w, eta_pct_per_w_cm2, length_cm, depleted=depleted) / pump_w


def dfg_power_w(
    pump_w: float,
    signal_w: float,
    eta_pct_per_w_cm2: float,
    length_cm: float,
    lam_signal_nm: float | None = None,
    lam_idler_nm: float | None = None,
    dk_l_half: float = 0.0,
) -> float:
    """Idler power from DFG in the small-signal limit.

    ``P_i = (eta/100) L^2 P_p P_s sinc^2(dk L/2) * (lam_s / lam_i)``

    The final factor converts the signal-photon conversion probability into a
    power ratio (each converted photon carries less energy at the longer
    idler wavelength). It is omitted when the wavelengths are not supplied.
    """
    if min(pump_w, signal_w, eta_pct_per_w_cm2, length_cm) < 0.0:
        raise ConversionError("powers, efficiencies and lengths must be non-negative")
    eta = eta_pct_per_w_cm2 / 100.0
    photon_factor = 1.0
    if lam_signal_nm is not None and lam_idler_nm is not None:
        photon_factor = lam_signal_nm / lam_idler_nm
    return eta * length_cm**2 * pump_w * signal_w * float(sinc_squared(dk_l_half)) * photon_factor


def sfg_power_w(
    p1_w: float,
    p2_w: float,
    eta_pct_per_w_cm2: float,
    length_cm: float,
    dk_l_half: float = 0.0,
) -> float:
    """Sum-frequency power in the small-signal limit (no photon-energy factor)."""
    return dfg_power_w(p1_w, p2_w, eta_pct_per_w_cm2, length_cm, dk_l_half=dk_l_half)


# --------------------------------------------------------------------------
# Cascaded SHG + DFG wavelength converter
# --------------------------------------------------------------------------

@dataclass
class ConverterParams:
    """All inputs for the cascaded converter model. Units in the field names."""

    eta_shg_pct_per_w_cm2: float = 1000.0
    eta_dfg_pct_per_w_cm2: float = 1000.0
    length_cm: float = 1.0
    pump_dbm: float = 20.0
    signal_dbm: float = 15.0
    lam_pump_nm: float = 1550.0
    lam_signal_nm: float = 1530.0
    in_coupling_db_pump: float = 1.5
    in_coupling_db_signal: float = 1.5
    out_coupling_db: float = 1.5
    prop_db_per_cm_pump: float = 0.1
    prop_db_per_cm_signal: float = 0.1
    prop_db_per_cm_sh: float = 0.3
    prop_db_per_cm_idler: float = 0.1
    filter_db_pump: float = 0.0
    filter_db_signal: float = 0.0
    filter_db_sh: float = 0.0
    depleted: bool = True

    _LIMITS: Mapping[str, tuple] = field(
        default_factory=lambda: {
            "eta_shg_pct_per_w_cm2": (0.0, 1e6),
            "eta_dfg_pct_per_w_cm2": (0.0, 1e6),
            "length_cm": (1e-3, 100.0),
            "pump_dbm": (-60.0, 50.0),
            "signal_dbm": (-60.0, 50.0),
            "lam_pump_nm": (300.0, 6000.0),
            "lam_signal_nm": (300.0, 6000.0),
        },
        repr=False,
        compare=False,
    )

    def validate(self) -> None:
        for name, (lo, hi) in self._LIMITS.items():
            v = getattr(self, name)
            if not (lo <= v <= hi):
                raise ConversionError(f"{name}={v} outside allowed range [{lo}, {hi}]")
        for f in fields(self):
            if f.name.startswith(("in_coupling", "out_coupling", "prop_", "filter_")):
                if getattr(self, f.name) < 0.0:
                    raise ConversionError(f"{f.name} must be non-negative")
        if self.lam_signal_nm <= self.lam_pump_nm / 2.0:
            raise ConversionError("signal wavelength must exceed half the pump wavelength")

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.pop("_LIMITS", None)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConverterParams":
        allowed = {f.name for f in fields(cls) if not f.name.startswith("_")}
        unknown = set(data) - allowed
        if unknown:
            raise ConversionError(f"unknown converter parameters: {sorted(unknown)}")
        return cls(**{k: data[k] for k in allowed if k in data})  # type: ignore[arg-type]


@dataclass(frozen=True)
class WavePowers:
    pump: float
    signal: float
    sh: float
    idler: float

    def as_dict(self) -> Dict[str, float]:
        return {"pump": self.pump, "signal": self.signal, "sh": self.sh, "idler": self.idler}

    def map(self, fn) -> "WavePowers":
        return WavePowers(fn(self.pump), fn(self.signal), fn(self.sh), fn(self.idler))


@dataclass(frozen=True)
class ConverterResult:
    params: ConverterParams
    wavelengths_nm: WavePowers
    on_chip_input_w: WavePowers
    after_interaction_w: WavePowers
    after_propagation_w: WavePowers
    output_w: WavePowers

    @property
    def output_dbm(self) -> WavePowers:
        return self.output_w.map(units.w_to_dbm)

    @property
    def idler_dbm(self) -> float:
        return units.w_to_dbm(self.output_w.idler)

    def conversion_efficiency_db(self) -> float:
        """Fibre-to-fibre conversion efficiency: idler out over signal in."""
        signal_in_w = units.dbm_to_w(self.params.signal_dbm)
        if signal_in_w <= 0.0 or self.output_w.idler <= 0.0:
            return -math.inf
        return 10.0 * math.log10(self.output_w.idler / signal_in_w)

    def to_dict(self) -> Dict[str, object]:
        return {
            "params": self.params.to_dict(),
            "wavelengths_nm": self.wavelengths_nm.as_dict(),
            "on_chip_input_w": self.on_chip_input_w.as_dict(),
            "after_interaction_w": self.after_interaction_w.as_dict(),
            "after_propagation_w": self.after_propagation_w.as_dict(),
            "output_w": self.output_w.as_dict(),
            "output_dbm": self.output_dbm.as_dict(),
            "conversion_efficiency_db": self.conversion_efficiency_db(),
        }


def cascaded_shg_dfg(params: ConverterParams) -> ConverterResult:
    """Evaluate the cascaded SHG+DFG converter described in the module docstring."""
    params.validate()
    p = params
    lam_i = cascaded_idler_wavelength(p.lam_pump_nm, p.lam_signal_nm)
    lam_sh = p.lam_pump_nm / 2.0
    wavelengths = WavePowers(p.lam_pump_nm, p.lam_signal_nm, lam_sh, lam_i)

    pump_in = units.dbm_to_w(p.pump_dbm) * units.db_to_linear(p.in_coupling_db_pump)
    signal_in = units.dbm_to_w(p.signal_dbm) * units.db_to_linear(p.in_coupling_db_signal)
    on_chip = WavePowers(pump_in, signal_in, 0.0, 0.0)

    sh_gen = shg_power_w(pump_in, p.eta_shg_pct_per_w_cm2, p.length_cm, depleted=p.depleted)
    pump_after = pump_in - sh_gen if p.depleted else pump_in
    idler_gen = dfg_power_w(
        sh_gen, signal_in, p.eta_dfg_pct_per_w_cm2, p.length_cm,
        lam_signal_nm=p.lam_signal_nm, lam_idler_nm=lam_i,
    )
    if p.depleted:
        # each idler photon consumes one SH photon; signal photons are conserved
        sh_consumed = idler_gen * (lam_i / lam_sh)
        sh_after = max(sh_gen - sh_consumed, 0.0)
    else:
        sh_after = sh_gen
    after_interaction = WavePowers(pump_after, signal_in, sh_after, idler_gen)

    def propagate(power: float, db_per_cm: float) -> float:
        return power * units.db_to_linear(db_per_cm * p.length_cm)

    after_prop = WavePowers(
        propagate(after_interaction.pump, p.prop_db_per_cm_pump),
        propagate(after_interaction.signal, p.prop_db_per_cm_signal),
        propagate(after_interaction.sh, p.prop_db_per_cm_sh),
        propagate(after_interaction.idler, p.prop_db_per_cm_idler),
    )

    out_cpl = units.db_to_linear(p.out_coupling_db)

    def filtered(power: float, extinction_db: float) -> float:
        return power * 10.0 ** (-extinction_db / 20.0)

    output = WavePowers(
        filtered(after_prop.pump * out_cpl, p.filter_db_pump),
        filtered(after_prop.signal * out_cpl, p.filter_db_signal),
        filtered(after_prop.sh * out_cpl, p.filter_db_sh),
        after_prop.idler * out_cpl,
    )
    return ConverterResult(p, wavelengths, on_chip, after_interaction, after_prop, output)


# --------------------------------------------------------------------------
# Loss budget helper
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LossBudget:
    """Itemised loss budget in dB for one wave through a device."""

    input_coupling_db: float
    propagation_db: float
    output_coupling_db: float
    filter_db: float = 0.0

    @property
    def total_db(self) -> float:
        return self.input_coupling_db + self.propagation_db + self.output_coupling_db + self.filter_db

    @property
    def transmission(self) -> float:
        return units.db_to_linear(self.total_db)

    def items(self) -> Dict[str, float]:
        return {
            "input_coupling_db": self.input_coupling_db,
            "propagation_db": self.propagation_db,
            "output_coupling_db": self.output_coupling_db,
            "filter_db": self.filter_db,
            "total_db": self.total_db,
        }


def loss_budget(params: ConverterParams, wave: str) -> LossBudget:
    """Loss budget for ``wave`` in ``{"pump", "signal", "sh", "idler"}``.

    Generated waves (sh, idler) have no input-coupling term.
    """
    p = params
    table = {
        "pump": (p.in_coupling_db_pump, p.prop_db_per_cm_pump, p.filter_db_pump),
        "signal": (p.in_coupling_db_signal, p.prop_db_per_cm_signal, p.filter_db_signal),
        "sh": (0.0, p.prop_db_per_cm_sh, p.filter_db_sh),
        "idler": (0.0, p.prop_db_per_cm_idler, 0.0),
    }
    try:
        in_cpl, prop, filt = table[wave]
    except KeyError:
        raise ConversionError(f"unknown wave {wave!r}") from None
    return LossBudget(in_cpl, prop * p.length_cm, p.out_coupling_db, filt)
