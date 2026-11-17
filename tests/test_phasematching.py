import math

import numpy as np
import pytest

from nlocalc.materials import MGO_LN
from nlocalc.phasematching import (
    Interaction,
    PhaseMatchingError,
    analytic_temperature_bandwidth_c,
    cascaded_idler_wavelength,
    coherence_length_um,
    dfg_wavelength,
    grating_fourier_amplitude,
    phase_matching_efficiency,
    qpm_mismatch,
    qpm_period_um,
    sfg_wavelength,
    shg_wavelength,
    sinc_squared,
    temperature_bandwidth_c,
    temperature_tuning_curve,
    wavelength_bandwidth_nm,
    wavelength_tuning_curve,
)


def test_wavelength_relations():
    assert shg_wavelength(1550.0) == 775.0
    assert sfg_wavelength(1550.0, 1550.0) == pytest.approx(775.0)
    assert sfg_wavelength(1064.0, 1550.0) == pytest.approx(630.9, abs=0.1)
    assert dfg_wavelength(775.0, 1530.0) == pytest.approx(1570.53, abs=0.01)
    assert cascaded_idler_wavelength(1550.0, 1530.0) == pytest.approx(dfg_wavelength(775.0, 1530.0))
    # energy conservation: 1/l3 = 1/l1 + 1/l2
    l3 = sfg_wavelength(1300.0, 1700.0)
    assert 1 / l3 == pytest.approx(1 / 1300.0 + 1 / 1700.0)


def test_wavelength_relation_errors():
    with pytest.raises(PhaseMatchingError):
        shg_wavelength(0.0)
    with pytest.raises(PhaseMatchingError):
        sfg_wavelength(-1.0, 1550.0)
    with pytest.raises(PhaseMatchingError):
        dfg_wavelength(1550.0, 775.0)  # pump must be shorter
    with pytest.raises(PhaseMatchingError):
        cascaded_idler_wavelength(1550.0, 700.0)


def test_interaction_constructors():
    shg = Interaction.shg("ppln", 1550.0)
    assert shg.wavelengths_nm() == pytest.approx((1550.0, 1550.0, 775.0))
    assert shg.material is MGO_LN

    dfg = Interaction.dfg("ppln", 775.0, 1530.0)
    l1, l2, l3 = dfg.wavelengths_nm()
    assert l1 == 1530.0
    assert l3 == pytest.approx(775.0)
    assert l2 == pytest.approx(1570.53, abs=0.01)

    sfg = Interaction.sfg("ktp", 1064.0, 1550.0)
    assert sfg.lam_3_nm == pytest.approx(630.9, abs=0.1)


def test_qpm_period_ppln_shg():
    # Standard PPLN SHG periods: ~19 um at 1550 nm and ~7 um at 1064 nm.
    assert qpm_period_um(Interaction.shg("ppln", 1550.0), 25.0) == pytest.approx(19.39, abs=0.05)
    assert qpm_period_um(Interaction.shg("ppln", 1064.0), 25.0) == pytest.approx(6.97, abs=0.05)
    # PPKTP SHG at 1064 nm uses ~9 um gratings.
    assert qpm_period_um(Interaction.shg("ktp", 1064.0), 25.0) == pytest.approx(9.01, abs=0.05)


def test_qpm_period_scales_with_order_and_temperature():
    inter = Interaction.shg("ppln", 1550.0)
    p1 = qpm_period_um(inter, 25.0, order=1)
    p3 = qpm_period_um(inter, 25.0, order=3)
    assert p3 == pytest.approx(3 * p1)
    # PPLN period needed shrinks as the crystal is heated
    assert qpm_period_um(inter, 100.0) < p1
    with pytest.raises(PhaseMatchingError):
        qpm_period_um(inter, 25.0, order=0)


def test_qpm_mismatch_zero_at_design_period():
    inter = Interaction.shg("ppln", 1550.0)
    period = qpm_period_um(inter, 50.0)
    assert qpm_mismatch(inter, period, 50.0) == pytest.approx(0.0, abs=1e-12)
    assert qpm_mismatch(inter, period, 60.0) != pytest.approx(0.0, abs=1e-6)
    with pytest.raises(PhaseMatchingError):
        qpm_mismatch(inter, 0.0)


def test_coherence_length():
    assert coherence_length_um(math.pi) == pytest.approx(1.0)
    assert coherence_length_um(-math.pi / 2) == pytest.approx(2.0)
    assert coherence_length_um(0.0) == math.inf
    inter = Interaction.shg("ppln", 1550.0)
    # the QPM period is twice the coherence length
    assert qpm_period_um(inter) == pytest.approx(2 * coherence_length_um(inter.bulk_mismatch()))


def test_grating_fourier_amplitude():
    assert grating_fourier_amplitude(1) == pytest.approx(2 / math.pi)
    assert grating_fourier_amplitude(2) == pytest.approx(0.0, abs=1e-12)
    assert grating_fourier_amplitude(3) == pytest.approx(2 / (3 * math.pi))
    assert grating_fourier_amplitude(1, duty_cycle=0.25) == pytest.approx(math.sqrt(2) / math.pi)
    with pytest.raises(PhaseMatchingError):
        grating_fourier_amplitude(1, duty_cycle=1.0)
    with pytest.raises(PhaseMatchingError):
        grating_fourier_amplitude(0)


def test_sinc_squared_basic_properties():
    assert sinc_squared(0.0) == 1.0
    x = np.linspace(-3.0, 3.0, 601)
    y = sinc_squared(x)
    assert y.shape == x.shape
    assert np.all(y <= 1.0) and np.all(y >= 0.0)
    assert np.allclose(y, y[::-1])  # symmetric
    assert np.all(np.diff(y[300:330]) < 0)  # decreasing just past the peak


def test_phase_matching_efficiency_peak():
    assert phase_matching_efficiency(0.0, 10.0) == pytest.approx(1.0)
    assert phase_matching_efficiency(1e-3, 10.0) < 1.0
    with pytest.raises(PhaseMatchingError):
        phase_matching_efficiency(0.0, 0.0)


def test_temperature_tuning_curve_peaks_at_design_temperature():
    inter = Interaction.shg("ppln", 1550.0)
    period = qpm_period_um(inter, 60.0)
    temps = np.linspace(20.0, 100.0, 801)
    eff = temperature_tuning_curve(inter, period, 10.0, temps)
    assert eff.shape == temps.shape
    assert temps[int(np.argmax(eff))] == pytest.approx(60.0, abs=0.1)
    assert eff.max() == pytest.approx(1.0, abs=1e-6)


def test_wavelength_tuning_curve_peaks_at_design_wavelength():
    inter = Interaction.shg("ppln", 1550.0)
    period = qpm_period_um(inter, 25.0)
    lams = np.linspace(1540.0, 1560.0, 2001)
    eff = wavelength_tuning_curve(inter, period, 10.0, lams, fixed_wave="shg")
    assert lams[int(np.argmax(eff))] == pytest.approx(1550.0, abs=0.01)


def test_bandwidths_shrink_with_length():
    inter = Interaction.shg("ppln", 1550.0)
    t10 = temperature_bandwidth_c(inter, 10.0)
    t20 = temperature_bandwidth_c(inter, 20.0)
    assert t20 == pytest.approx(t10 / 2.0, rel=0.02)
    w10 = wavelength_bandwidth_nm(inter, 10.0, fixed_wave="shg", span_nm=5.0)
    w20 = wavelength_bandwidth_nm(inter, 20.0, fixed_wave="shg", span_nm=5.0)
    assert w20 == pytest.approx(w10 / 2.0, rel=0.02)
    assert 0.1 < w10 < 5.0


def test_analytic_bandwidth_scales_inversely_with_length():
    inter = Interaction.shg("ppln", 1550.0)
    a10 = analytic_temperature_bandwidth_c(inter, 10.0)
    a40 = analytic_temperature_bandwidth_c(inter, 40.0)
    assert a40 == pytest.approx(a10 / 4.0, rel=1e-6)
    assert a10 > 0.0


def test_bandwidth_requires_curve_to_drop():
    inter = Interaction.shg("ppln", 1550.0)
    # a 0.1 mm crystal has a bandwidth of hundreds of degrees: span too small
    with pytest.raises(PhaseMatchingError):
        temperature_bandwidth_c(inter, 0.1, span_c=5.0)
