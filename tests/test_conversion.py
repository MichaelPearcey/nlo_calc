import math

import pytest

from nlocalc import units
from nlocalc.conversion import (
    ConversionError,
    ConverterParams,
    LossBudget,
    WavePowers,
    cascaded_shg_dfg,
    dfg_power_w,
    loss_budget,
    normalized_efficiency_from_deff,
    pct_per_w_cm2_to_si,
    sfg_power_w,
    shg_conversion_efficiency,
    shg_power_w,
)


def test_shg_undepleted_quadratic():
    p1 = shg_power_w(0.01, 1000.0, 1.0, depleted=False)
    p2 = shg_power_w(0.02, 1000.0, 1.0, depleted=False)
    assert p1 == pytest.approx(10.0 * 1.0 * 0.01**2)
    assert p2 == pytest.approx(4 * p1)
    # length squared
    assert shg_power_w(0.01, 1000.0, 2.0, depleted=False) == pytest.approx(4 * p1)


def test_shg_depleted_limits():
    # small signal: depleted and undepleted agree
    small_u = shg_power_w(1e-4, 100.0, 1.0, depleted=False)
    small_d = shg_power_w(1e-4, 100.0, 1.0, depleted=True)
    assert small_d == pytest.approx(small_u, rel=1e-3)
    # strong pump: cannot exceed the input power
    big = shg_power_w(1.0, 100.0, 1.0, depleted=True)
    assert big < 1.0
    assert big == pytest.approx(math.tanh(1.0) ** 2)
    assert shg_power_w(1e3, 100.0, 1.0, depleted=True) <= 1e3
    assert shg_conversion_efficiency(1.0, 100.0, 1.0) == pytest.approx(big)
    assert shg_conversion_efficiency(0.0, 5000.0, 2.0) == 0.0


def test_shg_phase_mismatch_reduces_output():
    matched = shg_power_w(0.1, 1000.0, 1.0, dk_l_half=0.0)
    detuned = shg_power_w(0.1, 1000.0, 1.0, dk_l_half=0.5)
    assert 0.0 < detuned < matched


def test_shg_rejects_negative():
    with pytest.raises(ConversionError):
        shg_power_w(-1.0, 1000.0, 1.0)
    with pytest.raises(ConversionError):
        dfg_power_w(1.0, -1.0, 1000.0, 1.0)


def test_dfg_bilinear_and_photon_factor():
    base = dfg_power_w(0.1, 0.01, 1000.0, 1.0)
    assert base == pytest.approx(10.0 * 0.1 * 0.01)
    assert dfg_power_w(0.2, 0.01, 1000.0, 1.0) == pytest.approx(2 * base)
    assert dfg_power_w(0.1, 0.02, 1000.0, 1.0) == pytest.approx(2 * base)
    with_photon = dfg_power_w(0.1, 0.01, 1000.0, 1.0, lam_signal_nm=1530.0, lam_idler_nm=1570.0)
    assert with_photon == pytest.approx(base * 1530.0 / 1570.0)
    assert sfg_power_w(0.1, 0.01, 1000.0, 1.0) == pytest.approx(base)


def test_normalized_efficiency_from_deff_scaling():
    eta_1 = normalized_efficiency_from_deff("ppln", 1550.0, 1.0)
    eta_50 = normalized_efficiency_from_deff("ppln", 1550.0, 50.0)
    assert eta_1 == pytest.approx(50 * eta_50)
    # TFLN-like micron-scale modes give ~1000 %/W/cm^2 at 1550 nm
    assert 500.0 < eta_1 < 2000.0
    # third order QPM is 9x weaker
    assert normalized_efficiency_from_deff("ppln", 1550.0, 1.0, qpm_order=3) == pytest.approx(eta_1 / 9.0)
    with pytest.raises(ConversionError):
        normalized_efficiency_from_deff("ppln", 1550.0, 0.0)


def test_pct_per_w_cm2_to_si():
    assert pct_per_w_cm2_to_si(100.0) == pytest.approx(1e4)


def test_converter_default_run_is_consistent():
    params = ConverterParams()
    res = cascaded_shg_dfg(params)
    lam = res.wavelengths_nm
    assert lam.sh == 775.0
    assert lam.idler == pytest.approx(1570.53, abs=0.01)
    out = res.output_dbm
    # signal only sees linear loss: 1.5 + 0.1 + 1.5 = 3.1 dB
    assert out.signal == pytest.approx(params.signal_dbm - 3.1, abs=1e-9)
    # pump is depleted and loses power
    assert out.pump < params.pump_dbm - 3.1
    # generated waves are present
    assert out.sh > -30.0 and out.idler > -30.0
    assert res.conversion_efficiency_db() == pytest.approx(out.idler - params.signal_dbm)
    # energy conservation on chip: total power after interaction <= total in
    total_in = res.on_chip_input_w.pump + res.on_chip_input_w.signal
    ai = res.after_interaction_w
    assert ai.pump + ai.signal + ai.sh + ai.idler <= total_in + 1e-12


def test_converter_undepleted_mode():
    params = ConverterParams(depleted=False)
    res = cascaded_shg_dfg(params)
    # pump passes through with linear loss only
    assert res.output_dbm.pump == pytest.approx(params.pump_dbm - 3.1, abs=1e-9)
    # idler grows ~linearly with signal power in the small-signal model
    params2 = ConverterParams(depleted=False, signal_dbm=params.signal_dbm + 3.0)
    res2 = cascaded_shg_dfg(params2)
    assert res2.output_dbm.idler - res.output_dbm.idler == pytest.approx(3.0, abs=1e-9)


def test_converter_idler_scales_with_pump_squared_undepleted():
    a = cascaded_shg_dfg(ConverterParams(depleted=False, pump_dbm=10.0))
    b = cascaded_shg_dfg(ConverterParams(depleted=False, pump_dbm=13.0))
    # SH ~ P_p^2, idler ~ SH * P_s -> +6 dB for +3 dB pump
    assert b.output_dbm.idler - a.output_dbm.idler == pytest.approx(6.0, abs=1e-9)
    assert b.output_dbm.sh - a.output_dbm.sh == pytest.approx(6.0, abs=1e-9)


def test_converter_result_to_dict_and_wavepowers():
    res = cascaded_shg_dfg(ConverterParams())
    d = res.to_dict()
    assert set(d) >= {"params", "wavelengths_nm", "output_w", "output_dbm", "conversion_efficiency_db"}
    assert d["output_dbm"]["idler"] == pytest.approx(res.idler_dbm)
    wp = WavePowers(1.0, 2.0, 3.0, 4.0).map(lambda x: 2 * x)
    assert wp.as_dict() == {"pump": 2.0, "signal": 4.0, "sh": 6.0, "idler": 8.0}


def test_params_validation():
    with pytest.raises(ConversionError):
        ConverterParams(length_cm=0.0).validate()
    with pytest.raises(ConversionError):
        ConverterParams(in_coupling_db_pump=-1.0).validate()
    with pytest.raises(ConversionError):
        ConverterParams(lam_pump_nm=1550.0, lam_signal_nm=700.0).validate()
    with pytest.raises(ConversionError):
        ConverterParams.from_dict({"bogus": 1.0})
    p = ConverterParams.from_dict({"pump_dbm": 5.0})
    assert p.pump_dbm == 5.0 and p.signal_dbm == 15.0
    assert "_LIMITS" not in p.to_dict()


def test_loss_budget():
    params = ConverterParams(filter_db_pump=30.0, length_cm=2.0)
    pump = loss_budget(params, "pump")
    assert pump.items()["propagation_db"] == pytest.approx(0.2)
    assert pump.total_db == pytest.approx(1.5 + 0.2 + 1.5 + 30.0)
    assert pump.transmission == pytest.approx(units.db_to_linear(pump.total_db))
    idler = loss_budget(params, "idler")
    assert idler.input_coupling_db == 0.0 and idler.filter_db == 0.0
    with pytest.raises(ConversionError):
        loss_budget(params, "noise")
    assert LossBudget(1.0, 1.0, 1.0).total_db == 3.0
