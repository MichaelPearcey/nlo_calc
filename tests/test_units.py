import math

import pytest

from nlocalc import units


@pytest.mark.parametrize("dbm,watts", [(0.0, 1e-3), (30.0, 1.0), (-30.0, 1e-6), (10.0, 0.01)])
def test_dbm_w_roundtrip(dbm, watts):
    assert units.dbm_to_w(dbm) == pytest.approx(watts)
    assert units.w_to_dbm(watts) == pytest.approx(dbm)
    assert units.w_to_dbm(units.dbm_to_w(dbm)) == pytest.approx(dbm)


def test_w_to_dbm_nonpositive_is_minus_inf():
    assert units.w_to_dbm(0.0) == -math.inf
    assert units.w_to_dbm(-1.0) == -math.inf


def test_db_to_linear():
    assert units.db_to_linear(0.0) == 1.0
    assert units.db_to_linear(3.0103) == pytest.approx(0.5, rel=1e-4)
    assert units.db_to_linear(10.0) == pytest.approx(0.1)
    assert units.db_to_linear(-10.0) == pytest.approx(10.0)


def test_linear_to_db_roundtrip():
    for loss in (0.0, 1.5, 20.0):
        assert units.linear_to_db(units.db_to_linear(loss)) == pytest.approx(loss)
    assert units.linear_to_db(0.0) == math.inf


def test_wavelength_frequency():
    assert units.nm_to_thz(1550.0) == pytest.approx(193.414, rel=1e-5)
    assert units.thz_to_nm(units.nm_to_thz(1310.0)) == pytest.approx(1310.0)
    assert units.nm_to_angular_frequency(1550.0) == pytest.approx(2 * math.pi * 193.414e12, rel=1e-5)
    with pytest.raises(ValueError):
        units.nm_to_thz(0.0)
    with pytest.raises(ValueError):
        units.thz_to_nm(-1.0)


def test_nm_um():
    assert units.nm_to_um(1550.0) == pytest.approx(1.55)
    assert units.um_to_nm(0.775) == pytest.approx(775.0)


def test_wavenumber():
    assert units.wavenumber_rad_per_um(1000.0, 1.0) == pytest.approx(2 * math.pi)
    assert units.wavenumber_rad_per_um(1000.0, 2.0) == pytest.approx(4 * math.pi)


def test_temperature():
    assert units.celsius_to_kelvin(25.0) == pytest.approx(298.15)
    assert units.kelvin_to_celsius(273.15) == pytest.approx(0.0)


def test_add_powers_dbm():
    assert units.add_powers_dbm(0.0, 0.0) == pytest.approx(10 * math.log10(2), abs=1e-9)
    assert units.add_powers_dbm(10.0, -math.inf) == pytest.approx(10.0)
    assert units.add_powers_dbm(10.0, 0.0) == pytest.approx(10.414, abs=1e-3)
    assert units.add_powers_dbm() == -math.inf


def test_clamp():
    assert units.clamp(5.0, 0.0, 1.0) == 1.0
    assert units.clamp(-5.0, 0.0, 1.0) == 0.0
    assert units.clamp(0.5, 0.0, 1.0) == 0.5
    with pytest.raises(ValueError):
        units.clamp(0.0, 1.0, 0.0)
