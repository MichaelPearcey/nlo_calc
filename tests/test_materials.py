import pytest

from nlocalc.materials import (
    CLN,
    KTP,
    MGO_LN,
    Material,
    MaterialError,
    TemperatureSellmeier,
    get_material,
    list_materials,
)


# Reference values for 5% MgO:LN n_e published alongside the Gayer (2008)
# coefficients (values rounded to 4 decimals).
@pytest.mark.parametrize(
    "lam_nm,temp_c,expected",
    [
        (532.0, 30.0, 2.2260),
        (780.0, 30.0, 2.1715),
        (1064.0, 30.0, 2.1496),
        (1550.0, 30.0, 2.1320),
        (3500.0, 30.0, 2.0732),
        (1064.0, 100.0, 2.1708),
        (1550.0, 100.0, 2.1530),
        (532.0, 150.0, 2.2673),
        (1550.0, 150.0, 2.1705),
    ],
)
def test_mgo_ln_extraordinary_reference_table(lam_nm, temp_c, expected):
    assert MGO_LN.refractive_index(lam_nm, temp_c) == pytest.approx(expected, abs=6e-5)


def test_mgo_ln_ordinary_room_temperature():
    # At the 24.5 C reference temperature the thermal factor vanishes and the
    # ordinary index reduces to the room-temperature Sellmeier fit.
    n_o = MGO_LN.refractive_index(1550.0, 24.5, axis="o")
    assert n_o == pytest.approx(2.2088, abs=2e-4)
    assert n_o > MGO_LN.refractive_index(1550.0, 24.5, axis="e")  # LN is negative uniaxial


def test_cln_jundt_reference():
    # Jundt (1997) congruent LN, n_e at 1064 nm is ~2.1560 at room temperature.
    assert CLN.refractive_index(1064.0, 24.5) == pytest.approx(2.1560, abs=5e-4)
    assert CLN.refractive_index(1550.0, 24.5) == pytest.approx(2.1381, abs=5e-4)


def test_ktp_reference():
    # Kato & Takaoka (2002): n_z(1064 nm) ~ 1.8297, n_y(1064 nm) ~ 1.7454 at 20 C.
    assert KTP.refractive_index(1064.0, 20.0, axis="z") == pytest.approx(1.8297, abs=5e-4)
    assert KTP.refractive_index(1064.0, 20.0, axis="y") == pytest.approx(1.7454, abs=5e-4)
    assert KTP.refractive_index(1064.0, 20.0, axis="x") < KTP.refractive_index(1064.0, 20.0, axis="y")


def test_ktp_thermo_optic_sign():
    # dn/dT for KTP z-axis is positive and of order 1e-5 /C in the near infrared.
    dndt = KTP.dn_dt(1064.0, axis="z")
    assert 1e-5 < dndt < 3e-5


def test_normal_dispersion_and_group_index():
    for mat in (MGO_LN, CLN, KTP):
        n1 = mat.refractive_index(1000.0)
        n2 = mat.refractive_index(2000.0)
        assert n1 > n2
        assert mat.group_index(1550.0) > mat.refractive_index(1550.0)
        assert mat.dn_dlambda(1550.0) < 0.0


def test_thermal_factor_zero_at_reference():
    assert TemperatureSellmeier.thermal_factor(24.5) == 0.0
    assert TemperatureSellmeier.thermal_factor(100.0) > 0.0


def test_registry_lookup_and_aliases():
    assert get_material("ppln") is MGO_LN
    assert get_material("MgO:LN") is MGO_LN
    assert get_material(" ktp ") is KTP
    assert get_material("ln") is CLN
    assert get_material(KTP) is KTP
    assert list_materials() == ("MgO:LN", "CLN", "KTP")


def test_unknown_material_and_axis():
    with pytest.raises(MaterialError):
        get_material("bbo")
    with pytest.raises(MaterialError):
        CLN.refractive_index(1550.0, axis="o")
    with pytest.raises(MaterialError):
        CLN.refractive_index(-1.0)


def test_strict_wavelength_range():
    assert MGO_LN.valid_range.contains(1550.0)
    with pytest.raises(MaterialError):
        MGO_LN.refractive_index(100.0, strict=True)
    # non-strict silently extrapolates
    assert MGO_LN.refractive_index(4500.0) > 1.0


def test_material_metadata():
    assert MGO_LN.axis_names() == ("e", "o")
    assert KTP.default_axis == "z"
    assert isinstance(MGO_LN, Material)
    assert MGO_LN.d_eff_pm_per_v > KTP.d_eff_pm_per_v
