import math

import numpy as np
import pytest

from nlocalc.conversion import ConverterParams, cascaded_shg_dfg
from nlocalc.spectrum import (
    OSASettings,
    Peak,
    Spectrum,
    SpectrumError,
    converter_peaks,
    find_peaks,
    gaussian_line_w,
    line_fwhm_nm,
    noise_floor_w,
    osnr_db,
    peak_table,
    simulate_spectrum,
)

SETTINGS = OSASettings(start_nm=1500.0, stop_nm=1600.0, points=10001, rbw_nm=0.1, noise_floor_dbm=-70.0)


def test_peak_validation():
    with pytest.raises(SpectrumError):
        Peak(0.0, 0.0)
    with pytest.raises(SpectrumError):
        Peak(1550.0, 0.0, linewidth_nm=-1.0)
    assert Peak(1550.0, 0.0).power_w == pytest.approx(1e-3)


def test_settings_validation_and_axis():
    with pytest.raises(SpectrumError):
        OSASettings(start_nm=1600.0, stop_nm=1500.0)
    with pytest.raises(SpectrumError):
        OSASettings(points=1)
    with pytest.raises(SpectrumError):
        OSASettings(rbw_nm=0.0)
    s = OSASettings(start_nm=0.0, stop_nm=10.0, points=11)
    assert s.step_nm == pytest.approx(1.0)
    assert np.allclose(s.wavelength_axis(), np.arange(11.0))


def test_gaussian_line_height_and_width():
    axis = np.linspace(1549.0, 1551.0, 20001)
    pk = Peak(1550.0, 0.0, linewidth_nm=0.0)
    y = gaussian_line_w(axis, pk, rbw_nm=0.2)
    assert y.max() == pytest.approx(1e-3)
    above_half = axis[y >= 0.5e-3]
    assert above_half[-1] - above_half[0] == pytest.approx(0.2, abs=2e-4)


def test_noise_floor_flat_and_rippled():
    flat = noise_floor_w(SETTINGS)
    assert np.allclose(flat, 1e-10)
    rippled_settings = OSASettings(start_nm=1500.0, stop_nm=1600.0, points=1001, noise_ripple_db=4.0, seed=7)
    r1 = noise_floor_w(rippled_settings)
    r2 = noise_floor_w(rippled_settings)
    assert np.array_equal(r1, r2)  # seeded => deterministic
    assert r1.min() < 1e-10 < r1.max()
    assert 10 * np.log10(r1.max() / r1.min()) <= 4.0 + 1e-9


def test_simulated_peak_heights_match_sources():
    peaks = [Peak(1530.0, 10.0, label="a"), Peak(1550.0, 0.0, label="b"), Peak(1570.0, -20.0, label="c")]
    spec = simulate_spectrum(peaks, SETTINGS)
    assert len(spec) == SETTINGS.points
    for pk in peaks:
        assert spec.power_at(pk.wavelength_nm) == pytest.approx(pk.power_dbm, abs=1e-3)
    # far from any line the trace sits on the noise floor
    assert spec.power_at(1510.0) == pytest.approx(-70.0, abs=1e-6)
    assert spec.peaks == peaks


def test_simulated_spectrum_ignores_infinite_and_out_of_span_peaks():
    spec = simulate_spectrum([Peak(1550.0, -math.inf), Peak(900.0, 10.0)], SETTINGS)
    assert spec.power_dbm.max() == pytest.approx(-70.0, abs=1e-6)
    assert len(spec.peaks) == 1


def test_find_peaks_detects_all_lines():
    peaks = [Peak(1530.0, 10.0), Peak(1550.0, 0.0), Peak(1570.0, -20.0)]
    spec = simulate_spectrum(peaks, SETTINGS)
    found = find_peaks(spec, threshold_db=10.0, min_separation_nm=0.5)
    assert [round(p.wavelength_nm, 3) for p in found] == [1530.0, 1550.0, 1570.0]
    assert found[0].power_dbm == pytest.approx(10.0, abs=1e-3)
    assert found[2].prominence_db == pytest.approx(50.0, abs=0.01)
    table = peak_table(found)
    assert "1530.0000" in table and table.count("\n") == 3


def test_find_peaks_merges_close_lines():
    peaks = [Peak(1550.0, 0.0), Peak(1550.2, -3.0)]
    spec = simulate_spectrum(peaks, SETTINGS)
    assert len(find_peaks(spec, min_separation_nm=0.5)) == 1
    assert len(find_peaks(spec, min_separation_nm=0.05)) == 2


def test_find_peaks_threshold():
    spec = simulate_spectrum([Peak(1550.0, -65.0)], SETTINGS)
    assert find_peaks(spec, threshold_db=10.0) == []
    assert len(find_peaks(spec, threshold_db=3.0)) == 1


def test_osnr_and_fwhm():
    spec = simulate_spectrum([Peak(1550.0, 0.0)], SETTINGS)
    assert osnr_db(spec, 1550.0, offset_nm=2.0) == pytest.approx(70.0, abs=1e-3)
    assert line_fwhm_nm(spec, 1550.0) == pytest.approx(0.1, abs=0.02)
    with pytest.raises(SpectrumError):
        osnr_db(spec, 1550.0, offset_nm=0.0)
    with pytest.raises(SpectrumError):
        spec.power_at(1700.0)


def test_spectrum_helpers_and_csv_roundtrip():
    spec = simulate_spectrum([Peak(1550.0, 0.0)], SETTINGS)
    lam, p = spec.max()
    assert lam == pytest.approx(1550.0) and p == pytest.approx(0.0, abs=1e-3)
    sl = spec.slice(1549.0, 1551.0)
    assert sl.wavelengths_nm[0] >= 1549.0 and sl.wavelengths_nm[-1] <= 1551.0
    text = sl.to_csv()
    back = Spectrum.from_csv(text)
    assert np.allclose(back.wavelengths_nm, sl.wavelengths_nm, atol=1e-6)
    assert np.allclose(back.power_dbm, sl.power_dbm, atol=1e-4)
    with pytest.raises(SpectrumError):
        Spectrum.from_csv("a,b\n1,2\n")
    with pytest.raises(SpectrumError):
        Spectrum(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    with pytest.raises(SpectrumError):
        spec.slice(1700.0, 1800.0)


def test_converter_peaks():
    res = cascaded_shg_dfg(ConverterParams())
    out = converter_peaks(res, "output")
    assert [p.label for p in out] == ["pump", "signal", "sh", "idler"]
    assert out[3].wavelength_nm == pytest.approx(res.wavelengths_nm.idler)
    inp = converter_peaks(res, "input")
    assert len(inp) == 2 and inp[0].power_dbm == res.params.pump_dbm
    with pytest.raises(SpectrumError):
        converter_peaks(res, "sideways")
    spec = simulate_spectrum(out, OSASettings(start_nm=700.0, stop_nm=1700.0, points=100001))
    assert len(find_peaks(spec)) == 4
