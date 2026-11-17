import json

import pytest

from nlocalc import __version__
from nlocalc.cli import main, run
from nlocalc.presets import load
from nlocalc.spectrum import Spectrum


def test_materials_command():
    out = run(["materials"])
    assert "MgO:LN" in out and "KTP" in out and "CLN" in out


def test_index_command():
    out = run(["index", "ppln", "1550", "--temperature", "100"])
    assert "2.153013" in out
    out = run(["index", "ktp", "1064", "--temperature", "20", "--axis", "z", "--group"])
    assert "n_g" in out


def test_period_command():
    out = run(["period", "--material", "ppln", "--shg", "1550"])
    assert "19.38" in out
    out = run(["period", "--dfg", "775", "1530", "--temperature", "40"])
    assert "1570.530" in out
    out = run(["period", "--sfg", "1064", "1550", "--material", "ktp"])
    assert "630.9" in out


def test_bandwidth_command():
    out = run(["bandwidth", "--shg", "1550", "--length-mm", "10"])
    assert "temperature FWHM" in out and "wavelength FWHM" in out


def test_convert_text_and_json(tmp_path):
    text = run(["convert", "--pump-dbm", "20", "--signal-dbm", "15"])
    assert "idler" in text and "conversion efficiency" in text
    js = json.loads(run(["convert", "--pump-dbm", "20", "--signal-dbm", "15", "--json"]))
    assert js["params"]["pump_dbm"] == 20.0
    assert js["wavelengths_nm"]["idler"] == pytest.approx(1570.53, abs=0.01)


def test_convert_with_preset_override_save_and_spectrum(legacy_preset_path, tmp_path):
    csv_path = tmp_path / "trace.csv"
    save_path = tmp_path / "new.ppln"
    out = run([
        "convert", "--preset", str(legacy_preset_path), "--signal-dbm", "0",
        "--no-depleted", "--spectrum", str(csv_path), "--save", str(save_path),
    ])
    assert "prominence_db" in out
    spec = Spectrum.from_csv(csv_path.read_text())
    assert len(spec) == 10001
    saved = load(save_path)
    assert saved.converter.signal_dbm == 0.0
    assert saved.converter.depleted is False
    assert saved.converter.eta_shg_pct_per_w_cm2 == pytest.approx(1592.417, abs=1e-3)


def test_preset_show_and_migrate(legacy_preset_path, tmp_path):
    shown = json.loads(run(["preset", "show", str(legacy_preset_path)]))
    assert shown["version"] == 2
    dst = tmp_path / "out.ppln"
    msg = run(["preset", "migrate", str(legacy_preset_path), str(dst)])
    assert "version 1" in msg and dst.exists()


def test_main_exit_codes(capsys):
    assert main(["materials"]) == 0
    assert main(["index", "unobtainium", "1550"]) == 1
    err = capsys.readouterr().err
    assert "unknown material" in err
    assert main(["period"]) == 1  # no interaction given
    assert main(["convert", "--length-cm", "-1"]) == 1


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
