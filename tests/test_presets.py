import json
from datetime import datetime

import pytest

from nlocalc.conversion import ConverterParams
from nlocalc.presets import (
    CURRENT_VERSION,
    FORMAT_TAG,
    LEGACY_KEY_MAP,
    Preset,
    PresetError,
    dumps,
    from_dict,
    load,
    loads,
    migrate_file,
    save,
)
from nlocalc.spectrum import OSASettings

STAMP = datetime(2026, 1, 2, 3, 4, 5)


def test_dumps_loads_roundtrip():
    preset = Preset(
        converter=ConverterParams(pump_dbm=12.5, lam_signal_nm=1540.0, depleted=False),
        osa=OSASettings(start_nm=1500.0, stop_nm=1600.0, points=1001),
        name="unit test",
    ).with_timestamp(STAMP)
    text = dumps(preset)
    data = json.loads(text)
    assert data["format"] == FORMAT_TAG and data["version"] == CURRENT_VERSION
    assert data["saved"] == "2026-01-02T03:04:05"
    back = loads(text)
    assert back.converter == preset.converter
    assert back.osa == preset.osa
    assert back.name == "unit test"
    assert back.source_version == CURRENT_VERSION


def test_save_and_load_file(tmp_path):
    path = save(Preset(name="disk"), tmp_path / "device", timestamp=STAMP)
    assert path.suffix == ".ppln"
    loaded = load(path)
    assert loaded.name == "disk"
    assert loaded.converter == ConverterParams()
    assert loaded.saved == "2026-01-02T03:04:05"
    with pytest.raises(PresetError):
        load(tmp_path / "missing.ppln")


def test_legacy_key_map_covers_all_converter_fields():
    targets = set(LEGACY_KEY_MAP.values())
    expected = set(ConverterParams().to_dict()) - {"depleted"}
    assert targets == expected


def test_load_legacy_v1(legacy_preset_path):
    preset = load(legacy_preset_path)
    assert preset.source_version == 1
    c = preset.converter
    assert c.eta_shg_pct_per_w_cm2 == pytest.approx(1592.417, abs=1e-3)
    assert c.pump_dbm == pytest.approx(10.0948, abs=1e-4)
    assert c.lam_signal_nm == 1530.0
    assert c.prop_db_per_cm_sh == 0.3
    assert c.depleted is True
    assert preset.saved == "2025-01-01T00:00:00"


def test_migrate_legacy_file(legacy_preset_path, tmp_path):
    dst = tmp_path / "migrated.ppln"
    migrate_file(legacy_preset_path, dst, timestamp=STAMP)
    data = json.loads(dst.read_text())
    assert data["version"] == CURRENT_VERSION
    assert data["converter"]["length_cm"] == 1.0
    assert load(dst).converter == load(legacy_preset_path).converter


def test_legacy_ignores_k_factor_and_rejects_unknown():
    base = {"parameters": {"k_factor": 1.0, "L_cm": 2.0}, "version": "1.0"}
    assert from_dict(base).converter.length_cm == 2.0
    with pytest.raises(PresetError):
        from_dict({"parameters": {"mystery": 1.0}})
    with pytest.raises(PresetError):
        from_dict({"parameters": []})


def test_bad_inputs():
    with pytest.raises(PresetError):
        loads("not json")
    with pytest.raises(PresetError):
        loads("[]")
    with pytest.raises(PresetError):
        from_dict({"converter": {}})  # missing format tag
    with pytest.raises(PresetError):
        from_dict({"format": FORMAT_TAG, "version": 99, "converter": {}})
    with pytest.raises(PresetError):
        from_dict({"format": FORMAT_TAG, "version": CURRENT_VERSION})
    with pytest.raises(PresetError):
        from_dict({"format": FORMAT_TAG, "version": CURRENT_VERSION, "converter": {}, "osa": {"nope": 1}})
    with pytest.raises(Exception):
        from_dict({"format": FORMAT_TAG, "version": CURRENT_VERSION, "converter": {"length_cm": -1.0}})
