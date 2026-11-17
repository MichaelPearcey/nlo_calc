"""Save and load converter presets (``.ppln`` JSON files).

Two on-disk formats are understood.

Version 2 (current)::

    {
      "format": "nlocalc-preset",
      "version": 2,
      "name": "my device",
      "saved": "2026-01-01T00:00:00",
      "converter": { ...ConverterParams fields... },
      "osa": { ...OSASettings fields... }        # optional
    }

Version 1 (legacy, written by the original Tk GUI)::

    {
      "parameters": {"eta_shg_pct_per_W_cm2": ..., "L_cm": ..., ...},
      "depletion": true,
      "saved_date": "...",
      "version": "1.0"
    }

Legacy files are migrated on load; saving always writes version 2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .conversion import ConverterParams
from .spectrum import OSASettings

FORMAT_TAG = "nlocalc-preset"
CURRENT_VERSION = 2


class PresetError(ValueError):
    pass


LEGACY_KEY_MAP: Mapping[str, str] = {
    "eta_shg_pct_per_W_cm2": "eta_shg_pct_per_w_cm2",
    "eta_dfg_pct_per_W_cm2": "eta_dfg_pct_per_w_cm2",
    "ce_pct_per_W_cm": "eta_dfg_pct_per_w_cm2",
    "L_cm": "length_cm",
    "pump_dBm": "pump_dbm",
    "signal_dBm": "signal_dbm",
    "lam_p_nm": "lam_pump_nm",
    "lam_s_nm": "lam_signal_nm",
    "in_cpl_dB_p": "in_coupling_db_pump",
    "in_cpl_dB_s": "in_coupling_db_signal",
    "out_cpl_dB_i": "out_coupling_db",
    "prop_dB_per_cm_p": "prop_db_per_cm_pump",
    "prop_dB_per_cm_s": "prop_db_per_cm_signal",
    "prop_dB_per_cm_2w": "prop_db_per_cm_sh",
    "prop_dB_per_cm_i": "prop_db_per_cm_idler",
    "filt_xt_dB_p": "filter_db_pump",
    "filt_xt_dB_s": "filter_db_signal",
    "filt_xt_dB_2w": "filter_db_sh",
}
"""Mapping from version-1 parameter names to ``ConverterParams`` fields."""

LEGACY_IGNORED_KEYS = frozenset({"k_factor"})


@dataclass
class Preset:
    converter: ConverterParams = field(default_factory=ConverterParams)
    osa: Optional[OSASettings] = None
    name: str = ""
    saved: Optional[str] = None
    source_version: int = CURRENT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "format": FORMAT_TAG,
            "version": CURRENT_VERSION,
            "name": self.name,
            "saved": self.saved,
            "converter": self.converter.to_dict(),
        }
        if self.osa is not None:
            d["osa"] = asdict(self.osa)
        return d

    def with_timestamp(self, when: Optional[datetime] = None) -> "Preset":
        stamp = (when or datetime.now()).replace(microsecond=0).isoformat()
        return replace(self, saved=stamp)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def dumps(preset: Preset, indent: int = 2) -> str:
    return json.dumps(preset.to_dict(), indent=indent, sort_keys=True) + "\n"


def loads(text: str) -> Preset:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PresetError(f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise PresetError("preset must be a JSON object")
    return from_dict(data)


def from_dict(data: Mapping[str, Any]) -> Preset:
    if "parameters" in data and "converter" not in data:
        return _load_v1(data)
    if data.get("format") != FORMAT_TAG:
        raise PresetError("not an nlocalc preset (missing format tag)")
    version = data.get("version")
    if version != CURRENT_VERSION:
        raise PresetError(f"unsupported preset version {version!r}")
    if not isinstance(data.get("converter"), dict):
        raise PresetError("preset is missing the 'converter' object")
    converter = ConverterParams.from_dict(data["converter"])
    converter.validate()
    osa = None
    if data.get("osa") is not None:
        if not isinstance(data["osa"], dict):
            raise PresetError("'osa' must be an object")
        try:
            osa = OSASettings(**data["osa"])
        except TypeError as exc:
            raise PresetError(f"bad OSA settings: {exc}") from None
    return Preset(
        converter=converter,
        osa=osa,
        name=str(data.get("name", "")),
        saved=data.get("saved"),
        source_version=CURRENT_VERSION,
    )


def _load_v1(data: Mapping[str, Any]) -> Preset:
    params = data.get("parameters")
    if not isinstance(params, dict):
        raise PresetError("legacy preset 'parameters' must be an object")
    mapped: Dict[str, Any] = {}
    unknown = []
    for key, value in params.items():
        if key in LEGACY_IGNORED_KEYS:
            continue
        target = LEGACY_KEY_MAP.get(key)
        if target is None:
            unknown.append(key)
            continue
        mapped[target] = float(value)
    if unknown:
        raise PresetError(f"legacy preset has unknown parameters: {sorted(unknown)}")
    if data.get("depletion"):
        mapped["depleted"] = bool(data["depletion"])
    converter = ConverterParams.from_dict(mapped)
    converter.validate()
    return Preset(
        converter=converter,
        osa=None,
        name=str(data.get("name", "")),
        saved=data.get("saved_date"),
        source_version=1,
    )


# --------------------------------------------------------------------------
# File helpers
# --------------------------------------------------------------------------

def save(preset: Preset, path: str | Path, timestamp: Optional[datetime] = None) -> Path:
    """Write ``preset`` as a version-2 file, stamping the save time."""
    path = Path(path)
    if path.suffix.lower() != ".ppln":
        path = path.with_suffix(".ppln")
    path.write_text(dumps(preset.with_timestamp(timestamp)), encoding="utf-8")
    return path


def load(path: str | Path) -> Preset:
    path = Path(path)
    if not path.exists():
        raise PresetError(f"preset file not found: {path}")
    return loads(path.read_text(encoding="utf-8"))


def migrate_file(src: str | Path, dst: str | Path, timestamp: Optional[datetime] = None) -> Preset:
    """Load any supported preset and re-save it in the current format."""
    preset = load(src)
    save(preset, dst, timestamp)
    return preset
