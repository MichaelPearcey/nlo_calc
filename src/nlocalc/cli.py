"""Command-line interface.

Examples::

    nlocalc index mgo:ln 1550 --temperature 50
    nlocalc period --material ppln --shg 1550
    nlocalc period --material ppln --dfg 775 1530 --temperature 40
    nlocalc bandwidth --material ppln --shg 1550 --length-mm 10
    nlocalc convert --pump-dbm 20 --signal-dbm 10 --json
    nlocalc convert --preset device.ppln --spectrum out.csv
    nlocalc preset migrate old.ppln new.ppln
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from typing import List, Optional, Sequence

from . import __version__
from .conversion import ConverterParams, cascaded_shg_dfg
from .materials import MaterialError, get_material, list_materials
from .phasematching import (
    Interaction,
    PhaseMatchingError,
    qpm_period_um,
    temperature_bandwidth_c,
    wavelength_bandwidth_nm,
)
from .presets import Preset, PresetError, load as load_preset, save as save_preset
from .spectrum import OSASettings, converter_peaks, find_peaks, peak_table, simulate_spectrum


class CliError(Exception):
    pass


def _interaction_from_args(args: argparse.Namespace) -> Interaction:
    if args.shg is not None:
        return Interaction.shg(args.material, args.shg, args.axis)
    if args.dfg is not None:
        return Interaction.dfg(args.material, *args.dfg)
    if args.sfg is not None:
        return Interaction.sfg(args.material, *args.sfg)
    raise CliError("specify one of --shg, --dfg or --sfg")


def _add_interaction_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--material", default="mgo:ln", help="material name (see 'materials')")
    p.add_argument("--temperature", type=float, default=25.0, help="crystal temperature [degC]")
    p.add_argument("--order", type=int, default=1, help="QPM order")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--shg", type=float, metavar="LAM_NM", help="SHG of this fundamental")
    g.add_argument("--dfg", type=float, nargs=2, metavar=("PUMP_NM", "SIGNAL_NM"), help="DFG pump/signal")
    g.add_argument("--sfg", type=float, nargs=2, metavar=("LAM1_NM", "LAM2_NM"), help="SFG inputs")
    p.add_argument("--axis", default=None, help="polarisation axis for SHG (material default if omitted)")


def _add_converter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", help="load parameters from a .ppln file first")
    for f in fields(ConverterParams):
        if f.name.startswith("_"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool" or f.type is bool:
            p.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(flag, dest=f.name, type=float, default=None)


def _converter_params(args: argparse.Namespace) -> ConverterParams:
    params = load_preset(args.preset).converter if args.preset else ConverterParams()
    overrides = {
        f.name: getattr(args, f.name)
        for f in fields(ConverterParams)
        if not f.name.startswith("_") and getattr(args, f.name, None) is not None
    }
    for k, v in overrides.items():
        setattr(params, k, v)
    params.validate()
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlocalc", description="QPM nonlinear optics calculator")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("materials", help="list known materials")

    p = sub.add_parser("index", help="refractive index")
    p.add_argument("material")
    p.add_argument("wavelength_nm", type=float)
    p.add_argument("--temperature", type=float, default=25.0)
    p.add_argument("--axis", default=None)
    p.add_argument("--group", action="store_true", help="also print the group index")

    p = sub.add_parser("period", help="QPM period for an interaction")
    _add_interaction_args(p)

    p = sub.add_parser("bandwidth", help="temperature and wavelength acceptance bandwidths")
    _add_interaction_args(p)
    p.add_argument("--length-mm", type=float, required=True)

    p = sub.add_parser("convert", help="evaluate the cascaded SHG+DFG converter")
    _add_converter_args(p)
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--spectrum", metavar="CSV", help="write the simulated output OSA trace to CSV")
    p.add_argument("--save", metavar="PPLN", help="save the effective parameters as a preset")

    p = sub.add_parser("preset", help="preset file utilities")
    ps = p.add_subparsers(dest="preset_command", required=True)
    q = ps.add_parser("show", help="print a preset")
    q.add_argument("path")
    q = ps.add_parser("migrate", help="rewrite a preset in the current format")
    q.add_argument("src")
    q.add_argument("dst")

    return parser


# --------------------------------------------------------------------------
# Command handlers
# --------------------------------------------------------------------------

def cmd_materials(_: argparse.Namespace) -> str:
    rows = []
    for name in list_materials():
        m = get_material(name)
        rows.append(f"{m.name:8} axes={','.join(m.axis_names()):6} default={m.default_axis}  {m.description}")
    return "\n".join(rows)


def cmd_index(args: argparse.Namespace) -> str:
    m = get_material(args.material)
    n = m.refractive_index(args.wavelength_nm, args.temperature, args.axis)
    out = f"n({args.wavelength_nm:g} nm, {args.temperature:g} C, {args.axis or m.default_axis}) = {n:.6f}"
    if args.group:
        out += f"\nn_g = {m.group_index(args.wavelength_nm, args.temperature, args.axis):.6f}"
    return out


def cmd_period(args: argparse.Namespace) -> str:
    inter = _interaction_from_args(args)
    period = qpm_period_um(inter, args.temperature, args.order)
    l1, l2, l3 = inter.wavelengths_nm()
    return (
        f"waves: {l1:.3f} nm + {l2:.3f} nm -> {l3:.3f} nm\n"
        f"QPM period (order {args.order}) at {args.temperature:g} C: {period:.4f} um"
    )


def cmd_bandwidth(args: argparse.Namespace) -> str:
    inter = _interaction_from_args(args)
    fixed = "shg" if args.shg is not None else "2"
    t_bw = temperature_bandwidth_c(inter, args.length_mm, args.temperature, args.order)
    w_bw = wavelength_bandwidth_nm(inter, args.length_mm, args.temperature, args.order, fixed_wave=fixed)
    return (
        f"length: {args.length_mm:g} mm\n"
        f"temperature FWHM: {t_bw:.3f} C\n"
        f"wavelength FWHM (wave 1): {w_bw:.4f} nm"
    )


def cmd_convert(args: argparse.Namespace) -> str:
    params = _converter_params(args)
    result = cascaded_shg_dfg(params)
    if args.save:
        save_preset(Preset(converter=params), args.save)
    if args.spectrum:
        settings = OSASettings(start_nm=700.0, stop_nm=1700.0, points=10001, rbw_nm=0.1)
        spec = simulate_spectrum(converter_peaks(result, "output"), settings)
        with open(args.spectrum, "w", encoding="utf-8") as fh:
            fh.write(spec.to_csv())
    if args.json:
        return json.dumps(result.to_dict(), indent=2)
    out = result.output_dbm
    lam = result.wavelengths_nm
    lines = [
        f"{'wave':8} {'nm':>10} {'out dBm':>10}",
        f"{'pump':8} {lam.pump:10.3f} {out.pump:10.2f}",
        f"{'signal':8} {lam.signal:10.3f} {out.signal:10.2f}",
        f"{'sh':8} {lam.sh:10.3f} {out.sh:10.2f}",
        f"{'idler':8} {lam.idler:10.3f} {out.idler:10.2f}",
        f"conversion efficiency: {result.conversion_efficiency_db():.2f} dB",
    ]
    if args.spectrum:
        settings = OSASettings(start_nm=700.0, stop_nm=1700.0, points=10001, rbw_nm=0.1)
        spec = simulate_spectrum(converter_peaks(result, "output"), settings)
        lines.append(peak_table(find_peaks(spec)))
    return "\n".join(lines)


def cmd_preset(args: argparse.Namespace) -> str:
    if args.preset_command == "show":
        preset = load_preset(args.path)
        return json.dumps(preset.to_dict(), indent=2, sort_keys=True)
    if args.preset_command == "migrate":
        preset = load_preset(args.src)
        path = save_preset(preset, args.dst)
        return f"migrated version {preset.source_version} preset to {path}"
    raise CliError(f"unknown preset command {args.preset_command!r}")


HANDLERS = {
    "materials": cmd_materials,
    "index": cmd_index,
    "period": cmd_period,
    "bandwidth": cmd_bandwidth,
    "convert": cmd_convert,
    "preset": cmd_preset,
}


def run(argv: Optional[Sequence[str]] = None) -> str:
    """Parse ``argv`` and return the command output as text (raises on error)."""
    args = build_parser().parse_args(argv)
    return HANDLERS[args.command](args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        print(run(argv))
        return 0
    except (CliError, MaterialError, PhaseMatchingError, PresetError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
