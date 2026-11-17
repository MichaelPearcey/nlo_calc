# nlocalc

A small, dependency-light Python library and CLI for designing quasi-phase-matched
(QPM) nonlinear optical devices such as PPLN and PPKTP frequency converters.

* **Sellmeier models** for 5% MgO:LiNbO3 (Gayer 2008), congruent LiNbO3 (Jundt 1997)
  and KTP (Kato & Takaoka 2002), with temperature dependence, group index and dn/dT.
* **QPM design**: wavevector mismatch, grating period, coherence length, temperature
  and wavelength tuning curves and acceptance bandwidths.
* **Power models**: undepleted and depleted SHG, small-signal DFG/SFG, and a
  cascaded SHG+DFG waveguide wavelength converter with a full loss budget.
* **OSA simulation**: synthetic optical spectrum analyser traces with resolution
  bandwidth, noise floor, peak detection, OSNR and linewidth measurement.
* **Presets**: `.ppln` JSON files (current v2 format plus migration of the legacy
  v1 format written by the original Tk GUI).

Only `numpy` is required at runtime; tests use `pytest`. Nothing touches the
network and no API keys or credentials are involved.

## Installation

```bash
pip install -e ".[test]"
```

or, without installing, run with `PYTHONPATH=src`.

## Quick start

```python
from nlocalc import Interaction, qpm_period_um, ConverterParams, cascaded_shg_dfg
from nlocalc.phasematching import temperature_bandwidth_c

shg = Interaction.shg("ppln", 1550.0)
print(qpm_period_um(shg, temperature_c=25.0))        # ~19.39 um
print(temperature_bandwidth_c(shg, length_mm=10.0))  # FWHM in degC

result = cascaded_shg_dfg(ConverterParams(pump_dbm=20.0, signal_dbm=15.0))
print(result.output_dbm.idler, result.conversion_efficiency_db())
```

## Command line

```bash
nlocalc materials
nlocalc index ppln 1550 --temperature 100 --group
nlocalc period --material ppln --shg 1550
nlocalc period --material ppln --dfg 775 1530 --temperature 40
nlocalc bandwidth --material ktp --shg 1064 --length-mm 10
nlocalc convert --pump-dbm 20 --signal-dbm 10 --json
nlocalc convert --preset device.ppln --spectrum trace.csv --save device_v2.ppln
nlocalc preset migrate old_gui_file.ppln new.ppln
```

## Conventions

| Quantity        | Unit            | Notes                                              |
|-----------------|-----------------|----------------------------------------------------|
| wavelength      | nm              | vacuum wavelength                                  |
| grating period  | um              |                                                    |
| crystal length  | mm (QPM), cm (conversion) | follows the convention of each field      |
| power           | W or dBm        | `*_w` / `*_dbm` suffixes                           |
| losses          | dB (positive)   | `db_to_linear(3) ~= 0.5`                           |
| efficiency      | %/(W cm^2)      | vendor-style normalised SHG efficiency             |
| temperature     | degC            |                                                    |

For a three-wave interaction the two long-wavelength waves are "1" and "2" and the
short-wavelength wave is "3" (`1/lam_3 = 1/lam_1 + 1/lam_2`). The bulk mismatch is
`dk = k_3 - k_1 - k_2` and the first-order QPM period is `2*pi/dk`.

## Layout

```
src/nlocalc/
  units.py          unit conversions
  materials.py      Sellmeier models and material registry
  phasematching.py  QPM period, mismatch, tuning curves, bandwidths
  conversion.py     SHG/DFG power models, cascaded converter, loss budget
  spectrum.py       OSA trace synthesis and analysis
  presets.py        .ppln file I/O and legacy migration
  cli.py            argparse CLI
tests/              pytest suite (deterministic, offline)
```

## Running the tests

```bash
python -m pytest
```

## License

MIT — see `LICENSE`.
