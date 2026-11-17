# Work package: nlocalc 0.4

You are working on `nlocalc`, a quasi-phase-matching calculator library (see
`README.md`). The existing test suite (`python -m pytest`) passes and must keep
passing. Do not add new runtime dependencies; `numpy` is the only one allowed.

There are two parts: five bug reports from users to investigate and fix, and one
new feature to implement. Where a bug report quotes an expected value, the fixed
code should reproduce it to the stated precision. Please add tests for every bug
you fix and for the new feature.

---

## Part 1 — Bug reports

### BUG-1: MgO:LN ordinary-axis index wrong when heated

> `MGO_LN.refractive_index(1550, 100, axis="o")` returns a value that disagrees
> with the Gayer (2008) ordinary-axis equation. At 24.5 °C the result is correct.
> Expected `n_o(1550 nm, 100 °C) = 2.2183` and `n_o(1064 nm, 100 °C) = 2.2391`
> (±1e-4). The extraordinary axis is fine.

### BUG-2: Numerical tuning-curve bandwidth disagrees with the analytic estimate

> For a 10 mm PPLN SHG device at 1550 nm, `temperature_bandwidth_c` and
> `analytic_temperature_bandwidth_c` should agree to within ~1 %. They don't —
> the numerical result is off by a large constant factor, and the first zero of
> `phase_matching_efficiency` is not where `dk·L = 2π`. The analytic function is
> believed to be correct.

### BUG-3: Output filter extinction only half applied

> Setting `filter_db_pump=30` in `ConverterParams` only reduces the residual
> pump at the output by 15 dB relative to `filter_db_pump=0`. The same happens
> for `filter_db_signal` and `filter_db_sh`. The `loss_budget()` helper reports
> the correct total, so the two disagree.

### BUG-4: Legacy presets lose the depletion flag

> Loading a version-1 `.ppln` file that contains `"depletion": false` yields a
> `ConverterParams` with `depleted=True`. Files with `"depletion": true` are fine.

### BUG-5: Weak OSA lines are displayed at the wrong level

> In `simulate_spectrum`, a line whose power equals the noise floor should
> appear 3.01 dB above the floor (two equal incoherent powers add), and a line
> 10 dB above the floor should read 0.41 dB above its nominal power. Instead the
> trace shows exactly the line power. Strong lines look fine, which is why this
> went unnoticed.

---

## Part 2 — Feature: thermal expansion and phase-matching temperature

Gratings are specified by the period fabricated at 25 °C, but the crystal expands
when heated, so the effective period changes. Add this to `nlocalc.phasematching`:

1. `EXPANSION_REFERENCE_C = 25.0` module constant.

2. `thermal_expansion_factor(material, temperature_c) -> float`
   returns `1 + alpha·ΔT + beta·ΔT²` with `ΔT = temperature_c − 25` and
   `(alpha, beta) = Material.thermal_expansion` (already present on every
   material; `material` may be a name or a `Material`).

3. `expanded_period_um(material, period_um, temperature_c) -> float`
   returns `period_um · thermal_expansion_factor(...)`. Raise
   `PhaseMatchingError` for a non-positive period.

4. Add a keyword argument `expansion: bool = False` to **both** `qpm_mismatch`
   and `temperature_tuning_curve`. When true, the given period is treated as the
   25 °C value and expanded to the evaluation temperature before computing the
   mismatch. Default behaviour must be unchanged.

5. `phase_matching_temperature_c(interaction, period_um, order=1, expansion=True,
   t_min_c=-20.0, t_max_c=250.0, tolerance_c=1e-4) -> float`
   finds the temperature in `[t_min_c, t_max_c]` at which `qpm_mismatch(...)`
   is zero, to within `tolerance_c`. Raise `PhaseMatchingError` if there is no
   zero crossing in the range, if there is more than one, or if
   `t_max_c <= t_min_c`. Any robust root finder is acceptable (no SciPy).

   Sanity checks: for `Interaction.shg("ppln", 1550.0)` and
   `period = qpm_period_um(inter, 80.0)`, calling with `expansion=False` must
   return 80.0 ± 0.01, and with `expansion=True` a temperature between 75 and
   77 °C (the expanded grating phase matches at a lower temperature).

6. CLI subcommand `pm-temperature` taking the same interaction arguments as
   `period` plus `--period-um` (required), `--no-expansion`, `--t-min`,
   `--t-max`. Print a single line containing the temperature formatted with two
   decimals and the unit `C`. A missing root must exit with status 1 and an
   `error:` message on stderr, like the other commands.

Update `README.md` with the new command.

---

## Deliverables

* All existing tests pass; new tests cover each bug and the feature.
* `python -m pytest` runs offline in well under a minute.
* A short summary of what you changed and why for each bug.
