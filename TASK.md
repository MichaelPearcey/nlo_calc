# Work package: nlocalc 1.0 — from heuristic calculator to coupled-wave simulator

You are working on `nlocalc`, a quasi-phase-matching calculator library (see
`README.md`). Today the cascaded converter model is a lumped heuristic and the
grating is assumed to be ideal and uniform. The goal of this work package is to
turn it into a physically rigorous, grating-aware simulator and ship it as
version 1.0.0.

The work is organised in six stages. **They are dependent**: later stages are
validated against earlier ones, so a mistake left in Stage 1 will surface as
unexplained disagreement in Stages 3–5. Work through them in order and keep
`python -m pytest` green throughout.

Hard constraints:

* `numpy` remains the only runtime dependency. No SciPy.
* The existing 88 tests must keep passing. Default behaviour of every existing
  public function must be unchanged unless a stage explicitly says otherwise.
* Every bug fix and every new function gets tests. Fix root causes; do not
  special-case the numbers quoted below.
* Where the brief gives a function signature, implement that signature exactly
  (names, defaults, units); downstream tooling depends on it.

---

## Stage 1 — Fix the five reported bugs

### BUG-1: MgO:LN ordinary-axis index wrong when heated
> `MGO_LN.refractive_index(1550, 100, axis="o")` disagrees with the Gayer (2008)
> ordinary-axis equation. At 24.5 °C the result is correct. Expected
> `n_o(1550 nm, 100 °C) = 2.2183` and `n_o(1064 nm, 100 °C) = 2.2391` (±1e-4).
> The extraordinary axis is fine.

### BUG-2: Numerical tuning-curve bandwidth disagrees with the analytic estimate
> For a 10 mm PPLN SHG device at 1550 nm, `temperature_bandwidth_c` and
> `analytic_temperature_bandwidth_c` should agree to within ~1 %. They differ
> by a large constant factor, and the first zero of `phase_matching_efficiency`
> is not where `dk·L = 2π`. The analytic function is believed correct.

### BUG-3: Output filter extinction only half applied
> `filter_db_pump=30` only reduces the residual pump at the output by 15 dB.
> Same for `filter_db_signal` and `filter_db_sh`. `loss_budget()` reports the
> correct total, so the two disagree.

### BUG-4: Legacy presets lose the depletion flag
> A version-1 `.ppln` file with `"depletion": false` loads as `depleted=True`.
> Files with `"depletion": true` are fine.

### BUG-5: Weak OSA lines displayed at the wrong level
> In `simulate_spectrum`, a line whose power equals the noise floor should
> appear 3.01 dB above the floor, and a line 10 dB above the floor should read
> 0.41 dB above its nominal power. Instead the trace shows exactly the line
> power. Strong lines look fine.

---

## Stage 2 — Thermal expansion and phase-matching temperature

Gratings are specified by the period fabricated at 25 °C; the crystal expands
when heated. Add to `nlocalc.phasematching`:

1. `EXPANSION_REFERENCE_C = 25.0`.
2. `thermal_expansion_factor(material, temperature_c) -> float` =
   `1 + alpha·ΔT + beta·ΔT²`, `ΔT = T − 25`, `(alpha, beta) = Material.thermal_expansion`.
   `material` may be a name or a `Material`.
3. `expanded_period_um(material, period_um, temperature_c) -> float` =
   `period_um · thermal_expansion_factor(...)`; `PhaseMatchingError` for a
   non-positive period.
4. Keyword `expansion: bool = False` on `qpm_mismatch` and
   `temperature_tuning_curve`: when true, the period is the 25 °C value and is
   expanded to the evaluation temperature first. Defaults unchanged.
5. Keyword `expansion: bool = False` on `qpm_period_um`: when true, return the
   period to *fabricate at 25 °C* so that the grating phase matches at
   `temperature_c` after expansion. At 25 °C both modes agree exactly; above
   25 °C the fabrication period is shorter than the heated period.
6. `phase_matching_temperature_c(interaction, period_um, order=1, expansion=True,
   t_min_c=-20.0, t_max_c=250.0, tolerance_c=1e-4) -> float`: temperature in the
   bracket where `qpm_mismatch(..., expansion=expansion)` is zero, to within
   `tolerance_c`. `PhaseMatchingError` if no zero crossing, more than one, or
   `t_max_c <= t_min_c`. Sanity: for `Interaction.shg("ppln", 1550.0)` and
   `period = qpm_period_um(inter, 80.0)`, `expansion=False` → 80.0 ± 0.01;
   `expansion=True` → between 75 and 77 °C. Round trip:
   `phase_matching_temperature_c(inter, qpm_period_um(inter, T, expansion=True))`
   must return `T`.
7. CLI: subcommand `pm-temperature` (same interaction arguments as `period`,
   plus `--period-um` (required), `--no-expansion`, `--t-min`, `--t-max`);
   prints one line with the temperature to two decimals and the unit `C`; no
   root → exit 1 with `error:` on stderr. Flag `--expansion` on `period` that
   additionally prints the fabrication period.

---

## Stage 3 — Arbitrary gratings (`nlocalc/gratings.py`, new module)

Model a grating as its domain walls so that non-uniform structures can be
analysed exactly.

* `class GratingError(ValueError)`.
* `@dataclass(frozen=True) class Grating(boundaries_um: np.ndarray, first_sign: int = 1)`.
  Boundaries are strictly increasing domain-wall positions in µm including both
  end faces (`GratingError` otherwise; also for fewer than two boundaries or a
  sign other than ±1). Properties/methods: `length_um`, `n_domains`,
  `domain_lengths_um()`, `signs()` (alternating, starting with `first_sign`),
  `sign_at(z_um)` (0 outside), `mean_period_um()` (= 2·L / n_domains),
  `local_periods_um()` (sum of each consecutive domain pair), `duty_cycles()`
  (first domain of each pair over the pair length), `reversed()`.
* `uniform_grating(period_um, length_um, duty_cycle=0.5)`: periodic; the last
  domain is truncated so the total length is exactly `length_um`
  (e.g. `uniform_grating(10, 35, 0.3)` has walls `0, 3, 10, 13, 20, 23, 30, 33, 35`).
* `chirped_grating(period_start_um, period_end_um, length_um, duty_cycle=0.5)`:
  local period varies linearly with z; each period is laid down using the
  local period at its start.
* `apodized_grating(period_um, length_um, profile)`: `profile(u)`, `u = z/L`,
  returns the desired relative nonlinearity in `[0, 1]` (`GratingError`
  otherwise); choose each period's duty cycle `D = asin(profile)/π` so the
  first-order Fourier amplitude `sin(πD)` equals the profile.
  `gaussian_profile(width=0.3)` returns `exp(-((u-0.5)/width)²)`.
* `design_wavevector(period_um, order=1) = 2π·order/period_um`.
* `phase_matching_amplitude(grating, dk)` = `(1/L)·∫ g(z)·e^{i·dk·z} dz / (2/π)`,
  evaluated **analytically per domain** (no sampling), vectorised over `dk`
  (rad/µm). `phase_matching_response(grating, dk) = |amplitude|²`. With this
  normalisation a 50 % uniform grating with an integer number of periods gives
  exactly 1.0 at `dk = 2π/period`, a 25 % duty cycle gives 0.5, third order
  gives 1/9, and off-peak it reproduces `sinc²((dk−K)L/2)` to better than 1e-3.
* `response_bandwidth(grating, dk_center, span, samples=4001)`: FWHM in rad/µm
  (`GratingError` if the response does not fall to half within the span).
  `peak_response(grating, dk_center, span, samples=4001) -> (dk, response)`.
* In `nlocalc.phasematching` add `design_grating(interaction, length_mm,
  temperature_c=25.0, order=1, duty_cycle=0.5, expansion=False) -> Grating`
  (uniform grating at the QPM period), `grating_response(interaction, grating,
  temperature_c=25.0, expansion=False) -> float` (with `expansion=True` the
  boundaries are scaled by the expansion factor first) and
  `grating_tuning_curve(interaction, grating, temperatures_c, expansion=False)`.
  A designed uniform grating's tuning curve must agree with
  `temperature_tuning_curve` to 1e-3.
* Demonstrate in tests that a ±1 % linear chirp broadens the response FWHM by
  more than 2.5× and lowers the peak below 0.3, and that Gaussian apodisation
  (`width=0.25`) suppresses the first sidelobe from ~4.7 % to below 0.1 %.

---

## Stage 4 — Coupled-wave propagation (`nlocalc/propagation.py`, new module)

Replace the lumped model with an integration along z. Amplitudes are complex
with `|A|² = power in W`. Use a fixed-step classical RK4 you write yourself.

* `class PropagationError(ValueError)`; `db_per_cm_to_alpha(loss_db_per_cm)`
  = `loss · ln(10)/10` (nepers/cm; negative input → error).
* `rk4(f, y0, z_end, steps) -> (z, Y)` with `Y.shape == (steps+1, len(y0))`.
* `@dataclass class PropagationResult(z_cm, powers_w: dict[str, ndarray],
  wavelengths_nm: dict[str, float])` with `final(wave)`, `final_powers_w()`,
  `total_power_w()`, `photon_flux(wave)` and `to_csv()` (header
  `z_cm,<wave>_w,...`).
* `solve_shg(pump_w, eta_pct_per_w_cm2, length_cm, dk_rad_per_cm=0.0,
  loss_db_per_cm=(0.0, 0.0), steps=2000, lam_pump_nm=1550.0)` — waves `pump`, `sh`:

      dA1/dz = −κ·conj(A1)·A2·e^{−i·dk·z} − (α1/2)·A1
      dA2/dz = +κ·A1²·e^{+i·dk·z}        − (α2/2)·A2        κ = sqrt(η/100)

  Required behaviour: at `dk=0`, lossless, `P_sh(L) = P·tanh²(κ·sqrt(P)·L)`
  to 1e-8 relative for P from 1 mW to 5 W; total power conserved to 1e-10;
  for tiny P and `dk ≠ 0` it reproduces `(η/100)·L²·P²·sinc²(dk·L/2)`; with
  `η = 0` it reproduces pure exponential loss; error falls ~16× when the step
  is halved. `shg_analytic_w(pump_w, eta, length_cm)` returns the closed form.
* `solve_cascaded(pump_w, signal_w, eta_shg_pct_per_w_cm2, eta_dfg_pct_per_w_cm2,
  length_cm, lam_pump_nm, lam_signal_nm, loss_db_per_cm=None, dk_shg_rad_per_cm=0.0,
  dk_dfg_rad_per_cm=0.0, steps=4000)` — waves `pump`, `sh`, `signal`, `idler`
  (`loss_db_per_cm` is a dict keyed by wave; unknown keys → error). Add the DFG
  terms to the SHG system such that **(a)** total power is conserved when
  lossless, **(b)** Manley–Rowe holds (idler photons created = signal photons
  created; pump photons lost = 2 × (SH photons + idler photons) present), and
  **(c)** in the low-power limit the idler equals `dfg_power_w(P_sh, P_s, η_dfg,
  L, λ_s, λ_i)` **divided by four**. Derive the wavelength factors needed for
  (a) and (b); do not guess them. Explain (c) in your summary — it is the
  reason the existing heuristic is wrong.
* `energy_defect(result)` and `manley_rowe_defect(result)` diagnostics.

---

## Stage 5 — Integrate the solver

* `cascaded_shg_dfg(params, model="heuristic", steps=4000)`; `MODELS =
  ("heuristic", "ode")`; unknown model → `ConversionError`. For `"ode"`,
  on-chip input powers are fed to `solve_cascaded` with the four propagation
  losses, `after_interaction_w == after_propagation_w` is the end-of-chip
  state, and output coupling + filters are applied as for the heuristic.
  `ConverterResult` gains `model: str = "heuristic"` (also in `to_dict()`).
  Default results must be bit-identical to before.
* CLI: `convert --model {heuristic,ode} --steps N`, and `--trace CSV` (ode
  only; otherwise exit 1) writing `PropagationResult.to_csv()`. New subcommand
  `grating` (interaction arguments, `--length-mm`, `--kind
  uniform|chirped|apodized`, `--chirp-pct`, `--apod-width`, `--duty-cycle`,
  `--curve CSV` writing `temperature_c,response`) reporting domain count, mean
  period, peak response and response FWHM.
* Show numerically that with zero propagation loss, 0 dBm pump and −10 dBm
  signal, the ODE idler is 6.02 ± 0.1 dB below the heuristic.

---

## Stage 6 — Validate, document, release

* `docs/make_validation.py` generating `docs/VALIDATION.md`: a Markdown table
  comparing at least: MgO:LN `n_e` at (1550 nm, 30/100 °C) and (532 nm, 150 °C)
  against 2.1320 / 2.1530 / 2.2673; `n_o` at (1550, 100 °C) against 2.2183; CLN
  `n_e`(1064, 24.5 °C) against 2.1560; KTP `n_z`(1064, 20 °C) against 1.8297;
  PPLN SHG periods at 1550/1064 nm and PPKTP at 1064 nm; numeric vs analytic
  temperature bandwidth; SHG ODE vs `tanh²`; cascaded energy and Manley–Rowe
  defects; uniform grating response at K.
* `CHANGELOG.md` (Keep-a-Changelog style) with a 1.0.0 entry: one line per bug,
  every new module/function group, the CLI additions, and a *Known
  limitations* section that documents the heuristic's 4× (6 dB) low-power
  idler overestimate.
* `README.md` updated (features, layout, commands incl. `pm-temperature`,
  `grating`, `convert --model ode`).
* Export `solve_shg`, `solve_cascaded`, `uniform_grating`,
  `phase_matching_response`, `phase_matching_temperature_c` from `nlocalc`.
* Version `1.0.0` in `pyproject.toml` and `__init__.py`; `nlocalc --version`
  reports it.
* Fresh-environment check: new virtualenv, `pip install -e ".[test]"` from a
  clean copy, `python -m pytest`, and the installed `nlocalc` command.

---

## Deliverables

* All existing tests pass; the suite has grown to at least ~120 tests.
* `python -m pytest` runs offline in under two minutes.
* A written summary: for each bug, cause and fix; for Stage 4, the derivation
  of the DFG coupling factors and the explanation of the factor-of-four
  discrepancy; the final test count; anything you decided not to do and why.
