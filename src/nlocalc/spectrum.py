"""Synthetic optical spectrum analyser (OSA) traces.

A trace is built from a list of narrow-line ``Peak`` objects. Each peak is
rendered as a Gaussian in the linear (Watt) domain whose FWHM is the
quadrature sum of the source linewidth and the OSA resolution bandwidth. The
peak *height* equals the source power, which is how an OSA displays a line
that is much narrower than the resolution bandwidth. All contributions and
the instrument noise floor are summed in Watts before conversion to dBm, so
overlapping lines combine correctly.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import units
from .conversion import ConverterResult

GAUSS_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


class SpectrumError(ValueError):
    pass


@dataclass(frozen=True)
class Peak:
    wavelength_nm: float
    power_dbm: float
    linewidth_nm: float = 0.01
    label: str = ""

    def __post_init__(self) -> None:
        if self.wavelength_nm <= 0.0:
            raise SpectrumError("peak wavelength must be positive")
        if self.linewidth_nm < 0.0:
            raise SpectrumError("linewidth must be non-negative")

    @property
    def power_w(self) -> float:
        return units.dbm_to_w(self.power_dbm)


@dataclass(frozen=True)
class OSASettings:
    start_nm: float = 700.0
    stop_nm: float = 1700.0
    points: int = 20001
    rbw_nm: float = 0.1
    noise_floor_dbm: float = -70.0
    noise_ripple_db: float = 0.0
    """Peak-to-peak random ripple applied to the noise floor (0 = flat)."""
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.stop_nm <= self.start_nm:
            raise SpectrumError("stop wavelength must exceed start wavelength")
        if self.points < 2:
            raise SpectrumError("need at least two sample points")
        if self.rbw_nm <= 0.0:
            raise SpectrumError("resolution bandwidth must be positive")
        if self.noise_ripple_db < 0.0:
            raise SpectrumError("noise ripple must be non-negative")

    @property
    def step_nm(self) -> float:
        return (self.stop_nm - self.start_nm) / (self.points - 1)

    def wavelength_axis(self) -> np.ndarray:
        return np.linspace(self.start_nm, self.stop_nm, self.points)


@dataclass
class Spectrum:
    wavelengths_nm: np.ndarray
    power_dbm: np.ndarray
    settings: Optional[OSASettings] = None
    peaks: List[Peak] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.wavelengths_nm = np.asarray(self.wavelengths_nm, dtype=float)
        self.power_dbm = np.asarray(self.power_dbm, dtype=float)
        if self.wavelengths_nm.shape != self.power_dbm.shape:
            raise SpectrumError("wavelength and power arrays must have the same shape")
        if self.wavelengths_nm.ndim != 1 or self.wavelengths_nm.size < 2:
            raise SpectrumError("spectrum must be a 1-D array with at least two samples")
        if np.any(np.diff(self.wavelengths_nm) <= 0.0):
            raise SpectrumError("wavelength axis must be strictly increasing")

    def __len__(self) -> int:
        return int(self.wavelengths_nm.size)

    @property
    def power_w(self) -> np.ndarray:
        return 10.0 ** ((self.power_dbm - 30.0) / 10.0)

    def power_at(self, lam_nm: float) -> float:
        """Linear interpolation of the trace (in the dBm domain) at ``lam_nm``."""
        lo, hi = self.wavelengths_nm[0], self.wavelengths_nm[-1]
        if not (lo <= lam_nm <= hi):
            raise SpectrumError(f"{lam_nm} nm is outside the trace [{lo}, {hi}] nm")
        return float(np.interp(lam_nm, self.wavelengths_nm, self.power_dbm))

    def max(self) -> Tuple[float, float]:
        """(wavelength_nm, power_dbm) of the highest sample."""
        i = int(np.argmax(self.power_dbm))
        return float(self.wavelengths_nm[i]), float(self.power_dbm[i])

    def slice(self, start_nm: float, stop_nm: float) -> "Spectrum":
        mask = (self.wavelengths_nm >= start_nm) & (self.wavelengths_nm <= stop_nm)
        if mask.sum() < 2:
            raise SpectrumError("slice must contain at least two samples")
        return Spectrum(self.wavelengths_nm[mask], self.power_dbm[mask], self.settings, list(self.peaks))

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["wavelength_nm", "power_dbm"])
        for lam, p in zip(self.wavelengths_nm, self.power_dbm):
            w.writerow([f"{lam:.6f}", f"{p:.4f}"])
        return buf.getvalue()

    @classmethod
    def from_csv(cls, text: str) -> "Spectrum":
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if header is None or [h.strip() for h in header] != ["wavelength_nm", "power_dbm"]:
            raise SpectrumError("CSV must start with header 'wavelength_nm,power_dbm'")
        lams, pows = [], []
        for row in reader:
            if not row:
                continue
            if len(row) != 2:
                raise SpectrumError(f"malformed CSV row: {row!r}")
            lams.append(float(row[0]))
            pows.append(float(row[1]))
        return cls(np.array(lams), np.array(pows))


# --------------------------------------------------------------------------
# Trace synthesis
# --------------------------------------------------------------------------

def gaussian_line_w(axis_nm: np.ndarray, peak: Peak, rbw_nm: float) -> np.ndarray:
    """Linear-power profile of one line as seen through the OSA."""
    fwhm = math.hypot(peak.linewidth_nm, rbw_nm)
    sigma = fwhm * GAUSS_FWHM_TO_SIGMA
    return peak.power_w * np.exp(-0.5 * ((axis_nm - peak.wavelength_nm) / sigma) ** 2)


def noise_floor_w(settings: OSASettings) -> np.ndarray:
    """Noise floor in Watts, optionally with seeded random ripple."""
    base = units.dbm_to_w(settings.noise_floor_dbm)
    if settings.noise_ripple_db == 0.0:
        return np.full(settings.points, base)
    rng = np.random.default_rng(settings.seed)
    ripple_db = rng.uniform(-0.5, 0.5, settings.points) * settings.noise_ripple_db
    return base * 10.0 ** (ripple_db / 10.0)


def simulate_spectrum(peaks: Iterable[Peak], settings: OSASettings = OSASettings()) -> Spectrum:
    """Render ``peaks`` into an OSA trace. Peaks outside the span are ignored
    but retained in ``Spectrum.peaks`` for bookkeeping."""
    peaks = [p for p in peaks if math.isfinite(p.power_dbm)]
    axis = settings.wavelength_axis()
    lines_w = np.zeros(settings.points)
    for pk in peaks:
        lines_w = lines_w + gaussian_line_w(axis, pk, settings.rbw_nm)
    total_w = np.maximum(lines_w, noise_floor_w(settings))
    power_dbm = 10.0 * np.log10(total_w) + 30.0
    return Spectrum(axis, power_dbm, settings, peaks)


def converter_peaks(result: ConverterResult, side: str = "output", linewidth_nm: float = 0.01) -> List[Peak]:
    """Build the OSA peak list for the input or output of a cascaded converter."""
    lam = result.wavelengths_nm
    if side == "input":
        return [
            Peak(lam.pump, result.params.pump_dbm, linewidth_nm, "pump"),
            Peak(lam.signal, result.params.signal_dbm, linewidth_nm, "signal"),
        ]
    if side == "output":
        out = result.output_dbm
        return [
            Peak(lam.pump, out.pump, linewidth_nm, "pump"),
            Peak(lam.signal, out.signal, linewidth_nm, "signal"),
            Peak(lam.sh, out.sh, linewidth_nm, "sh"),
            Peak(lam.idler, out.idler, linewidth_nm, "idler"),
        ]
    raise SpectrumError("side must be 'input' or 'output'")


# --------------------------------------------------------------------------
# Trace analysis
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedPeak:
    wavelength_nm: float
    power_dbm: float
    prominence_db: float
    index: int


def find_peaks(
    spectrum: Spectrum,
    threshold_db: float = 10.0,
    min_separation_nm: float = 0.5,
) -> List[DetectedPeak]:
    """Locate local maxima that rise at least ``threshold_db`` above the
    trace median (a robust estimate of the noise floor).

    Maxima closer than ``min_separation_nm`` are merged, keeping the stronger
    one. The returned list is sorted by wavelength.
    """
    y = spectrum.power_dbm
    x = spectrum.wavelengths_nm
    floor = float(np.median(y))
    level = floor + threshold_db
    candidates: List[DetectedPeak] = []
    for i in range(1, y.size - 1):
        if y[i] >= level and y[i] >= y[i - 1] and y[i] > y[i + 1]:
            candidates.append(DetectedPeak(float(x[i]), float(y[i]), float(y[i] - floor), i))
    # handle plateaus at the array ends
    if y.size >= 2 and y[0] >= level and y[0] > y[1]:
        candidates.insert(0, DetectedPeak(float(x[0]), float(y[0]), float(y[0] - floor), 0))
    if y.size >= 2 and y[-1] >= level and y[-1] > y[-2]:
        candidates.append(DetectedPeak(float(x[-1]), float(y[-1]), float(y[-1] - floor), y.size - 1))

    merged: List[DetectedPeak] = []
    for c in sorted(candidates, key=lambda d: d.power_dbm, reverse=True):
        if all(abs(c.wavelength_nm - m.wavelength_nm) >= min_separation_nm for m in merged):
            merged.append(c)
    return sorted(merged, key=lambda d: d.wavelength_nm)


def osnr_db(spectrum: Spectrum, lam_nm: float, offset_nm: float = 1.0) -> float:
    """Optical signal-to-noise ratio: peak level minus the mean of the trace
    (in dBm) at +-``offset_nm`` from the peak."""
    if offset_nm <= 0.0:
        raise SpectrumError("offset must be positive")
    peak = spectrum.power_at(lam_nm)
    lo, hi = spectrum.wavelengths_nm[0], spectrum.wavelengths_nm[-1]
    sides = [p for p in (lam_nm - offset_nm, lam_nm + offset_nm) if lo <= p <= hi]
    if not sides:
        raise SpectrumError("offset positions fall outside the trace")
    noise_w = np.mean([units.dbm_to_w(spectrum.power_at(p)) for p in sides])
    return peak - units.w_to_dbm(float(noise_w))


def line_fwhm_nm(spectrum: Spectrum, lam_nm: float) -> float:
    """Measured -3 dB width of the line nearest ``lam_nm``."""
    x, y = spectrum.wavelengths_nm, spectrum.power_dbm
    i = int(np.argmin(np.abs(x - lam_nm)))
    # climb to the local maximum
    while 0 < i < y.size - 1 and (y[i + 1] > y[i] or y[i - 1] > y[i]):
        i = i + 1 if y[i + 1] > y[i - 1] else i - 1
    half = y[i] - 10.0 * math.log10(2.0)
    left = i
    while left > 0 and y[left] > half:
        left -= 1
    right = i
    while right < y.size - 1 and y[right] > half:
        right += 1
    if y[left] > half or y[right] > half:
        raise SpectrumError("line does not fall by 3 dB within the trace")

    def cross(i0: int, i1: int) -> float:
        return float(x[i0] + (half - y[i0]) * (x[i1] - x[i0]) / (y[i1] - y[i0]))

    return cross(right - 1, right) - cross(left, left + 1)


def peak_table(peaks: Sequence[DetectedPeak]) -> str:
    """Fixed-width text table for CLI output."""
    lines = [f"{'wavelength_nm':>14} {'power_dbm':>10} {'prominence_db':>14}"]
    for p in peaks:
        lines.append(f"{p.wavelength_nm:14.4f} {p.power_dbm:10.2f} {p.prominence_db:14.2f}")
    return "\n".join(lines)
