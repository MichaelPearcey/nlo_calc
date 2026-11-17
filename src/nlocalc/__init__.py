"""nlocalc: quasi-phase-matched nonlinear optics calculator.

Modules
-------
units          unit conversions (dBm/W, nm/THz, dB)
materials      Sellmeier refractive-index models (MgO:LN, CLN, KTP)
phasematching  QPM period, mismatch and tuning curves
conversion     SHG / DFG / cascaded converter power models
spectrum       synthetic OSA traces and peak analysis
presets        .ppln preset files (v1 legacy and v2)
cli            command-line entry point
"""

__version__ = "0.3.0"

from .conversion import ConverterParams, ConverterResult, cascaded_shg_dfg  # noqa: E402,F401
from .materials import Material, get_material, list_materials  # noqa: E402,F401
from .phasematching import Interaction, qpm_period_um  # noqa: E402,F401
from .presets import Preset  # noqa: E402,F401
from .spectrum import OSASettings, Peak, Spectrum, simulate_spectrum  # noqa: E402,F401
