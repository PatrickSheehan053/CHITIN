"""
CHITIN · src/
──────────────
Causal Heuristics Isolating Transcriptomic Intervention Noise
"""

from .utils import load_config, setup_logger, apply_chitin_style
from .utils import log_phase, snapshot, log_memory, force_gc

from . import engine
from . import diagnostics
from . import plotting
