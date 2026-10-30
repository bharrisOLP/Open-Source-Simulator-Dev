import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
