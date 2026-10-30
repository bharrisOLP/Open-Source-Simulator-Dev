"""The device for the solver arrays.

Some packages change the global JAX default: klujax (imported by sax, imported by meow) calls
``jax.config.update("jax_platform_name", "cpu")`` at import. So meow_gpu does not use the global default. It puts
its arrays on the first GPU (ROCm or CUDA), else on the CPU. Set MEOW_GPU_DEVICE=cpu to force the CPU.
"""

from __future__ import annotations

import functools
import os

import jax


@functools.cache
def solver_device() -> jax.Device:
    if os.environ.get("MEOW_GPU_DEVICE", "").lower() != "cpu":
        try:
            return jax.devices("gpu")[0]
        except RuntimeError:
            pass
    return jax.devices("cpu")[0]


def put(x, dtype=None):
    """A device array on the solver device (optionally converted to ``dtype``)."""
    import jax.numpy as jnp
    if dtype is not None:
        x = jnp.asarray(x, dtype=dtype) if not isinstance(x, jax.Array) else x.astype(dtype)
    return jax.device_put(x, solver_device())
