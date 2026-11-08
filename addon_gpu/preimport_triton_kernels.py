# SPDX-License-Identifier: GPL-2.0-or-later
"""A pytest plugin that imports every Triton kernel module of the add-on, with the real Triton, before any test.

It shows an order dependence in a test of the add-on. ``meep_gpu/test_dispatch_chunk_division.py`` replaces
``sys.modules["triton"]`` with a double that has no ``jit`` (``governable_triton`` of the add-on), and then plans a
run with the real composers. A kernel module of the add-on that is imported for the first time under that double
fails at ``@triton.jit`` ("module 'triton' has no attribute 'jit'"), and the planner refuses with an internal
error. In a run of the whole suite an earlier module has already imported the kernel modules; a run of the module
alone has not. With this plugin the module passes alone too. Use it from the clone folder, with this folder on the
Python path:

    PYTHONPATH=<this folder> bash tools/gpu_run.sh .local/upstream/erised-fork \\
        python -m pytest -q -p preimport_triton_kernels meep_gpu/test_dispatch_chunk_division.py
"""

from __future__ import annotations

import importlib
import pkgutil

import meep_gpu.triton_kernels as _kernels

IMPORTED = [importlib.import_module(f"meep_gpu.triton_kernels.{info.name}").__name__
            for info in pkgutil.iter_modules(_kernels.__path__) if not info.name.startswith("test_")]
print(f"preimport_triton_kernels: {len(IMPORTED)} Triton kernel modules imported", flush=True)
