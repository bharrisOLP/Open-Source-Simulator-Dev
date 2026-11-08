# SPDX-License-Identifier: GPL-2.0-or-later
"""The GPU time stepping of the add-on on the AMD GPU, against independent references.

Three references, as the add-on uses them on NVIDIA hardware:

* the array path of the same GPU route (CuPy, dispatch off, under the policy the kernel run reported): the
  kernels must give the same float32 words, bit for bit (the identity the add-on certifies on NVIDIA);
* the NumPy reference of the add-on (host CPU);
* the CPU solver itself (its own time stepping, double precision): the fields and fluxes must agree to the band
  of the add-on examples, 1e-4 relative (README of the add-on, "Bit identity, and agreement with MEEP").

Cases (addon_cases.py): the 2-D slab waveguide with two flux planes, the 2-D ring resonator of the tutorial with
the field at a point and Harminv, and the 3-D dielectric sphere with a flux plane.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import differing_words, rel_l2, run_leg

BAND = 1e-4          # the band of the add-on examples for a run against the CPU solver (relative L2)
# The resonances of the ring, against the CPU solver. Measured on 2026-11-07 (both tables, the same modes):
# |delta(f + i decay)| / f at most 7.7e-6, and Q at most 0.67 % (the mode with Q near 1700). The bands are the
# tightest that keep a margin over these logged values: 2.6 times for the complex frequency, 1.5 times for Q (the
# float32 NumPy reference of the add-on is at 4.4e-7 of f and 0.14 % of Q, so the floor of float32 against float64
# stepping is below both bands).
RESONANCE_BAND = 2e-5
Q_BAND = 1e-2
TABLES = ("triton", "cuda")


@pytest.fixture(scope="module")
def slab(legs_dir):
    out = {}
    for table in TABLES:
        out[table] = run_leg(legs_dir, "slab2d", "gpu", table=table, until=20)
    out["array"] = run_leg(legs_dir, "slab2d", "array", policy="keep", until=20)
    out["cuda-per-point"] = run_leg(legs_dir, "slab2d", "gpu", table="cuda", until=20,
                                    extra=["--chi1inv-read", "per-point"])
    out["reference"] = run_leg(legs_dir, "slab2d", "reference", until=20)
    out["cpu"] = run_leg(legs_dir, "slab2d", "cpu", until=20)
    return out


@pytest.mark.parametrize("table", TABLES)
def test_slab_kernels_dispatch_on_the_amd_gpu(slab, table):
    rec, _ = slab[table]
    assert rec["gpu"] == "cuda" and rec["versions"]["cupy_is_hip"]
    assert rec["active_step_path"] == "fused"
    assert rec["dispatch"]["table"] == table
    assert set(rec["dispatch"]["dispatched_slots"]) == {"step_B", "update_H", "step_D", "update_E"}
    assert rec["certified"] is False            # outside the certified identities (MEEP_GPU_HARDWARE=any)
    assert rec["subnormal_policy"] == "keep"


@pytest.mark.parametrize("table", TABLES)
def test_slab_kernels_equal_the_array_path_bit_for_bit(slab, table):
    _, gpu = slab[table]
    _, array = slab["array"]
    assert differing_words(gpu["field"], array["field"]) == 0
    for name in ("flux_0", "flux_1"):
        np.testing.assert_array_equal(gpu[name], array[name])


def test_slab_array_path_and_reference_against_the_cpu_solver(slab):
    _, cpu = slab["cpu"]
    for leg in ("array", "reference", *TABLES):
        _, arrays = slab[leg]
        assert arrays["field"].shape == cpu["field"].shape
        assert rel_l2(arrays["field"], cpu["field"]) < BAND, leg
        assert rel_l2(arrays["flux_0"], cpu["flux_0"]) < BAND, leg
    assert slab["cpu"][0]["steps"] == slab["triton"][0]["steps"] == 800


@pytest.fixture(scope="module")
def ring(legs_dir):
    return {"triton": run_leg(legs_dir, "ring2d", "gpu", table="triton"),
            "cuda": run_leg(legs_dir, "ring2d", "gpu", table="cuda"),
            "cpu": run_leg(legs_dir, "ring2d", "cpu")}


@pytest.mark.parametrize("table", TABLES)
def test_ring_point_series_and_resonances_against_the_cpu_solver(ring, table):
    rec, gpu = ring[table]
    crec, cpu = ring["cpu"]
    assert rec["active_step_path"] == "fused"
    assert rec["steps"] == crec["steps"] and len(gpu["series"]) == len(cpu["series"]) > 1000
    assert rel_l2(gpu["series"], cpu["series"]) < BAND
    # The resonances: the complex frequency f + i decay and Q of each mode against the CPU solver.
    modes, cmodes = rec["harminv"], crec["harminv"]
    resonances = [m for m in cmodes if m["Q"] > 50]
    assert len(resonances) >= 3, cmodes
    for m in resonances:
        match = min(modes, key=lambda g: abs(g["freq"] - m["freq"]))
        assert abs(complex(match["freq"] - m["freq"], match["decay"] - m["decay"])) / m["freq"] < RESONANCE_BAND
        assert abs(match["Q"] - m["Q"]) / m["Q"] < Q_BAND


@pytest.fixture(scope="module")
def sphere(legs_dir):
    out = {table: run_leg(legs_dir, "sphere3d", "gpu", table=table, until=10) for table in TABLES}
    out["array"] = run_leg(legs_dir, "sphere3d", "array", policy="keep", until=10)
    out["cuda-per-point"] = run_leg(legs_dir, "sphere3d", "gpu", table="cuda", until=10,
                                    extra=["--chi1inv-read", "per-point"])
    out["cpu"] = run_leg(legs_dir, "sphere3d", "cpu", until=10)
    return out


@pytest.mark.parametrize("table", TABLES)
def test_sphere_3d_against_the_array_path_and_the_cpu_solver(sphere, table):
    rec, gpu = sphere[table]
    _, array = sphere["array"]
    _, cpu = sphere["cpu"]
    assert rec["active_step_path"] == "fused" and rec["dispatch"]["table"] == table
    assert differing_words(gpu["field"], array["field"]) == 0
    np.testing.assert_array_equal(gpu["flux_0"], array["flux_0"])
    assert rel_l2(gpu["field"], cpu["field"]) < BAND
    assert rel_l2(gpu["flux_0"], cpu["flux_0"]) < BAND


@pytest.mark.parametrize("case", ["slab", "sphere"])
def test_the_bulk_read_of_the_permittivity_changes_no_word(request, case):
    """Patch 0003: the lift reads the permittivity in bulk on this solver build, and the fields after the run are
    the same words as after a lift that reads every point one at a time (the reference of the add-on)."""
    legs = request.getfixturevalue(case)
    rec, bulk = legs["cuda"]
    prec, per_point = legs["cuda-per-point"]
    assert rec["chi1inv_read"]["path"] == "bulk", rec["chi1inv_read"]
    assert not rec["chi1inv_read"]["refusal"] and not rec["chi1inv_read"]["restarted_by"], rec["chi1inv_read"]
    assert prec["chi1inv_read"]["path"] == "per-point", prec["chi1inv_read"]
    assert differing_words(bulk["field"], per_point["field"]) == 0
    for name in [k for k in bulk if k.startswith("flux_")]:
        np.testing.assert_array_equal(bulk[name], per_point[name])
