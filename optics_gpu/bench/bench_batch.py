"""Time of the batches of optics_gpu.analysis (image_quality: fields x wavelengths x planes in one call) against the
per-call functions and against Optiland 0.6.3 on its NumPy backend (CPU), for systems.dispersive_singlet: the
object heights 0, 0.5 and 1 mm (--fields), the wavelengths F, d, C, and the planes dz (--dz, um).

For each pupil sampling N (FFT grid M = 2 N), one JSON line with the best of --repeat runs (after one untimed run),
times in s:
- batch_s: image_quality with the weights 1, 2, 1 (FFT PSF and MTF of all F W Z planes; polychromatic PSF and MTF
  of the F Z field points and planes); batch_mono_s: image_quality without weights (no polychromatic results);
- percall_s: the same monochromatic PSFs and MTFs by one call for each field, wavelength and plane, with a new trace
  for each plane: pupil_wavefront of the optic with its image surface moved by dz, fft_psf, fft_mtf;
- percall_defocus_s: pupil_wavefront for each field and wavelength, then defocus, fft_psf and fft_mtf for each plane;
- single_s and single_cold_s: one pupil_wavefront call (field 0, wavelength d) with the kept copy of the optic,
  and after clear_copy_cache (a new copy);
- optiland_s: Optiland's ScalarFFTPSF and ScalarFFTMTF (it traces its own wavefront and PSF) for each field,
  wavelength and plane, on the optic with its image surface moved by dz; optiland_mtf_s: ScalarFFTMTF alone.
The row also gives the largest differences of the batch from the per-call results with a new trace for each plane
(d_percall_psf, d_percall_mtf) and from Optiland (d_optiland_psf in Strehl units, d_optiland_mtf), the peak of the
polychromatic PSF of the last field at dz = 0, the clock time, the versions, the thread counts and max_rss_gb (the
peak resident memory of this process). Optiland runs with --optiland-max N and below (its time grows as N^2).

    bash tools/gpu_run.sh optics_gpu python bench/bench_batch.py --pupil 64 128 256 512
"""
import argparse
import json
import resource
import time
import warnings

import numpy as np
import torch

from optics_gpu.analysis import (clear_copy_cache, defocus, fft_mtf, fft_psf, image_quality, pupil_wavefront)
from optics_gpu.systems import dispersive_singlet

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
FCD = (0.4861, 0.5876, 0.6563)
WEIGHTS = (1.0, 2.0, 1.0)

ap = argparse.ArgumentParser()
ap.add_argument("--pupil", type=int, nargs="+", default=[64, 128, 256])
ap.add_argument("--fields", type=float, nargs="+", default=[0.0, 0.5, 1.0])
ap.add_argument("--dz", type=float, nargs="+", default=[-100.0, -50.0, 0.0, 50.0, 100.0])
ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--optiland-max", type=int, default=256)
a = ap.parse_args()
optic = dispersive_singlet(tuple(a.fields))
FIELDS = [(0.0, y / max(a.fields)) if max(a.fields) > 0 else (0.0, 0.0) for y in a.fields]


def sync():
    if a.device == "cuda":
        torch.cuda.synchronize()


def best(fn):
    out = fn()
    sync()
    times = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return min(times), out


def moved(dz_um: float):
    o = dispersive_singlet(tuple(a.fields))
    k = len(o.surfaces.surfaces) - 2
    o.updater.set_thickness(float(o.surfaces[k].thickness) + dz_um * 1e-3, k)
    return o


MOVED = {dz: (optic if dz == 0 else moved(dz)) for dz in a.dz}


def provenance() -> dict:
    from importlib.metadata import version

    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "optiland": version("optiland"), "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}


for N in a.pupil:
    M = 2 * N

    def percall():
        out = {}
        for f, field in enumerate(FIELDS):
            for i, wl in enumerate(FCD):
                for z, dz in enumerate(a.dz):
                    w = pupil_wavefront(MOVED[dz], field, wl, n_pupil=N, device=a.device)
                    q = fft_psf(w, M)
                    out[f, i, z] = (q, fft_mtf(w, q))
        return out

    def percall_defocus():
        out = {}
        for f, field in enumerate(FIELDS):
            for i, wl in enumerate(FCD):
                w0 = pupil_wavefront(optic, field, wl, n_pupil=N, device=a.device)
                for z, dz in enumerate(a.dz):
                    w = defocus(w0, dz)
                    q = fft_psf(w, M)
                    out[f, i, z] = (q, fft_mtf(w, q))
        return out

    t_batch, res = best(lambda: image_quality(optic, FIELDS, FCD, a.dz, WEIGHTS, n_pupil=N, grid_size=M,
                                              device=a.device))
    t_mono, _ = best(lambda: image_quality(optic, FIELDS, FCD, a.dz, None, n_pupil=N, grid_size=M,
                                           device=a.device))
    t_pc, pc = best(percall)
    t_pd, _ = best(percall_defocus)
    t_single, _ = best(lambda: pupil_wavefront(optic, FIELDS[0], FCD[1], n_pupil=N, device=a.device))

    def cold():
        clear_copy_cache()
        return pupil_wavefront(optic, FIELDS[0], FCD[1], n_pupil=N, device=a.device)

    t_cold, _ = best(cold)
    d_psf = d_mtf = 0.0
    for (f, i, z), (q, m) in pc.items():
        d_psf = max(d_psf, float((res.psfs[f][i][z].psf - q.psf).abs().max()))
        d_mtf = max(d_mtf, float((res.mtfs[f][i][z].tangential - m.tangential).abs().max()),
                    float((res.mtfs[f][i][z].sagittal - m.sagittal).abs().max()))
    row = {"system": "dispersive_singlet", "device": a.device, "pupil": N, "grid": M, "fields": len(FIELDS),
           "wavelengths": len(FCD), "planes": len(a.dz), "dz_um": a.dz, "rays": res.wavefronts[0][0][0].n_rays,
           "batch_s": t_batch, "batch_mono_s": t_mono, "percall_s": t_pc, "percall_defocus_s": t_pd,
           "single_s": t_single, "single_cold_s": t_cold, "speedup_percall": t_pc / t_mono,
           "d_percall_psf": d_psf, "d_percall_mtf": d_mtf,
           "poly_peak": float(res.poly_psfs[-1][list(a.dz).index(0.0)].psf.max()) if 0.0 in a.dz else None}
    if N <= a.optiland_max:
        from optiland.mtf import ScalarFFTMTF
        from optiland.psf.fft import ScalarFFTPSF

        def optiland_all(mtf_only=False):
            out = {}
            for f, field in enumerate(FIELDS):
                for i, wl in enumerate(FCD):
                    for z, dz in enumerate(a.dz):
                        p = None if mtf_only else ScalarFFTPSF(MOVED[dz], field, wl, num_rays=N, grid_size=M)
                        out[f, i, z] = (p, ScalarFFTMTF(MOVED[dz], [field], wl, num_rays=N, grid_size=M))
            return out

        t_opt, opt = best(optiland_all)
        t_opt_mtf, _ = best(lambda: optiland_all(True))
        d_opsf = d_omtf = 0.0
        for (f, i, z), (p, m) in opt.items():
            d_opsf = max(d_opsf, float(np.max(np.abs(res.psfs[f][i][z].psf.cpu().numpy().T - np.asarray(p.psf) / 100))))
            d_omtf = max(d_omtf, float(np.max(np.abs(res.mtfs[f][i][z].tangential.cpu().numpy()
                                                     - np.asarray(m.mtf[0][0])))),
                         float(np.max(np.abs(res.mtfs[f][i][z].sagittal.cpu().numpy() - np.asarray(m.mtf[0][1])))))
        row.update({"optiland_s": t_opt, "optiland_mtf_s": t_opt_mtf, "speedup_optiland": t_opt / t_mono,
                    "d_optiland_psf": d_opsf, "d_optiland_mtf": d_omtf})
    row.update(provenance())
    print(json.dumps(row), flush=True)
