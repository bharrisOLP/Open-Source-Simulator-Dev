"""Time of the image-quality analyses (optics_gpu.analysis) on a PyTorch device against Optiland 0.6.3 on its NumPy
backend (CPU), for the aberrated singlet (default) or the perfect lens (--system relay), on the axis.

For each pupil sampling N (an N x N grid, the points inside the unit circle traced) and the FFT grid M = 2 N, one
JSON line with the best of --repeat runs (after one untimed run) of:
- Optiland: Wavefront (uniform, N), ZernikeFit (Fringe, 37 terms, on the points of that Wavefront),
  ScalarFFTPSF (N, M; it traces its own wavefront), ScalarFFTMTF (N, M; it traces its own wavefront and PSF);
- optics_gpu: pupil_wavefront, zernike_fit (Fringe, 37), fft_psf, fft_mtf (each step alone, with
  torch.cuda.synchronize), and `ours_total_s` (all four steps in one run).
`speedup_psf` is the time of ScalarFFTPSF over pupil_wavefront + fft_psf; `speedup_mtf` that of ScalarFFTMTF over
pupil_wavefront + fft_psf + fft_mtf. Each row gives the largest differences from Optiland (OPD in waves, Zernike
coefficients in waves, PSF in Strehl units, MTF). With --huygens, one more line per N <= 256: Optiland's HuygensPSF
(pupil N, 80 x 80 points of 1 um) against huygens_psf (the image sum of a point source on the same points),
after all other rows (with the kept rays of the image sum and the rays of HuygensPSF). Each row also logs the clock
time, the versions of Optiland and PyTorch, and the thread counts of PyTorch and numba with the numba threading
layer (the rows of benchmarks/2026-11-08/optics-analysis.jsonl do not have these fields).

    bash tools/gpu_run.sh optics_gpu python bench/bench_analysis.py --pupil 64 128 256 512 1024
"""
import argparse
import json
import time
import warnings

import numpy as np
import torch

from optics_gpu.analysis import fft_mtf, fft_psf, huygens_psf, pupil_wavefront, zernike_fit
from optics_gpu.systems import aberrated_singlet, ideal_relay

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
WL = 1.55

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=("singlet", "relay"), default="singlet")
ap.add_argument("--pupil", type=int, nargs="+", default=[64, 128, 256, 512])
ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--huygens", action="store_true")
a = ap.parse_args()
optic = aberrated_singlet() if a.system == "singlet" else ideal_relay()
FIELD = (0.0, 0.0)


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


def np_(t):
    return t.detach().cpu().numpy()


def provenance() -> dict:
    """The clock time, the versions, and the thread counts of PyTorch and numba (numba's threading layer is known
    only after a parallel numba kernel has run, for example Optiland's HuygensPSF)."""
    from importlib.metadata import version

    import numba

    try:
        layer = numba.threading_layer()
    except ValueError:
        layer = "not initialized"
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "optiland": version("optiland"),
            "torch": torch.__version__, "threads": torch.get_num_threads(), "numba_threads": numba.get_num_threads(),
            "numba_threading_layer": layer}


for N in a.pupil:
    from optiland.mtf import ScalarFFTMTF
    from optiland.psf.fft import ScalarFFTPSF
    from optiland.wavefront import Wavefront
    from optiland.zernike import ZernikeFit

    M = 2 * N
    t_owf, owf = best(lambda: Wavefront(optic, fields=[FIELD], wavelengths=[WL], num_rays=N,
                                        distribution="uniform").get_data(FIELD, WL))
    p = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(p, p)
    ins = X**2 + Y**2 <= 1
    use = np.asarray(owf.intensity) > 0
    t_ozk, ozk = best(lambda: ZernikeFit(X[ins][use], Y[ins][use], np.asarray(owf.opd)[use], "fringe", 37))
    t_opsf, opsf = best(lambda: ScalarFFTPSF(optic, FIELD, WL, num_rays=N, grid_size=M))
    t_omtf, omtf = best(lambda: ScalarFFTMTF(optic, [FIELD], WL, num_rays=N, grid_size=M))

    t_wf, wf = best(lambda: pupil_wavefront(optic, FIELD, WL, n_pupil=N, device=a.device))
    t_zk, zk = best(lambda: zernike_fit(wf, 37, "fringe"))
    t_psf, psf = best(lambda: fft_psf(wf, M))
    t_mtf, mtf = best(lambda: fft_mtf(wf, psf))

    def total():
        w = pupil_wavefront(optic, FIELD, WL, n_pupil=N, device=a.device)
        zernike_fit(w, 37, "fringe")
        q = fft_psf(w, M)
        return fft_mtf(w, q)

    t_all, _ = best(total)
    d_opd = float(np.max(np.abs(np_(wf.opd).T[ins] - np.asarray(owf.opd))))
    d_zk = float(np.max(np.abs(np_(zk.coeffs) - np.asarray(ozk.coeffs))))
    d_psf = float(np.max(np.abs(np_(psf.psf).T - np.asarray(opsf.psf) / 100)))
    d_mtf = max(float(np.max(np.abs(np_(mtf.tangential) - np.asarray(omtf.mtf[0][0])))),
                float(np.max(np.abs(np_(mtf.sagittal) - np.asarray(omtf.mtf[0][1])))))
    row = {"system": a.system, "device": a.device, "pupil": N, "grid": M, "rays": wf.n_rays,
           "optiland_wavefront_s": t_owf, "optiland_zernike_s": t_ozk, "optiland_fft_psf_s": t_opsf,
           "optiland_fft_mtf_s": t_omtf, "ours_wavefront_s": t_wf, "ours_zernike_s": t_zk, "ours_fft_psf_s": t_psf,
           "ours_fft_mtf_s": t_mtf, "ours_total_s": t_all, "speedup_psf": t_opsf / (t_wf + t_psf),
           "speedup_mtf": t_omtf / (t_wf + t_psf + t_mtf), "d_opd_waves": d_opd, "d_zernike_waves": d_zk,
           "d_psf": d_psf, "d_mtf": d_mtf, "strehl": float(np_(psf.psf).max()),
           **provenance()}
    print(json.dumps(row), flush=True)

# the Huygens rows after all FFT rows: Optiland's HuygensPSF runs numba kernels in parallel, and in a development run
# torch.get_num_threads() changed from 4 to 16 after it
for N in (n for n in a.pupil if a.huygens and n <= 256):
    from optiland.psf import HuygensPSF

    t_oh, oh = best(lambda: HuygensPSF(optic, FIELD, WL, num_rays=N, image_size=80, pixel_pitch=1e-3))
    xs = np.linspace(float(oh.cx) - 0.04, float(oh.cx) + 0.04, 80) * 1e3
    t_h, (h, bundle) = best(lambda: huygens_psf(optic, WL, xs, xs, device=a.device, return_bundle=True))
    I, R = np_(h), np.asarray(oh.psf).T / 100
    corr = float((I * R).sum() / np.sqrt((I**2).sum() * (R**2).sum()))
    print(json.dumps({"system": a.system, "device": a.device, "huygens_pupil": N, "image": 80,
                      "optiland_huygens_s": t_oh, "ours_huygens_s": t_h, "speedup": t_oh / t_h,
                      "correlation": corr, "peak_ours": float(I.max()), "peak_optiland": float(R.max()),
                      "image_sum_rays": bundle.n_rays,
                      "optiland_huygens_rays": int(np.size(oh.get_data(FIELD, WL).opd)),
                      **provenance()}), flush=True)
