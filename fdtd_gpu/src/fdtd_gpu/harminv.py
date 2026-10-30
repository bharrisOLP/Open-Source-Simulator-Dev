"""Resonant modes of a time signal (frequency, decay, Q, amplitude), as the Harminv tool of the MIT package.

Method: the signal is shifted to the center frequency of the band, filtered with a windowed-sinc low-pass FIR,
sampled at about 4 x the band width, and fitted with a sum of damped exponentials by the matrix pencil method
(Hua and Sarkar, IEEE Trans. Antennas Propag. 38, 814, 1990). This is not the filter-diagonalization method of
Harminv; the results agree for well separated modes with a good signal-to-noise ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarminvMode:
    freq: float          # real part of the complex frequency (1/um)
    decay: float         # imaginary part of the complex frequency (negative for a decaying mode)
    Q: float             # freq / (-2 decay)
    amp: complex         # NOT the amplitude of the MIT package: the complex amplitude of the shifted and filtered
    #                      signal (about half of the real cosine amplitude, times the gain of the filter)
    err: float           # NOT the error of the MIT package: one relative residual of the whole fit (all modes)


def harminv(signal, dt: float, fmin: float, fmax: float, max_modes: int = 30, rel_sv: float = 1e-5,
            oversample: float = 4.0) -> list[HarminvMode]:
    """Modes of a real signal sampled at the step dt, with frequencies in [fmin, fmax], sorted by frequency."""
    s = np.asarray(signal, float)
    if not 0 < fmin < fmax:
        raise ValueError("need 0 < fmin < fmax")
    fc, half = 0.5 * (fmin + fmax), 0.5 * (fmax - fmin)
    t = np.arange(len(s)) * dt
    x = s * np.exp(2j * np.pi * fc * t)                 # the band moves to [-half, half]
    fcut = 1.5 * half
    ntap = int(np.ceil(6.0 / (fcut * dt))) | 1         # about 6 periods of the cutoff
    k = (np.arange(ntap) - ntap // 2) * dt
    taps = 2 * fcut * dt * np.sinc(2 * fcut * k) * np.blackman(ntap)
    if len(x) <= 2 * ntap:
        raise ValueError("signal too short for this band: run longer or use a wider band")
    y = np.convolve(x, taps, mode="valid")
    q = max(1, int(1.0 / (2 * oversample * half * dt)))
    y = y[::q]
    delta = q * dt
    n = len(y)
    if n < 8:
        raise ValueError("too few samples after decimation: run longer")
    L = n // 3
    H = np.lib.stride_tricks.sliding_window_view(y, L + 1)          # (n - L, L + 1) Hankel matrix
    _, sv, vh = np.linalg.svd(H, full_matrices=False)
    M = int(min(max_modes, np.count_nonzero(sv > rel_sv * sv[0])))
    if M == 0:
        return []
    vm = vh[:M]
    z = np.linalg.eigvals(vm[:, 1:] @ np.linalg.pinv(vm[:, :-1]))
    z = z[np.abs(z) > 0]
    V = z[None, :] ** np.arange(n)[:, None]
    amp, *_ = np.linalg.lstsq(V, y, rcond=None)
    err = float(np.linalg.norm(V @ amp - y) / np.linalg.norm(y))
    freq = fc - np.angle(z) / (2 * np.pi * delta)
    rate = -np.log(np.abs(z)) / delta                    # amplitude decay rate (1/time)
    decay = -rate / (2 * np.pi)
    modes = []
    for f, d, a in zip(freq, decay, amp):
        if fmin <= f <= fmax:
            Q = f / (-2 * d) if d != 0 else np.inf
            modes.append(HarminvMode(float(f), float(d), float(Q), complex(a), err))
    return sorted(modes, key=lambda m: m.freq)
