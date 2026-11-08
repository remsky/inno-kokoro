"""Autocorrelation pitch tracker for the prosody head, in NumPy; no torch, so a torch-free build can use it as is.

The algorithm is Boersma (1993), "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise
ratio of a sampled sound". The head was fitted on Praat's To Pitch (ac) at its defaults, so the thresholds, the unvoiced
candidate's strength and the octave cost follow the Praat manual's description of that command, and the rest (frame
times, window, candidate refinement, voicing) was tuned against Praat's output to read the same. No Praat code is used.
"""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import irfft, next_fast_len, rfft

HOP_S = 0.01  # Praat time step
VOICING = 0.45  # voicing threshold on the normalized autocorrelation peak
SILENCE = 0.03  # silence threshold, frame peak over clip peak
OCTAVE = 0.01  # strength cost per octave below the ceiling: ties go to the higher candidate
OCTAVE_JUMP = 0.35  # cost per octave between consecutive voiced frames
VUV = 0.14  # voiced/unvoiced transition cost
K = 14  # voiced candidates kept per frame; with the unvoiced one, Praat's 15
UP = 4  # 1/UP-lag refinement grid: whole-lag parabolas underestimate sharp peaks, losing an octave; finer buys nothing
DEPTH = 16  # sin(x)/x taps each side of the grid: Praat's candidates to ~1e-4, head stats to ~3e-4 st; 32 closes that

F0_MIN, F0_MAX = 60.0, 400.0  # Hz, the floor and ceiling the head was fitted with; prosody.py's own tracker shares them
ST_REF_HZ = 100.0  # semitones are measured relative to 100 Hz
CHUNK_S = 6.0  # the head was fitted on per-clip stats, so a whole reference is measured in chunks
MIN_CHUNK_S = 1.5  # a trailing chunk shorter than this is dropped
ENERGY_GATE = 0.1  # voiced frames below this fraction of the loudest frame's RMS are dropped, as in the head's fit


def frame_times(n_samples, sr, fmin=F0_MIN):
    """Centre times (s) of the analysis frames: as many whole windows as fit, HOP_S apart, centred in the clip."""
    dur, wdur = n_samples * (1.0 / sr), 3.0 / fmin  # n * dx, rounded as Praat rounds it, so the frame count agrees
    n = math.floor((dur - wdur) / HOP_S) + 1
    return 0.5 * dur - 0.5 * (n - 1) * HOP_S + HOP_S * np.arange(max(n, 0))


def _candidates(w, sr, fmin, fmax):
    """Pitch candidates at frame_times(), Hann window of 3 periods of fmin: each frame's autocorrelation
    normalized by the window's own, the top K local maxima over whole lags, each refined by sin(x)/x interpolation
    onto a 1/UP-lag grid and a parabola through its best point.
    Returns (f0 [N, K] Hz, strength [N, K], frame peak over clip peak [N]); empty or out-of-range slots have f0 0."""
    win, period = int(3 * sr / fmin), int(sr / fmin)  # counts np.hanning's zero endpoints; parity was tuned on that
    taps = np.arange(-DEPTH - 1, DEPTH + 2)  # whole lags read around a candidate; kernel Hann zero at ±(DEPTH + 2)
    nlag = period + DEPTH + 4  # lags 2 .. period + 1 are searched; the taps around the last reach period + DEPTH + 2
    start = np.clip(np.round((frame_times(len(w), sr, fmin) - 1.5 / fmin) * sr).astype(int), 0, len(w) - win)
    frames = sliding_window_view(w.astype(np.float32), win)[start]  # a copy (fancy index), float32 from here on
    mid = win // 2
    hann = np.hanning(win)
    frames -= frames[:, mid - period : mid + period].mean(1, keepdims=True)  # local mean, a period each side
    frames *= hann.astype(np.float32)
    peak = np.abs(frames[:, mid - period // 2 : mid + period // 2]).max(1)  # windowed peak, half a period each side
    peak = peak / max(np.abs(w - w.mean()).max(), 1e-12)  # over the whole clip's peak: the silence test's input
    n = next_fast_len(win + nlag)  # linear autocorrelation is exact up to the longest lag at this length

    def autocorr(x):
        s = rfft(x, n)
        a = irfft(s.real**2 + s.imag**2, n)[..., :nlag]
        return a / np.maximum(a[..., :1], 1e-12)

    corr = autocorr(frames) / autocorr(hann).astype(np.float32)  # [N, nlag] over whole lags from 0
    band = corr[:, 2 : period + 2]  # column j is lag j + 2; maxima are tested at lags 3 .. period
    cand = np.full_like(band, -1.0)
    m = band[:, 1:-1]
    local_max = (m > band[:, :-2]) & (m >= band[:, 2:]) & (m > 0.5 * VOICING)
    cand[:, 1:-1] = np.where(local_max, m, -1.0)  # peaks up to the floor's lag
    jc = np.argpartition(-cand, K, axis=1)[:, :K]  # fewer than K peaks: r = -1 fillers, dropped by `voiced` at the end
    r = np.take_along_axis(cand, jc, 1)
    lag = jc + 2
    x = np.arange(-UP, UP + 1)[:, None] / UP - taps  # [2 UP + 1, taps]: grid points within a lag of the peak
    kernel = (np.sinc(x) * (0.5 + 0.5 * np.cos(np.pi * x / (DEPTH + 2)))).astype(np.float32)  # Hann-windowed sin(x)/x
    rows = np.arange(len(corr))[:, None, None]
    fine = corr[rows, np.abs(lag[..., None] + taps)] @ kernel.T  # [N, K, 2 UP + 1]; the autocorrelation is even
    i = np.clip(fine.argmax(-1), 1, 2 * UP - 1)[..., None]
    a, b, c = (np.take_along_axis(fine, i + d, -1)[..., 0] for d in (-1, 0, 1))
    off = np.clip(0.5 * (a - c) / np.minimum(a - 2 * b + c, -1e-12), -0.5, 0.5)  # the curvature at a peak is negative
    r = np.where(r > 0, b - 0.25 * (a - c) * off, r)  # interpolated peak height
    f0 = np.maximum(sr / (lag + (i[..., 0] - UP + off) / UP), fmin)  # a peak at the floor's lag can refine past it
    r = np.where(r > 1, 1 / r, r)
    strength = r - OCTAVE * np.log2(fmax / f0)
    voiced = (r > 0) & (f0 < fmax)
    return f0 * voiced, np.where(voiced, strength, -1.0), peak


def _path(f0, strength, peak):
    """Viterbi over the candidates plus an unvoiced candidate per frame. Returns f0 per frame, 0 = unvoiced."""
    n = len(f0)
    unvoiced = VOICING + np.maximum(0, 2 - peak / (SILENCE / (1 + VOICING)))  # unvoiced strength, Praat manual's form
    f = np.concatenate([f0, np.zeros((n, 1))], 1)  # [N, K+1], last column unvoiced
    voiced = f > 0  # filler slots (f 0) become extra unvoiced candidates below
    s = np.where(voiced, np.concatenate([strength, unvoiced[:, None]], 1), unvoiced[:, None]).astype(np.float64)
    lf = np.where(voiced, np.log2(np.maximum(f, 1e-8)), 0.0)
    prev, cur = voiced[:-1, :, None], voiced[1:, None, :]
    jump = OCTAVE_JUMP * np.abs(lf[:-1, :, None] - lf[1:, None, :])
    trans = np.where(prev & cur, jump, VUV * (prev | cur))  # [N-1, prev, cur]; both unvoiced costs nothing
    cost = -s[0]
    back = np.zeros((n, K + 1), dtype=np.int64)
    for t in range(1, n):
        total = cost[:, None] + trans[t - 1]
        back[t] = total.argmin(0)
        cost = total.min(0) - s[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = cost.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return f[np.arange(n), path]


def track_f0(w, sr, fmin=F0_MIN, fmax=F0_MAX):
    """(F0 in Hz per frame, 0 where unvoiced; frame centre times in s)."""
    if not 0 < fmin < fmax or sr < (K + 1) * fmin:  # K + 1 lags per period, for argpartition to have K to pick
        raise ValueError(f"need 0 < fmin < fmax and sr >= {K + 1} * fmin, got fmin {fmin}, fmax {fmax}, sr {sr}")
    w = np.asarray(w, dtype=np.float64)
    t = frame_times(len(w), sr, fmin)
    if not len(t):
        return np.zeros(0), t
    return _path(*_candidates(w, sr, fmin, fmax)), t


def st(f):
    return 12 * np.log2(f / ST_REF_HZ)


def clip_stats(x, sr, fmin=F0_MIN, fmax=F0_MAX, gate=ENERGY_GATE):
    """(F0 mean st, population F0 sd st) over voiced frames above `gate` of the loudest frame's RMS, or None: the
    per-clip unit the head was fitted on."""
    x = np.asarray(x, dtype=np.float64)
    f0, t = track_f0(x, sr, fmin, fmax)
    hop = int(sr * HOP_S)
    e = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    if not len(f0) or not len(e):
        return None
    e = e[np.clip((t / HOP_S).astype(int), 0, len(e) - 1)]  # frame holding each centre; truncates like the head's fit
    v = f0[(f0 > 0) & (e > gate * e.max())]
    if not len(v):
        return None
    s = st(v)
    return s.mean(), s.std()


def head_stats(w, sr, fmin=F0_MIN, fmax=F0_MAX, gate=ENERGY_GATE):
    """(F0 mean st, F0 sd st) for the prosody head: clip_stats over CHUNK_S chunks, length-weighted; None if no chunk
    has anything voiced."""
    w = np.asarray(w, dtype=np.float64)
    n, out = int(CHUNK_S * sr), []
    for i in range(0, len(w), n):
        x = w[i : i + n]
        if len(x) >= MIN_CHUNK_S * sr and (s := clip_stats(x, sr, fmin, fmax, gate)):
            out.append((*s, len(x)))
    if not out:
        return None
    total = sum(c[2] for c in out)
    return tuple(float(sum(c[i] * c[2] for c in out) / total) for i in (0, 1))
