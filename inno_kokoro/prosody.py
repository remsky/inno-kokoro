"""Prosody measurements used at enrollment: log-F0 mean/std in semitones, voiced fraction, speaking rate, spectral tilt.

CLI diagnostic: python -m inno_kokoro.prosody <ref.wav> <clone.wav> ...
"""

import math
import sys

import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

SR_ENC = 16000  # pitch tracking runs at the speaker-encoder rate
FRAME_S = 0.02  # pitch frame length
HOP = int(SR_ENC * FRAME_S)  # samples per pitch frame (320)
F0_MAX = 400.0  # Hz, upper bound for voiced speech
F0_MIN = 60.0  # Hz, first-pass lower bound
MEDIAN_FRAMES = 30  # median-smoothing window over the tracked lag, 600 ms
LAG_TOL = 0.9  # a submultiple must reach this fraction of the peak to count as the period
ST_REF_HZ = 100.0  # semitones are measured relative to 100 Hz
SR_TILT = 24000  # tilt is measured at Kokoro's rate with a fixed window, so references and renders read alike
TILT_WIN, TILT_HOP = 1024, 256
TILT_BAND = (300.0, 4000.0)  # Hz, the slope is fitted over this band


def st(hz):
    """Hz -> semitones relative to ST_REF_HZ."""
    return 12 * torch.log2(hz / ST_REF_HZ)


def hz(semitones):
    """Semitones relative to ST_REF_HZ -> Hz."""
    return ST_REF_HZ * 2 ** (semitones / 12)


def resample(w, sr, target):
    """[T] float tensor at sr -> [T'] at target, polyphase."""
    if sr == target:
        return w
    g = math.gcd(sr, target)
    return torch.from_numpy(resample_poly(w.numpy(), target // g, sr // g)).float()


def _wav(path, sr):
    """Accept a wav path (sr=None) or a [T] tensor with its sample rate. Returns (mono float tensor, sr)."""
    if sr is None:
        w, sr = sf.read(path)
        if w.ndim > 1:
            w = w.mean(1)
        return torch.tensor(w, dtype=torch.float32), sr
    return path.detach().float().cpu(), sr


def _frame_rms(w, hop):
    """RMS energy per hop-sized frame, dropping the partial last frame."""
    n = len(w) // hop * hop
    return w[:n].reshape(-1, hop).pow(2).mean(1).sqrt()


def _fundamental(rows, off, tol):
    """Peak index per row of [N, lags], resolved to the period rather than a multiple of it. A 2x or 3x lag scores
    as well as the true period, and better when the period falls between samples, so the shortest submultiple
    scoring within tol of the peak wins. `off` is the lag the first column stands for."""
    best = rows.argmax(1)
    peak = rows.gather(1, best[:, None]).squeeze(1)
    idx = best
    for k in (2, 3, 4):
        c = torch.round((best + off) / k).long() - off
        near = (c[:, None] + torch.tensor([-1, 0, 1])).clamp(0, rows.shape[1] - 1)
        v, j = rows.gather(1, near).max(1)
        ok = (c >= 0) & (peak > 0) & (v >= tol * peak)
        idx = torch.where(ok, near.gather(1, j[:, None]).squeeze(1), idx)
    return idx


def _track_f0(w, lo, hi, win=2 * HOP):
    """Normalized cross-correlation pitch tracker at SR_ENC, one estimate per HOP. Returns (f0 Hz, rms) per frame."""
    lags = range(int(SR_ENC / hi), int(SR_ENC / lo) + 1)
    length = win + lags[-1]
    frames = F.pad(w, (0, length)).unfold(0, length, HOP)
    x = frames[:, :win]
    xx = x.pow(2).sum(1)
    corr = []
    for lag in lags:
        y = frames[:, lag : lag + win]
        corr.append((x * y).sum(1) / (xx * y.pow(2).sum(1)).sqrt().clamp_min(1e-8))
    corr = torch.stack(corr, 1)
    best_lag = (lags[0] + _fundamental(corr, lags[0], LAG_TOL))[: len(w) // HOP].float()
    pad = MEDIAN_FRAMES // 2
    best_lag = F.pad(best_lag[None, None], (pad, MEDIAN_FRAMES - 1 - pad), mode="replicate")[0, 0]
    best_lag = best_lag.unfold(0, MEDIAN_FRAMES, 1).median(1).values
    return SR_ENC / best_lag, _frame_rms(w, HOP)


def rate(path, sr=None, limit=60):
    """Speaking-rate proxy: smoothed-energy peaks (roughly syllables) per second of speech, over the first `limit` s."""
    w, sr = _wav(path, sr)
    hop = sr // 100  # 10 ms frames
    e = _frame_rms(w[: limit * sr], hop)
    e = F.avg_pool1d(e[None, None], 3, 1, 1)[0, 0]  # 3-frame smoothing
    speech = e > 0.05 * e.max()  # energy gate for "is speech"

    is_peak = (e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]) & speech[1:-1]
    peaks = is_peak.nonzero().flatten()
    if len(peaks) == 0:
        return 0.0
    # Peaks closer than 80 ms (8 frames) belong to the same syllable.
    syllables = 1 + (peaks[1:] - peaks[:-1] >= 8).sum().item()
    speech_seconds = speech.float().mean().item() * len(e) / 100
    return syllables / (speech_seconds + 1e-9)


def tilt(path, sr=None):
    """Spectral tilt in dB per octave: least-squares slope of the long-term power spectrum over TILT_BAND.
    Pink noise reads -3, speech about -8; less negative is brighter, more vocal effort."""
    w, sr = _wav(path, sr)
    w = resample(w, sr, SR_TILT)
    spec = (
        torch.stft(
            w,
            TILT_WIN,
            TILT_HOP,
            window=torch.hann_window(TILT_WIN),
            return_complex=True,
        )
        .abs()
        .pow(2)
        .mean(1)
    )
    f = torch.linspace(0, SR_TILT / 2, len(spec))
    band = (f > TILT_BAND[0]) & (f < TILT_BAND[1])
    x = torch.log2(f[band])
    y = 10 * torch.log10(spec[band] + 1e-12)
    x = x - x.mean()
    return ((x * (y - y.mean())).sum() / (x * x).sum()).item()


CEPS_N = 4096  # cepstrum frame length, ~4 Hz resolution at SR_ENC
CEPS_MIN_PEAK = 0.01  # comb strength below this = no reliable fundamental, leave the ceiling alone
CEPS_CEIL = 1.6  # tracking ceiling in multiples of the cepstral fundamental
CEPS_TOL = 0.25  # rahmonics can outweigh the fundamental, and a ceiling set too low is unrecoverable


def cepstral_f0(w):
    """Fundamental from harmonic spacing: the cepstral peak over the loud frames. Immune to the octave
    errors that fool the lag tracker on band-limited audio, but biased toward the modal (not mean) F0.
    Returns (hz, peak strength); (0, 0) if the clip is too short or has no loud frames."""
    frames = w[: len(w) // CEPS_N * CEPS_N].reshape(-1, CEPS_N)
    if len(frames):
        e = frames.pow(2).mean(1)
        frames = frames[e > e.quantile(0.7)]
    if not len(frames):
        return 0.0, 0.0
    spec = (torch.fft.rfft(frames * torch.hann_window(CEPS_N)).abs() + 1e-8).log()
    ceps = torch.fft.irfft(spec - spec.mean(1, keepdim=True)).mean(0)
    q = torch.arange(len(ceps)) / SR_ENC  # quefrency in seconds
    m = (q > 1 / 300) & (q < 1 / F0_MIN)
    band, q0 = ceps[m], int(m.nonzero()[0])
    i = int(_fundamental(band[None], q0, CEPS_TOL)[0])
    return SR_ENC / (q0 + i), band[i].item()


def ceiling(path, sr=None, fmax=None):
    """Tracking ceiling in Hz. Broadcast or telephone band audio can make the tracker read frames an octave high;
    when the harmonic comb is clear, the ceiling is set from it. Pass fmax to override."""
    if fmax is not None:
        return fmax
    w, sr = _wav(path, sr)
    c, peak = cepstral_f0(resample(w, sr, SR_ENC))
    return min(F0_MAX, CEPS_CEIL * c) if peak > CEPS_MIN_PEAK else F0_MAX


def stats(path, sr=None, fmax=None):
    """Log-F0 statistics of the voiced frames. Returns (mean st, std st, voiced fraction); zeros if nothing is voiced.
    fmax as in ceiling(); enroll() resolves it once and passes the same value here and to head_stats()."""
    w, sr = _wav(path, sr)
    fmax = ceiling(w, sr, fmax)
    w = resample(w, sr, SR_ENC)

    # Pass 2 raises the floor to 0.65x the upper-quartile F0, killing octave-down picks on high voices.
    lo = F0_MIN
    for _ in range(2):
        f0, e = _track_f0(w, lo, fmax)
        voiced = (e > 0.1 * e.max()) & (f0 > lo) & (f0 < fmax)
        voiced_st = st(f0[voiced])
        if not len(voiced_st):
            return 0.0, 0.0, 0.0
        lo = min(0.65 * hz(voiced_st.quantile(0.75).item()), 180)

    return voiced_st.mean().item(), voiced_st.std().item(), voiced.float().mean().item()


HEAD_CHUNK_S = 6.0  # the head was fitted on per-clip stats, so a whole reference is measured in chunks


def head_stats(path, sr=None, fmin=F0_MIN, fmax=F0_MAX):
    """(F0 mean st, F0 sd st) for the prosody head: Praat pitch at 10 ms over energy-gated frames, averaged over
    HEAD_CHUNK_S chunks. stats() smooths over 600 ms and reads the spread ~1.3 st low, so it cannot feed the head."""
    import parselmouth

    w, sr = _wav(path, sr)
    w = w.double().numpy()
    hop, out = 0.01, []
    for i in range(0, len(w), int(HEAD_CHUNK_S * sr)):
        x = w[i : i + int(HEAD_CHUNK_S * sr)]
        if len(x) < 1.5 * sr:
            continue
        p = parselmouth.Sound(x, sampling_frequency=sr).to_pitch_ac(time_step=hop, pitch_floor=fmin, pitch_ceiling=fmax)
        f0 = p.selected_array["frequency"]
        n = int(sr * hop)
        e = ((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(1)) ** 0.5
        e = e[((p.xs() / hop).astype(int)).clip(0, len(e) - 1)]
        v = f0[(f0 > 0) & (e > 0.1 * e.max())]
        if len(v):
            s = 12 * torch.log2(torch.tensor(v) / ST_REF_HZ)
            out.append((s.mean().item(), s.std(correction=0).item(), len(x)))
    if not out:
        return 0.0, 0.0
    n = sum(c[2] for c in out)  # length-weighted
    return tuple(float(sum(c[i] * c[2] for c in out) / n) for i in (0, 1))


if __name__ == "__main__":
    print(f"{'file':40s} {'F0 mean(st)':>11s} {'F0 std':>7s} {'voiced%':>8s} {'peaks/s':>8s} {'tilt':>6s}")
    for p in sys.argv[1:]:
        mean, std, voiced = stats(p)
        print(f"{p[-40:]:40s} {mean:11.1f} {std:7.1f} {100 * voiced:8.0f} {rate(p):8.1f} {tilt(p):6.1f}")
