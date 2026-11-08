"""Kaldi fbank front end of the speaker encoder, in NumPy: the input of the ONNX graph.

The recipe the weights were trained on, the same as encoder.py's torch fbank: int16-scale samples, 25/10 ms frames,
snip edges, DC removal, preemphasis, hamming, 512-point power spectrum, 80 kaldi mel filters from 20 Hz to nyquist,
log, per-utterance mean normalization. tests/test_fbank.py holds it to the torch one.
"""

import functools

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 16000  # input rate: the front end and the weights are fixed to it
N_MELS = 80
WIN, HOP = 400, 160  # frame and step in samples, 25 / 10 ms at SR
N_FFT = 512
MEL_LOW_HZ = 20.0  # kaldi's default low cutoff; the high cutoff is nyquist
PREEMPH = 0.97


def _mel(f):
    return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)


@functools.lru_cache(maxsize=1)
def mel_banks():
    """[N_MELS, N_FFT // 2 + 1] float32 kaldi triangular mel filters, MEL_LOW_HZ to nyquist."""
    d = (_mel(SR / 2) - _mel(MEL_LOW_HZ)) / (N_MELS + 1)
    left = _mel(MEL_LOW_HZ) + np.arange(N_MELS)[:, None] * d
    f = _mel(np.arange(N_FFT // 2) * (SR / N_FFT))  # FFT bin centers on the mel scale
    banks = np.maximum(np.minimum((f - left) / d, (left + 2 * d - f) / d), 0)
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)  # kaldi drops the nyquist bin


def fbank(w):
    """w [T] float in [-1, 1] at SR -> [frames, N_MELS] float32 log mel features, mean-normalized over the clip."""
    frames = sliding_window_view(np.asarray(w, dtype=np.float32) * 32768, WIN)[::HOP]  # [F, WIN] on the int16 scale
    frames = frames - frames.mean(-1, keepdims=True)
    frames = frames - PREEMPH * np.concatenate([frames[:, :1], frames[:, :-1]], 1)
    spec = np.abs(np.fft.rfft(frames * np.hamming(WIN), N_FFT)) ** 2
    feat = np.log(np.maximum(spec @ mel_banks().T, np.finfo(np.float32).eps))
    return (feat - feat.mean(0, keepdims=True)).astype(np.float32)
