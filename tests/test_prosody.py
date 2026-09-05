"""Pitch tracking reads the right octave, and each measurement reads back its synthetic input."""

import math

import pytest
import torch

from inno_kokoro.prosody import SR_ENC, cepstral_f0, head_stats, hz, rate, stats, tilt

SWEEP = range(90, 265, 5)


def tone(f0, ks=range(1, 6), seconds=2):
    """Harmonic tone at f0, partials ks, 1/k amplitudes."""
    t = torch.arange(SR_ENC * seconds) / SR_ENC
    return sum(torch.sin(2 * torch.pi * f0 * k * t) / k for k in ks)


def test_f0_octave():
    """A 220 Hz tone sits at 13.7 st, and neither measurement may drop it an octave."""
    mean, std, _ = stats(tone(220), SR_ENC)
    assert abs(mean - 13.7) < 0.5 and std < 0.5, (mean, std)
    hm, hs = head_stats(tone(220), SR_ENC)
    assert abs(hm - 13.7) < 0.5 and hs < 0.5, (hm, hs)


def test_missing_fundamental():
    """Harmonics 2..6 of 110 Hz: the lag tracker alone reads 220, the cepstral ceiling pulls it back."""
    mean = stats(tone(110, ks=range(2, 7)), SR_ENC)[0]
    assert abs(mean - 1.65) < 0.7, mean


@pytest.mark.parametrize("f", SWEEP)
def test_f0_sweep(f):
    """No frequency in the speech range may be read a whole semitone off, at either layer.

    The cepstral peak can land on a rahmonic at f0/k, which the tracker often survives because a
    ceiling set too high is recoverable, so the ceiling is asserted separately from the tracked F0."""
    w = tone(f)
    ceps = cepstral_f0(w)[0]
    assert abs(12 * math.log2(ceps / f)) < 1.5, (f, ceps)
    got = hz(torch.tensor(stats(w, SR_ENC)[0])).item()
    assert abs(12 * math.log2(got / f)) < 1, (f, got)


def test_silence():
    """Nothing voiced, nothing to report, and no division by zero on the way there."""
    assert stats(torch.zeros(SR_ENC * 4), SR_ENC) == (0.0, 0.0, 0.0)
    assert cepstral_f0(torch.zeros(SR_ENC * 4)) == (0.0, 0.0)


def test_rate():
    """Noise amplitude-modulated at 5 Hz reads as 5 syllables per second."""
    torch.manual_seed(0)
    t = torch.arange(SR_ENC * 4) / SR_ENC
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * torch.cos(2 * torch.pi * 5 * t))
    r = rate(envelope * torch.randn(len(t)), SR_ENC)
    assert abs(r - 5) < 0.7, r


def test_tilt():
    """White noise is flat, pink noise (1/f power) falls 3 dB per octave."""
    torch.manual_seed(0)
    white = torch.randn(SR_ENC * 4)
    pink = torch.fft.irfft(torch.fft.rfft(white) / torch.arange(1, SR_ENC * 2 + 2).sqrt())
    tw, tp = tilt(white, SR_ENC), tilt(pink, SR_ENC)
    assert abs(tw) < 0.3, tw
    assert abs(tp + 3) < 0.3, tp
