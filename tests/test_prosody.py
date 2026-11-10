"""Pitch tracking reads the right octave, and each measurement reads back its synthetic input."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from inno_kokoro import pitch
from inno_kokoro.pitch import track_f0
from inno_kokoro.prosody import SR_ENC, cepstral_f0, head_stats, hz, rate, stats, tilt

SWEEP = range(90, 265, 5)
FIXTURES = Path(__file__).parent / "fixtures"


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
    assert head_stats(torch.zeros(SR_ENC * 4), SR_ENC) is None
    assert stats(torch.zeros(SR_ENC * 4, requires_grad=True), SR_ENC) == (0.0, 0.0, 0.0)  # detached, not refused


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


@pytest.mark.parametrize(
    "f, sr, ks",
    [(220, 16000, [1]), (390, 8000, [1]), (137, 24000, [1]), (360, 16000, range(4, 10))]
    + [(f, 8000, range(4, 10)) for f in (180, 220, 225, 240)],
)
def test_track_f0(f, sr, ks):
    """Reads f as Praat does: sines whose period is not a whole number of samples, and harmonics 4..9 with no
    fundamental, which a parabola on whole lags reads an octave down."""
    t = np.arange(2 * sr) / sr
    f0, _ = track_f0(sum(np.sin(2 * np.pi * f * k * t) / k for k in ks), sr)
    assert (f0 > 0).all(), "tone read as unvoiced"
    assert abs(np.median(f0) - f) < 1e-3 * f, np.median(f0)


def test_track_f0_unvoiced():
    """Noise is not voiced, and neither is a tone below Praat's silence threshold after a full-scale click."""
    t = np.arange(2 * SR_ENC) / SR_ENC
    click = 1e-3 * np.sin(2 * np.pi * 220 * t)
    click[0] = 1.0
    for w in (np.random.default_rng(0).standard_normal(2 * SR_ENC), click):
        assert not track_f0(w, SR_ENC)[0].any()


def test_track_f0_rejects_bad_range():
    for fmin, fmax in [(2000, 4000), (300, 200), (0, 400)]:
        with pytest.raises(ValueError):
            track_f0(np.zeros(SR_ENC), SR_ENC, fmin, fmax)


@pytest.mark.parametrize(
    "praat",
    [False, pytest.param(True, marks=pytest.mark.skipif(pitch.parselmouth is None, reason="no praat extra"))],
)
@pytest.mark.parametrize(
    "utterance, mean, sd",
    [
        ("libritts_r_1462_170142_000046_000004", 13.8587, 3.4486),
        ("libritts_r_1272_128104_000006_000009", 2.7042, 2.8947),
    ],
)
def test_head_stats_matches_praat(utterance, mean, sd, praat, monkeypatch):
    """The head was fitted on Praat's To Pitch (ac), so on real speech the port has to read the same numbers. Pinned
    from parselmouth 0.4.7 on the first 8 s (two chunks, 24 kHz) of two LibriTTS-R dev-clean utterances, CC BY 4.0:
    a high voice (1462, ~223 Hz) and a low one (1272, ~117 Hz), whose candidates sit near the floor's lag where the
    refinement grid and the octave cost matter. The port lands within 3e-4 st; 0.02 st is well inside the 0.1 head-sd
    (0.07 st on the sd) it was gated on. With the praat extra installed, the Praat path is held to the same pins."""
    monkeypatch.setattr(pitch, "PRAAT", praat)
    m, s = head_stats(FIXTURES / f"{utterance}.flac")
    assert abs(m - mean) < 0.02 and abs(s - sd) < 0.02, (m, s)
