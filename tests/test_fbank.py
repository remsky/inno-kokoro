"""The NumPy fbank reads the same features as the torch one the weights were trained on."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from inno_kokoro.encoder import _ResNet34
from inno_kokoro.fbank import HOP, N_MELS, SR, WIN, fbank
from inno_kokoro.prosody import resample

FIXTURES = Path(__file__).parent / "fixtures"


def test_fbank_matches_torch():
    w, sr = sf.read(FIXTURES / "libritts_r_1462_170142_000046_000004.flac")
    w = resample(w, sr, SR)
    got = fbank(w)
    want = _ResNet34().fbank(torch.as_tensor(w, dtype=torch.float32)[None])[0].numpy()
    assert got.shape == want.shape == ((len(w) - WIN) // HOP + 1, N_MELS), got.shape
    assert np.abs(got - want).max() < 1e-3, np.abs(got - want).max()
