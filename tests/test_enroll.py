"""The baked weights load, and a synthetic reference makes it through enroll() and the CLI."""

import sys

import pytest
import soundfile as sf
import torch

import inno_kokoro.enroll as enroll


def reference(seconds=12, f0=130, sr=enroll.SR_ENC):
    """Harmonic tone under a slow amplitude sweep, with enough partials to fill the tilt band."""
    t = torch.arange(sr * seconds) / sr
    partials = sum(torch.sin(2 * torch.pi * f0 * k * t) / k for k in range(1, 30))
    return partials * (0.6 + 0.4 * torch.cos(2 * torch.pi * 3 * t))


@pytest.fixture(scope="session")
def tuner():
    return enroll.Tuner()


def test_encoder_baked(tuner):
    """The baked encoder yields a unit embedding in the UniSpeech-SAT-sv space."""
    torch.manual_seed(0)
    e = enroll._speaker_embedding(torch.randn(enroll.SR_ENC * 4) * 0.1, enroll.SR_ENC, tuner)
    assert e.shape == (1, 512), e.shape
    assert abs(e.norm().item() - 1) < 1e-4 and torch.isfinite(e).all()


def test_enroll(tuner):
    """Reference in, stock-shaped pack out, over a blend that is a real convex combination."""
    pack, weights = enroll.enroll(reference(), enroll.SR_ENC, tuner)
    assert pack.shape == (enroll.ROWS, 1, enroll.STYLE), pack.shape
    assert torch.isfinite(pack).all()
    assert abs(sum(weights.values()) - 1) < 1e-3, weights
    assert all(type(k) is str and v > 0 for k, v in weights.items()), weights


def test_enroll_rejects_short_reference(tuner):
    with pytest.raises(ValueError, match="at least 3 s"):
        enroll.enroll(reference(seconds=2), enroll.SR_ENC, tuner)


def test_read_wav(tmp_path):
    """read() downmixes to mono float, which is what the README example feeds enroll()."""
    p = tmp_path / "ref.wav"
    stereo = torch.stack([torch.full((enroll.SR_KOKORO,), 0.8), torch.zeros(enroll.SR_KOKORO)], -1)
    sf.write(p, stereo.numpy(), enroll.SR_KOKORO)
    w, sr = enroll.read(str(p))
    assert (sr, w.shape, w.dtype) == (enroll.SR_KOKORO, (enroll.SR_KOKORO,), torch.float32)
    assert abs(w.mean().item() - 0.4) < 1e-4, w.mean()


def test_cli(tmp_path, monkeypatch, capsys):
    """The whole shipped command: reference in, pack and a real Kokoro render of it out."""
    ref = tmp_path / "ref.wav"
    sf.write(ref, reference().numpy(), enroll.SR_ENC)
    out = tmp_path / "voices"
    monkeypatch.setattr(sys, "argv", ["inno-kokoro", str(ref), "am_test", str(out)])
    enroll.main()

    pack = torch.load(out / "am_test.pt", weights_only=True)
    assert pack.shape == (enroll.ROWS, 1, enroll.STYLE), pack.shape
    assert torch.isfinite(pack).all()
    wav, sr = sf.read(out / "am_test_test.wav")
    assert sr == enroll.SR_KOKORO and len(wav) > sr, (sr, len(wav))
    assert "blend" in capsys.readouterr().out


def test_cli_fetch(tmp_path, monkeypatch, capsys):
    """--fetch prints the weights path and is a no-op the second time, as a Dockerfile layer needs."""
    monkeypatch.setattr(sys, "argv", ["inno-kokoro", "--fetch", str(tmp_path)])
    with pytest.raises(SystemExit):
        enroll.main()
    first = capsys.readouterr().out.strip()
    assert first.endswith("model.safetensors")
    with pytest.raises(SystemExit):
        enroll.main()
    assert capsys.readouterr().out.strip() == first
