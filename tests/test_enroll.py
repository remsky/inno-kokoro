"""The baked weights load, and a synthetic reference makes it through enroll() and the CLI."""

import sys

import numpy as np
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
    return enroll.Tuner(backend="torch")


def test_encoder_baked(tuner):
    """The baked encoder yields a unit embedding in the UniSpeech-SAT-sv space."""
    torch.manual_seed(0)
    e, s = enroll._speaker_embedding(torch.randn(enroll.SR_ENC * 4) * 0.1, enroll.SR_ENC, tuner)
    assert (e.shape, s.shape) == ((512,), (enroll.STYLE,)), (e.shape, s.shape)
    assert abs(np.linalg.norm(e) - 1) < 1e-4 and np.isfinite(e).all() and np.isfinite(s).all()


def test_enroll(tuner):
    """Reference in, stock-shaped pack out, over a blend that is a real convex combination."""
    pack, weights = enroll.enroll(reference(), enroll.SR_ENC, tuner)
    assert (pack.shape, pack.dtype) == ((enroll.ROWS, 1, enroll.STYLE), np.float32), (pack.shape, pack.dtype)
    assert np.isfinite(pack).all()
    assert torch.equal(enroll.enroll(reference(), enroll.SR_ENC, tuner, return_pt=True)[0], torch.from_numpy(pack))
    assert abs(sum(weights.values()) - 1) < 1e-3, weights
    assert all(type(k) is str and v > 0 for k, v in weights.items()), weights


def test_enroll_unvoiced(tuner):
    """Nothing voiced: the head is skipped rather than fed an F0 spread far outside its training range."""
    torch.manual_seed(0)
    noise = torch.randn(enroll.SR_ENC * 6) * 0.1
    with_head = enroll.enroll(noise, enroll.SR_ENC, tuner)[0]
    assert np.array_equal(with_head, enroll.enroll(noise, enroll.SR_ENC, tuner, head=False)[0])


def test_onnx_backend(tuner):
    """model.onnx in onnxruntime gives the torch backend's pack."""
    ort = enroll.Tuner(check_update=False, backend="onnx")
    assert ort.session and not tuner.session
    want, w_want = enroll.enroll(reference(), enroll.SR_ENC, tuner)
    got, w_got = enroll.enroll(reference(), enroll.SR_ENC, ort)
    assert w_got == w_want and np.abs(got - want).max() < 1e-4, np.abs(got - want).max()


def test_enroll_rejects_short_reference(tuner):
    with pytest.raises(ValueError, match="at least 3 s"):
        enroll.enroll(reference(seconds=2), enroll.SR_ENC, tuner)


def test_backend_required():
    """No default backend: nothing is picked, or fetched, for the caller."""
    with pytest.raises(TypeError, match="backend"):
        enroll.Tuner()
    with pytest.raises(ValueError, match="'onnx' or 'torch'"):
        enroll.Tuner(backend="auto")


def test_return_pt_without_torch(tuner, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError from here on
    with pytest.raises(ImportError, match=r"inno-kokoro\[torch\]"):
        enroll.enroll(reference(), enroll.SR_ENC, tuner, return_pt=True)


def test_read_wav(tmp_path):
    """read() downmixes to mono float32 NumPy, which is what the README example feeds enroll()."""
    p = tmp_path / "ref.wav"
    stereo = np.stack([np.full(enroll.SR_KOKORO, 0.8), np.zeros(enroll.SR_KOKORO)], -1)
    sf.write(p, stereo, enroll.SR_KOKORO)
    w, sr = enroll.read(str(p))
    assert (sr, w.shape, w.dtype) == (enroll.SR_KOKORO, (enroll.SR_KOKORO,), np.float32)
    assert abs(w.mean() - 0.4) < 1e-4, w.mean()


def test_cli(tmp_path, monkeypatch, capsys):
    """The whole shipped command: reference in, pack and a real Kokoro render of it out."""
    ref = tmp_path / "ref.wav"
    sf.write(ref, reference().numpy(), enroll.SR_ENC)
    out = tmp_path / "voices"
    monkeypatch.setattr(sys, "argv", ["inno-kokoro", str(ref), "am_test", str(out), "--backend", "torch"])
    enroll.main()

    pack = torch.load(out / "am_test.pt", weights_only=True)
    assert pack.shape == (enroll.ROWS, 1, enroll.STYLE), pack.shape
    assert torch.isfinite(pack).all()
    wav, sr = sf.read(out / "am_test_test.wav")
    assert sr == enroll.SR_KOKORO and len(wav) > sr, (sr, len(wav))
    assert "blend" in capsys.readouterr().out


def test_cli_without_torch(tmp_path, monkeypatch, capsys):
    """Base install: the onnxruntime encoder, a .npy pack and no test render."""
    ref = tmp_path / "ref.wav"
    sf.write(ref, reference().numpy(), enroll.SR_ENC)
    out = tmp_path / "voices"
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError from here on
    monkeypatch.setitem(sys.modules, "kokoro", None)
    monkeypatch.setattr(sys, "argv", ["inno-kokoro", str(ref), "am_test", str(out), "--backend", "onnx"])
    enroll.main()

    pack = np.load(out / "am_test.npy")
    assert (pack.shape, pack.dtype) == ((enroll.ROWS, 1, enroll.STYLE), np.float32), (pack.shape, pack.dtype)
    assert not (out / "am_test_test.wav").exists() and "blend" in capsys.readouterr().out


def test_cli_fetch(tmp_path, monkeypatch, capsys):
    """--fetch prints the weights path and is a no-op the second time, as a Dockerfile layer needs."""
    monkeypatch.setattr(sys, "argv", ["inno-kokoro", "--fetch", str(tmp_path), "--backend", "torch"])
    with pytest.raises(SystemExit):
        enroll.main()
    first = capsys.readouterr().out.strip()
    assert first.endswith("model.safetensors")
    with pytest.raises(SystemExit):
        enroll.main()
    assert capsys.readouterr().out.strip() == first
    assert not (tmp_path / "model.onnx").exists()  # a backend fetches its own files only


def test_fetch_weights_checksum(tmp_path):
    """A fetched file that does not match the pinned sha256 is removed, not left for the next start to load."""
    with pytest.raises(RuntimeError, match="sha256"):
        enroll.fetch_weights(str(tmp_path), backend="torch", sha256="0" * 64)
    assert not (tmp_path / "model.safetensors").exists()
