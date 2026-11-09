"""Reference clip -> stock-shaped Kokoro voice pack, in one pass with no renders.

inno-kokoro ref.wav name [voices_dir] --backend torch  ->  voices_dir/name.pt ([ROWS, 1, STYLE]) + name_test.wav
The pack can then be used like any plain Kokoro voice: KPipeline(...)(text, voice="voices/name.pt").
--backend onnx needs no torch: the pack is name.npy, float32 for kokoro-onnx, and there is no test render.

Decoder half (timbre): the speaker embedding goes through the style head, along with the reference's spectral tilt
along a learned direction.

Predictor half (prosody): stock packs' predictor rows, blended by nonnegative least squares to the reference's
measured F0 mean, F0 spread, syllable rate. Packs too far from the reference's pitch are excluded and low-grade packs
are weighted to help keep quality up, but can still be chosen if they are the best match.
A prosody head then nudges the blend: a linear map from the reference's F0 mean and spread to a predictor-half
delta, fitted through Kokoro's frozen predictor.

Everything the blending needs (e.g. rows, stats, grades), the head and the torch speaker encoder are baked in
model.safetensors on the Hub; model.onnx beside it is the same encoder for onnxruntime. The backend is always named
and only its files are fetched: Tuner(backend="onnx") pulls both into the huggingface cache, Tuner(backend="torch")
the safetensors alone, fetch_weights("/some/dir", backend=...) into a directory of your own.
"""

import hashlib
import logging
import os
import sys
import threading

import numpy as np
import soundfile as sf
from safetensors import safe_open
from scipy.optimize import nnls

from .fbank import fbank
from .prosody import SR_ENC, ceiling, head_stats, rate, resample, stats, tilt

HUB_REPO = "remsky/kokoro-inno-clone-tuner"
HUB_REVISION = "v0.2.0-onnx"
HUB_SHA256 = {  # at HUB_REVISION
    "model.safetensors": "71cb8e93544f697043197f27fa7f13f0f9f9161076b092aaff6d0894fec2b0e8",
    "model.onnx": "8ae286e103b9ef59eb47ca578f4c82c05b5c48a1ae614832c39c463eb5ccd27d",
}
BACKENDS = {"onnx": ("model.safetensors", "model.onnx"), "torch": ("model.safetensors",)}  # files each reads
log = logging.getLogger(__name__)
SR_KOKORO = 24000
ROWS, STYLE, HALF = 510, 256, 128  # stock pack: one row per phoneme count, decoder half then predictor half
REF_MIN_S, REF_WARN_S = 3, 5  # s of reference: rejected below the first, warned below the second
REF_MAX_S = 30  # only the first 30 s of the reference are embedded
MIN_PACKS = 3  # the pitch gate widens to the nearest packs for references outside the stock range
GRADE_C = 2.0  # pack grades run A = 4.0 to C = 2.0; the grade penalty applies below C
SUM_WEIGHT = 10.0  # weight of the sum-to-1 row in the blend's least squares, against the unit-scaled stat rows
WEIGHT_MIN = 1e-3  # packs below this blend weight are dropped from the result
TILT_SD_MAX = 2.5  # warn past this many sd of reference brightness
TEST_TEXT = (
    "It is a bright, calm morning, and the air is cool and clear. "
    "Down by the river, a few boats drift slowly past the old stone bridge. Some days move quickly, others take their time, "
    "but every one of them ends with the sun going down behind the hills. The quick brown fox jumps over the lazy dog."
)


def _files(backend):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be 'onnx' or 'torch', got {backend!r}")
    return BACKENDS[backend]


def _torch(what):
    """import torch, or an ImportError that names the extra."""
    try:
        import torch
    except ImportError:
        raise ImportError(f"{what} needs torch: pip install 'inno-kokoro[torch]'") from None
    return torch


def fetch_weights(dest=None, *, backend, repo=HUB_REPO, revision=HUB_REVISION, sha256=HUB_SHA256):
    """Hub -> the files `backend` reads, returns the model.safetensors path. "torch": model.safetensors. "onnx": that
    and model.onnx beside it. dest defaults to the huggingface cache; a file already in a directory of your own is
    left alone (delete it to refetch). sha256 maps file name to pin (a plain string pins model.safetensors): a fetched
    file that does not match is removed and raises RuntimeError; None skips the check."""
    if isinstance(sha256, str):
        sha256 = {"model.safetensors": sha256}
    paths = []
    for name in _files(backend):
        if dest and os.path.exists(os.path.join(dest, name)):
            paths.append(os.path.abspath(os.path.join(dest, name)))
            continue
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, name, revision=revision, local_dir=dest)
        want = (sha256 or {}).get(name)
        if want:
            with open(path, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
            if got != want:
                # the blob itself when the cache symlinks, so the next fetch redownloads
                os.remove(os.path.realpath(path))
                raise RuntimeError(f"{path}: sha256 {got}, expected {want}")
        paths.append(path)
    return paths[0]


def _check_update(local_version):
    """Background GET of the Hub config.json: counts usage and logs if newer weights exist.
    Writes nothing to disk, never raises or blocks. Skipped under HF_HUB_OFFLINE or HF_HUB_DISABLE_TELEMETRY."""
    from huggingface_hub import constants

    if constants.HF_HUB_OFFLINE or constants.HF_HUB_DISABLE_TELEMETRY:
        return

    def go():
        try:
            from huggingface_hub import hf_hub_url
            from huggingface_hub.utils import get_session

            latest = get_session().get(hf_hub_url(HUB_REPO, "config.json"), timeout=3).json()["version"]
            if tuple(map(int, latest.split("."))) > tuple(map(int, local_version.split("."))):
                log.warning(
                    "weights %s available (loaded %s): refetch to update",
                    latest,
                    local_version,
                )
        except Exception:
            pass

    threading.Thread(target=go, daemon=True).start()


class Tuner:
    """The style head, tilt direction and baked blend tables read from model.safetensors, plus the speaker encoder.
    embed() returns NumPy with either backend.

    backend, required: "onnx" runs model.onnx beside the weights in onnxruntime, on CPU. "torch" runs the encoder
    from the safetensors on `device` (torch extra). Nothing is picked for you, and a missing file or package raises.
    path defaults to fetching that backend's files into the huggingface cache; a Tuner given a path downloads
    nothing. check_update=False skips the background GET of the Hub config.json that logs when newer weights exist.
    Drop the last reference to free it."""

    def __init__(self, path=None, device="cpu", check_update=True, *, backend):
        _files(backend)
        path = path or fetch_weights(backend=backend)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        graph = os.path.join(os.path.dirname(os.path.abspath(path)), "model.onnx")
        if backend == "onnx" and not os.path.exists(graph):
            raise FileNotFoundError(f"{graph} not found: fetch_weights(dir, backend='onnx') puts it there")
        with safe_open(path, "np") as f:
            meta = f.metadata() or {}
            keys = list(f.keys())
            if not any(k.startswith("enc.") for k in keys):
                raise KeyError(f"{path} has no enc.* speaker-encoder keys; update the weights file")
            self.tilt_dir = f.get_tensor("heads.tilt.weight")[:, 0]  # [STYLE] per tilt sd
            self.rows = f.get_tensor("blend.rows")  # [V, ROWS, HALF] stock predictor halves
            self.stats = f.get_tensor("blend.stats")  # [V, 3]: F0 mean st, F0 sd st, syllables/s
            self.grades = f.get_tensor("blend.grades")  # [V] pack quality, A = 4.0, C = 2.0
            self.head = {
                k.removeprefix("head."): f.get_tensor(k) for k in keys if k.startswith("head.")
            }  # W [3, HALF] (f0 mean, sd, bias), mu, sd
            self.session = None
            if backend == "onnx":
                import onnxruntime as ort

                self.session = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
            else:  # load_state_dict casts the fp16 encoder weights to the fp32 parameters
                torch = _torch("backend='torch'")
                from .encoder import EMBED_DIM, SpeakerEncoder

                self.encoder = SpeakerEncoder()
                self.encoder.load_state_dict(
                    {k.removeprefix("enc."): torch.tensor(f.get_tensor(k)) for k in keys if k.startswith("enc.")}
                )
                self.style = torch.nn.Sequential(
                    torch.nn.Linear(EMBED_DIM, 512), torch.nn.GELU(), torch.nn.Linear(512, STYLE)
                )
                self.style.load_state_dict(
                    {
                        k.removeprefix("heads.style."): torch.tensor(f.get_tensor(k))
                        for k in keys
                        if k.startswith("heads.style.")
                    }
                )
                self.encoder.to(device).eval().requires_grad_(False)
                self.style.to(device).eval().requires_grad_(False)
        self.version = meta.get("version", "0.1.0")
        if check_update:
            _check_update(self.version)
        self.names = meta["blend_names"].split(",")
        self.gate = float(meta["blend_gate"])  # st
        self.scale = np.array([float(v) for v in meta["blend_scale"].split(",")])  # per-stat cost scale
        self.grade_pen = float(meta["grade_pen"])  # cost per unit of weight on a pack below C
        self.tilt_norm = float(meta["tilt_mean"]), float(meta["tilt_sd"])
        self.backend, self.device = backend, device if backend == "torch" else "cpu"

    def embed(self, w):
        """[T] float at SR_ENC -> (unit embedding [512] in the UniSpeech-SAT-sv space, style [STYLE]), float32 NumPy."""
        if self.session:
            e, s = self.session.run(None, {"fbank": fbank(w)[None]})
            return e[0], s[0]
        import torch

        e = self.encoder(torch.as_tensor(np.asarray(w, dtype=np.float32), device=self.device)[None])
        return e[0].cpu().numpy(), self.style(e)[0].cpu().numpy()


def read(path):
    """wav path -> (mono float32 array, sr)."""
    w, sr = sf.read(path, dtype="float32")
    return (w.mean(-1) if w.ndim > 1 else w), sr


def _speaker_embedding(wav, sr, tuner):
    """(unit embedding [512], style [STYLE]) of the first REF_MAX_S s of wav, as float32 NumPy."""
    return tuner.embed(resample(wav, sr, SR_ENC)[: SR_ENC * REF_MAX_S])


def blend_weights(tuner, z):
    """z: (F0 mean st, F0 sd st, syllables/s) of the reference -> {stock pack: weight}, nonnegative, summing to 1.
    Least squares on the scaled stats over the pitch-gated packs, with the grade penalty as extra rows."""
    S = tuner.stats
    d = np.abs(S[:, 0] - z[0])
    ok = d <= tuner.gate
    if ok.sum() < MIN_PACKS:  # outside the stock range: widen the gate until it reaches the nearest packs
        ok = d <= np.sort(d)[MIN_PACKS - 1]
    pen = tuner.grade_pen * np.maximum(0.0, GRADE_C - tuner.grades[ok])
    A = np.vstack([S[ok].T / tuner.scale[:, None], SUM_WEIGHT * np.ones((1, ok.sum())), np.diag(pen)])
    b = np.concatenate([z / tuner.scale, [SUM_WEIGHT], np.zeros(ok.sum())])  # the SUM_WEIGHT row: weights sum to 1
    w, _ = nnls(A, b)
    w = w / w.sum()
    return {str(n): float(v) for n, v in zip(np.array(tuner.names)[ok], w, strict=True) if v > WEIGHT_MIN}


def enroll(wav, sr, tuner, fmax=None, head=True, return_pt=False):
    """wav: [T] float mono at sr, array or tensor. Returns (the [ROWS, 1, STYLE] float32 pack, the blend
    weights). The pack is a NumPy array; return_pt=True gives a CPU torch tensor (torch extra), what kokoro's voice=
    and torch.save take. head=False: plain stock blend (v0.1)."""
    if return_pt:
        torch = _torch("return_pt=True")
    if hasattr(wav, "detach"):  # torch tensor on any device
        wav = wav.detach().cpu()
    wav = np.asarray(wav, dtype=np.float64)
    dur = len(wav) / sr
    if dur < REF_MIN_S:
        raise ValueError(f"reference is {dur:.1f} s; need at least {REF_MIN_S} s to measure prosody")
    if dur < REF_WARN_S:
        log.warning("reference is %.1f s, %d s or more recommended", dur, REF_WARN_S)
    wav = wav[: REF_MAX_S * sr]
    e, s = _speaker_embedding(wav, sr, tuner)
    mean, sd = tuner.tilt_norm
    z_tilt = (tilt(wav, sr) - mean) / sd  # reference brightness in training sd; the shift is a linear extrapolation
    if abs(z_tilt) > TILT_SD_MAX:
        log.warning(
            "reference is unusually %s (%+.1f sd); archival or heavily filtered sources push the timbre past what was trained",
            "dark" if z_tilt < 0 else "bright",
            z_tilt,
        )
    s = s + tuner.tilt_dir * z_tilt
    fmax = ceiling(wav, sr, fmax)  # one tracking ceiling for the blend and the head, or they can read different octaves
    z = np.array([*stats(wav, sr, fmax)[:2], rate(wav, sr)])
    weights = blend_weights(tuner, z)
    pred = sum(v * tuner.rows[tuner.names.index(k)] for k, v in weights.items())  # [ROWS, HALF]
    hs = head_stats(wav, sr, fmax=fmax) if head and tuner.head else None
    if hs is not None:  # nothing voiced: the head would read (0, 0) as an F0 spread far outside its training range
        x = (np.array(hs, dtype=np.float32) - tuner.head["mu"]) / tuner.head["sd"]
        pred = pred + np.append(x, 1) @ tuner.head["W"]  # the same [HALF] delta on every row
    pack = np.concatenate([np.broadcast_to(s[:HALF], (ROWS, HALF)), pred], -1)[:, None].astype(np.float32)
    return (torch.from_numpy(pack) if return_pt else pack), weights


def main():
    usage = (
        "usage: inno-kokoro ref.wav name [voices_dir] --backend onnx|torch [--fmax HZ]\n"
        "       inno-kokoro --fetch [dir] --backend onnx|torch"
    )
    opts = {}  # --fmax: the tracking ceiling is set automatically from the harmonic comb, this overrides it
    argv = sys.argv[1:]
    for flag in ("--backend", "--fmax"):
        if flag in argv[:-1]:
            i = argv.index(flag)
            opts[flag] = argv[i + 1]
            del argv[i : i + 2]
    backend = opts.get("--backend")
    if backend not in BACKENDS:
        sys.exit(usage)
    if argv[:1] == ["--fetch"]:
        print(fetch_weights(argv[1] if len(argv) > 1 else None, backend=backend))
        sys.exit()
    if len(argv) < 2:
        sys.exit(usage)
    fmax = float(opts["--fmax"]) if "--fmax" in opts else None
    ref, name = argv[:2]
    out = argv[2] if len(argv) > 2 else "voices"

    try:
        torch = _torch("--backend torch") if backend == "torch" else None  # onnx: .npy pack, no test render
    except ImportError as e:
        sys.exit(str(e))
    tuner = Tuner(device="cuda" if torch and torch.cuda.is_available() else "cpu", backend=backend)
    wav, sr = read(ref)
    pack, weights = enroll(wav, sr, tuner, fmax, return_pt=bool(torch))

    os.makedirs(out, exist_ok=True)
    pack_path = os.path.join(out, f"{name}.pt" if torch else f"{name}.npy")
    if torch:
        torch.save(pack, pack_path)
    else:
        np.save(pack_path, pack)
    f0 = f"{stats(wav[: REF_MAX_S * sr], sr, fmax)[0]:.1f}"
    try:
        if not torch:
            raise ImportError
        from kokoro import KPipeline
    except ImportError:
        pass
    else:
        test = torch.cat([r.audio for r in KPipeline(lang_code="a")(TEST_TEXT, voice=pack_path)]).cpu()
        sf.write(os.path.join(out, f"{name}_test.wav"), test.numpy(), SR_KOKORO)
        f0 += f" / tuned {stats(test, SR_KOKORO)[0]:.1f}"
    blend = " ".join(f"{k}:{v:.2f}" for k, v in sorted(weights.items(), key=lambda kv: -kv[1]))
    print(f"{pack_path}  f0 mean ref {f0} st  blend {blend}")


if __name__ == "__main__":
    main()
