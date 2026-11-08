"""Reference clip -> stock-shaped Kokoro voice pack, in one pass with no renders.

inno-kokoro ref.wav name [voices_dir]  ->  voices_dir/name.pt ([ROWS, 1, STYLE]) + name_test.wav
The pack can then be used like any plain Kokoro voice: KPipeline(...)(text, voice="voices/name.pt")

Decoder half (timbre): the speaker embedding goes through the style head, along with the reference's spectral tilt
along a learned direction.

Predictor half (prosody): stock packs' predictor rows, blended by nonnegative least squares to the reference's
measured F0 mean, F0 spread, syllable rate. Packs too far from the reference's pitch are excluded and low-grade packs
are weighted to help keep quality up, but can still be chosen if they are the best match.
A prosody head then nudges the blend: a linear map from the reference's F0 mean and spread to a predictor-half
delta, fitted through Kokoro's frozen predictor.

Everything the blending needs (e.g. rows, stats, grades) and the head are baked in model.safetensors on the Hub:
Tuner() pulls it into the huggingface cache, Tuner(fetch_weights("/some/dir")) into a directory of your own.
"""

import hashlib
import logging
import os
import sys
import threading

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from scipy.optimize import nnls

from .encoder import EMBED_DIM, SpeakerEncoder
from .fbank import fbank
from .prosody import SR_ENC, ceiling, head_stats, rate, resample, stats, tilt

HUB_REPO = "remsky/kokoro-inno-clone-tuner"
HUB_REVISION = "v0.2.0"
HUB_SHA256 = "71cb8e93544f697043197f27fa7f13f0f9f9161076b092aaff6d0894fec2b0e8"  # model.safetensors at HUB_REVISION
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


def fetch_weights(dest=None, repo=HUB_REPO, revision=HUB_REVISION, sha256=HUB_SHA256):
    """Hub -> dest/model.safetensors, returns the path. dest defaults to the huggingface cache; a directory of your own
    is left alone if it already has the file (delete it to refetch). A fetched file whose sha256 is not `sha256` is
    removed and raises RuntimeError; sha256=None skips the check."""
    if dest and os.path.exists(os.path.join(dest, "model.safetensors")):
        return os.path.abspath(os.path.join(dest, "model.safetensors"))
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, "model.safetensors", revision=revision, local_dir=dest)
    if sha256:
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != sha256:
            os.remove(os.path.realpath(path))  # the blob itself when the cache symlinks, so the next fetch redownloads
            raise RuntimeError(f"{path}: sha256 {got}, expected {sha256}")
    return path


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
    """The style head, tilt direction and baked blend tables read from model.safetensors, plus the speaker encoder:
    the torch model from the same file, or, when `onnx` is the path of the graph tools/export_onnx.py writes, that
    graph in onnxruntime. embed() returns NumPy either way.

    path defaults to fetching the weights into the huggingface cache. check_update=False skips the background GET
    of the Hub config.json that logs when newer weights exist. Drop the last reference to free it."""

    def __init__(self, path=None, device="cpu", check_update=True, onnx=None):
        path = path or fetch_weights()
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
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
            if onnx:
                import onnxruntime as ort

                self.session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
            else:  # load_state_dict casts the fp16 encoder weights to the fp32 parameters
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
        self.device = device

    def embed(self, w):
        """[T] float at SR_ENC -> (unit embedding [512] in the UniSpeech-SAT-sv space, style [STYLE]), float32 NumPy."""
        if self.session:
            e, s = self.session.run(None, {"fbank": fbank(w)[None]})
            return e[0], s[0]
        e = self.encoder(torch.as_tensor(np.asarray(w, dtype=np.float32), device=self.device)[None])
        return e[0].cpu().numpy(), self.style(e)[0].cpu().numpy()


def read(path):
    """wav path -> (mono float tensor, sr)."""
    w, sr = sf.read(path)
    w = torch.tensor(w, dtype=torch.float32)
    if w.ndim > 1:
        w = w.mean(-1)
    return w, sr


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


def enroll(wav, sr, tuner, fmax=None, head=True):
    """wav: [T] float mono at sr, array or tensor. Returns (the [ROWS, 1, STYLE] float32 NumPy pack, the blend
    weights). head=False: plain stock blend (v0.1)."""
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
    pack = np.concatenate([np.broadcast_to(s[:HALF], (ROWS, HALF)), pred], -1)[:, None]
    return pack.astype(np.float32), weights


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--fetch":
        print(fetch_weights(sys.argv[2] if len(sys.argv) > 2 else None))
        sys.exit()
    fmax = None  # the tracking ceiling is set automatically from the harmonic comb; --fmax overrides it
    if "--fmax" in sys.argv:
        i = sys.argv.index("--fmax")
        fmax = float(sys.argv[i + 1])
        del sys.argv[i : i + 2]
    if len(sys.argv) < 3:
        sys.exit("usage: inno-kokoro ref.wav name [voices_dir] [--fmax HZ]   (or --fetch [dir])")
    ref, name = sys.argv[1:3]
    out = sys.argv[3] if len(sys.argv) > 3 else "voices"

    tuner = Tuner(device="cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = read(ref)
    pack, weights = enroll(wav, sr, tuner, fmax)

    os.makedirs(out, exist_ok=True)
    pack_path = os.path.join(out, f"{name}.pt")
    torch.save(torch.from_numpy(pack), pack_path)
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a")
    test = torch.cat([r.audio for r in pipe(TEST_TEXT, voice=pack_path)])
    sf.write(os.path.join(out, f"{name}_test.wav"), test.cpu().numpy(), SR_KOKORO)
    print(
        f"{pack_path}  f0 mean ref {stats(wav[: REF_MAX_S * sr], sr, fmax)[0]:.1f} / tuned {stats(test.cpu(), SR_KOKORO)[0]:.1f} st  blend "
        + " ".join(f"{k}:{v:.2f}" for k, v in sorted(weights.items(), key=lambda kv: -kv[1]))
    )


if __name__ == "__main__":
    main()
