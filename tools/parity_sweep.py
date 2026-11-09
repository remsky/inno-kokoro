"""Torch against onnxruntime: the speaker encoder and the style head on a set of audio directories.

    uv run python tools/parity_sweep.py dist/onnx/inno_fbank.onnx DIR [DIR ...]

Each DIR is a directory or a quoted glob pattern; wav and flac files under it are one set. PER_DIR clips of 3 s or more
are sampled from each set, and Tuner.embed() from both backends runs on each. EXTREMES clips per set also run as a 3 s
cut and as a 30 s loop, the shortest and the longest input enrollment accepts. Two clips per set go through the full
enroll() with both backends. Prints a markdown table of the differences.
"""

import glob
import random
import sys
from pathlib import Path

import numpy as np

from inno_kokoro import enroll
from inno_kokoro.prosody import SR_ENC, resample

PER_DIR = 40
EXTREMES = 10


def clips(pattern):
    files = sorted(p for d in glob.glob(pattern) for p in Path(d).rglob("*") if p.suffix in (".wav", ".flac"))
    random.shuffle(files)
    out = []
    for f in files:
        w, sr = enroll.read(f)
        w16 = resample(w, sr, SR_ENC).astype(np.float32)
        if len(w16) >= 3 * SR_ENC:
            out.append((f, w16))
        if len(out) == PER_DIR:
            break
    return out


def main(graph, patterns):
    random.seed(0)
    torch_t, ort_t = enroll.Tuner(check_update=False, onnx=False), enroll.Tuner(onnx=graph, check_update=False)

    def diff(w16):
        (e_t, s_t), (e_o, s_o) = torch_t.embed(w16), ort_t.embed(w16)
        return np.abs(e_t - e_o).max(), np.abs(s_t - s_o).max(), float(e_t @ e_o)

    print("| set | clips | max embedding diff | max style diff | min cosine |\n|---|---|---|---|---|")
    rows, packs = [], []
    for pattern in patterns:
        cs = clips(pattern)
        r = [diff(w[: 30 * SR_ENC]) for _, w in cs]
        for _, w in cs[:EXTREMES]:
            r.append(diff(w[: 3 * SR_ENC]))
            r.append(diff(np.tile(w, 30 * SR_ENC // len(w) + 1)[: 30 * SR_ENC]))
        rows += r
        name = "/".join(Path(pattern).parts[-2:])
        print(
            f"| {name} | {len(r)} | {max(x[0] for x in r):.1e} | {max(x[1] for x in r):.1e} | {min(x[2] for x in r):.7f} |"
        )
        for f, _ in cs[:2]:
            w, sr = enroll.read(f)
            (p_t, w_t), (p_o, w_o) = enroll.enroll(w, sr, torch_t), enroll.enroll(w, sr, ort_t)
            assert w_t == w_o, (f, w_t, w_o)
            packs.append(np.abs(p_t - p_o).max())
    print(
        f"| all | {len(rows)} | {max(x[0] for x in rows):.1e} | {max(x[1] for x in rows):.1e} | {min(x[2] for x in rows):.7f} |"
    )
    print(f"\nfull enroll() packs, 2 clips per set: max diff {max(packs):.1e}, blend weights identical")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
