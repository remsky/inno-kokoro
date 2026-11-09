"""Export the speaker encoder and style head to ONNX, then check each graph against torch in onnxruntime.

    uv run --with onnxruntime --with onnx --with onnxscript python tools/export_onnx.py [out_dir]

Two graphs, batch fixed at 1, both with outputs embedding [1, 512] (unit, UniSpeech-SAT-sv space) and style [1, 256]:
  inno_fbank.onnx  input fbank [1, frames, 80], the features inno_kokoro.fbank computes. Standard opset 20 ops only
                   (Conv, Gemm, Relu, Gelu, reductions, elementwise, shape ops), no DFT: the one to ship.
  inno_wav.onnx    input wav [1, samples] at 16 kHz, the fbank inside the graph as an opset 20 DFT node. onnxruntime
                   runs it on CPU; smaller runtimes often lack DFT. A convenience, nothing depends on it.
Prints the max difference against torch and the timing of both on the test fixtures and a 30 s clip.
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from inno_kokoro.enroll import REF_MAX_S, Tuner, read
from inno_kokoro.prosody import SR_ENC, resample

FIXTURES = Path(__file__).parents[1] / "tests" / "fixtures"


class Graph(torch.nn.Module):
    """x -> (embedding, style). x is wav [B, samples], or fbank features [B, frames, 80] with features=True, which
    swaps the backbone's fbank for the identity."""

    def __init__(self, tuner, features=False):
        super().__init__()
        self.enc, self.style = tuner.encoder, tuner.style
        if features:
            self.enc.backbone.fbank = lambda feat: feat

    def forward(self, x):
        e = self.enc(x)
        return e, self.style(e)


def bench(f, n=5):
    f()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return 1e3 * min(times)


def check(path, g, inputs, name, axis):
    sample = next(iter(inputs.values()))
    torch.onnx.export(
        g,
        (sample,),
        str(path),
        input_names=[name],
        output_names=["embedding", "style"],
        dynamic_shapes={"x": {1: axis}},
        dynamo=True,
        external_data=False,
    )
    print(f"{path}: {path.stat().st_size / 1e6:.1f} MB")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for k, x in inputs.items():
        with torch.no_grad():
            e_t, s_t = (v.numpy() for v in g(x))
        e_o, s_o = sess.run(None, {name: x.numpy()})
        print(
            f"  {k}: {axis} {x.shape[1]}  max|diff| embedding {np.abs(e_t - e_o).max():.1e} style "
            f"{np.abs(s_t - s_o).max():.1e}  torch {bench(lambda: g(x)):.0f} ms  "
            f"onnxruntime {bench(lambda: sess.run(None, {name: x.numpy()})):.0f} ms"
        )


def main(out_dir="dist/onnx"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tuner = Tuner(check_update=False, onnx=False)
    wavs = {}
    for p in sorted(FIXTURES.glob("*.flac")):
        w, sr = read(p)
        w = torch.as_tensor(resample(w, sr, SR_ENC)[: SR_ENC * REF_MAX_S], dtype=torch.float32)
        wavs[p.stem.split("_")[2]] = w[None]
    wavs["30 s"] = torch.cat(list(wavs.values()) * 2, 1)[:, : SR_ENC * REF_MAX_S]
    feats = {k: tuner.encoder.backbone.fbank(w) for k, w in wavs.items()}
    check(out / "inno_wav.onnx", Graph(tuner).eval(), wavs, "wav", "samples")
    check(out / "inno_fbank.onnx", Graph(tuner, features=True).eval(), feats, "fbank", "frames")


if __name__ == "__main__":
    main(*sys.argv[1:])
