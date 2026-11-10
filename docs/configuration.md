# Configuration

## Backends

(new in v0.4.0) `backend` is required on `Tuner`, `fetch_weights` and the command line. 

| backend | install | runs on | downloads | 
|---|---|---|---|
| `"torch"` | `pip install "inno-kokoro[torch]"` | CPU or GPU (`device=`) | `model.safetensors` |
| `"onnx"` | `pip install inno-kokoro` | CPU, onnxruntime | `model.safetensors` + `model.onnx` |

- same pack from either, to within 1e-6
- `enroll()` returns float32 NumPy `[510, 1, 256]`; `return_pt=True` returns a torch tensor (needs `[torch]`)
- a missing file or package raises, no fallback to the other backend
- `backend="onnx"` with a `device` other than `"cpu"` raises

## Weights

- default: downloaded to the huggingface cache on first use, pinned by tag and sha256
- `fetch_weights("/models", backend=...)`: download to a directory of your own, returns the `model.safetensors` path
- `Tuner("/models/model.safetensors", backend=...)`: load from that path, downloads nothing
- `backend="onnx"` reads `model.onnx` from the same directory
- `check_update=True` (default): one background GET of the Hub `config.json` per `Tuner`, with either backend or a path of your own. Logs if newer weights exist, writes nothing
- `Tuner(..., check_update=False)` or `HF_HUB_OFFLINE=1`: skips it. With a path of your own, no network at all

## Pitch tracker: NumPy vs Praat

`[praat]`: optional pitch tracker via praat-parselmouth (GPLv3). Combine: `[torch,praat]`

- Either route will generate the same timing; pitch may differ by <1%
- Praat can be a bit faster overall, though:
    - it has license considerations (GPL)
    - install has to build from source on Linux arm64 w/ the C++ toolchain
- Default install has moved to NumPy to simplify deployment options.

Pitch tracking step timing differences:

| reference | NumPy | Praat |
|---|---|---|
| 8 s, 24 kHz | 82 ms | 31 ms |
| 30 s, 44.1 kHz | 262 ms | 53 ms |
| 32 s, 48 kHz | 221 ms | 69 ms |

CPU (best of 7 runs)
