# Changelog

Notable changes to this project will be documented in this file.

## [v0.4.0] - unreleased: ONNX backend, torch-free install
### Breaking
- `backend` is required on `Tuner`, `fetch_weights` and the command line (`--backend`): `"torch"` or `"onnx"`. Each downloads only its own files
- torch and kokoro moved to the `[torch]` extra: `pip install "inno-kokoro[torch]"`
- `enroll()` returns float32 NumPy unless `return_pt=True`; `read()` returns NumPy

### Added
- `backend="onnx"`: the speaker encoder runs in onnxruntime, no torch needed. For [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) users
- `enroll(..., return_pt=True)`: returns a torch tensor (needs `[torch]`)
- command line: `--backend onnx` writes a `.npy` pack, no test render

### Changed
- weights pinned to the `v0.2.0-onnx` tag: same `model.safetensors`, plus `model.onnx`
- `backend="onnx"` with a `device` other than `"cpu"` raises

See [docs/configuration.md](docs/configuration.md)

## [v0.3.0] - 2026-11-07: NumPy pitch tracker
### Added
- `[praat]` extra: runs the pitch step on praat-parselmouth (GPLv3)
- `Tuner(check_update=False)`: skip the background check for newer weights
- `fetch_weights` verifies a pinned sha256

### Changed
- pitch tracker reimplemented in NumPy, praat-parselmouth no longer in the default install
