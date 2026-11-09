![Inno Clone Tuner](https://raw.githubusercontent.com/remsky/inno-kokoro/main/docs/banner.png)

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://pypi.org/project/inno-kokoro/)
[![codecov](https://codecov.io/gh/remsky/inno-kokoro/graph/badge.svg)](https://codecov.io/gh/remsky/inno-kokoro)
[![CI](https://github.com/remsky/inno-kokoro/actions/workflows/ci.yml/badge.svg)](https://github.com/remsky/inno-kokoro/actions/workflows/ci.yml)


[![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-purple)](https://huggingface.co/spaces/remsky/kokoro-clone-tune-generator) 
[![Weights](https://img.shields.io/badge/Weights-v0.2.0-purple)](https://huggingface.co/remsky/kokoro-inno-clone-tuner)

[![PyPI](https://img.shields.io/pypi/v/inno-kokoro)](https://pypi.org/project/inno-kokoro/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/remsky/inno-kokoro/blob/main/LICENSE)


Zero-shot voice tuner for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

Generate a custom voicepack in under a second from a 5-30 second reference clip. 
- stock-compatble `[510, 1, 256]`
- [write-up](https://remsky.art/writeups/kokoro-fastapi/inno-clone) of the development process
- captures about 1/3 of identity while maintaining high quality 


## Usage

```bash
pip install inno-kokoro # onnxruntime encoder, NumPy pitch tracker
pip install "inno-kokoro[torch]" # torch encoder + kokoro; the example below needs it
pip install "inno-kokoro[praat]" # optional: pitch tracker via praat-parselmouth (GPLv3)
```

```python
import torch
from inno_kokoro.enroll import Tuner, enroll, read
from kokoro import KPipeline

tuner = Tuner() # downloads the weights to the huggingface cache on first use
pack, _ = enroll(*read("my_ref.wav"), tuner) # [510, 1, 256] float32 NumPy

pipe = KPipeline(lang_code="a")
wav = next(pipe("Hello from a tuned voice.", voice=torch.from_numpy(pack))).audio
```

`Tuner("/models/model.safetensors")` loads from a path of your own instead of the cache.

Currently only available for English; prefixes work like the stock packs e.g: `af_`, `am_`, `bf_`, `bm_`.

<details>
<summary>Command line</summary>

Write a voice pack plus a test render to disk:

```bash
inno-kokoro my_ref.wav am_me # -> voices/am_me.pt + voices/am_me_test.wav
inno-kokoro my_ref.wav am_me packs/ # write somewhere else
inno-kokoro my_ref.wav am_me --fmax 300 # override the pitch ceiling (set automatically otherwise)
```

Without `[torch]`: `am_me.npy`, no test render.

Weights download on first use. To pre-fetch them instead (Dockerfile layer, offline machine):

```bash
inno-kokoro --fetch /models # -> /models/model.safetensors
```

</details>

Integrated into [remsky/Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) (v0.9.0+) with `ENABLE_INNO_TUNER=true`

---

For best results, the reference audio should be:
- 3-second minimum, up to the first 30-seconds
- Single speaker (english). 
- Reasonably clear of audio artifacts

Voice pack generation time:
- about 0.05 s per second of reference on CPU
- 0.1 to 0.5 s total on a GPU (after the model is loaded), 0.3 with `[praat]`

<details>
<summary>(v0.3.0) Pitch tracking step, NumPy vs Praat</summary>

Functionally: 
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

</details>


## Identity benchmarking

LibriSpeech test-clean, F5-TTS cross-sentence split: 1127 utterances, 39 held-out speakers. 

Scored against the speaker's real recording. Normalized scores compare the render between sounding like a stranger (0) and a second sample of the same benchmark speaker (1). 

RTF on an RTX 4060 Ti.

| system | SIM-o | normalized | UTMOS | RTF |
|---|---|---|---|---|
| ground truth (second recording) | 0.695 | 1.00 | 4.10 | |
| F5-TTS v1 base | 0.650 | 0.94 | 3.86 | 0.48 |
| StyleTTS2 (LibriTTS zero-shot) | 0.386 | 0.46 | 4.40 | 0.06 |
| **Inno v0.2** | 0.288 | 0.32 | 4.45 | 0.07 |
| OpenVoice v2 | 0.227 | 0.23 | 3.80 | 0.12 |
| Kokoro, nearest stock pack | 0.167 | 0.15 | 4.25 | 0.06 |

---

UTMOS scoring stays high which reflects the priority placed on maintaining Kokoro's voice quality. 

Inno can match about a third of the way to most identities, and avoids copying recording artifacts etc. 

## License and Attributions

Apache-2.0. The speaker encoder is CC BY-SA 3.0. 
- Pitch tracker: Boersma (1993), "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound", *Proc. Institute of Phonetic Sciences* 17, reimplemented in NumPy. The prosody head was fitted on pitch measured with [Praat](https://www.praat.org) (Boersma & Weenink). No Praat code is included in default install.
- `[praat]` extra: praat-parselmouth, GPLv3.
- Test fixtures: first 8 s of two LibriTTS-R utterances (Koizumi et al., 2023), CC BY 4.0.

Full model card available on HuggingFace: [remsky/kokoro-inno-clone-tuner](https://huggingface.co/remsky/kokoro-inno-clone-tuner). 

**Only clone voices you have permission to clone, even the shallow cloning provided by this model and technique.** 
