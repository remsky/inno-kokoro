![Inno Clone Tuner](https://raw.githubusercontent.com/remsky/inno-kokoro/main/docs/banner.png)

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://pypi.org/project/inno-kokoro/)
[![codecov](https://codecov.io/gh/remsky/inno-kokoro/graph/badge.svg)](https://codecov.io/gh/remsky/inno-kokoro)
[![CI](https://github.com/remsky/inno-kokoro/actions/workflows/ci.yml/badge.svg)](https://github.com/remsky/inno-kokoro/actions/workflows/ci.yml)


[![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-purple)](https://huggingface.co/spaces/Remsky/FastKoko) 
[![Weights](https://img.shields.io/badge/Weights-v0.2.0-purple)](https://huggingface.co/remsky/kokoro-inno-clone-tuner)

[![PyPI](https://img.shields.io/pypi/v/inno-kokoro)](https://pypi.org/project/inno-kokoro/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/remsky/inno-kokoro/blob/main/LICENSE)


Zero-shot voice tuner for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). 

A 5-30 second reference clip in, a stock-shaped `[510, 1, 256]` voice pack out in about 0.3 seconds.

## Usage
```bash
pip install inno-kokoro
inno-kokoro --fetch /models # -> /models/model.safetensors, e.g. in a Dockerfile; skips if present
inno-kokoro my_ref.wav am_me # -> voices/am_me.pt + voices/am_me_test.wav
inno-kokoro my_ref.wav am_me --fmax 300 # override pitch ceiling
```

Currently only available for English; prefixes work like the stock packs e.g: `af_`, `am_`, `bf_`, `bm_`.

```python
from inno_kokoro.enroll import Tuner, enroll, read
from kokoro import KPipeline

tuner = Tuner()  # huggingface cache; or Tuner("/models/model.safetensors") for a path of your own
pack, _ = enroll(*read("my_ref.wav"), tuner)
pipe = KPipeline(lang_code="a")
wav = next(pipe("Hello from a tuned voice.", voice=pack)).audio
```

---

For best results, the reference audio should be:
- 3-second minimum, up to the first 30-seconds
- Single speaker (english). 
- Reasonably clear of audio artifacts

Voice pack generation time:
- about 0.05 s per second of reference on CPU
- 0.1 to 0.3 s total on a GPU (after the model is loaded).


## Identity benchmarking

LibriSpeech test-clean, F5-TTS cross-sentence split: 1127 utterances, 39 held-out speakers. 

Reference in, new sentence out, scored against the speaker's real recording. Normalized scores compare the render between sounding like a stranger (0) and a second sample of the same benchmark speaker (1). 

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

Full model card available on HuggingFace: [remsky/kokoro-inno-clone-tuner](https://huggingface.co/remsky/kokoro-inno-clone-tuner). 

**Only clone voices you have permission to clone, even the shallow cloning provided by this model and technique.** 
