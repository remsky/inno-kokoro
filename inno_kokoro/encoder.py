"""Distilled speaker encoder: WeSpeaker ResNet34 trained to imitate UniSpeech-SAT-sv.

The network is wespeaker's ResNet34 (models/resnet.py + pooling_layers.TSTP, Apache-2.0)
plus a 256 to 1024 to 512 projection, initialised from the VoxCeleb-trained ResNet34-LM
weights (CC BY 4.0) and fine-tuned so its output matches microsoft/unispeech-sat-base-plus-sv
embeddings (CC BY-SA 3.0 on the teacher; see README licensing). The style head trained on
UniSpeech embeddings is therefore unchanged. Weights ship as the enc.* keys of
model.safetensors (fp16, about 14 MB), so enrollment downloads nothing.

Front end: kaldi fbank on int16-scale samples, 80 mel, 25/10 ms, hamming, no dither,
per-utterance CMN. Vendored in pure torch (verified to 1e-4 against
torchaudio.compliance.kaldi.fbank, which trained the weights) so torchaudio is not a
dependency. Input 16 kHz mono float in [-1, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_MEL_BANKS = None


def _mel(f):
    return 1127.0 * torch.log1p(torch.as_tensor(f, dtype=torch.float64) / 700.0)


def _mel_banks():
    """[80, 257] kaldi triangular mel filters, 20 Hz to nyquist over a 512 FFT at 16 kHz."""
    global _MEL_BANKS
    if _MEL_BANKS is None:
        d = (_mel(8000.0) - _mel(20.0)) / 81  # 80 bins + 1
        left = _mel(20.0) + torch.arange(80)[:, None] * d
        f = _mel(torch.arange(256) * 16000.0 / 512)  # FFT bin centers on the mel scale
        banks = torch.minimum((f - left) / d, (left + 2 * d - f) / d).clamp(min=0)
        _MEL_BANKS = F.pad(banks, (0, 1)).float()  # kaldi drops the nyquist bin
    return _MEL_BANKS


class _Block(nn.Module):
    def __init__(self, cin, c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != c:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, c, 1, stride, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class _ResNet34(nn.Module):
    def __init__(self, m=32, feat_dim=80, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, m, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(m)
        cin = m
        for i, (c, s) in enumerate([(m, 1), (m * 2, 2), (m * 4, 2), (m * 8, 2)]):
            n = [3, 4, 6, 3][i]
            blocks = []
            for st in [s] + [1] * (n - 1):
                blocks.append(_Block(cin, c, st))
                cin = c
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))
        self.seg_1 = nn.Linear(feat_dim // 8 * m * 8 * 2, embed_dim)

    def fbank(self, wav):
        """wav [B,T] float in [-1,1] @16k -> [B,F,80]. Kaldi fbank (int16 scale, snip edges,
        DC removal, 0.97 preemphasis, hamming, power spectrum) plus per-utterance CMN."""
        frames = (wav * 32768).unfold(-1, 400, 160)  # [B, F, 400], 25/10 ms
        frames = frames - frames.mean(-1, keepdim=True)
        frames = frames - 0.97 * F.pad(frames, (1, 0), mode="replicate")[..., :-1]
        win = torch.hamming_window(400, periodic=False, dtype=frames.dtype, device=frames.device)
        spec = torch.fft.rfft(frames * win, 512).abs().pow(2)  # [B, F, 257]
        feat = (spec @ _mel_banks().to(frames).T).clamp(min=torch.finfo(torch.float).eps).log()
        return feat - feat.mean(1, keepdim=True)

    def forward(self, wav):
        x = self.fbank(wav).permute(0, 2, 1)[:, None]  # B,1,F,T
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(4):
            out = getattr(self, f"layer{i + 1}")(out)
        stats = torch.cat([out.mean(-1).flatten(1), torch.sqrt(out.var(-1) + 1e-7).flatten(1)], 1)
        return self.seg_1(stats)


class SpeakerEncoder(nn.Module):
    """wav [B,T] float in [-1,1] at 16 kHz -> unit embedding [B,512] in UniSpeech-SAT-sv space."""

    def __init__(self):
        super().__init__()
        self.backbone = _ResNet34()
        self.proj = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 512))

    def forward(self, wav):
        return F.normalize(self.proj(self.backbone(wav)), dim=-1)
