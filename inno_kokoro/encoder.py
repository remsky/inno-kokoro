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
dependency. fbank.py is the same front end in NumPy, the input of the ONNX export. Input 16 kHz mono float in [-1, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fbank import HOP, N_FFT, N_MELS, PREEMPH, WIN, mel_banks

EMBED_DIM = 512  # output size, UniSpeech-SAT-sv's


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
    def __init__(self, m=32, feat_dim=N_MELS, embed_dim=256):
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
        # a buffer rather than built in fbank(): torch.hamming_window has no ONNX translation
        self.register_buffer("win", torch.hamming_window(WIN, periodic=False), persistent=False)

    def fbank(self, wav):
        """wav [B, T] float in [-1, 1] at SR -> [B, F, N_MELS]. Kaldi fbank (int16 scale, snip edges,
        DC removal, preemphasis, hamming, power spectrum) plus per-utterance CMN."""
        frames = (wav * 32768).unfold(-1, WIN, HOP)  # [B, F, WIN] on the int16 scale
        frames = frames - frames.mean(-1, keepdim=True)
        frames = frames - PREEMPH * F.pad(frames, (1, 0), mode="replicate")[..., :-1]
        spec = torch.fft.rfft(frames * self.win.to(frames), N_FFT).abs().pow(2)  # [B, F, N_FFT // 2 + 1]
        feat = (spec @ torch.from_numpy(mel_banks()).to(frames).T).clamp(min=torch.finfo(torch.float).eps).log()
        return feat - feat.mean(1, keepdim=True)

    def forward(self, wav):
        x = self.fbank(wav).permute(0, 2, 1)[:, None]  # B,1,F,T
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(4):
            out = getattr(self, f"layer{i + 1}")(out)
        stats = torch.cat([out.mean(-1).flatten(1), torch.sqrt(out.var(-1) + 1e-7).flatten(1)], 1)
        return self.seg_1(stats)


class SpeakerEncoder(nn.Module):
    """wav [B, T] float in [-1, 1] at SR -> unit embedding [B, EMBED_DIM] in UniSpeech-SAT-sv space."""

    def __init__(self):
        super().__init__()
        self.backbone = _ResNet34()
        self.proj = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, EMBED_DIM))

    def forward(self, wav):
        return F.normalize(self.proj(self.backbone(wav)), dim=-1)
