"""Export the ECAPA-TDNN speaker encoder (mel -> 1024-d x-vector) to ncnn.

Trace-time rewrites (all verified equal to the original forward):
- Res2NetBlock.forward: torch.chunk -> static channel slices (chunk becomes a
  parameterless ncnn Slice via pnnx = null-deref, pitfall P5)
- AttentiveStatisticsPooling.forward: the length mask is all-ones for our
  single-utterance use, so drop _length_to_mask/.item()/masked_fill entirely;
  replace repeat(1,1,T) (shape-frozen under trace) with broadcast-add.
"""
import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import common
from common import S

OUT = os.path.join(common.WORK_DIR, "ncnn_models")
cfg_all = common.load_config()

torch.manual_seed(0)
spk_cfg = S.configuration.Qwen3TTSSpeakerEncoderConfig(**cfg_all["speaker_encoder_config"])
model = S.modeling.Qwen3TTSSpeakerEncoder(spk_cfg)
model.eval()
model.load_state_dict(common.load_weights("speaker_encoder."), strict=True)

# reference BEFORE patching
T = 300
mels = torch.randn(1, T, 128)
with torch.no_grad():
    ref = model(mels)


def res2net_forward(self, x):
    ch = x.shape[1] // self.scale  # python int at trace; channels are static
    outputs = []
    output_part = None
    for i in range(self.scale):
        part = x[:, i * ch:(i + 1) * ch]
        if i == 0:
            output_part = part
        elif i == 1:
            output_part = self.blocks[i - 1](part)
        else:
            output_part = self.blocks[i - 1](part + output_part)
        outputs.append(output_part)
    return torch.cat(outputs, dim=1)


def asp_forward(self, x):
    # all-ones mask: total = T, statistics are plain means over time
    mean = x.mean(dim=2)
    std = torch.sqrt((x - mean.unsqueeze(2)).pow(2).mean(dim=2).clamp(self.eps))
    zeros = x * 0.0
    attention = torch.cat([x, zeros + mean.unsqueeze(2), zeros + std.unsqueeze(2)], dim=1)
    attention = self.conv(self.tanh(self.tdnn(attention)))
    attention = F.softmax(attention, dim=2)
    mean2 = (attention * x).sum(2)
    std2 = torch.sqrt((attention * (x - mean2.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
    return torch.cat((mean2, std2), dim=1).unsqueeze(2)


S.modeling.Res2NetBlock.forward = res2net_forward
S.modeling.AttentiveStatisticsPooling.forward = asp_forward

with torch.no_grad():
    patched = model(mels)
d = (patched - ref).abs().max().item()
print("patched vs original:", d)
assert d < 1e-5

traced = torch.jit.trace(model, (mels,))
with torch.no_grad():
    # dynamic-length check in torch before pnnx
    T2 = 173
    m2 = torch.randn(1, T2, 128)
    d2 = (traced(m2) - model(m2)).abs().max().item()
print("traced dynamic check (T=173):", d2)
assert d2 < 1e-5, "trace froze the length"

pt = os.path.join(OUT, "speaker_encoder.pt")
traced.save(pt)
r = subprocess.run(["pnnx", pt, f"inputshape=[1,{T},128]", "inputshape2=[1,173,128]", "fp16=0"],
                   cwd=OUT, capture_output=True, text=True)
print("\n".join(r.stdout.splitlines()[-3:]))
assert r.returncode == 0, r.stderr[-2000:]

param = os.path.join(OUT, "speaker_encoder.ncnn.param")
bad = [l.split()[0] for l in open(param) if l.strip() and "." in l.split()[0]]
print("unconvertible:", set(bad) if bad else "none")

# ---- ncnn parity ----
import numpy as np
import ncnn

net = ncnn.Net()
assert net.load_param(param) == 0
assert net.load_model(os.path.join(OUT, "speaker_encoder.ncnn.bin")) == 0
ok = True
for TT in (300, 173):
    torch.manual_seed(TT)
    m = torch.randn(1, TT, 128)
    with torch.no_grad():
        r_t = model(m)[0].numpy()
    a = np.ascontiguousarray(m[0].numpy())
    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(a).clone())
        ret, o = ex.extract("out0")
        assert ret == 0
        got = np.array(o).copy().reshape(-1)
    cos_sim = float(np.dot(got, r_t) / (np.linalg.norm(got) * np.linalg.norm(r_t)))
    print(f"T={TT}: max_abs {np.abs(got - r_t).max():.3e} | cos_sim {cos_sim:.7f}")
    ok &= cos_sim > 0.9999
print("SPEAKER ENCODER:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
