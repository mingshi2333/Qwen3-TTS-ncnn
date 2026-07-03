"""Export the talker (28L) or code-predictor (5L) decoder stack to ncnn via pnnx,
then apply the kv-cache param patch.

Usage:
    python export_decoder.py talker
    python export_decoder.py predictor
"""
import os
import subprocess
import sys

import torch
import torch.nn as nn

import common
from common import S

common.apply_export_patches()

COMPONENT = sys.argv[1] if len(sys.argv) > 1 else "talker"
OUT_DIR = os.path.join(common.WORK_DIR, "ncnn_models")
os.makedirs(OUT_DIR, exist_ok=True)

cfg_all = common.load_config()

if COMPONENT == "talker":
    cfg = S.configuration.Qwen3TTSTalkerConfig(
        **{k: v for k, v in cfg_all["talker_config"].items() if k not in ("code_predictor_config", "architectures")}
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = S.modeling.Qwen3TTSTalkerModel(cfg)
    sd = common.load_weights("talker.model.")
    name = "talker_decoder"
elif COMPONENT == "predictor":
    cfg = S.configuration.Qwen3TTSTalkerCodePredictorConfig(
        **{k: v for k, v in cfg_all["talker_config"]["code_predictor_config"].items() if k != "architectures"}
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = S.modeling.Qwen3TTSTalkerCodePredictorModel(cfg, embedding_dim=cfg_all["talker_config"]["hidden_size"])
    sd = common.load_weights("talker.code_predictor.model.")
    name = "predictor_decoder"
else:
    raise SystemExit(f"unknown component {COMPONENT}")

model.eval()
missing, unexpected = model.load_state_dict(sd, strict=False)
real = [m for m in missing if not m.startswith(("codec_embedding", "text_embedding"))]
print(f"{name}: loaded {len(sd)} tensors | missing(non-embed): {real} | unexpected: {unexpected}")
assert not real, "missing decoder weights!"

head_dim = cfg.head_dim


class DecoderWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.layers = m.layers
        self.norm = m.norm

    def forward(self, hidden_states, attn_mask, cos, sin):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_embeddings=(cos, sin),
            )[0]
        return self.norm(hidden_states)


wrapper = DecoderWrapper(model)
wrapper.eval()

T = 37
hidden = torch.randn(1, T, cfg.hidden_size)
mask = torch.full((1, 1, T, T), float("-inf")).triu(1)
if COMPONENT == "talker":
    rot = S.modeling.Qwen3TTSTalkerRotaryEmbedding(config=cfg)
    pos = torch.arange(T).unsqueeze(0)
    cos3, sin3 = rot(hidden, pos.unsqueeze(0).expand(3, -1, -1))
    cos, sin = cos3[0], sin3[0]
else:
    rot = S.modeling.Qwen3TTSRotaryEmbedding(config=cfg)
    pos = torch.arange(T).unsqueeze(0)
    cos, sin = rot(hidden, pos)

with torch.no_grad():
    ref = wrapper(hidden, mask, cos, sin)
traced = torch.jit.trace(wrapper, (hidden, mask, cos, sin))
with torch.no_grad():
    tdiff = (traced(hidden, mask, cos, sin) - ref).abs().max().item()
print(f"trace ok, diff={tdiff}")
assert tdiff == 0.0

pt_path = os.path.join(OUT_DIR, f"{name}.pt")
traced.save(pt_path)

H = cfg.hidden_size
D = head_dim
shape1 = f"[1,{T},{H}],[1,1,{T},{T}],[1,{T},{D}],[1,{T},{D}]"
shape2 = f"[1,5,{H}],[1,1,5,5],[1,5,{D}],[1,5,{D}]"
cmd = ["pnnx", pt_path, f"inputshape={shape1}", f"inputshape2={shape2}", "fp16=0"]
print("running:", " ".join(cmd))
r = subprocess.run(cmd, cwd=OUT_DIR, capture_output=True, text=True)
tail = "\n".join(r.stdout.splitlines()[-5:])
print(tail)
assert r.returncode == 0, f"pnnx failed: {r.stderr[-2000:]}"

param_path = os.path.join(OUT_DIR, f"{name}.ncnn.param")
assert os.path.exists(param_path)

r = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "add_kvcache.py"), param_path],
    capture_output=True, text=True,
)
print(r.stdout.strip())
assert r.returncode == 0, r.stderr

# leftover-op sanity check: everything in the param must be a real ncnn layer
bad = []
for line in open(param_path):
    t = line.split()[0] if line.strip() else ""
    if "." in t and t not in ("7767517",):
        bad.append(t)
print("unconvertible ops:", set(bad) if bad else "none")
assert not bad, f"graph contains non-ncnn ops: {set(bad)}"
print(f"DONE {name}")
