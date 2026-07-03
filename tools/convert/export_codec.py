"""Export the Qwen3-TTS-Tokenizer-12Hz codec DECODER (codes -> 24kHz wav) to ncnn.

Wrapper design (SOLUTION.md §3.3):
- whole graph in one net: RVQ dequant (16 Embed sum + 2 output_proj) -> pre_conv
  -> 8-layer sliding-window transformer (mask/cos/sin as explicit inputs)
  -> upsample convnext -> DAC/SnakeBeta vocoder -> clamp
- pre-folded before trace: EuclideanCodebook embedding (embedding_sum/usage),
  SnakeBeta exp(alpha)/exp(beta)
- input codes as int32 (1,16,T); mask (1,1,T,T) band-causal window 72; rope theta 1e4
"""
import json
import os
import subprocess
import sys

import torch
import torch.nn as nn

import common
from common import S

MODEL_DIR = os.path.join(common.WORK_DIR, "models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer")
OUT_DIR = os.path.join(common.WORK_DIR, "ncnn_models")
os.makedirs(OUT_DIR, exist_ok=True)

tok_mod = S.tokenizer_v2
cfg_all = json.load(open(os.path.join(MODEL_DIR, "config.json")))
dec_cfg_cls = sys.modules["qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2"]
# config classes live next to the modeling file
import importlib
cfgmod = importlib.import_module("qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2")
dec_cfg = cfgmod.Qwen3TTSTokenizerV2DecoderConfig(**cfg_all["decoder_config"])
dec_cfg._attn_implementation = "sdpa"

torch.manual_seed(0)
decoder = tok_mod.Qwen3TTSTokenizerV2Decoder(dec_cfg)
decoder.eval()

from safetensors import safe_open
sd = {}
with safe_open(os.path.join(MODEL_DIR, "model.safetensors"), framework="pt") as f:
    for key in f.keys():
        if key.startswith("decoder."):
            sd[key[len("decoder."):]] = f.get_tensor(key).float()
missing, unexpected = decoder.load_state_dict(sd, strict=False)
print(f"codec decoder: loaded {len(sd)} | missing: {missing} | unexpected: {unexpected}")
assert not missing, "missing codec weights"

# ---- pre-fold EuclideanCodebook: embedding = embedding_sum / clamp(usage) ----
def fold_codebooks(module):
    n = 0
    for m in module.modules():
        if isinstance(m, tok_mod.EuclideanCodebook):
            emb = (m.embedding_sum / m.cluster_usage.clamp(min=m.epsilon)[:, None]).detach()
            m.register_buffer("folded_embedding", emb)
            n += 1
    return n

def folded_codebook_decode(self, codes):
    return torch.nn.functional.embedding(codes, self.folded_embedding)

print("folded codebooks:", fold_codebooks(decoder))
tok_mod.EuclideanCodebook.decode = folded_codebook_decode

# ---- pre-fold SnakeBeta exp ----
def fold_snake(module):
    n = 0
    for m in module.modules():
        if isinstance(m, tok_mod.SnakeBeta):
            m.alpha.data = m.alpha.exp()
            m.beta.data = m.beta.exp()
            n += 1
    return n

def folded_snake_forward(self, x):
    alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
    beta = self.beta.unsqueeze(0).unsqueeze(-1)
    return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

print("folded snakes:", fold_snake(decoder))
tok_mod.SnakeBeta.forward = folded_snake_forward

# ---- trace-safe causal conv forwards ----
# original CausalConvNet.forward computes extra_padding from hidden.shape[-1] via
# math.ceil — under jit.trace this bakes "trace_T - length" into the graph, which
# TRUNCATES at other lengths (T=50 in -> 25 out). stride=1 everywhere in the
# decoder ⇒ extra_padding ≡ 0, so a fixed left pad is exact.
import torch.nn.functional as F

def causal_conv_forward(self, hidden_state):
    hidden_state = F.pad(hidden_state, (self.padding, 0), mode="constant", value=0)
    return self.conv(hidden_state).contiguous()

# original TransConv crops with shape[-1] - right_pad (constant-baked end); a
# python negative slice gets silently DROPPED by pnnx. Negative F.pad however
# converts to a proper relative ncnn Crop (-23310 ends) — verified dynamic-safe.
def causal_transconv_forward(self, hidden_state):
    hidden_state = self.conv(hidden_state)
    if self.right_pad > 0:
        hidden_state = F.pad(hidden_state, (0, -self.right_pad))
    return hidden_state.contiguous()

for m in decoder.modules():
    if isinstance(m, tok_mod.Qwen3TTSTokenizerV2CausalConvNet):
        assert m.stride == 1, f"stride {m.stride} != 1 breaks the fixed-pad assumption"
tok_mod.Qwen3TTSTokenizerV2CausalConvNet.forward = causal_conv_forward
tok_mod.Qwen3TTSTokenizerV2CausalTransConvNet.forward = causal_transconv_forward

# ---- fused RVQ dequant tables ----
# Any per-level slicing of the codes tensor (enumerate/unbind, tensor_split,
# repeated select) becomes a PARAMETERLESS ncnn Slice layer via pnnx, which
# null-derefs at runtime (pnnx bug — report upstream). Instead: fuse the 15
# acoustic codebooks into one (15*2048, 256) table; the caller pre-offsets
# level k codes by k*2048. Semantic level embeds directly from its own table.
sem_table = decoder.quantizer.rvq_first.vq.layers[0]._codebook.folded_embedding  # (2048, 256)
ac_table = torch.cat(
    [l._codebook.folded_embedding for l in decoder.quantizer.rvq_rest.vq.layers], dim=0
)  # (15*2048, 256)


class CodecDecoderWrapper(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
        self.register_buffer("sem_table", sem_table)
        self.register_buffer("ac_table", ac_table)
        self.register_buffer("level_ones", torch.ones(1, 15))

    def forward(self, codes0, codes_rest_flat, attn_mask, cos, sin):
        # codes0: (1,T) semantic codes. codes_rest_flat: (15*T,) acoustic codes,
        # level-major, PRE-OFFSET by level*2048 AND PRE-FLATTENED by the caller —
        # int32 bits must not pass through any layout-transforming ncnn layer
        # (Reshape's SIMD packing mangles them), and ncnn Embed reads 1D only.
        # The level sum uses ones@x (MatMul) because pnnx drops Reduction axes.
        dec = self.dec
        sem = torch.nn.functional.embedding(codes0, self.sem_table).transpose(1, 2)          # (1,256,T)
        ac = torch.nn.functional.embedding(codes_rest_flat, self.ac_table)                   # (15*T,256)
        ac = torch.matmul(self.level_ones, ac.reshape(15, -1))                               # (1, T*256)
        ac = ac.reshape(1, -1, 256).transpose(1, 2)                                          # (1,256,T)
        hidden = dec.quantizer.rvq_first.output_proj(sem) + dec.quantizer.rvq_rest.output_proj(ac)
        hidden = dec.pre_conv(hidden).transpose(1, 2)
        t = dec.pre_transformer
        hidden = t.input_proj(hidden)
        # inline the layer forward: position_embeddings must reach attention as a
        # positional arg — routed through **kwargs it gets frozen to the trace-time
        # shape by torch.jit.trace, breaking dynamic seqlen
        for layer in t.layers:
            residual = hidden
            x = layer.input_layernorm(hidden)
            attn_out, _ = layer.self_attn(x, (cos, sin), attn_mask)
            hidden = residual + layer.self_attn_layer_scale(attn_out)
            residual = hidden
            x = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp_layer_scale(layer.mlp(x))
        hidden = t.norm(hidden)
        hidden = t.output_proj(hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in dec.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in dec.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


wrapper = CodecDecoderWrapper(decoder)
wrapper.eval()


def band_mask(T, window):
    m = torch.full((1, 1, T, T), float("-inf"))
    for i in range(T):
        lo = max(0, i - window + 1)
        m[0, 0, i, lo:i + 1] = 0.0
    return m


def rope_tables(T, dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.arange(T).float()[:, None] * inv[None]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos()[None], emb.sin()[None]  # (1,T,dim)


T = 25
codes = torch.randint(0, dec_cfg.codebook_size, (1, dec_cfg.num_quantizers, T), dtype=torch.int32)
mask = band_mask(T, dec_cfg.sliding_window)
cos, sin = rope_tables(T, dec_cfg.head_dim, dec_cfg.rope_theta)

OFF = torch.arange(15, dtype=torch.long)[:, None] * dec_cfg.codebook_size
def prep(codes):
    return codes[0, :1].long(), (codes[0, 1:].long() + OFF).reshape(-1)
c0, c1 = prep(codes)
with torch.no_grad():
    ref = wrapper(c0, c1, mask, cos, sin)
print("torch forward ok:", tuple(ref.shape), "expected T*1920 =", T * 1920)

# sanity: wrapper must equal the original decoder.forward on same codes
torch.manual_seed(0)
with torch.no_grad():
    ref_orig = decoder(codes.long())
d = (ref - ref_orig).abs().max().item()
print("wrapper vs original decoder.forward max_abs:", d)
assert d < 1e-5, "wrapper mismatch vs original forward"

traced = torch.jit.trace(wrapper, (c0.int(), c1.int(), mask, cos, sin))
with torch.no_grad():
    tdiff = (traced(c0.int(), c1.int(), mask, cos, sin) - ref).abs().max().item()
print("trace ok, diff:", tdiff)

pt_path = os.path.join(OUT_DIR, "codec_decoder.pt")
traced.save(pt_path)

Q = dec_cfg.num_quantizers
D = dec_cfg.head_dim
shape1 = f"[1,{T}]i32,[{(Q-1)*T}]i32,[1,1,{T},{T}],[1,{T},{D}],[1,{T},{D}]"
T2 = 50
shape2 = f"[1,{T2}]i32,[{(Q-1)*T2}]i32,[1,1,{T2},{T2}],[1,{T2},{D}],[1,{T2},{D}]"
cmd = ["pnnx", pt_path, f"inputshape={shape1}", f"inputshape2={shape2}", "fp16=0"]
print("running:", " ".join(cmd))
r = subprocess.run(cmd, cwd=OUT_DIR, capture_output=True, text=True)
print("\n".join(r.stdout.splitlines()[-6:]))
assert r.returncode == 0, f"pnnx failed: {r.stderr[-3000:]}"

param_path = os.path.join(OUT_DIR, "codec_decoder.ncnn.param")
bad = []
for line in open(param_path):
    t = line.split()[0] if line.strip() else ""
    if "." in t and t != "7767517":
        bad.append(t)
print("unconvertible ops:", set(bad) if bad else "none")
print("DONE codec_decoder")
