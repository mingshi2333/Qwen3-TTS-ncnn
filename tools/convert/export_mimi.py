"""Export the Mimi encoder (wav -> 512-d latent @12.5Hz) to ncnn + dump VQ tables.

Graph: SEANet conv encoder -> 8-layer transformer (25Hz, sliding window 250,
mask/cos/sin as explicit inputs) -> stride-2 downsample conv (replicate pad).
VQ stays in C++ (split semantics: 1 semantic level ∥ 15-level acoustic residual
chain, each side through its own input_proj; encoder_valid_num_quantizers=16).

Constraints: input wav length must be a multiple of 1920 (the caller右零填充) —
then every conv's extra_padding is exactly 0 and the static-pad patch is exact.
"""
import json
import os
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
from common import S  # noqa: F401  (shim loads qwen_tts)

from transformers.models.mimi import modeling_mimi as MM
from transformers.models.mimi.configuration_mimi import MimiConfig

W = common.WORK_DIR
OUT = os.path.join(W, "ncnn_models")
MODELS = "/home/mingshi/Project/AI/Qwen3-TTS-ncnn/models"
ST = os.path.join(W, "models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer")

enc_cfg = MimiConfig(**json.load(open(os.path.join(ST, "config.json")))["encoder_config"])
enc_cfg._attn_implementation = "eager"

torch.manual_seed(0)
tok_mod = S.tokenizer_v2
encoder = tok_mod.Qwen3TTSTokenizerV2Encoder(enc_cfg)
encoder.eval()

from safetensors import safe_open
sd = {}
with safe_open(os.path.join(ST, "model.safetensors"), framework="pt") as f:
    for k in f.keys():
        if k.startswith("encoder."):
            sd[k[len("encoder."):]] = f.get_tensor(k).float()
missing, unexpected = encoder.load_state_dict(sd, strict=False)
real_missing = [m for m in missing if not m.startswith(("decoder", "upsample", "decoder_transformer"))]
print(f"mimi encoder: {len(sd)} tensors | missing: {real_missing[:4]} | unexpected: {unexpected[:4]}")
assert not real_missing

# ---- reference BEFORE patching (full torch encode path, pre-quantizer) ----
def torch_latent(model, wav):
    with torch.no_grad():
        emb = model.encoder(wav)
        h = model.encoder_transformer(emb.transpose(1, 2), return_dict=True)[0]
        return model.downsample(h.transpose(1, 2))

L = 1920 * 25
wav = torch.randn(1, 1, L) * 0.1
ref = torch_latent(encoder, wav)
print("torch latent:", tuple(ref.shape))

# ---- patches ----
def conv_forward_static(self, hidden_states, padding_cache=None):
    # exact when stride divides the incoming length (guaranteed by the 1920-multiple rule)
    return self.conv(F.pad(hidden_states, (int(self.padding_total), 0), mode=self.pad_mode))

MM.MimiConv1d.forward = conv_forward_static

with torch.no_grad():
    d = (torch_latent(encoder, wav) - ref).abs().max().item()
print("patched conv vs original:", d)
assert d < 1e-5


class MimiEncoderWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, wav, attn_mask, cos, sin):
        m = self.m
        h = m.encoder(wav).transpose(1, 2)          # (1, T2, 512) @25Hz
        for layer in m.encoder_transformer.layers:  # inline: rope/mask external (P4)
            a = layer.self_attn
            residual = h
            x = layer.input_layernorm(h)
            B, T, _ = 1, x.shape[1], 0
            q = a.q_proj(x).view(1, -1, a.num_heads, a.head_dim).transpose(1, 2)
            k = a.k_proj(x).view(1, -1, a.num_key_value_heads, a.head_dim).transpose(1, 2)
            v = a.v_proj(x).view(1, -1, a.num_key_value_heads, a.head_dim).transpose(1, 2)
            q, k = MM.apply_rotary_pos_emb(q, k, cos, sin)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=a.scaling)
            o = o.transpose(1, 2).reshape(1, -1, a.num_heads * a.head_dim)
            h = residual + layer.self_attn_layer_scale(a.o_proj(o))
            residual = h
            h = residual + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
        return m.downsample(h.transpose(1, 2))      # (1, 512, T2/2) @12.5Hz


def band_mask(T, window):
    m = torch.full((1, 1, T, T), float("-inf"))
    for i in range(T):
        m[0, 0, i, max(0, i - window + 1):i + 1] = 0.0
    return m


def rope_tables(T, dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2).double() / dim))
    ang = torch.arange(T).double()[:, None] * inv[None]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos().float()[None], emb.sin().float()[None]


wrapper = MimiEncoderWrapper(encoder).eval()
T2 = L // 960
mask = band_mask(T2, enc_cfg.sliding_window)
cos, sin = rope_tables(T2, enc_cfg.head_dim, enc_cfg.rope_theta)
with torch.no_grad():
    wref = wrapper(wav, mask, cos, sin)
d = (wref - ref).abs().max().item()
print("wrapper vs torch encode path:", d)
assert d < 1e-4

traced = torch.jit.trace(wrapper, (wav, mask, cos, sin))
L2 = 1920 * 13
wav2 = torch.randn(1, 1, L2) * 0.1
T2b = L2 // 960
m2 = band_mask(T2b, enc_cfg.sliding_window)
c2, s2 = rope_tables(T2b, enc_cfg.head_dim, enc_cfg.rope_theta)
with torch.no_grad():
    d2 = (traced(wav2, m2, c2, s2) - torch_latent(encoder, wav2)).abs().max().item()
print("traced dynamic check:", d2)
assert d2 < 1e-4, "trace froze a length"

pt = os.path.join(OUT, "mimi_encoder.pt")
traced.save(pt)
r = subprocess.run(["pnnx", pt,
                    f"inputshape=[1,1,{L}],[1,1,{T2},{T2}],[1,{T2},64],[1,{T2},64]",
                    f"inputshape2=[1,1,{L2}],[1,1,{T2b},{T2b}],[1,{T2b},64],[1,{T2b},64]",
                    "fp16=0"], cwd=OUT, capture_output=True, text=True)
print("\n".join(r.stdout.splitlines()[-3:]))
assert r.returncode == 0, r.stderr[-2000:]
param = os.path.join(OUT, "mimi_encoder.ncnn.param")
bad = [l.split()[0] for l in open(param) if l.strip() and "." in l.split()[0]]
print("unconvertible:", set(bad) if bad else "none")
assert not bad

# ---- VQ tables ----
q = encoder.quantizer
sem = q.semantic_residual_vector_quantizer
ac = q.acoustic_residual_vector_quantizer
def folded(cb):
    return (cb.embed_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]).numpy().astype(np.float32)

np.save(os.path.join(MODELS, "mimi_sem_proj.npy"), sem.input_proj.weight.detach().numpy()[:, :, 0].astype(np.float32))
np.save(os.path.join(MODELS, "mimi_sem_codebook.npy"), folded(sem.layers[0].codebook))
np.save(os.path.join(MODELS, "mimi_ac_proj.npy"), ac.input_proj.weight.detach().numpy()[:, :, 0].astype(np.float32))
np.save(os.path.join(MODELS, "mimi_ac_codebooks.npy"),
        np.stack([folded(ac.layers[i].codebook) for i in range(15)]))
print("VQ tables saved (sem proj/cb, ac proj + 15 cbs)")

# ---- ncnn parity: latent + full codes ----
import ncnn
net = ncnn.Net()
assert net.load_param(param) == 0
assert net.load_model(os.path.join(OUT, "mimi_encoder.ncnn.bin")) == 0

def ncnn_latent(w):
    T2x = w.shape[-1] // 960
    mx = band_mask(T2x, enc_cfg.sliding_window)
    cx, sx = rope_tables(T2x, enc_cfg.head_dim, enc_cfg.rope_theta)
    a0 = np.ascontiguousarray(w[0].numpy())
    a1 = np.ascontiguousarray(mx[0].numpy())
    a2 = np.ascontiguousarray(cx[0].numpy())
    a3 = np.ascontiguousarray(sx[0].numpy())
    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(a0).clone())
        ex.input("in1", ncnn.Mat(a1).clone())
        ex.input("in2", ncnn.Mat(a2).clone())
        ex.input("in3", ncnn.Mat(a3).clone())
        ret, o = ex.extract("out0")
        assert ret == 0
        return np.array(o).copy()

def np_split_vq(latent):  # latent (512, T)
    sem_p = np.load(os.path.join(MODELS, "mimi_sem_proj.npy"))
    sem_c = np.load(os.path.join(MODELS, "mimi_sem_codebook.npy"))
    ac_p = np.load(os.path.join(MODELS, "mimi_ac_proj.npy"))
    ac_c = np.load(os.path.join(MODELS, "mimi_ac_codebooks.npy"))
    codes = []
    z = sem_p @ latent
    codes.append(np.argmin(((z.T[:, None, :] - sem_c[None]) ** 2).sum(-1), axis=1))
    r = (ac_p @ latent).T
    for k in range(15):
        idx = np.argmin(((r[:, None, :] - ac_c[k][None]) ** 2).sum(-1), axis=1)
        codes.append(idx)
        r = r - ac_c[k][idx]
    return np.stack(codes, 1)  # (T, 16)

ok = True
for w in (wav, wav2):
    lat = ncnn_latent(w)
    tl = torch_latent(encoder, w)[0].numpy()
    dl = np.abs(lat - tl).max()
    with torch.no_grad():
        tcodes = encoder.encode(w, return_dict=True).audio_codes[0, :16].numpy().T  # (T,16)
    ccodes = np_split_vq(lat)
    match = (ccodes == tcodes).mean()
    print(f"L={w.shape[-1]}: latent max_abs {dl:.3e} | codes match {match*100:.2f}%")
    ok &= match == 1.0
print("MIMI ENCODER:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
