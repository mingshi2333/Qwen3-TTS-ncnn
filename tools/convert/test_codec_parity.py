"""Codec decoder parity: ncnn vs PyTorch on random codes, two lengths (dynamic check).

PyTorch side uses the ORIGINAL unpatched decoder.forward (ground truth).
"""
import importlib
import json
import os

import numpy as np
import torch

import common
from common import S

import ncnn


def to_mat(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return ncnn.Mat(arr).clone()


def to_mat_i32(arr):
    # ncnn Embed reinterprets input bits as int32 — keep the integer dtype
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    return ncnn.Mat(arr).clone()


tok_mod = S.tokenizer_v2
cfgmod = importlib.import_module("qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2")
MODEL_DIR = os.path.join(common.WORK_DIR, "models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer")
dec_cfg = cfgmod.Qwen3TTSTokenizerV2DecoderConfig(**json.load(open(os.path.join(MODEL_DIR, "config.json")))["decoder_config"])
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
decoder.load_state_dict(sd, strict=False)


def band_mask(T, window):
    m = np.full((1, T, T), -np.inf, dtype=np.float32)
    for i in range(T):
        m[0, i, max(0, i - window + 1):i + 1] = 0.0
    return m


def rope_tables(T, dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    ang = np.arange(T, dtype=np.float64)[:, None] * inv[None]
    emb = np.concatenate([ang, ang], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


net = ncnn.Net()
assert net.load_param(os.path.join(common.WORK_DIR, "ncnn_models/codec_decoder.ncnn.param")) == 0
assert net.load_model(os.path.join(common.WORK_DIR, "ncnn_models/codec_decoder.ncnn.bin")) == 0

ok_all = True
for T in (25, 44):
    torch.manual_seed(T)
    codes = torch.randint(0, dec_cfg.codebook_size, (1, dec_cfg.num_quantizers, T))
    with torch.no_grad():
        ref = decoder(codes)[0, 0].numpy()  # ORIGINAL forward, unpatched

    codes_np = codes.numpy().astype(np.int32)
    mask = band_mask(T, dec_cfg.sliding_window)
    cos, sin = rope_tables(T, dec_cfg.head_dim, dec_cfg.rope_theta)

    off = (np.arange(15, dtype=np.int32)[:, None] * dec_cfg.codebook_size)
    with net.create_extractor() as ex:
        ex.input("in0", to_mat_i32(codes_np[0, :1]))          # semantic (1,T)
        ex.input("in1", to_mat_i32((codes_np[0, 1:] + off).reshape(-1)))  # acoustic (15T,), pre-offset + pre-flattened
        ex.input("in2", to_mat(mask))
        ex.input("in3", to_mat(cos))
        ex.input("in4", to_mat(sin))
        ret, out0 = ex.extract("out0")
        assert ret == 0, "extract failed"
        got = np.array(out0).copy().reshape(-1)

    assert got.shape == ref.shape, f"shape mismatch {got.shape} vs {ref.shape}"
    d = np.abs(got - ref)
    # wav in [-1,1]; SOLUTION.md acceptance: max_abs < 1e-2, SNR > 40 dB
    noise = np.mean((got - ref) ** 2)
    snr = 10 * np.log10(np.mean(ref ** 2) / noise) if noise > 0 else float("inf")
    ok = d.max() < 1e-2 and snr > 40
    ok_all &= ok
    print(f"T={T}: wav len {len(got)} | max_abs {d.max():.3e} | mean_abs {d.mean():.3e} | SNR {snr:.1f} dB | {'PASS' if ok else 'FAIL'}")

print("CODEC PARITY:", "PASS" if ok_all else "FAIL")
raise SystemExit(0 if ok_all else 1)
