"""Shared helpers for Qwen3-TTS → ncnn export scripts.

Applies two mandatory trace-time patches (see SOLUTION.md §3.2/§3.3):
1. mrope → standard 1D RoPE (verified exactly equal when all 3 position rows match)
2. sdpa attention → torch.repeat_interleave GQA so pnnx's
   F_scaled_dot_product_attention_fb_mask_gqa pattern matches (HF's default
   unsqueeze/expand/reshape repeat_kv leaves unloadable pnnx.Expression layers)
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)

import shim_qwen_tts as S  # noqa: E402

MODEL_DIR = os.path.join(WORK_DIR, "models/Qwen3-TTS-12Hz-0.6B-Base")


def apply_export_patches():
    _orig_rope = S.modeling.apply_rotary_pos_emb

    def _plain_rope(q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1):
        return _orig_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    S.modeling.apply_multimodal_rotary_pos_emb = _plain_rope

    def _export_sdpa_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        if module.num_key_value_groups > 1:
            key = torch.repeat_interleave(key, module.num_key_value_groups, dim=-3)
            value = torch.repeat_interleave(value, module.num_key_value_groups, dim=-3)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    S.modeling.ALL_ATTENTION_FUNCTIONS["sdpa"] = _export_sdpa_forward


def load_config(model_dir=MODEL_DIR):
    return json.load(open(os.path.join(model_dir, "config.json")))


def load_weights(prefix, model_dir=MODEL_DIR, strip=True):
    """Load fp32 tensors whose key starts with prefix; optionally strip the prefix."""
    from safetensors import safe_open
    sd = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                out_key = key[len(prefix):] if strip else key
                sd[out_key] = f.get_tensor(key).float()
    return sd
