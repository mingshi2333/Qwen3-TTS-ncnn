"""KV-cache parity for the exported 5-layer code-predictor decoder.

Same scheme as the talker test; predictor uses standard RoPE (θ=1e6) and
sequences ≤17 (2-token prefill + 14 steps per frame).
"""
import os

import numpy as np
import torch

import common
from common import S

import ncnn


def to_mat(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return ncnn.Mat(arr).clone()


N_LAYERS = 5
T = 2  # realistic prefill: [past_hidden, emb(code0)]

cfg_all = common.load_config()
cfg = S.configuration.Qwen3TTSTalkerCodePredictorConfig(
    **{k: v for k, v in cfg_all["talker_config"]["code_predictor_config"].items() if k != "architectures"}
)
cfg._attn_implementation = "sdpa"
torch.manual_seed(0)
model = S.modeling.Qwen3TTSTalkerCodePredictorModel(cfg, embedding_dim=cfg_all["talker_config"]["hidden_size"])
model.eval()
model.load_state_dict(common.load_weights("talker.code_predictor.model."), strict=False)

STEPS = 3  # verify a few AR steps beyond prefill
TOTAL = T + STEPS
embeds = torch.randn(1, TOTAL, cfg.hidden_size)
mask_full = torch.full((1, 1, TOTAL, TOTAL), float("-inf")).triu(1)
pos = torch.arange(TOTAL).unsqueeze(0)
rot = S.modeling.Qwen3TTSRotaryEmbedding(config=cfg)
cos, sin = rot(embeds, pos)

with torch.no_grad():
    h = embeds
    for layer in model.layers:
        h = layer(h, attention_mask=mask_full, position_embeddings=(cos, sin))[0]
    ref = model.norm(h)

net = ncnn.Net()
assert net.load_param(os.path.join(common.WORK_DIR, "ncnn_models/predictor_decoder.ncnn.param")) == 0
assert net.load_model(os.path.join(common.WORK_DIR, "ncnn_models/predictor_decoder.ncnn.bin")) == 0

caches = {}
with net.create_extractor() as ex:
    ex.input("in0", to_mat(embeds[:, :T].squeeze(0).numpy()))
    ex.input("in1", to_mat(mask_full[0, :, :T, :T].numpy()))
    ex.input("in2", to_mat(cos[:, :T].squeeze(0).numpy()))
    ex.input("in3", to_mat(sin[:, :T].squeeze(0).numpy()))
    ret, out0 = ex.extract("out0")
    assert ret == 0
    prefill_out = torch.from_numpy(np.array(out0).copy())
    for i in range(N_LAYERS):
        rk, mk = ex.extract(f"cache_k_out_{i}")
        rv, mv = ex.extract(f"cache_v_out_{i}")
        caches[i] = (np.array(mk).copy(), np.array(mv).copy())

d_pre = (prefill_out - ref[0, :T]).abs().max().item()
rel_pre = d_pre / ref[0, :T].abs().max().item()

max_rel_step = 0.0
for s in range(STEPS):
    p = T + s
    with net.create_extractor() as ex:
        ex.input("in0", to_mat(embeds[:, p:p + 1].squeeze(0).numpy()))
        ex.input("in1", to_mat(np.zeros((1, 1, p + 1), dtype=np.float32)))
        ex.input("in2", to_mat(cos[:, p:p + 1].squeeze(0).numpy()))
        ex.input("in3", to_mat(sin[:, p:p + 1].squeeze(0).numpy()))
        for i in range(N_LAYERS):
            ex.input(f"cache_k_in_{i}", to_mat(caches[i][0]))
            ex.input(f"cache_v_in_{i}", to_mat(caches[i][1]))
        ret, out0 = ex.extract("out0")
        assert ret == 0
        step_out = torch.from_numpy(np.array(out0).copy())
        for i in range(N_LAYERS):
            rk, mk = ex.extract(f"cache_k_out_{i}")
            rv, mv = ex.extract(f"cache_v_out_{i}")
            caches[i] = (np.array(mk).copy(), np.array(mv).copy())
    d = (step_out.squeeze() - ref[0, p]).abs().max().item()
    rel = d / ref[0, p].abs().max().item()
    max_rel_step = max(max_rel_step, rel)
    print(f"step {s} (pos {p}): max_abs {d:.3e} | rel {rel:.2e}")

print(f"prefill rel {rel_pre:.2e} | worst step rel {max_rel_step:.2e}")
ok = rel_pre < 1e-5 and max_rel_step < 1e-5
print("PREDICTOR PARITY:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
