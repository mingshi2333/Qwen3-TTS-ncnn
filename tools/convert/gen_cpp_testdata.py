"""Generate C++ smoke-test vectors from the VALIDATED python-ncnn pipeline.

The C++ runtime must reproduce these bit-for-bit-ish (same libncnn, same graphs):
  talker_in.npy        (10,1024) random prefill embeds
  talker_prefill.npy   (10,1024) expected prefill hidden
  talker_step_in.npy   (1,1024)  step embed
  talker_step.npy      (1,1024)  expected step hidden (with cache from prefill)
  ids.npy              (7,)      random text token ids
  text_embed.npy       (7,1024)  expected talker_text_embed output
"""
import os

import numpy as np

import common
import ncnn

W = common.WORK_DIR
NM = os.path.join(W, "ncnn_models")
OUT = "/home/mingshi/Project/AI/Qwen3-TTS-ncnn/tests/data"
os.makedirs(OUT, exist_ok=True)

TC = common.load_config()["talker_config"]


def to_mat(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return ncnn.Mat(a).clone()


def to_mat_i32(a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return ncnn.Mat(a).clone()


def rope(pos, theta=1e6, dim=128):
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None]
    emb = np.concatenate([ang, ang], 1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


rng = np.random.RandomState(42)

# --- talker prefill + step ---
net = ncnn.Net()
assert net.load_param(os.path.join(NM, "talker_decoder.ncnn.param")) == 0
assert net.load_model(os.path.join(NM, "talker_decoder.ncnn.bin")) == 0

T = 10
embeds = (rng.randn(T, 1024) * 0.05).astype(np.float32)
step_embed = (rng.randn(1, 1024) * 0.05).astype(np.float32)

mask = np.triu(np.full((1, T, T), -np.inf, dtype=np.float32), 1)
cos, sin = rope(np.arange(T))
caches = {}
with net.create_extractor() as ex:
    ex.input("in0", to_mat(embeds))
    ex.input("in1", to_mat(mask))
    ex.input("in2", to_mat(cos))
    ex.input("in3", to_mat(sin))
    ret, o = ex.extract("out0")
    assert ret == 0
    prefill = np.array(o).copy()
    for i in range(28):
        rk, mk = ex.extract(f"cache_k_out_{i}")
        rv, mv = ex.extract(f"cache_v_out_{i}")
        caches[i] = (np.array(mk).copy(), np.array(mv).copy())

cos1, sin1 = rope([T])
with net.create_extractor() as ex:
    ex.input("in0", to_mat(step_embed))
    ex.input("in1", to_mat(np.zeros((1, 1, T + 1), dtype=np.float32)))
    ex.input("in2", to_mat(cos1))
    ex.input("in3", to_mat(sin1))
    for i in range(28):
        ex.input(f"cache_k_in_{i}", to_mat(caches[i][0]))
        ex.input(f"cache_v_in_{i}", to_mat(caches[i][1]))
    ret, o = ex.extract("out0")
    assert ret == 0
    step = np.array(o).copy()

# --- text embed ---
tnet = ncnn.Net()
assert tnet.load_param(os.path.join(NM, "talker_text_embed.ncnn.param")) == 0
assert tnet.load_model(os.path.join(NM, "talker_text_embed.ncnn.bin")) == 0
ids = rng.randint(0, 151936, 7).astype(np.int32)
with tnet.create_extractor() as ex:
    ex.input("in0", to_mat_i32(ids))
    ret, o = ex.extract("out0")
    text_out = np.array(o).copy()

np.save(os.path.join(OUT, "talker_in.npy"), embeds)
np.save(os.path.join(OUT, "talker_prefill.npy"), prefill)
np.save(os.path.join(OUT, "talker_step_in.npy"), step_embed)
np.save(os.path.join(OUT, "talker_step.npy"), step)
np.save(os.path.join(OUT, "ids.npy"), ids)
np.save(os.path.join(OUT, "text_embed.npy"), text_out)
print("saved test vectors:", prefill.shape, step.shape, text_out.shape)
