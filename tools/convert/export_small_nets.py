"""Export + verify the small nets (SOLUTION.md §3.1 #1,2,4,6,7):

  talker_text_embed  : ids (T,) i32  -> text_embedding(151936x2048) -> ResizeMLP -> (T,1024)
  talker_codec_embed : ids (T,) i32  -> codec_embedding(3072x1024)  -> (T,1024)
  talker_codec_head  : hidden (T,1024) -> Linear(1024->3072)        -> (T,3072)
  predictor_embeds   : ids (n,) i32 PRE-OFFSET by group*2048 -> fused table (15*2048,1024) -> (n,1024)
  predictor_heads    : hidden (T,1024) -> 15 branches Linear(1024->2048) -> out0..out14 (lazy extract)

Conventions: id inputs are 1D int32 BITS (ncnn Embed reinterprets), fed directly
to Embed without any transform layer in between.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
from common import S

import ncnn

OUT = os.path.join(common.WORK_DIR, "ncnn_models")
cfg_all = common.load_config()
HID = cfg_all["talker_config"]["hidden_size"]


def run_pnnx(name, mod, example, shape1, shape2):
    traced = torch.jit.trace(mod, example)
    pt = os.path.join(OUT, f"{name}.pt")
    traced.save(pt)
    r = subprocess.run(["pnnx", pt, f"inputshape={shape1}", f"inputshape2={shape2}", "fp16=0"],
                       cwd=OUT, capture_output=True, text=True)
    assert r.returncode == 0, f"pnnx {name} failed: {r.stderr[-1500:]}"
    param = os.path.join(OUT, f"{name}.ncnn.param")
    bad = [l.split()[0] for l in open(param) if l.strip() and "." in l.split()[0]]
    assert not bad, f"{name}: non-ncnn ops {set(bad)}"
    return param, os.path.join(OUT, f"{name}.ncnn.bin")


def to_mat_i32(a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return ncnn.Mat(a).clone()


def to_mat(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return ncnn.Mat(a).clone()


def check(name, param, bin_, feeds, ref_by_out):
    net = ncnn.Net()
    assert net.load_param(param) == 0 and net.load_model(bin_) == 0
    worst = 0.0
    with net.create_extractor() as ex:
        for blob, mat in feeds:
            ex.input(blob, mat)
        for out_blob, ref in ref_by_out.items():
            ret, o = ex.extract(out_blob)
            assert ret == 0, f"{name}: extract {out_blob} failed"
            g = torch.from_numpy(np.array(o).copy())
            if g.shape != ref.shape and g.numel() == ref.numel():
                g = g.reshape(ref.shape)
            d = (g - ref).abs().max().item()
            worst = max(worst, d)
    status = "PASS" if worst < 1e-4 else "FAIL"
    print(f"{name}: max_abs {worst:.3e} [{status}]")
    return worst < 1e-4


results = []

# ---------------- talker_text_embed ----------------
w = common.load_weights("talker.model.text_embedding.")
proj = common.load_weights("talker.text_projection.")

class TextEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", w["weight"])
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc1.weight.data = proj["linear_fc1.weight"]
        self.fc1.bias.data = proj["linear_fc1.bias"]
        self.fc2.weight.data = proj["linear_fc2.weight"]
        self.fc2.bias.data = proj["linear_fc2.bias"]

    def forward(self, ids):
        return self.fc2(F.silu(self.fc1(F.embedding(ids, self.table))))

m = TextEmbed().eval()
ids = torch.randint(0, 151936, (7,), dtype=torch.int32)
with torch.no_grad():
    ref = m(ids.long())
param, bin_ = run_pnnx("talker_text_embed", m, (ids,), "[7]i32", "[13]i32")
results.append(check("talker_text_embed", param, bin_, [("in0", to_mat_i32(ids.numpy()))], {"out0": ref}))

# ---------------- talker_codec_embed ----------------
w = common.load_weights("talker.model.codec_embedding.")

class CodecEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", w["weight"])

    def forward(self, ids):
        return F.embedding(ids, self.table)

m = CodecEmbed().eval()
ids = torch.randint(0, 3072, (7,), dtype=torch.int32)
with torch.no_grad():
    ref = m(ids.long())
param, bin_ = run_pnnx("talker_codec_embed", m, (ids,), "[7]i32", "[13]i32")
results.append(check("talker_codec_embed", param, bin_, [("in0", to_mat_i32(ids.numpy()))], {"out0": ref}))

# ---------------- talker_codec_head ----------------
w = common.load_weights("talker.codec_head.")

class CodecHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 3072, bias=False)
        self.fc.weight.data = w["weight"]

    def forward(self, x):
        return self.fc(x)

m = CodecHead().eval()
x = torch.randn(3, 1024)
with torch.no_grad():
    ref = m(x)
param, bin_ = run_pnnx("talker_codec_head", m, (x,), "[3,1024]", "[1,1024]")
results.append(check("talker_codec_head", param, bin_, [("in0", to_mat(x.numpy()))], {"out0": ref}))

# ---------------- predictor_embeds (fused, pre-offset ids) ----------------
pw = common.load_weights("talker.code_predictor.model.codec_embedding.")
tables = torch.cat([pw[f"{i}.weight"] for i in range(15)], dim=0)  # (15*2048, 1024)

class PredEmbeds(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", tables)

    def forward(self, ids):
        return F.embedding(ids, self.table)

m = PredEmbeds().eval()
ids = (torch.randint(0, 2048, (5,)) + torch.arange(5) * 2048).to(torch.int32)
with torch.no_grad():
    ref = m(ids.long())
param, bin_ = run_pnnx("predictor_embeds", m, (ids,), "[5]i32", "[15]i32")
results.append(check("predictor_embeds", param, bin_, [("in0", to_mat_i32(ids.numpy()))], {"out0": ref}))

# ---------------- predictor_heads (15 lazy branches) ----------------
hw = common.load_weights("talker.code_predictor.lm_head.")

class PredHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(1024, 2048, bias=False) for _ in range(15)])
        for i, h in enumerate(self.heads):
            h.weight.data = hw[f"{i}.weight"]

    def forward(self, x):
        return tuple(h(x) for h in self.heads)

m = PredHeads().eval()
x = torch.randn(1, 1024)
with torch.no_grad():
    refs = m(x)
param, bin_ = run_pnnx("predictor_heads", m, (x,), "[1,1024]", "[2,1024]")
results.append(check("predictor_heads", param, bin_, [("in0", to_mat(x.numpy()))],
                     {f"out{i}": refs[i] for i in range(15)}))

print("SMALL NETS:", "ALL PASS" if all(results) else "SOME FAILED")
raise SystemExit(0 if all(results) else 1)
