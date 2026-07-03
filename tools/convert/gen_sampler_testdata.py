"""Sampler parity vectors: random logits -> HF processor+warper chain -> final
probability distribution. C++ must reproduce the distribution exactly (the
multinomial draw itself is seeded independently and not compared).

File sampler_test.bin: uint32 n_cases, per case:
  int32 step, float temp, int32 top_k, float top_p,
  uint32 n_hist, int32 hist..., float32 logits[3072],
  uint32 n_keep, (int32 idx, float32 prob) * n_keep   (sorted desc by prob)
"""
import os
import struct

import numpy as np
import torch
from transformers import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                          SuppressTokensLogitsProcessor, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper)

import common

VOCAB, EOS = 3072, 2150
SUPPRESS = [i for i in range(VOCAB - 1024, VOCAB) if i != EOS]
OUT = "/home/mingshi/Project/AI/Qwen3-TTS-ncnn/tests/data/sampler_test.bin"

rng = np.random.RandomState(3)
cases = []
for c in range(24):
    step = int(rng.randint(0, 5))
    temp = float(rng.choice([0.9, 1.0, 0.7, 1.3]))
    top_k = int(rng.choice([50, 20, 0]))
    top_p = float(rng.choice([1.0, 0.95, 0.8]))
    hist = rng.randint(0, 2048, rng.randint(0, 12)).astype(np.int64)
    logits = (rng.randn(VOCAB) * 3).astype(np.float32)
    cases.append((step, temp, top_k, top_p, hist, logits))

with open(OUT, "wb") as f:
    f.write(struct.pack("<I", len(cases)))
    for step, temp, top_k, top_p, hist, logits in cases:
        l = torch.from_numpy(logits)[None].float()
        ids = torch.from_numpy(hist)[None]
        procs = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(1.05)])
        l2 = procs(ids, l.clone())
        if step < 2:
            l2[0, EOS] = float("-inf")
        l2 = SuppressTokensLogitsProcessor(SUPPRESS, device="cpu")(ids, l2)
        warpers = LogitsProcessorList([TemperatureLogitsWarper(temp)])
        if top_k > 0:
            warpers.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p))
        l3 = warpers(ids, l2)
        probs = torch.softmax(l3[0], dim=-1).numpy()
        keep = np.where(probs > 0)[0]
        order = keep[np.argsort(-probs[keep], kind="stable")]

        f.write(struct.pack("<ifif", step, temp, top_k, top_p))
        f.write(struct.pack("<I", len(hist)))
        f.write(hist.astype(np.int32).tobytes())
        f.write(logits.tobytes())
        f.write(struct.pack("<I", len(order)))
        for i in order:
            f.write(struct.pack("<if", int(i), float(probs[i])))
print(f"{len(cases)} sampler cases -> {OUT}")
