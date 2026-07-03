# Vulkan autoregressive decode: submit overhead, and how we made GPU beat CPU

This note documents a performance problem we hit running the Qwen3-TTS decode
loop on ncnn's Vulkan backend, the root cause (an ncnn execution-model
characteristic, not a bug in our graphs), how we diagnosed it, and the fix.
It is written to be reusable for anyone driving an autoregressive (token-by-token)
decode loop on ncnn Vulkan, and to serve as the basis for an upstream report.

All numbers: Ryzen 7745HX (8 physical cores) + RTX 4060 Laptop, 0.6B model,
fp32, 44-frame utterance (3.52 s audio). fp32 is pinned for token-exact parity
(`use_fp16_* = false`).

## Symptom

The Vulkan build produced **bit-identical tokens** (0/704) but was **slower than
the CPU build**:

| tier | RTF (lower = faster) |
| --- | --- |
| ncnn CPU fp32 (8 threads) | ~2.5 |
| ncnn Vulkan fp32 (initial) | ~3.95 |

A GPU losing to an 8-core CPU on an LLM-shaped workload is a red flag. The codec
decoder — a single feed-forward pass over the whole sequence — was clearly
*faster* on the GPU in the same run (3.4× RT vs CPU 1.9× RT). Only the
autoregressive (AR) loop lost.

## Root cause: one blocking submit per extracted blob

The AR loop is **latency-bound by the number of GPU submits**, and our code was
generating an enormous number of them.

ncnn's default CPU-`Mat` extract path creates a **fresh `VkCompute` and calls
`submit_and_wait()` (a blocking GPU sync) for every extracted blob**
(`src/net.cpp` ~2860–2908: if the requested CPU blob isn't materialized yet, it
builds a `VkCompute`, records, `record_download`, `submit_and_wait`). There is
**no command-buffer reuse / no record-once-replay** (every command buffer is
begun with `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` and reset+re-recorded;
`src/command.cpp:323`, `vkResetCommandBuffer` in `VkCompute::reset`). Only the
shader *pipeline* cache is reused, not command buffers.

Our `KvCacheNet::forward` extracted `out0` plus all `2 * n_layers` KV-cache
outputs **one blob at a time**, so each talker step issued **`1 + 2*28 = 57`
blocking submits**. The Code Predictor made it worse: 15 MTP forwards/frame, each
also extracting its cache blobs, plus 15 separate head extracts. Across the
44-frame run this is on the order of **~10,000 blocking submits**, each paying
fixed launch + queue + fence-wait latency.

## Diagnosis (why we know it's submits, not transfer or compute)

Two cheap experiments, both reproducible with the `Q3TTS_PROFILE=1` per-step
timer built into the runtime:

1. **Per-step cost is flat across sequence length.** If the cost were dominated
   by moving the (quadratically growing) KV cache over PCIe, per-step time would
   grow with position. It didn't — talker steps stayed ~flat — so it is
   **fixed per-submit overhead**, not transfer volume. (Back-of-envelope: the
   whole run moves ~650 MB of talker cache, but spread over the submits that is
   <1 MB / submit, sub-millisecond; the measured ~ms-scale per-submit cost is
   launch/sync latency.)

2. **The codec is the control.** The codec decoder is *one* extractor (one big
   feed-forward). On the same GPU, same fp32, same run, it **wins** (3.4× RT vs
   CPU 1.9×). When ncnn runs a GPU workload as one submit with enough work to
   amortize launch overhead, the GPU pulls ahead. Only the ~10k-submit AR loop
   loses. That contrast isolates the cause to the *submit pattern*.

## The fix: collapse the submits

Both changes use ncnn's public `Extractor::extract(blob, VkMat&, VkCompute&)`
overload, which **records into a caller-owned `VkCompute` without submitting**,
so many outputs can be batched into a single `submit_and_wait()`.

**1. Batch the KV-cache extraction (`KvCacheNet::forward`).**
Extract `out0` and every cache output as `VkMat` into one shared `VkCompute`,
`record_download` them all, then submit **once**. `57 → 1` submit per talker
step. The cache still lands in CPU `Mat`s, so nothing else in the pipeline
changes.

```
generation 290 ms/frame → 164 ms/frame ; Vulkan RTF 3.95 → 2.4 (now beats CPU)
```

**2. Fuse the predictor forward + head (`KvCacheNet::forward_head`).**
In the MTP loop, `head[s]` is always applied to the hidden produced by the
`forward` at step `s`, so the two fuse. Keep the hidden as a `VkMat` (never
downloaded), feed it cross-net into the head's extractor sharing the same
`VkCompute`, and download only the logits + cache. `30 → 15` submits per
predictor frame.

```
Vulkan RTF 2.4 → 2.25
```

Both preserve token-exact parity: **0/704** (x-vector) and **0/656**
(CustomVoice) on Vulkan, CPU `ctest` 4/4 unchanged.

## Result

| tier | RTF | vs PyTorch fp32 CPU (10.5) |
| --- | --- | --- |
| ncnn CPU fp32 (8 threads) | 2.5 | ~4× |
| **ncnn Vulkan fp32 (fixed)** | **2.25** | **~4.7×** |

After the fix the AR loop is ~tied CPU/GPU (~156 ms/frame each) and the codec
wins on GPU, so Vulkan is slightly faster overall. 2.25 is close to the
practical floor for this architecture on ncnn Vulkan: the remaining cost is the
15 *sequential* MTP forwards/frame (each still one submit — they are
data-dependent through a CPU-side argmax, so they can't be batched without
moving argmax onto the GPU), and the batch-1 GEMV workload is memory-bandwidth-
bound, capping any GPU win at roughly the GPU/CPU bandwidth ratio (~3× here),
most of which the per-submit latency still eats.

## Reusable guidance for ncnn Vulkan autoregressive decode

- **Never extract blob-by-blob in a hot loop.** Each CPU-`Mat` `extract` is its
  own `VkCompute` + blocking `submit_and_wait`. Use
  `extract(blob, VkMat&, cmd)` with one caller-owned `VkCompute` and submit once
  per step.
- **Keep intermediates that only feed the next op on the GPU** (as `VkMat`), and
  download only what the host actually needs (logits for sampling, cache for the
  next step). A `VkMat` from one net can be fed as input to another net's
  extractor recording into the *same* `VkCompute`.
- **Profile per-step, not just end-to-end.** Flat-vs-growing per-step time
  distinguishes launch-bound from transfer-bound; a one-shot control stage
  (here, the codec) isolates submit overhead from raw GPU capability.
- **fp16 is not a free lever here.** We tested fp16 storage-only (fp32
  accumulate): it still diverges (703/704) *and* is slower (fp16→fp32 unpack
  overhead dominates the tiny per-step GEMVs). fp32 stays.

## Relation to known ncnn behavior

The general "ncnn Vulkan is slower than CPU" observation is documented upstream
(issue [#2743](https://github.com/Tencent/ncnn/issues/2743); the official
[FAQ-ncnn-vulkan](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan) notes the
Vulkan path "is far from the preferred state"), and it is not ncnn-specific
(e.g. llama.cpp [#20603](https://github.com/ggml-org/llama.cpp/issues/20603)).
ncnn's KV cache is also known to be concat + full copy each step
([#6512](https://github.com/Tencent/ncnn/issues/6512)), with a persistent
KV-cache mode still WIP and CPU-only
([#6776](https://github.com/Tencent/ncnn/pull/6776)). But the specific
mechanism here — an AR decode loop paying one blocking submit per extracted
blob, and the caller-side batching that removes it — does not appear to be
described in any existing issue, and the structural remedy (record-once /
command-buffer reuse, i.e. a CUDA-graph equivalent) is not in ncnn's public API.
