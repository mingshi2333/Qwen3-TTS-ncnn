// Core ncnn plumbing for the Qwen3-TTS runtime.
//
// Conventions (mirror export/test_e2e_greedy.py — the executable spec):
//  * float tensors (T, D)      -> ncnn::Mat(w=D, h=T)         [2D]
//  * masks (1, rows, cols)     -> ncnn::Mat(w=cols, h=rows, c=1) [3D]
//  * token ids                 -> 1D ncnn::Mat whose bits ARE int32
//    (ncnn Embed reinterprets input memory as int32; ids must be fed
//     directly to Embed without passing through any transform layer)
//  * KV cache blobs are named cache_k_in_{i}/cache_v_in_{i} and
//    cache_k_out_{i}/cache_v_out_{i} by export/add_kvcache.py; prefill
//    feeds no cache (empty past), each step feeds back the previous ones.
#pragma once

#include <string>
#include <vector>

#include "net.h"

namespace q3tts {

// Fill an ncnn Mat with rope cos/sin rows for absolute positions
// [pos_begin, pos_begin+count). Layout (w=head_dim, h=count).
void rope_tables(int pos_begin, int count, int head_dim, double theta,
                 ncnn::Mat& cos_out, ncnn::Mat& sin_out);

// int32-bit 1D Mat from ids (see Embed convention above).
ncnn::Mat ids_mat(const std::vector<int>& ids);

// Single-shot net: one float input -> one output (codec_head, ...).
// For nets with several outputs (predictor_heads) pass the blob name.
class SimpleNet {
public:
    void load(const std::string& param, const std::string& bin);
    // float (T,D) input
    ncnn::Mat run(const ncnn::Mat& in, const char* out_blob = "out0") const;
    // id-input variant (text_embed, codec_embed, predictor_embeds)
    ncnn::Mat run_ids(const std::vector<int>& ids, const char* out_blob = "out0") const;

    ncnn::Net net;
};

// Autoregressive decoder net with SDPA kv cache (talker / predictor).
class KvCacheNet {
public:
    void load(const std::string& param, const std::string& bin,
              int n_layers, int head_dim, double theta);

    void reset();

    // Run `embeds` (h=T, w=hidden) at positions past..past+T-1.
    // past==0 -> causal prefill; otherwise single/multi-token step with
    // an all-zero (T, past+T) mask. Returns hidden states (h=T, w=hidden)
    // and advances the internal cache.
    ncnn::Mat forward(const ncnn::Mat& embeds);

    // Fused forward + follow-on head for the MTP predictor step: run forward,
    // feed the resulting hidden's LAST row into `head` (a stateless SimpleNet,
    // input blob "in0"), and return head's `head_out` logits (1 row). On Vulkan
    // the whole thing is ONE submit — the hidden never leaves the GPU, only the
    // logits and updated cache download. Numerically identical to
    // forward() followed by a separate head run.
    ncnn::Mat forward_head(const ncnn::Mat& embeds, ncnn::Net& head, const char* head_out);

    int past() const { return past_; }

private:
    ncnn::Net net_;
    int n_layers_ = 0;
    int head_dim_ = 0;
    double theta_ = 0.0;
    int past_ = 0;
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> caches_;
};

}  // namespace q3tts
