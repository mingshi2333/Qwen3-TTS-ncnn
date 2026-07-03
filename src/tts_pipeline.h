// End-to-end Qwen3-TTS pipeline over the 8 exported ncnn graphs.
// This is a line-by-line port of export/test_e2e_greedy.py (the executable
// spec validated at 100% token parity against PyTorch).
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "net_utils.h"
#include "sampler.h"
#include "split_rvq.h"
#include "manifest.h"
#include "tts_config.h"

namespace q3tts {

struct Frame {
    int codes[16];
};

// generation_config.json defaults; do_sample=false gives the parity-grade path
struct GenOpts {
    bool do_sample = false;
    float temperature = 0.9f, top_p = 1.0f;
    int top_k = 50;
    bool sub_do_sample = false;
    float sub_temperature = 0.9f, sub_top_p = 1.0f;
    int sub_top_k = 50;
    uint64_t seed = 0;
};

class TtsPipeline {
public:
    // model_dir contains the 8 .ncnn.param/.bin pairs from export/
    void load(const std::string& model_dir);

    // Prompt construction for x-vector voice clone, streaming text mode
    // (modeling_qwen3_tts.py generate() lines 2124-2233, non-ICL branch).
    //   text_ids : tokenized "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    //   xvec     : 1024-dim speaker embedding
    //   lang_id  : talker codec language id (e.g. chinese 2055), or -1 for auto
    // Fills prompt embeddings and the per-step trailing text embeddings.
    void build_prompt_xvec(const std::vector<int>& text_ids, const float* xvec,
                           int lang_id, ncnn::Mat& prompt, ncnn::Mat& trailing);

    // AR generation (greedy or seeded sampling per opts).
    std::vector<Frame> generate(const ncnn::Mat& prompt, const ncnn::Mat& trailing,
                                int max_frames, const GenOpts& opts = {});
    std::vector<Frame> generate_greedy(const ncnn::Mat& prompt, const ncnn::Mat& trailing,
                                       int max_frames) {
        return generate(prompt, trailing, max_frames, {});
    }

    // codes -> 24 kHz waveform (single chunk; chunked_decode lands with M4)
    std::vector<float> decode(const std::vector<Frame>& frames);

    // ---- ICL voice clone ----
    // 24 kHz wav (length padded to a 1920 multiple internally) -> ref codes (T,16)
    std::vector<Frame> encode_reference(const std::vector<float>& wav24k);
    // prompt for ICL mode (modeling generate() ICL branch + generate_icl_prompt,
    // streaming): ref_ids = tokenized "<|im_start|>assistant\n{ref}<|im_end|>\n"
    void build_prompt_icl(const std::vector<int>& text_ids, const std::vector<int>& ref_ids,
                          const std::vector<Frame>& ref_codes, const float* xvec, int lang_id,
                          ncnn::Mat& prompt, ncnn::Mat& trailing);
    // decode(ref ++ gen) and cut the reference span (wrapper lines 612-631)
    std::vector<float> decode_icl(const std::vector<Frame>& ref_codes,
                                  const std::vector<Frame>& gen_frames);

    // CustomVoice built-in speaker, non-streaming text mode (the checkpoint's
    // default: modeling generate() lines 2199-2227). speaker row is the talker
    // codec-embedding of spk_id instead of an x-vector.
    void build_prompt_speaker(const std::vector<int>& text_ids, int spk_id, int lang_id,
                              ncnn::Mat& prompt, ncnn::Mat& trailing);

    TtsConfig cfg;
    Manifest manifest;  // spk_id / language_ids name tables (model.json)

private:
    // one frame of the MTP sub-model: 2-token prefill emits codebook-1 via
    // head[0], then 14 steps with table[s-1]/head[s]  (modeling lines 1277-1299)
    void predictor_frame(const float* past_hidden, int code0, int* rest_out,
                         const GenOpts& opts, std::mt19937_64* rng);

    // next talker input = talker_emb(code0) + sum_i pred_emb_i(code_{i+1})
    ncnn::Mat frame_sum_embed(const Frame& f);

    SimpleNet text_embed_, codec_embed_, codec_head_;
    ncnn::Mat tts_pad_embed_;  // cached, set by build_prompt_*
    SimpleNet pred_embeds_, pred_heads_, codec_dec_;
    SimpleNet mimi_;           // lazy: only loaded for ICL
    bool mimi_loaded_ = false;
    std::string model_dir_;
    KvCacheNet talker_;
    KvCacheNet predictor_;
    SplitRvq rvq_;
};

}  // namespace q3tts
