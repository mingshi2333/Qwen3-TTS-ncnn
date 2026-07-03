// Qwen3-TTS-12Hz-0.6B-Base constants, verified against the checkpoint
// config.json (NOT the python code defaults, which disagree!).
// TODO(M4): load from a model.json manifest instead (export/make_model_json.py).
#pragma once

namespace q3tts {

struct TtsConfig {
    // talker
    int talker_layers = 28;
    int head_dim = 128;
    double talker_theta = 1e6;
    int hidden = 1024;
    int codec_vocab = 3072;

    // codec special ids (talker vocabulary)
    int codec_eos = 2150;
    int codec_pad = 2148;
    int codec_bos = 2149;
    int codec_think = 2154;
    int codec_nothink = 2155;
    int codec_think_bos = 2156;
    int codec_think_eos = 2157;

    // text special ids (Qwen2 vocabulary)
    int tts_bos = 151672;
    int tts_eos = 151673;
    int tts_pad = 151671;

    // predictor (MTP)
    int pred_layers = 5;
    double pred_theta = 1e6;
    int num_code_groups = 16;
    int pred_vocab = 2048;

    // sampling (generation_config.json)
    float repetition_penalty = 1.05f;
    int min_new_tokens = 2;
    int max_new_tokens = 8192;

    // codec decoder
    int codec_window = 72;
    int codec_head_dim = 64;
    double codec_theta = 1e4;
    int upsample = 1920;
    int sample_rate = 24000;
};

}  // namespace q3tts
