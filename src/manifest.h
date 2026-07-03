// Minimal reader for the model.json manifest (tools/convert/make_model_json.py).
// Purpose-built for that flat schema — string-scans "key": value pairs and the
// two name→id maps. Not a general JSON parser; the ncnn_llm PR variant uses
// nlohmann::json instead (docs/ncnn_llm_pr_plan.md).
#pragma once

#include <cstdio>
#include <map>
#include <string>

#include "tts_config.h"

namespace q3tts {

struct Manifest {
    TtsConfig cfg;                       // defaults overridden by the file
    std::map<std::string, int> spk_id;
    std::map<std::string, int> language_ids;
    bool loaded = false;
};

inline std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

// find `"key": <number>` anywhere after `from`; returns fallback if absent
inline double json_num(const std::string& s, const std::string& key, double fallback,
                       size_t from = 0) {
    const std::string pat = "\"" + key + "\":";
    size_t p = s.find(pat, from);
    if (p == std::string::npos) return fallback;
    p += pat.size();
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
    return atof(s.c_str() + p);
}

// parse a flat {"name": int, ...} object that follows `"section":`
inline void json_int_map(const std::string& s, const std::string& section,
                         std::map<std::string, int>& out) {
    size_t p = s.find("\"" + section + "\":");
    if (p == std::string::npos) return;
    size_t open = s.find('{', p);
    size_t close = s.find('}', open);
    if (open == std::string::npos || close == std::string::npos) return;
    size_t i = open + 1;
    while (i < close) {
        size_t k0 = s.find('"', i);
        if (k0 == std::string::npos || k0 > close) break;
        size_t k1 = s.find('"', k0 + 1);
        size_t colon = s.find(':', k1);
        out[s.substr(k0 + 1, k1 - k0 - 1)] = atoi(s.c_str() + colon + 1);
        i = s.find(',', colon);
        if (i == std::string::npos || i > close) break;
        i++;
    }
}

inline Manifest load_manifest(const std::string& model_dir) {
    Manifest m;
    const std::string s = read_file(model_dir + "/model.json");
    if (s.empty()) return m;  // fall back to built-in defaults
    TtsConfig& c = m.cfg;

    c.talker_theta = json_num(s, "theta", c.talker_theta);   // first theta = talker rope
    c.head_dim = (int)json_num(s, "head_dim", c.head_dim);
    c.codec_eos = (int)json_num(s, "codec_eos", c.codec_eos);
    c.codec_pad = (int)json_num(s, "codec_pad", c.codec_pad);
    c.codec_bos = (int)json_num(s, "codec_bos", c.codec_bos);
    c.codec_think = (int)json_num(s, "codec_think", c.codec_think);
    c.codec_nothink = (int)json_num(s, "codec_nothink", c.codec_nothink);
    c.codec_think_bos = (int)json_num(s, "codec_think_bos", c.codec_think_bos);
    c.codec_think_eos = (int)json_num(s, "codec_think_eos", c.codec_think_eos);
    c.tts_bos = (int)json_num(s, "tts_bos", c.tts_bos);
    c.tts_eos = (int)json_num(s, "tts_eos", c.tts_eos);
    c.tts_pad = (int)json_num(s, "tts_pad", c.tts_pad);
    c.talker_layers = (int)json_num(s, "talker_layers", c.talker_layers);
    c.pred_layers = (int)json_num(s, "pred_layers", c.pred_layers);
    c.hidden = (int)json_num(s, "hidden", c.hidden);
    c.codec_vocab = (int)json_num(s, "codec_vocab", c.codec_vocab);
    c.pred_vocab = (int)json_num(s, "pred_vocab", c.pred_vocab);
    c.num_code_groups = (int)json_num(s, "num_code_groups", c.num_code_groups);
    c.pred_theta = json_num(s, "pred_rope_theta", c.pred_theta);
    c.max_new_tokens = (int)json_num(s, "max_new_tokens", c.max_new_tokens);
    c.repetition_penalty = (float)json_num(s, "repetition_penalty", c.repetition_penalty);
    c.min_new_tokens = (int)json_num(s, "min_new_tokens", c.min_new_tokens);

    json_int_map(s, "spk_id", m.spk_id);
    json_int_map(s, "language_ids", m.language_ids);
    m.loaded = true;
    return m;
}

}  // namespace q3tts
