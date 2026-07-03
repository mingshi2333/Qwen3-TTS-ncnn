#include "net_utils.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace q3tts {

void rope_tables(int pos_begin, int count, int head_dim, double theta,
                 ncnn::Mat& cos_out, ncnn::Mat& sin_out) {
    // ncnn RotaryEmbed convention (tests/test_rotaryembed.cpp): cache rows are
    // HALFDIM wide. (The CPU layer strides by the Mat's own row width so it
    // also tolerates full-dim rows — the Vulkan shader does not. Feed halfdim.)
    const int half = head_dim / 2;
    cos_out.create(half, count);
    sin_out.create(half, count);
    for (int r = 0; r < count; r++) {
        float* c = cos_out.row(r);
        float* s = sin_out.row(r);
        const double pos = (double)(pos_begin + r);
        for (int j = 0; j < half; j++) {
            // inv_freq computed in double, matching the python reference
            const double inv = 1.0 / std::pow(theta, (double)(2 * j) / head_dim);
            const double ang = pos * inv;
            c[j] = (float)std::cos(ang);
            s[j] = (float)std::sin(ang);
        }
    }
}

ncnn::Mat ids_mat(const std::vector<int>& ids) {
    ncnn::Mat m((int)ids.size());
    memcpy(m.data, ids.data(), ids.size() * sizeof(int32_t));
    return m;
}

void SimpleNet::load(const std::string& param, const std::string& bin) {
    // alignment-grade fp32: the exported graphs were validated fp32; fp16
    // kernels (enabled by default on capable CPUs) cost ~1e-3 drift which is
    // enough to flip an argmax. Performance tier can relax this later.
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    if (const char* t = getenv("Q3TTS_THREADS")) net.opt.num_threads = atoi(t);
    if (const char* h = getenv("Q3TTS_FP16"); h && h[0] == '1') {
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
    }
#if NCNN_VULKAN
    if (const char* v = getenv("Q3TTS_VULKAN"); v && v[0] == '1') {
        net.opt.use_vulkan_compute = true;
        net.set_vulkan_device(0);
    }
#endif
    if (net.load_param(param.c_str()) != 0 || net.load_model(bin.c_str()) != 0)
        throw std::runtime_error("failed to load " + param);
}

ncnn::Mat SimpleNet::run(const ncnn::Mat& in, const char* out_blob) const {
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract(out_blob, out) != 0)
        throw std::runtime_error(std::string("extract failed: ") + out_blob);
    return out;
}

ncnn::Mat SimpleNet::run_ids(const std::vector<int>& ids, const char* out_blob) const {
    return run(ids_mat(ids), out_blob);
}

void KvCacheNet::load(const std::string& param, const std::string& bin,
                      int n_layers, int head_dim, double theta) {
    // alignment-grade fp32: the exported graphs were validated fp32; fp16
    // kernels (enabled by default on capable CPUs) cost ~1e-3 drift which is
    // enough to flip an argmax. Performance tier can relax this later.
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;
    net_.opt.use_bf16_storage = false;
    if (const char* t = getenv("Q3TTS_THREADS")) net_.opt.num_threads = atoi(t);
    if (const char* h = getenv("Q3TTS_FP16"); h && h[0] == '1') {
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;
        net_.opt.use_fp16_arithmetic = true;
    }
#if NCNN_VULKAN
    if (const char* v = getenv("Q3TTS_VULKAN"); v && v[0] == '1') {
        net_.opt.use_vulkan_compute = true;
        net_.set_vulkan_device(0);
    }
#endif
    if (net_.load_param(param.c_str()) != 0 || net_.load_model(bin.c_str()) != 0)
        throw std::runtime_error("failed to load " + param);
    n_layers_ = n_layers;
    head_dim_ = head_dim;
    theta_ = theta;
    reset();
}

void KvCacheNet::reset() {
    past_ = 0;
    caches_.assign(n_layers_, {});
}

ncnn::Mat KvCacheNet::forward(const ncnn::Mat& embeds) {
    const int T = embeds.h;

    // 2D mask per ncnn SDPA convention (tests/test_sdpa.cpp); a 3D c==1 mask is
    // handled inconsistently across the CPU/FA/non-FA paths (upstream issue).
    ncnn::Mat mask;
    if (past_ == 0) {
        // causal prefill mask: -inf strictly above the diagonal
        mask.create(T, T);
        mask.fill(0.0f);
        for (int i = 0; i < T; i++) {
            float* p = mask.row(i);
            for (int j = i + 1; j < T; j++) p[j] = -INFINITY;
        }
    } else {
        // decode step: every past position (and self) is visible
        mask.create(past_ + T, T);
        mask.fill(0.0f);
    }

    ncnn::Mat cos, sin;
    rope_tables(past_, T, head_dim_, theta_, cos, sin);

    ncnn::Extractor ex = net_.create_extractor();
    ex.input("in0", embeds);
    ex.input("in1", mask);
    ex.input("in2", cos);
    ex.input("in3", sin);
    char name[64];
    if (past_ > 0) {
        for (int i = 0; i < n_layers_; i++) {
            snprintf(name, sizeof(name), "cache_k_in_%d", i);
            ex.input(name, caches_[i].first);
            snprintf(name, sizeof(name), "cache_v_in_%d", i);
            ex.input(name, caches_[i].second);
        }
    }

    ncnn::Mat hidden;
    if (ex.extract("out0", hidden) != 0)
        throw std::runtime_error("KvCacheNet: extract out0 failed");
    for (int i = 0; i < n_layers_; i++) {
        snprintf(name, sizeof(name), "cache_k_out_%d", i);
        if (ex.extract(name, caches_[i].first) != 0)
            throw std::runtime_error("KvCacheNet: cache extract failed");
        snprintf(name, sizeof(name), "cache_v_out_%d", i);
        if (ex.extract(name, caches_[i].second) != 0)
            throw std::runtime_error("KvCacheNet: cache extract failed");
    }

    past_ += T;
    return hidden;
}

}  // namespace q3tts
