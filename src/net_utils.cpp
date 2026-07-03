#include "net_utils.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif

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
    // fp16 storage only (halve weight bandwidth) with fp32 accumulate — test
    // whether it keeps token parity while cutting GPU memory traffic.
    if (const char* h = getenv("Q3TTS_FP16_STORAGE"); h && h[0] == '1') {
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = false;
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
    if (const char* h = getenv("Q3TTS_FP16_STORAGE"); h && h[0] == '1') {
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;
        net_.opt.use_fp16_arithmetic = false;
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
#if NCNN_VULKAN
    if (net_.opt.use_vulkan_compute) {
        // Batched GPU extraction: record hidden + all KV-cache outputs into ONE
        // command buffer and submit once, instead of the default CPU-Mat extract
        // path which does a separate VkCompute + submit_and_wait per blob
        // (1 + 2*n_layers blocking submits per step). The AR loop is latency-
        // bound by the number of submits, so collapsing them is the win.
        // Cache still downloads to CPU Mats — pipeline semantics unchanged.
        const ncnn::VulkanDevice* vkdev = net_.vulkan_device();
        ncnn::VkAllocator* blob_alloc = vkdev->acquire_blob_allocator();
        ncnn::VkAllocator* staging_alloc = vkdev->acquire_staging_allocator();
        ex.set_blob_vkallocator(blob_alloc);
        ex.set_workspace_vkallocator(blob_alloc);
        ex.set_staging_vkallocator(staging_alloc);

        ncnn::Option dl = net_.opt;
        dl.blob_vkallocator = blob_alloc;
        dl.workspace_vkallocator = blob_alloc;
        dl.staging_vkallocator = staging_alloc;

        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat hidden_gpu;
        std::vector<ncnn::VkMat> ck(n_layers_), cv(n_layers_);
        int ret = ex.extract("out0", hidden_gpu, cmd);
        for (int i = 0; ret == 0 && i < n_layers_; i++) {
            snprintf(name, sizeof(name), "cache_k_out_%d", i);
            ret = ex.extract(name, ck[i], cmd);
            if (ret != 0) break;
            snprintf(name, sizeof(name), "cache_v_out_%d", i);
            ret = ex.extract(name, cv[i], cmd);
        }
        if (ret == 0) {
            cmd.record_download(hidden_gpu, hidden, dl);
            for (int i = 0; i < n_layers_; i++) {
                cmd.record_download(ck[i], caches_[i].first, dl);
                cmd.record_download(cv[i], caches_[i].second, dl);
            }
            ret = cmd.submit_and_wait();
        }
        vkdev->reclaim_blob_allocator(blob_alloc);
        vkdev->reclaim_staging_allocator(staging_alloc);
        if (ret != 0)
            throw std::runtime_error("KvCacheNet: vk batched extract failed");
        past_ += T;
        return hidden;
    }
#endif
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

ncnn::Mat KvCacheNet::forward_head(const ncnn::Mat& embeds, ncnn::Net& head,
                                   const char* head_out) {
#if NCNN_VULKAN
    if (net_.opt.use_vulkan_compute) {
        const int T = embeds.h;
        ncnn::Mat mask;
        if (past_ == 0) {
            mask.create(T, T);
            mask.fill(0.0f);
            for (int i = 0; i < T; i++) {
                float* p = mask.row(i);
                for (int j = i + 1; j < T; j++) p[j] = -INFINITY;
            }
        } else {
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

        const ncnn::VulkanDevice* vkdev = net_.vulkan_device();
        ncnn::VkAllocator* blob_alloc = vkdev->acquire_blob_allocator();
        ncnn::VkAllocator* staging_alloc = vkdev->acquire_staging_allocator();
        ex.set_blob_vkallocator(blob_alloc);
        ex.set_workspace_vkallocator(blob_alloc);
        ex.set_staging_vkallocator(staging_alloc);

        ncnn::Extractor hex = head.create_extractor();
        hex.set_blob_vkallocator(blob_alloc);
        hex.set_workspace_vkallocator(blob_alloc);
        hex.set_staging_vkallocator(staging_alloc);

        ncnn::Option dl = net_.opt;
        dl.blob_vkallocator = blob_alloc;
        dl.workspace_vkallocator = blob_alloc;
        dl.staging_vkallocator = staging_alloc;

        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat hidden_gpu, logits_gpu;
        std::vector<ncnn::VkMat> ck(n_layers_), cv(n_layers_);
        // forward records into cmd; hidden stays on GPU and feeds the head
        // directly (never downloaded) — only logits + cache come back.
        int ret = ex.extract("out0", hidden_gpu, cmd);
        for (int i = 0; ret == 0 && i < n_layers_; i++) {
            snprintf(name, sizeof(name), "cache_k_out_%d", i);
            ret = ex.extract(name, ck[i], cmd);
            if (ret != 0) break;
            snprintf(name, sizeof(name), "cache_v_out_%d", i);
            ret = ex.extract(name, cv[i], cmd);
        }
        if (ret == 0) {
            hex.input("in0", hidden_gpu);
            ret = hex.extract(head_out, logits_gpu, cmd);
        }
        ncnn::Mat logits;
        if (ret == 0) {
            cmd.record_download(logits_gpu, logits, dl);
            for (int i = 0; i < n_layers_; i++) {
                cmd.record_download(ck[i], caches_[i].first, dl);
                cmd.record_download(cv[i], caches_[i].second, dl);
            }
            ret = cmd.submit_and_wait();
        }
        vkdev->reclaim_blob_allocator(blob_alloc);
        vkdev->reclaim_staging_allocator(staging_alloc);
        if (ret != 0)
            throw std::runtime_error("KvCacheNet: vk fused forward_head failed");
        past_ += T;
        ncnn::Mat last(logits.w, 1);
        memcpy(last.row(0), logits.row(logits.h - 1), logits.w * sizeof(float));
        return last;
    }
#endif
    // CPU: no submit overhead, reuse the verified forward() then run the head.
    ncnn::Mat hidden = forward(embeds);
    ncnn::Mat last(hidden.w, 1);
    memcpy(last.row(0), hidden.row(hidden.h - 1), hidden.w * sizeof(float));
    ncnn::Extractor hex = head.create_extractor();
    hex.input("in0", last);
    ncnn::Mat logits;
    if (hex.extract(head_out, logits) != 0)
        throw std::runtime_error("KvCacheNet: head extract failed");
    return logits;
}

}  // namespace q3tts
