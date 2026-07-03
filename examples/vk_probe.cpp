// Vulkan-vs-CPU blob-level probe: run the same graph on both devices with the
// same inputs and report per-blob max_abs. Used to bisect broken Vulkan layers.
//
// Usage: vk_probe <param> <bin> <talker_in.npy> [blob ...]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "net.h"
#include "net_utils.h"
#include "npy.h"

using namespace q3tts;

static ncnn::Mat run_one(bool vulkan, const std::string& param, const std::string& bin,
                         const ncnn::Mat& embeds, const ncnn::Mat& mask, const ncnn::Mat& cos,
                         const ncnn::Mat& sin, const char* blob) {
    ncnn::Net net;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    if (vulkan) {
#if NCNN_VULKAN
        net.opt.use_vulkan_compute = true;
        net.set_vulkan_device(0);
#else
        throw std::runtime_error("this build has no Vulkan support (NCNN_VULKAN=OFF)");
#endif
    }
    if (net.load_param(param.c_str()) != 0 || net.load_model(bin.c_str()) != 0)
        throw std::runtime_error("load failed");
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", embeds);
    ex.input("in1", mask);
    ex.input("in2", cos);
    ex.input("in3", sin);
    ncnn::Mat out;
    if (ex.extract(blob, out) != 0)
        throw std::runtime_error(std::string("extract failed: ") + blob);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <param> <bin> <in.npy> [blob ...]\n", argv[0]);
        return 1;
    }
    NpyArray in = load_npy(argv[3]);
    const int T = in.shape[0];
    ncnn::Mat embeds(1024, T);
    memcpy(embeds.data, in.f32(), in.numel() * 4);

    ncnn::Mat mask(T, T);
    mask.fill(0.0f);
    for (int i = 0; i < T; i++) {
        float* mp = mask.row(i);
        for (int j = i + 1; j < T; j++) mp[j] = -INFINITY;
    }
    ncnn::Mat cos, sin;
    rope_tables(0, T, 128, 1e6, cos, sin);

    std::vector<const char*> blobs;
    for (int a = 4; a < argc; a++) blobs.push_back(argv[a]);
    if (blobs.empty()) blobs.push_back("out0");

    for (const char* b : blobs) {
        try {
            ncnn::Mat c = run_one(false, argv[1], argv[2], embeds, mask, cos, sin, b);
            ncnn::Mat v = run_one(true, argv[1], argv[2], embeds, mask, cos, sin, b);
            double worst = 0, cmax = 0;
            const size_t n = (size_t)c.total();
            const float* cp = (const float*)c.data;
            const float* vp = (const float*)v.data;
            for (size_t i = 0; i < n && i < (size_t)v.total(); i++) {
                worst = std::max(worst, (double)std::fabs(cp[i] - vp[i]));
                cmax = std::max(cmax, (double)std::fabs(cp[i]));
            }
            printf("blob %-6s cpu(%d,%d,%d) vk(%d,%d,%d): max_abs %.3e (cpu scale %.2e)\n",
                   b, c.c, c.h, c.w, v.c, v.h, v.w, worst, cmax);
        } catch (const std::exception& e) {
            printf("blob %-6s ERROR: %s\n", b, e.what());
        }
    }
    return 0;
}
