// C++ plumbing smoke test: run the exported talker (prefill + one cached step)
// and the text-embed net, compare against vectors produced by the validated
// python-ncnn pipeline (export/gen_cpp_testdata.py). Same libncnn, same graphs
// => differences here mean OUR plumbing (Mat layout / mask / rope / cache) is
// wrong, not the model.
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "net_utils.h"
#include "npy.h"

using namespace q3tts;

static double max_abs_diff(const ncnn::Mat& got, const NpyArray& ref) {
    const int rows = got.h, cols = got.w;
    if ((size_t)rows * cols != ref.numel()) {
        fprintf(stderr, "shape mismatch: got (%d,%d) vs ref %zu elems\n", rows, cols, ref.numel());
        return 1e30;
    }
    double worst = 0.0;
    for (int r = 0; r < rows; r++) {
        const float* g = got.row(r);
        const float* e = ref.f32() + (size_t)r * cols;
        for (int c = 0; c < cols; c++)
            worst = std::max(worst, (double)std::fabs(g[c] - e[c]));
    }
    return worst;
}

int main(int argc, char** argv) {
    const std::string models = argc > 1 ? argv[1] : "../qwen3-tts-ncnn-work/ncnn_models";
    const std::string data = argc > 2 ? argv[2] : "tests/data";

    int fails = 0;

    // ---- talker prefill + cached step ----
    KvCacheNet talker;
    talker.load(models + "/talker_decoder.ncnn.param", models + "/talker_decoder.ncnn.bin",
                /*n_layers=*/28, /*head_dim=*/128, /*theta=*/1e6);

    NpyArray in = load_npy(data + "/talker_in.npy");
    ncnn::Mat embeds(1024, in.shape[0]);
    memcpy(embeds.data, in.f32(), in.numel() * 4);

    ncnn::Mat prefill = talker.forward(embeds);
    // tolerance note: reference vectors come from the pip ncnn wheel; a locally
    // built libncnn uses different (equally valid) kernel blocking, so expect
    // ~1e-3 absolute drift over 28 layers. E2E argmax parity is the real gate.
    double d1 = max_abs_diff(prefill, load_npy(data + "/talker_prefill.npy"));
    printf("talker prefill : max_abs %.3e %s\n", d1, d1 < 5e-3 ? "[PASS]" : "[FAIL]");
    fails += d1 >= 5e-3;

    NpyArray sin_ = load_npy(data + "/talker_step_in.npy");
    ncnn::Mat semb(1024, 1);
    memcpy(semb.data, sin_.f32(), sin_.numel() * 4);
    ncnn::Mat step = talker.forward(semb);
    double d2 = max_abs_diff(step, load_npy(data + "/talker_step.npy"));
    printf("talker step    : max_abs %.3e %s\n", d2, d2 < 5e-3 ? "[PASS]" : "[FAIL]");
    fails += d2 >= 5e-3;

    // ---- text embed (id input convention) ----
    SimpleNet text_embed;
    text_embed.load(models + "/talker_text_embed.ncnn.param", models + "/talker_text_embed.ncnn.bin");
    NpyArray ids = load_npy(data + "/ids.npy");
    std::vector<int> idv(ids.i32(), ids.i32() + ids.numel());
    ncnn::Mat t = text_embed.run_ids(idv);
    double d3 = max_abs_diff(t, load_npy(data + "/text_embed.npy"));
    printf("text embed     : max_abs %.3e %s\n", d3, d3 < 1e-5 ? "[PASS]" : "[FAIL]");
    fails += d3 >= 1e-5;

    printf("SMOKE: %s\n", fails == 0 ? "ALL PASS" : "FAILED");
    return fails == 0 ? 0 : 1;
}
