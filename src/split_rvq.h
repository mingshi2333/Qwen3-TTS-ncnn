// Mimi Split-RVQ encoder-side quantizer (C++; the neural encoder runs in ncnn).
// Split semantics: the semantic level and the 15-level acoustic residual chain
// run IN PARALLEL on the same latent, each through its own 1x1 input_proj.
// Only encoder_valid_num_quantizers=16 levels are computed.
#pragma once

#include <cfloat>
#include <string>
#include <vector>

#include "npy.h"

namespace q3tts {

class SplitRvq {
public:
    void load(const std::string& models_dir) {
        sem_proj_ = load_npy(models_dir + "/mimi_sem_proj.npy");      // (256,512)
        sem_cb_ = load_npy(models_dir + "/mimi_sem_codebook.npy");    // (2048,256)
        ac_proj_ = load_npy(models_dir + "/mimi_ac_proj.npy");        // (256,512)
        ac_cbs_ = load_npy(models_dir + "/mimi_ac_codebooks.npy");    // (15,2048,256)
    }

    // latent: (512, T) row-major -> codes (T, 16)
    std::vector<int> encode(const float* latent, int T) const {
        const int D = 512, K = 256, N = 2048;
        std::vector<int> codes((size_t)T * 16);
        std::vector<float> z(K), r(K);
        for (int t = 0; t < T; t++) {
            // column t of latent
            auto project = [&](const NpyArray& proj, float* out) {
                for (int i = 0; i < K; i++) {
                    double acc = 0;
                    const float* w = proj.f32() + (size_t)i * D;
                    for (int d = 0; d < D; d++) acc += (double)w[d] * latent[(size_t)d * T + t];
                    out[i] = (float)acc;
                }
            };
            project(sem_proj_, z.data());
            codes[(size_t)t * 16] = nearest(z.data(), sem_cb_.f32(), N, K);

            project(ac_proj_, r.data());
            for (int lvl = 0; lvl < 15; lvl++) {
                const float* cb = ac_cbs_.f32() + (size_t)lvl * N * K;
                const int idx = nearest(r.data(), cb, N, K);
                codes[(size_t)t * 16 + 1 + lvl] = idx;
                const float* e = cb + (size_t)idx * K;
                for (int i = 0; i < K; i++) r[i] -= e[i];
            }
        }
        return codes;
    }

private:
    static int nearest(const float* v, const float* cb, int n, int k) {
        int best = 0;
        float best_d = FLT_MAX;
        for (int i = 0; i < n; i++) {
            const float* c = cb + (size_t)i * k;
            float d = 0;
            for (int j = 0; j < k; j++) {
                const float t = v[j] - c[j];
                d += t * t;
            }
            if (d < best_d) { best_d = d; best = i; }
        }
        return best;
    }

    NpyArray sem_proj_, sem_cb_, ac_proj_, ac_cbs_;
};

}  // namespace q3tts
