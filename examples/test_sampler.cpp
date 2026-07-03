// Sampler distribution parity vs HF processors+warpers (gen_sampler_testdata.py).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <vector>

#include "sampler.h"

using namespace q3tts;

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "tests/data/sampler_test.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }

    uint32_t n;
    f.read((char*)&n, 4);
    int fails = 0;
    for (uint32_t c = 0; c < n; c++) {
        int32_t step, top_k;
        float temp, top_p;
        f.read((char*)&step, 4);
        f.read((char*)&temp, 4);
        f.read((char*)&top_k, 4);
        f.read((char*)&top_p, 4);
        uint32_t nh;
        f.read((char*)&nh, 4);
        std::vector<int32_t> hist(nh);
        f.read((char*)hist.data(), nh * 4);
        std::vector<float> logits(3072);
        f.read((char*)logits.data(), 3072 * 4);
        uint32_t nk;
        f.read((char*)&nk, 4);
        std::map<int, double> ref;
        for (uint32_t i = 0; i < nk; i++) {
            int32_t idx;
            float p;
            f.read((char*)&idx, 4);
            f.read((char*)&p, 4);
            ref[idx] = p;
        }

        Code0Sampler s(3072, 2150, 1.05f, 2);
        // seed the history via argmax on synthetic one-hot logits
        for (int t : hist) {
            std::vector<float> oh(3072, -1e30f);
            oh[t] = 0.0f;
            s.argmax(oh.data(), 100);
        }
        auto got = s.warped_probs(logits.data(), step, temp, top_k, top_p);
        double s2 = 0;
        for (auto& [i, p] : got) s2 += p;

        double worst = 0;
        bool keys_ok = got.size() == ref.size();
        for (auto& [i, p] : got) {
            auto it = ref.find(i);
            if (it == ref.end()) { keys_ok = false; break; }
            worst = std::max(worst, std::fabs(p / s2 - it->second));
        }
        if (!keys_ok || worst > 1e-5) {
            fails++;
            printf("case %u FAIL: keys %s, worst %.3e (kept %zu vs %zu)\n", c,
                   keys_ok ? "ok" : "MISMATCH", worst, got.size(), ref.size());
        }
    }
    printf("sampler parity: %u/%u cases %s\n", n - fails, n, fails == 0 ? "[PASS]" : "[FAIL]");
    return fails == 0 ? 0 : 1;
}
