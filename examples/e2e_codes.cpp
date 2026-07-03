// M3 acceptance (tokenizer-less): run the full C++ pipeline from pre-tokenized
// text ids + x-vector, compare codec tokens against the PyTorch golden
// reference token-for-token, then decode and report wav stats.
//
// Inputs (tests/data, from export/gen_e2e_inputs.py):
//   e2e_text_ids.npy  (N,)    tokenized assistant template
//   e2e_xvec.npy      (1024,) speaker embedding
//   e2e_codes.npy     (T,16)  golden codec tokens (PyTorch fp32 greedy)
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "npy.h"
#include "tts_pipeline.h"

using namespace q3tts;

int main(int argc, char** argv) {
    const std::string models = argc > 1 ? argv[1] : "models";
    const std::string data = argc > 2 ? argv[2] : "tests/data";
    const int lang_id = argc > 3 ? atoi(argv[3]) : 2055;  // chinese

    NpyArray ids_a = load_npy(data + "/e2e_text_ids.npy");
    NpyArray xvec = load_npy(data + "/e2e_xvec.npy");
    NpyArray golden = load_npy(data + "/e2e_codes.npy");
    std::vector<int> ids(ids_a.i32(), ids_a.i32() + ids_a.numel());
    const int GT = golden.shape[0];

    TtsPipeline tts;
    tts.load(models);

    ncnn::Mat prompt, trailing;
    tts.build_prompt_xvec(ids, xvec.f32(), lang_id, prompt, trailing);
    printf("prompt %d x %d | trailing %d\n", prompt.h, prompt.w, trailing.h);

    // golden may itself be capped at max_new_tokens; cap identically for comparison
    std::vector<Frame> frames = tts.generate_greedy(prompt, trailing, GT);
    printf("generated %zu frames | golden %d\n", frames.size(), GT);

    int mismatch = 0;
    const int n = std::min((int)frames.size(), GT);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < 16; k++)
            if (frames[t].codes[k] != golden.i32()[t * 16 + k]) mismatch++;
    const bool exact = (int)frames.size() == GT && mismatch == 0;
    printf("token match: %d/%d mismatches | length %s\n", mismatch, n * 16,
           (int)frames.size() == GT ? "OK" : "DIFFERS");

    std::vector<float> wav = tts.decode(frames);
    printf("decoded %zu samples (%.2f s)\n", wav.size(), wav.size() / 24000.0);

    // write raw pcm f32 for inspection (sox -t raw -r 24000 -e float -b 32 ...)
    FILE* fp = fopen((data + "/e2e_cpp.f32").c_str(), "wb");
    fwrite(wav.data(), 4, wav.size(), fp);
    fclose(fp);

    printf("C++ E2E: %s\n", exact ? "PASS" : "FAIL");
    return exact ? 0 : 1;
}
