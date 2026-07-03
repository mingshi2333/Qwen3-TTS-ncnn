// CustomVoice acceptance: built-in speaker, non-streaming prompt, same-domain
// golden (CPU-torch). Inputs: cv_text_ids.npy, cv_codes.npy.
// Usage: e2e_cv <models_dir> <data_dir> <spk_id> [lang_id=2055]
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "npy.h"
#include "tts_pipeline.h"
#include "wav_io.h"

using namespace q3tts;

int main(int argc, char** argv) {
    const std::string models = argc > 1 ? argv[1] : "models-cv";
    const std::string data = argc > 2 ? argv[2] : "tests/data";
    const int spk_id = argc > 3 ? atoi(argv[3]) : 3066;  // serena
    const int lang_id = argc > 4 ? atoi(argv[4]) : 2055;

    NpyArray tid = load_npy(data + "/cv_text_ids.npy");
    NpyArray golden = load_npy(data + "/cv_codes.npy");
    std::vector<int> ids(tid.i32(), tid.i32() + tid.numel());
    const int GT = golden.shape[0];

    TtsPipeline tts;
    tts.load(models);

    ncnn::Mat prompt, trailing;
    tts.build_prompt_speaker(ids, spk_id, lang_id, prompt, trailing);
    printf("prompt %d rows | trailing %d\n", prompt.h, trailing.h);

    std::vector<Frame> frames = tts.generate_greedy(prompt, trailing, GT);
    int mism = 0;
    const int n = std::min((int)frames.size(), GT);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < 16; k++)
            if (frames[t].codes[k] != golden.i32()[t * 16 + k]) mism++;
    printf("gen: %zu/%d frames, %d/%d token mismatches\n", frames.size(), GT, mism, n * 16);

    std::vector<float> wav = tts.decode(frames);
    write_wav(data + "/cv_cpp.wav", wav, 24000);
    const bool ok = mism == 0 && (int)frames.size() == GT;
    printf("CUSTOMVOICE E2E: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
