// Performance benchmark with per-stage breakdown. Uses the e2e test inputs so
// the run is deterministic and token accuracy vs the golden can be reported
// alongside speed (relevant for the fp16 tier).
//
// Usage: bench <models_dir> <data_dir> [lang_id=2055]
// Env:   Q3TTS_THREADS=<n>  Q3TTS_FP16=1
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#endif

#include "mel.h"
#include "npy.h"
#include "tts_pipeline.h"
#include "wav_io.h"

using namespace q3tts;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// instrumented copy of the generation loop (pipeline internals timed from outside
// would need hooks; here we re-use pipeline pieces via its public API and time
// the aggregate stages)
int main(int argc, char** argv) {
    const std::string models = argc > 1 ? argv[1] : "models";
    const std::string data = argc > 2 ? argv[2] : "tests/data";
    const int lang_id = argc > 3 ? atoi(argv[3]) : 2055;

    NpyArray ids_a = load_npy(data + "/e2e_text_ids.npy");
    NpyArray xvec = load_npy(data + "/e2e_xvec.npy");
    NpyArray golden = load_npy(data + "/e2e_codes.npy");
    std::vector<int> ids(ids_a.i32(), ids_a.i32() + ids_a.numel());
    const int GT = golden.shape[0];

    auto t0 = clk::now();
    TtsPipeline tts;
    tts.load(models);
    printf("load           : %8.1f ms\n", ms_since(t0));

    // x-vector stage (mel front-end + ECAPA), measured on the clone reference
    {
        int sr = 0;
        std::vector<float> ref = read_wav(data + "/../../../qwen3-tts-ncnn-work/ref_clone_2.wav", sr);
        if (sr == 24000) {
            t0 = clk::now();
            int frames = 0;
            std::vector<float> mel = mel_spectrogram_24k(ref, frames);
            const double mel_ms = ms_since(t0);
            SimpleNet spk;
            spk.load(models + "/speaker_encoder.ncnn.param", models + "/speaker_encoder.ncnn.bin");
            ncnn::Mat m(128, frames);
            memcpy(m.data, mel.data(), mel.size() * sizeof(float));
            t0 = clk::now();
            spk.run(m);
            printf("x-vector       : mel %6.1f ms + ecapa %6.1f ms (%.2fs audio)\n",
                   mel_ms, ms_since(t0), ref.size() / 24000.0);
        }
    }

    ncnn::Mat prompt, trailing;
    t0 = clk::now();
    tts.build_prompt_xvec(ids, xvec.f32(), lang_id, prompt, trailing);
    printf("prompt build   : %8.1f ms (%d rows)\n", ms_since(t0), prompt.h);

    t0 = clk::now();
    std::vector<Frame> frames = tts.generate_greedy(prompt, trailing, GT + 8);
    const double gen_ms = ms_since(t0);

    t0 = clk::now();
    std::vector<float> wav = tts.decode(frames);
    const double dec_ms = ms_since(t0);

    int mismatch = 0;
    const int n = std::min((int)frames.size(), GT);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < 16; k++)
            if (frames[t].codes[k] != golden.i32()[t * 16 + k]) mismatch++;

#ifndef _WIN32
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#endif

    const double audio_s = wav.size() / 24000.0;
    printf("generation     : %8.1f ms (%zu frames, %.1f ms/frame)\n", gen_ms, frames.size(),
           gen_ms / std::max<size_t>(frames.size(), 1));
    printf("codec decode   : %8.1f ms (%.1fx realtime alone)\n", dec_ms, audio_s / (dec_ms / 1000.0));
    printf("audio          : %8.2f s\n", audio_s);
    printf("RTF            : %8.2f (generation+decode)\n", (gen_ms + dec_ms) / 1000.0 / audio_s);
    printf("token accuracy : %d/%d mismatches vs fp32 golden (len %zu vs %d)\n",
           mismatch, n * 16, frames.size(), GT);
#ifndef _WIN32
    printf("peak RSS       : %8.1f MB\n", ru.ru_maxrss / 1024.0);
#endif
    return 0;
}
