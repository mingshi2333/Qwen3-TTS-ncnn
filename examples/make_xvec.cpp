// Extract a 1024-d x-vector from a 24 kHz wav using the exported ECAPA net.
// With a reference .npy (torch-computed) it doubles as the M4.1 parity test.
//
// Usage: make_xvec <models_dir> <in.wav> <out.npy-raw> [ref.npy]
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mel.h"
#include "net_utils.h"
#include "npy.h"
#include "wav_io.h"

using namespace q3tts;

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <models_dir> <in.wav> <out.f32> [ref.npy]\n", argv[0]);
        return 1;
    }
    int sr = 0;
    std::vector<float> wav = read_wav(argv[2], sr);
    if (sr != 24000) {
        fprintf(stderr, "expect 24 kHz input, got %d (resampling lands later)\n", sr);
        return 1;
    }

    int frames = 0;
    std::vector<float> mel = mel_spectrogram_24k(wav, frames);
    printf("wav %zu samples -> mel %d x 128\n", wav.size(), frames);

    SimpleNet spk;
    const std::string dir = argv[1];
    spk.load(dir + "/speaker_encoder.ncnn.param", dir + "/speaker_encoder.ncnn.bin");
    ncnn::Mat m(128, frames);
    memcpy(m.data, mel.data(), mel.size() * sizeof(float));
    ncnn::Mat xv = spk.run(m);
    const float* x = xv;
    const size_t n = (size_t)xv.w * xv.h * xv.c;

    FILE* f = fopen(argv[3], "wb");
    fwrite(x, 4, n, f);
    fclose(f);
    printf("x-vector dim %zu -> %s\n", n, argv[3]);

    if (argc > 4) {
        NpyArray ref = load_npy(argv[4]);
        double dot = 0, na = 0, nb = 0, worst = 0;
        for (size_t i = 0; i < n; i++) {
            dot += (double)x[i] * ref.f32()[i];
            na += (double)x[i] * x[i];
            nb += (double)ref.f32()[i] * ref.f32()[i];
            worst = std::max(worst, (double)std::fabs(x[i] - ref.f32()[i]));
        }
        const double cos_sim = dot / (std::sqrt(na) * std::sqrt(nb));
        printf("vs torch ref: cos_sim %.7f | max_abs %.3e | %s\n", cos_sim, worst,
               cos_sim > 0.9999 ? "PASS" : "FAIL");
        return cos_sim > 0.9999 ? 0 : 1;
    }
    return 0;
}
