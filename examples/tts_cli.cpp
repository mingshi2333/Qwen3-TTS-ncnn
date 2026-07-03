// Text -> 24 kHz wav, fully in C++ (tokenizer + talker + MTP + codec on ncnn).
// Voice reference: a 24 kHz wav (x-vector computed on the fly) or a .npy x-vector.
//
// Usage:
//   tts_cli <models_dir> <voice> "<text>" <out.wav> [language=chinese] [seed]
// Voice: 24 kHz wav (x-vector clone), .npy x-vector, a speaker name from
// model.json (CustomVoice, e.g. serena), or a raw integer speaker id.
// Language: a name from model.json (chinese/english/...) or a raw id. Seed enables sampling.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "mel.h"
#include "npy.h"
#include "qwen_bpe.h"
#include "tts_pipeline.h"
#include "wav_io.h"

using namespace q3tts;

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <models_dir> <xvec.npy> \"<text>\" <out.wav> [lang_id]\n", argv[0]);
        return 1;
    }
    const std::string models = argv[1], xvec_path = argv[2], text = argv[3], out_path = argv[4];
    const std::string lang_arg = argc > 5 ? argv[5] : "chinese";
    GenOpts opts;
    if (argc > 6) {
        opts.do_sample = opts.sub_do_sample = true;
        opts.seed = strtoull(argv[6], nullptr, 10);
    }

    QwenBpe bpe;
    bpe.load(models + "/tokenizer.txt");
    const std::string templ =
        "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> ids = bpe.encode(templ);
    printf("tokens: %zu\n", ids.size());

    // built-in speaker mode: integer id or a name resolved via model.json
    TtsPipeline probe;  // manifest lookup needs the model dir only
    probe.manifest = load_manifest(models);
    const bool is_int = !xvec_path.empty() &&
                        xvec_path.find_first_not_of("0123456789") == std::string::npos;
    const bool is_name = probe.manifest.spk_id.count(xvec_path) > 0;
    auto resolve_lang = [&](const Manifest& mf) {
        if (!lang_arg.empty() && lang_arg.find_first_not_of("0123456789") == std::string::npos)
            return atoi(lang_arg.c_str());
        auto it = mf.language_ids.find(lang_arg);
        if (it == mf.language_ids.end()) { fprintf(stderr, "unknown language %s\n", lang_arg.c_str()); exit(1); }
        return it->second;
    };
    if (is_int || is_name) {
        TtsPipeline tts;
        tts.load(models);
        const int spk = is_name ? tts.manifest.spk_id.at(xvec_path) : atoi(xvec_path.c_str());
        const int lang_id = resolve_lang(tts.manifest);
        auto t0 = std::chrono::steady_clock::now();
        ncnn::Mat prompt, trailing;
        tts.build_prompt_speaker(ids, spk, lang_id, prompt, trailing);
        std::vector<Frame> frames = tts.generate(prompt, trailing, tts.cfg.max_new_tokens, opts);
        std::vector<float> wav = tts.decode(frames);
        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        write_wav(out_path, wav, tts.cfg.sample_rate);
        printf("%zu frames -> %.2f s audio in %.1f s (RTF %.2f) -> %s\n", frames.size(),
               wav.size() / 24000.0, dt, dt / (wav.size() / 24000.0), out_path.c_str());
        return 0;
    }

    std::vector<float> xvec_data;
    if (xvec_path.size() > 4 && xvec_path.substr(xvec_path.size() - 4) == ".wav") {
        int sr = 0;
        std::vector<float> ref = read_wav(xvec_path, sr);
        if (sr != 24000) { fprintf(stderr, "reference must be 24 kHz\n"); return 1; }
        int frames = 0;
        std::vector<float> mel = mel_spectrogram_24k(ref, frames);
        SimpleNet spk;
        spk.load(models + "/speaker_encoder.ncnn.param", models + "/speaker_encoder.ncnn.bin");
        ncnn::Mat m(128, frames);
        memcpy(m.data, mel.data(), mel.size() * sizeof(float));
        ncnn::Mat xv = spk.run(m);
        xvec_data.assign((const float*)xv, (const float*)xv + 1024);
        printf("x-vector from %s (%d mel frames)\n", xvec_path.c_str(), frames);
    } else {
        NpyArray a = load_npy(xvec_path);
        xvec_data.assign(a.f32(), a.f32() + a.numel());
    }

    TtsPipeline tts;
    tts.load(models);
    const int lang_id = resolve_lang(tts.manifest);

    auto t0 = std::chrono::steady_clock::now();
    ncnn::Mat prompt, trailing;
    tts.build_prompt_xvec(ids, xvec_data.data(), lang_id, prompt, trailing);
    std::vector<Frame> frames = tts.generate(prompt, trailing, tts.cfg.max_new_tokens, opts);
    std::vector<float> wav = tts.decode(frames);
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    write_wav(out_path, wav, tts.cfg.sample_rate);
    printf("%zu frames -> %.2f s audio in %.1f s (RTF %.2f) -> %s\n",
           frames.size(), wav.size() / 24000.0, dt, dt / (wav.size() / 24000.0), out_path.c_str());
    return 0;
}
