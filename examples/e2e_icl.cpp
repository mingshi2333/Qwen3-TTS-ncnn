// M4.3 acceptance: ICL voice clone, C++ vs PyTorch golden (tokenizer-less).
// Inputs from ref_e2e_icl.py: icl_{text_ids,ref_ids,ref_codes,xvec,codes}.npy
// Also verifies encode_reference (mimi+split-RVQ) against the torch ref codes.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "npy.h"
#include "tts_pipeline.h"
#include "wav_io.h"

using namespace q3tts;

int main(int argc, char** argv) {
    const std::string models = argc > 1 ? argv[1] : "models";
    const std::string data = argc > 2 ? argv[2] : "tests/data";
    const std::string ref_wav_path = argc > 3 ? argv[3] : "";
    const int lang_id = argc > 4 ? atoi(argv[4]) : 2055;

    NpyArray tid = load_npy(data + "/icl_text_ids.npy");
    NpyArray rid = load_npy(data + "/icl_ref_ids.npy");
    NpyArray rc = load_npy(data + "/icl_ref_codes.npy");
    NpyArray xv = load_npy(data + "/icl_xvec.npy");
    NpyArray golden = load_npy(data + "/icl_codes.npy");

    std::vector<int> text_ids(tid.i32(), tid.i32() + tid.numel());
    std::vector<int> ref_ids(rid.i32(), rid.i32() + rid.numel());
    const int Tr = rc.shape[0], GT = golden.shape[0];
    std::vector<Frame> ref_codes(Tr);
    for (int t = 0; t < Tr; t++)
        for (int k = 0; k < 16; k++)
            ref_codes[t].codes[k] = rc.i32()[t * 16 + k];

    TtsPipeline tts;
    tts.load(models);

    int fails = 0;

    // 1) C++ mimi + split-RVQ vs CPU-torch ref codes (same numeric domain;
    //    GPU-torch differs from CPU-torch by ~7% near-tie RVQ flips)
    if (!ref_wav_path.empty()) {
        NpyArray rc_cpu = load_npy(data + "/icl_ref_codes_cpu.npy");
        int sr = 0;
        std::vector<float> wav = read_wav(ref_wav_path, sr);
        std::vector<Frame> got = tts.encode_reference(wav);
        int mism = 0;
        const int n = std::min((int)got.size(), rc_cpu.shape[0]);
        for (int t = 0; t < n; t++)
            for (int k = 0; k < 16; k++)
                if (got[t].codes[k] != rc_cpu.i32()[t * 16 + k]) mism++;
        printf("ref codes vs CPU-torch: %d/%d frames, %d/%d token mismatches %s\n", (int)got.size(),
               rc_cpu.shape[0], mism, n * 16, (mism == 0 && (int)got.size() == rc_cpu.shape[0]) ? "[PASS]" : "[FAIL]");
        fails += !(mism == 0 && (int)got.size() == rc_cpu.shape[0]);
    }

    // 2) ICL generation vs golden
    ncnn::Mat prompt, trailing;
    tts.build_prompt_icl(text_ids, ref_ids, ref_codes, xv.f32(), lang_id, prompt, trailing);
    printf("prompt %d rows | trailing %d\n", prompt.h, trailing.h);
    // golden was capped at max_new_tokens; cap C++ identically for the comparison
    std::vector<Frame> frames = tts.generate_greedy(prompt, trailing, GT);
    int mism = 0;
    const int n = std::min((int)frames.size(), GT);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < 16; k++)
            if (frames[t].codes[k] != golden.i32()[t * 16 + k]) mism++;
    printf("gen codes: %zu/%d frames, %d/%d token mismatches %s\n", frames.size(), GT,
           mism, n * 16, (mism == 0 && (int)frames.size() == GT) ? "[PASS]" : "[FAIL]");
    fails += !(mism == 0 && (int)frames.size() == GT);

    std::vector<float> wav = tts.decode_icl(ref_codes, frames);
    write_wav(data + "/icl_cpp.wav", wav, 24000);
    printf("decoded %zu samples -> %s/icl_cpp.wav\n", wav.size(), data.c_str());

    printf("ICL E2E: %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
