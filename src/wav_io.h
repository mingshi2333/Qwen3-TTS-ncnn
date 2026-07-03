// Minimal 16-bit PCM mono WAV writer.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace q3tts {

inline void write_wav(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);

    const uint32_t data_size = (uint32_t)samples.size() * 2;
    const uint32_t byte_rate = (uint32_t)sample_rate * 2;
    uint8_t hdr[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E',
                       'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 1, 0};
    auto put32 = [&](int off, uint32_t v) { memcpy(hdr + off, &v, 4); };
    auto put16 = [&](int off, uint16_t v) { memcpy(hdr + off, &v, 2); };
    put32(4, 36 + data_size);
    put32(24, (uint32_t)sample_rate);
    put32(28, byte_rate);
    put16(32, 2);   // block align
    put16(34, 16);  // bits
    memcpy(hdr + 36, "data", 4);
    put32(40, data_size);
    fwrite(hdr, 1, 44, f);

    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); i++)
        pcm[i] = (int16_t)std::lround(std::clamp(samples[i], -1.0f, 1.0f) * 32767.0f);
    fwrite(pcm.data(), 2, pcm.size(), f);
    fclose(f);
}


// 16-bit PCM WAV reader (mono or first channel of stereo).
inline std::vector<float> read_wav(const std::string& path, int& sample_rate) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    uint8_t hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4))
        throw std::runtime_error("not a wav: " + path);
    uint16_t channels = 1, bits = 16, audio_fmt = 1;
    uint32_t sr = 0;
    std::vector<float> out;
    uint8_t ck[8];
    while (fread(ck, 1, 8, f) == 8) {
        uint32_t sz;
        memcpy(&sz, ck + 4, 4);
        if (!memcmp(ck, "fmt ", 4)) {
            uint8_t fmt[16];
            fread(fmt, 1, 16, f);
            memcpy(&audio_fmt, fmt, 2);
            memcpy(&channels, fmt + 2, 2);
            memcpy(&sr, fmt + 4, 4);
            memcpy(&bits, fmt + 14, 2);
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(ck, "data", 4)) {
            if (audio_fmt == 3 && bits == 32) {          // IEEE float
                std::vector<float> pcm(sz / 4);
                fread(pcm.data(), 4, pcm.size(), f);
                out.resize(pcm.size() / channels);
                for (size_t i = 0; i < out.size(); i++)
                    out[i] = pcm[i * channels];
            } else if (audio_fmt == 1 && bits == 16) {   // PCM16
                std::vector<int16_t> pcm(sz / 2);
                fread(pcm.data(), 2, pcm.size(), f);
                out.resize(pcm.size() / channels);
                for (size_t i = 0; i < out.size(); i++)
                    out[i] = pcm[i * channels] / 32768.0f;
            } else {
                throw std::runtime_error("unsupported wav format");
            }
            break;
        } else {
            fseek(f, sz + (sz & 1), SEEK_CUR);
        }
    }
    fclose(f);
    sample_rate = (int)sr;
    return out;
}

}  // namespace q3tts
