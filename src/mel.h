// Mel front-end for the speaker encoder — exact replica of
// modeling_qwen3_tts.mel_spectrogram (n_fft 1024, hop 256, win 1024,
// periodic hann, reflect pad (n_fft-hop)/2 = 384 both sides, center=False,
// |X| = sqrt(re^2+im^2+1e-9), librosa slaney mel (fmin 0, fmax 12000,
// norm='slaney'), log(clamp(x, 1e-5))).
#pragma once

#include <cmath>
#include <complex>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace q3tts {

// iterative radix-2 complex FFT, n must be a power of two
inline void fft_inplace(std::vector<std::complex<double>>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / (double)len;
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0);
            for (size_t k = 0; k < len / 2; k++) {
                std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

// librosa slaney mel filterbank (n_mels, n_fft/2+1), norm='slaney', htk=False
inline std::vector<float> slaney_mel_filters(int sr, int n_fft, int n_mels,
                                             double fmin, double fmax) {
    auto hz_to_mel = [](double hz) {
        const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0;
        const double min_log_mel = min_log_hz / f_sp;
        const double logstep = std::log(6.4) / 27.0;
        return hz < min_log_hz ? hz / f_sp : min_log_mel + std::log(hz / min_log_hz) / logstep;
    };
    auto mel_to_hz = [](double mel) {
        const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0;
        const double min_log_mel = min_log_hz / f_sp;
        const double logstep = std::log(6.4) / 27.0;
        return mel < min_log_mel ? mel * f_sp : min_log_hz * std::exp(logstep * (mel - min_log_mel));
    };
    const int n_bins = n_fft / 2 + 1;
    std::vector<double> pts(n_mels + 2);
    const double m0 = hz_to_mel(fmin), m1 = hz_to_mel(fmax);
    for (int i = 0; i < n_mels + 2; i++)
        pts[i] = mel_to_hz(m0 + (m1 - m0) * i / (n_mels + 1));

    std::vector<float> w((size_t)n_mels * n_bins, 0.0f);
    for (int m = 0; m < n_mels; m++) {
        const double lo = pts[m], mid = pts[m + 1], hi = pts[m + 2];
        const double norm = 2.0 / (hi - lo);  // slaney norm
        for (int k = 0; k < n_bins; k++) {
            const double f = (double)k * sr / n_fft;
            const double up = (f - lo) / (mid - lo);
            const double dn = (hi - f) / (hi - mid);
            const double v = std::max(0.0, std::min(up, dn));
            w[(size_t)m * n_bins + k] = (float)(v * norm);
        }
    }
    return w;
}

// wav (float, [-1,1]) -> mel (frames x n_mels), row-major
inline std::vector<float> mel_spectrogram_24k(const std::vector<float>& wav, int& frames_out,
                                              int n_fft = 1024, int hop = 256, int n_mels = 128,
                                              int sr = 24000, double fmax = 12000.0) {
    const int pad = (n_fft - hop) / 2;  // 384
    std::vector<double> y(wav.size() + 2 * pad);
    for (size_t i = 0; i < wav.size(); i++) y[pad + i] = wav[i];
    for (int i = 0; i < pad; i++) {                       // reflect (no edge repeat)
        y[pad - 1 - i] = wav[i + 1];
        y[pad + wav.size() + i] = wav[wav.size() - 2 - i];
    }

    std::vector<double> window(n_fft);                    // periodic hann
    for (int i = 0; i < n_fft; i++)
        window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / n_fft));

    const int n_bins = n_fft / 2 + 1;
    const int frames = 1 + ((int)y.size() - n_fft) / hop;
    frames_out = frames;
    static thread_local std::vector<float> mel_fb;
    if (mel_fb.empty()) mel_fb = slaney_mel_filters(sr, n_fft, n_mels, 0.0, fmax);

    std::vector<float> mel((size_t)frames * n_mels);
    std::vector<std::complex<double>> buf(n_fft);
    std::vector<double> mag(n_bins);
    for (int t = 0; t < frames; t++) {
        for (int i = 0; i < n_fft; i++)
            buf[i] = {y[(size_t)t * hop + i] * window[i], 0.0};
        fft_inplace(buf);
        for (int k = 0; k < n_bins; k++)
            mag[k] = std::sqrt(std::norm(buf[k]) + 1e-9);
        for (int m = 0; m < n_mels; m++) {
            double acc = 0.0;
            const float* fw = &mel_fb[(size_t)m * n_bins];
            for (int k = 0; k < n_bins; k++) acc += fw[k] * mag[k];
            mel[(size_t)t * n_mels + m] = (float)std::log(std::max(acc, 1e-5));
        }
    }
    return mel;
}

}  // namespace q3tts
