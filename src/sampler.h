// Logits processing — an exact replica of the HF processor->warper split
// (see SOLUTION.md §5.4). Order matters for parity:
//   1. repetition penalty (history = generated codebook-0 tokens)
//   2. min-new-tokens: eos logit = -inf for the first N steps
//      (greedy then picks the SECOND best — "skip eos and continue" diverges!)
//   3. suppress mask [vocab-1024, vocab) \ {eos}
//   4. only when sampling: temperature -> top-k -> top-p -> multinomial
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

namespace q3tts {

class Code0Sampler {
public:
    Code0Sampler(int vocab, int eos, float rep_penalty, int min_new_tokens)
        : vocab_(vocab), eos_(eos), penalty_(rep_penalty), min_new_(min_new_tokens) {}

    // processors (always applied, greedy or sampling)
    void process(std::vector<double>& l, int step) const {
        for (int t : history_) l[t] = l[t] > 0 ? l[t] / penalty_ : l[t] * penalty_;
        if (step < min_new_) l[eos_] = -INFINITY;
        if (suppress_tail_)
            for (int t = vocab_ - 1024; t < vocab_; t++)
                if (t != eos_) l[t] = -INFINITY;
    }

    int argmax(const float* logits, int step) {
        std::vector<double> l(logits, logits + vocab_);
        process(l, step);
        int best = 0;
        for (int t = 1; t < vocab_; t++)
            if (l[t] > l[best]) best = t;
        history_.insert(best);
        return best;
    }

    // HF warper order: temperature -> top-k -> top-p -> softmax -> multinomial
    int sample(const float* logits, int step, float temperature, int top_k, float top_p,
               std::mt19937_64& rng) {
        std::vector<double> l(logits, logits + vocab_);
        process(l, step);
        return warp_and_draw(l, temperature, top_k, top_p, rng, &history_);
    }

    // shared with the sub-talker (which has no processors/history)
    static int warp_and_draw(std::vector<double>& l, float temperature, int top_k, float top_p,
                             std::mt19937_64& rng, std::unordered_set<int>* history = nullptr) {
        const int vocab = (int)l.size();
        if (temperature != 1.0f)
            for (auto& v : l) v /= temperature;
        // top-k: keep the k largest
        std::vector<int> idx(vocab);
        for (int i = 0; i < vocab; i++) idx[i] = i;
        if (top_k > 0 && top_k < vocab) {
            std::nth_element(idx.begin(), idx.begin() + top_k, idx.end(),
                             [&](int a, int b) { return l[a] > l[b]; });
            idx.resize(top_k);
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        // softmax over survivors
        std::vector<double> p(idx.size());
        double mx = l[idx[0]], sum = 0;
        for (size_t i = 0; i < idx.size(); i++) { p[i] = std::exp(l[idx[i]] - mx); sum += p[i]; }
        for (auto& v : p) v /= sum;
        // top-p: smallest prefix with cumulative >= top_p (HF keeps the crossing token)
        if (top_p < 1.0f) {
            double cum = 0;
            size_t keep = p.size();
            for (size_t i = 0; i < p.size(); i++) {
                cum += p[i];
                if (cum >= top_p) { keep = i + 1; break; }
            }
            p.resize(keep);
            idx.resize(keep);
            double s2 = 0;
            for (double v : p) s2 += v;
            for (auto& v : p) v /= s2;
        }
        std::discrete_distribution<int> dist(p.begin(), p.end());
        const int tok = idx[dist(rng)];
        if (history) history->insert(tok);
        return tok;
    }

    void reset() { history_.clear(); }

    // exposed for the parity test: processed+warped probability vector
    std::vector<std::pair<int, double>> warped_probs(const float* logits, int step,
                                                     float temperature, int top_k, float top_p) const {
        std::vector<double> l(logits, logits + vocab_);
        process(l, step);
        if (temperature != 1.0f) for (auto& v : l) v /= temperature;
        std::vector<int> idx(vocab_);
        for (int i = 0; i < vocab_; i++) idx[i] = i;
        if (top_k > 0 && top_k < vocab_) {
            std::nth_element(idx.begin(), idx.begin() + top_k, idx.end(),
                             [&](int a, int b) { return l[a] > l[b]; });
            idx.resize(top_k);
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        std::vector<std::pair<int, double>> out;
        double mx = l[idx[0]], sum = 0;
        for (int i : idx) sum += std::exp(l[i] - mx);
        double cum = 0;
        for (int i : idx) {
            if (std::isinf(l[i])) break;  // suppressed tail (prob 0), sorted last
            double pv = std::exp(l[i] - mx) / sum;
            if (top_p < 1.0f && cum >= top_p) break;
            cum += pv;
            out.emplace_back(i, pv);
        }
        return out;
    }

    void set_suppress(bool on) { suppress_tail_ = on; }

private:
    int vocab_, eos_, min_new_;
    float penalty_;
    bool suppress_tail_ = true;
    std::unordered_set<int> history_;
};

}  // namespace q3tts
