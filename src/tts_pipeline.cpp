#include "tts_pipeline.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace q3tts {

static void copy_row(ncnn::Mat& dst, int dst_row, const ncnn::Mat& src, int src_row) {
    memcpy(dst.row(dst_row), src.row(src_row), src.w * sizeof(float));
}

static void add_row(ncnn::Mat& dst, int dst_row, const ncnn::Mat& src, int src_row) {
    float* d = dst.row(dst_row);
    const float* s = src.row(src_row);
    for (int i = 0; i < src.w; i++) d[i] += s[i];
}

void TtsPipeline::load(const std::string& dir) {
    model_dir_ = dir;
    manifest = load_manifest(dir);
    if (manifest.loaded) cfg = manifest.cfg;
    auto p = [&](const char* n) { return dir + "/" + n + std::string(".ncnn.param"); };
    auto b = [&](const char* n) { return dir + "/" + n + std::string(".ncnn.bin"); };
    text_embed_.load(p("talker_text_embed"), b("talker_text_embed"));
    codec_embed_.load(p("talker_codec_embed"), b("talker_codec_embed"));
    codec_head_.load(p("talker_codec_head"), b("talker_codec_head"));
    pred_embeds_.load(p("predictor_embeds"), b("predictor_embeds"));
    pred_heads_.load(p("predictor_heads"), b("predictor_heads"));
    codec_dec_.load(p("codec_decoder"), b("codec_decoder"));
    talker_.load(p("talker_decoder"), b("talker_decoder"), cfg.talker_layers, cfg.head_dim, cfg.talker_theta);
    predictor_.load(p("predictor_decoder"), b("predictor_decoder"), cfg.pred_layers, cfg.head_dim, cfg.pred_theta);
}

void TtsPipeline::build_prompt_xvec(const std::vector<int>& ids, const float* xvec,
                                    int lang_id, ncnn::Mat& prompt, ncnn::Mat& trailing) {
    const int n = (int)ids.size();
    if (n < 9) throw std::runtime_error("prompt too short");

    // tts_bos/eos/pad embeddings through the text channel
    ncnn::Mat bep = text_embed_.run_ids({cfg.tts_bos, cfg.tts_eos, cfg.tts_pad});
    tts_pad_embed_.create(cfg.hidden, 1);
    copy_row(tts_pad_embed_, 0, bep, 2);

    // codec channel: think prefix (+language) | x-vector | pad,bos
    std::vector<int> prefix;
    if (lang_id >= 0)
        prefix = {cfg.codec_think, cfg.codec_think_bos, lang_id, cfg.codec_think_eos};
    else
        prefix = {cfg.codec_nothink, cfg.codec_think_bos, cfg.codec_think_eos};
    ncnn::Mat emb0 = codec_embed_.run_ids(prefix);
    ncnn::Mat emb1 = codec_embed_.run_ids({cfg.codec_pad, cfg.codec_bos});
    const int codec_len = emb0.h + 1 + emb1.h;  // (+1 x-vector row)

    ncnn::Mat codec_emb(cfg.hidden, codec_len);
    for (int r = 0; r < emb0.h; r++) copy_row(codec_emb, r, emb0, r);
    memcpy(codec_emb.row(emb0.h), xvec, cfg.hidden * sizeof(float));
    for (int r = 0; r < emb1.h; r++) copy_row(codec_emb, emb0.h + 1 + r, emb1, r);

    // role prefix "<|im_start|>assistant\n" = first 3 ids, text channel only
    ncnn::Mat role = text_embed_.run_ids({ids[0], ids[1], ids[2]});
    // first text token (id[3]) rides on the codec_bos position
    ncnn::Mat first_text = text_embed_.run_ids({ids[3]});

    // prompt = [role(3)] ++ [tts_pad*(codec_len-2), tts_bos] + codec_emb[:-1] ++ [first_text + codec_bos]
    prompt.create(cfg.hidden, 3 + (codec_len - 1) + 1);
    for (int r = 0; r < 3; r++) copy_row(prompt, r, role, r);
    for (int r = 0; r < codec_len - 1; r++) {
        copy_row(prompt, 3 + r, bep, r == codec_len - 2 ? 0 : 2);  // tts_bos on last, else tts_pad
        add_row(prompt, 3 + r, codec_emb, r);
    }
    copy_row(prompt, 3 + codec_len - 1, first_text, 0);
    add_row(prompt, 3 + codec_len - 1, codec_emb, codec_len - 1);

    // trailing (streaming): remaining text ids[4:n-5] ++ tts_eos
    std::vector<int> rest(ids.begin() + 4, ids.end() - 5);
    trailing.create(cfg.hidden, (int)rest.size() + 1);
    if (!rest.empty()) {
        ncnn::Mat re = text_embed_.run_ids(rest);
        for (int r = 0; r < re.h; r++) copy_row(trailing, r, re, r);
    }
    copy_row(trailing, (int)rest.size(), bep, 1);  // tts_eos
}

void TtsPipeline::predictor_frame(const float* past_hidden, int code0, int* rest,
                                  const GenOpts& opts, std::mt19937_64* rng) {
    predictor_.reset();

    ncnn::Mat emb_c0 = codec_embed_.run_ids({code0});
    ncnn::Mat pre(cfg.hidden, 2);
    memcpy(pre.row(0), past_hidden, cfg.hidden * sizeof(float));
    copy_row(pre, 1, emb_c0, 0);

    // MTP loop: at step s, head[s] is applied to the hidden produced by the
    // forward at step s, so forward+head[s] fuse into one op (one GPU submit).
    ncnn::Mat cur = pre;
    char out_name[16];
    for (int s = 0; s < cfg.num_code_groups - 1; s++) {
        snprintf(out_name, sizeof(out_name), "out%d", s);
        ncnn::Mat logits = predictor_.forward_head(cur, pred_heads_.net, out_name);
        const float* l = logits.row(0);
        int best;
        if (opts.sub_do_sample) {
            std::vector<double> ld(l, l + cfg.pred_vocab);
            best = Code0Sampler::warp_and_draw(ld, opts.sub_temperature, opts.sub_top_k,
                                               opts.sub_top_p, *rng);
        } else {
            best = 0;
            for (int t = 1; t < cfg.pred_vocab; t++)
                if (l[t] > l[best]) best = t;
        }
        rest[s] = best;
        if (s == cfg.num_code_groups - 2) break;
        cur = pred_embeds_.run_ids({best + s * cfg.pred_vocab});
    }
}

ncnn::Mat TtsPipeline::frame_sum_embed(const Frame& f) {
    ncnn::Mat sum = codec_embed_.run_ids({f.codes[0]}).clone();
    std::vector<int> ids(cfg.num_code_groups - 1);
    for (int i = 0; i < cfg.num_code_groups - 1; i++)
        ids[i] = f.codes[i + 1] + i * cfg.pred_vocab;
    ncnn::Mat pe = pred_embeds_.run_ids(ids);
    for (int r = 0; r < pe.h; r++) add_row(sum, 0, pe, r);
    return sum;
}

std::vector<Frame> TtsPipeline::generate(const ncnn::Mat& prompt, const ncnn::Mat& trailing,
                                         int max_frames, const GenOpts& opts) {
    talker_.reset();
    Code0Sampler sampler(cfg.codec_vocab, cfg.codec_eos, cfg.repetition_penalty, cfg.min_new_tokens);
    std::mt19937_64 rng(opts.seed);

    ncnn::Mat hidden = talker_.forward(prompt);
    ncnn::Mat past(cfg.hidden, 1);
    copy_row(past, 0, hidden, hidden.h - 1);
    ncnn::Mat logits = codec_head_.run(past);

    const bool prof = getenv("Q3TTS_PROFILE") && getenv("Q3TTS_PROFILE")[0] == '1';
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    std::vector<Frame> frames;
    for (int step = 0; step < max_frames; step++) {
        int code0 = opts.do_sample
                        ? sampler.sample(logits.row(0), step, opts.temperature, opts.top_k, opts.top_p, rng)
                        : sampler.argmax(logits.row(0), step);
        if (code0 == cfg.codec_eos) break;

        Frame f;
        f.codes[0] = code0;
        auto tp0 = clk::now();
        predictor_frame(past.row(0), code0, f.codes + 1, opts, &rng);
        auto tp1 = clk::now();
        frames.push_back(f);

        ncnn::Mat nxt = frame_sum_embed(f);
        // trailing text injection: one text embedding per step while text
        // remains (last trailing row is tts_eos), afterwards tts_pad
        // (modeling_qwen3_tts.py lines 1689-1692)
        if (step < trailing.h)
            add_row(nxt, 0, trailing, step);
        else
            add_row(nxt, 0, tts_pad_embed_, 0);
        auto tt0 = clk::now();
        ncnn::Mat h = talker_.forward(nxt);
        auto tt1 = clk::now();
        copy_row(past, 0, h, h.h - 1);
        logits = codec_head_.run(past);
        if (prof)
            fprintf(stderr, "PROF step=%2d talker=%.2fms predictor=%.2fms\n",
                    step, ms(tt0, tt1), ms(tp0, tp1));
    }
    return frames;
}

void TtsPipeline::build_prompt_speaker(const std::vector<int>& ids, int spk_id, int lang_id,
                                       ncnn::Mat& prompt, ncnn::Mat& trailing) {
    const int n = (int)ids.size();
    if (n < 9) throw std::runtime_error("prompt too short");

    ncnn::Mat bep = text_embed_.run_ids({cfg.tts_bos, cfg.tts_eos, cfg.tts_pad});
    tts_pad_embed_.create(cfg.hidden, 1);
    copy_row(tts_pad_embed_, 0, bep, 2);

    // codec channel: think prefix (+language) | speaker embedding row | pad,bos
    std::vector<int> prefix;
    if (lang_id >= 0)
        prefix = {cfg.codec_think, cfg.codec_think_bos, lang_id, cfg.codec_think_eos};
    else
        prefix = {cfg.codec_nothink, cfg.codec_think_bos, cfg.codec_think_eos};
    ncnn::Mat emb0 = codec_embed_.run_ids(prefix);
    ncnn::Mat spk = codec_embed_.run_ids({spk_id});
    ncnn::Mat emb1 = codec_embed_.run_ids({cfg.codec_pad, cfg.codec_bos});
    const int codec_len = emb0.h + 1 + emb1.h;
    ncnn::Mat codec_emb(cfg.hidden, codec_len);
    for (int r = 0; r < emb0.h; r++) copy_row(codec_emb, r, emb0, r);
    copy_row(codec_emb, emb0.h, spk, 0);
    for (int r = 0; r < emb1.h; r++) copy_row(codec_emb, emb0.h + 1 + r, emb1, r);

    ncnn::Mat role = text_embed_.run_ids({ids[0], ids[1], ids[2]});

    // non-streaming: full text (+tts_eos) rides on codec_pad rows, then
    // a final row of tts_pad + codec_bos; trailing is a single tts_pad row
    std::vector<int> text(ids.begin() + 3, ids.end() - 5);
    ncnn::Mat te = text_embed_.run_ids(text);
    ncnn::Mat pad_e = codec_embed_.run_ids({cfg.codec_pad});
    ncnn::Mat bos_e = codec_embed_.run_ids({cfg.codec_bos});

    const int n_text = te.h;
    prompt.create(cfg.hidden, 3 + (codec_len - 1) + (n_text + 1) + 1);
    for (int r = 0; r < 3; r++) copy_row(prompt, r, role, r);
    for (int r = 0; r < codec_len - 1; r++) {
        copy_row(prompt, 3 + r, bep, r == codec_len - 2 ? 0 : 2);  // tts_bos on last, else tts_pad
        add_row(prompt, 3 + r, codec_emb, r);
    }
    const int base = 3 + codec_len - 1;
    for (int r = 0; r < n_text + 1; r++) {
        if (r < n_text) copy_row(prompt, base + r, te, r);
        else copy_row(prompt, base + r, bep, 1);  // tts_eos
        add_row(prompt, base + r, pad_e, 0);
    }
    copy_row(prompt, base + n_text + 1, bep, 2);   // tts_pad
    add_row(prompt, base + n_text + 1, bos_e, 0);  // + codec_bos

    trailing.create(cfg.hidden, 1);
    copy_row(trailing, 0, bep, 2);  // every step adds tts_pad
}

std::vector<Frame> TtsPipeline::encode_reference(const std::vector<float>& wav24k) {
    if (!mimi_loaded_) {
        mimi_.load(model_dir_ + "/mimi_encoder.ncnn.param", model_dir_ + "/mimi_encoder.ncnn.bin");
        rvq_.load(model_dir_);
        mimi_loaded_ = true;
    }
    std::vector<float> wav = wav24k;
    wav.resize(((wav.size() + 1919) / 1920) * 1920, 0.0f);  // right zero-pad
    const int T25 = (int)wav.size() / 960;                  // 25 Hz frames
    const int T = T25 / 2;                                  // 12.5 Hz frames

    ncnn::Mat in(wav.size() /*w*/, 1 /*h*/);
    memcpy(in.data, wav.data(), wav.size() * sizeof(float));
    ncnn::Mat mask(T25, T25);
    for (int i = 0; i < T25; i++) {
        float* mp = mask.row(i);
        for (int j = 0; j < T25; j++)
            mp[j] = (j <= i && j >= i - 250 + 1) ? 0.0f : -INFINITY;
    }
    ncnn::Mat cos, sin;
    rope_tables(0, T25, 64, 1e4, cos, sin);

    ncnn::Extractor ex = mimi_.net.create_extractor();
    ex.input("in0", in);
    ex.input("in1", mask);
    ex.input("in2", cos);
    ex.input("in3", sin);
    ncnn::Mat latent;
    if (ex.extract("out0", latent) != 0)
        throw std::runtime_error("mimi encode failed");
    // latent (h=512, w=T)
    std::vector<int> codes = rvq_.encode((const float*)latent.data, T);
    std::vector<Frame> out(T);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < 16; k++)
            out[t].codes[k] = codes[(size_t)t * 16 + k];
    return out;
}

void TtsPipeline::build_prompt_icl(const std::vector<int>& text_ids, const std::vector<int>& ref_ids,
                                   const std::vector<Frame>& ref_codes, const float* xvec, int lang_id,
                                   ncnn::Mat& prompt, ncnn::Mat& trailing) {
    // shared head: role + codec think prefix + x-vector + (pad,bos)[:-1]
    ncnn::Mat bep = text_embed_.run_ids({cfg.tts_bos, cfg.tts_eos, cfg.tts_pad});
    tts_pad_embed_.create(cfg.hidden, 1);
    copy_row(tts_pad_embed_, 0, bep, 2);

    std::vector<int> prefix;
    if (lang_id >= 0)
        prefix = {cfg.codec_think, cfg.codec_think_bos, lang_id, cfg.codec_think_eos};
    else
        prefix = {cfg.codec_nothink, cfg.codec_think_bos, cfg.codec_think_eos};
    ncnn::Mat emb0 = codec_embed_.run_ids(prefix);
    ncnn::Mat emb1 = codec_embed_.run_ids({cfg.codec_pad, cfg.codec_bos});
    const int codec_len = emb0.h + 1 + emb1.h;
    ncnn::Mat codec_emb(cfg.hidden, codec_len);
    for (int r = 0; r < emb0.h; r++) copy_row(codec_emb, r, emb0, r);
    memcpy(codec_emb.row(emb0.h), xvec, cfg.hidden * sizeof(float));
    for (int r = 0; r < emb1.h; r++) copy_row(codec_emb, emb0.h + 1 + r, emb1, r);

    ncnn::Mat role = text_embed_.run_ids({text_ids[0], text_ids[1], text_ids[2]});

    // ICL text channel: ref_ids[3:-2] ++ text_ids[3:-5] -> text_embed ++ tts_eos
    std::vector<int> icl_text(ref_ids.begin() + 3, ref_ids.end() - 2);
    icl_text.insert(icl_text.end(), text_ids.begin() + 3, text_ids.end() - 5);
    ncnn::Mat te = text_embed_.run_ids(icl_text);
    const int T1 = te.h + 1;  // + tts_eos

    // ICL codec channel: bos ++ per-frame 16-code embedding sums
    const int Tr = (int)ref_codes.size();
    ncnn::Mat ce(cfg.hidden, Tr + 1);
    {
        ncnn::Mat bos = codec_embed_.run_ids({cfg.codec_bos});
        copy_row(ce, 0, bos, 0);
        for (int t = 0; t < Tr; t++) {
            ncnn::Mat s = frame_sum_embed(ref_codes[t]);
            copy_row(ce, 1 + t, s, 0);
        }
    }

    // streaming branch (modeling 2014-2019): overlap length = Tr+1
    prompt.create(cfg.hidden, 3 + (codec_len - 1) + (Tr + 1));
    for (int r = 0; r < 3; r++) copy_row(prompt, r, role, r);
    for (int r = 0; r < codec_len - 1; r++) {
        copy_row(prompt, 3 + r, bep, r == codec_len - 2 ? 0 : 2);
        add_row(prompt, 3 + r, codec_emb, r);
    }
    const int base = 3 + codec_len - 1;
    for (int r = 0; r < Tr + 1; r++) {
        // text side: te rows, then tts_eos at index te.h, then tts_pad beyond
        if (r < te.h) copy_row(prompt, base + r, te, r);
        else if (r == te.h) copy_row(prompt, base + r, bep, 1);
        else copy_row(prompt, base + r, bep, 2);
        add_row(prompt, base + r, ce, r);
    }

    if (T1 > Tr + 1) {  // leftover text becomes trailing
        const int n = T1 - (Tr + 1);
        trailing.create(cfg.hidden, n);
        for (int r = 0; r < n; r++) {
            const int src = Tr + 1 + r;
            if (src < te.h) copy_row(trailing, r, te, src);
            else copy_row(trailing, r, bep, 1);  // tts_eos is the final text row
        }
    } else {
        trailing.create(cfg.hidden, 1);
        copy_row(trailing, 0, bep, 2);  // tts_pad
    }
}

std::vector<float> TtsPipeline::decode_icl(const std::vector<Frame>& ref_codes,
                                           const std::vector<Frame>& gen_frames) {
    std::vector<Frame> all(ref_codes);
    all.insert(all.end(), gen_frames.begin(), gen_frames.end());
    std::vector<float> wav = decode(all);
    const size_t cut = (size_t)((double)ref_codes.size() / all.size() * wav.size());
    return std::vector<float>(wav.begin() + cut, wav.end());
}

std::vector<float> TtsPipeline::decode(const std::vector<Frame>& frames) {
    const int T = (int)frames.size();

    std::vector<int> sem(T), ac((size_t)(cfg.num_code_groups - 1) * T);
    for (int t = 0; t < T; t++) sem[t] = frames[t].codes[0];
    for (int i = 0; i < cfg.num_code_groups - 1; i++)
        for (int t = 0; t < T; t++)
            ac[(size_t)i * T + t] = frames[t].codes[i + 1] + i * cfg.pred_vocab;

    ncnn::Mat mask(T, T);
    for (int i = 0; i < T; i++) {
        float* mp = mask.row(i);
        for (int j = 0; j < T; j++)
            mp[j] = (j <= i && j >= i - cfg.codec_window + 1) ? 0.0f : -INFINITY;
    }

    ncnn::Mat cos, sin;
    rope_tables(0, T, cfg.codec_head_dim, cfg.codec_theta, cos, sin);

    ncnn::Extractor ex = codec_dec_.net.create_extractor();
    ex.input("in0", ids_mat(sem));
    ex.input("in1", ids_mat(ac));
    ex.input("in2", mask);
    ex.input("in3", cos);
    ex.input("in4", sin);
    ncnn::Mat wav;
    if (ex.extract("out0", wav) != 0)
        throw std::runtime_error("codec decode failed");

    const size_t n = (size_t)wav.w * wav.h * wav.c;
    std::vector<float> out(n);
    memcpy(out.data(), wav.data, n * sizeof(float));
    return out;
}

}  // namespace q3tts
