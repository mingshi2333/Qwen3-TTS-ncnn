"""M3 numeric validation: full ncnn pipeline, greedy, vs the PyTorch golden reference.

Everything neural runs in ncnn (text embed, talker+KV, predictor+KV, codec).
Torch/HF only for: tokenizer, x-vector (speaker encoder export is M4 scope).

Pass criteria: codec token matrix (T,16) matches ref_output/greedy_base_xvec.npz
token-for-token; decoded wav SNR > 40 dB vs golden wav.
"""
import json
import os

import numpy as np
import torch

import common
from common import S

import ncnn

W = common.WORK_DIR
NM = os.path.join(W, "ncnn_models")
MODEL_DIR = os.path.join(W, "models/Qwen3-TTS-12Hz-0.6B-Base")

cfg_all = common.load_config()
TC = cfg_all["talker_config"]
EOS = TC["codec_eos_token_id"]
VOCAB = TC["vocab_size"]  # 3072
SUPPRESS = [i for i in range(VOCAB - 1024, VOCAB) if i != EOS]
N_LAYERS, N_PRED_LAYERS, HEAD_DIM = 28, 5, 128
NCG = TC["num_code_groups"]  # 16

meta = json.load(open(os.path.join(W, "ref_output/greedy_base_xvec.json")))
golden = np.load(os.path.join(W, "ref_output/greedy_base_xvec.npz"))
g_codes, g_wav = golden["codes"], golden["wav"]
print(f"golden: {g_codes.shape} codes, {len(g_wav)} wav samples")


def to_mat(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return ncnn.Mat(a).clone()


def to_mat_i32(a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return ncnn.Mat(a).clone()


def load_net(name):
    net = ncnn.Net()
    assert net.load_param(os.path.join(NM, f"{name}.ncnn.param")) == 0, name
    assert net.load_model(os.path.join(NM, f"{name}.ncnn.bin")) == 0, name
    return net


nets = {n: load_net(n) for n in [
    "talker_text_embed", "talker_codec_embed", "talker_codec_head",
    "talker_decoder", "predictor_decoder", "predictor_embeds", "predictor_heads",
    "codec_decoder",
]}


def run1(name, feeds, out="out0"):
    with nets[name].create_extractor() as ex:
        for blob, mat in feeds:
            ex.input(blob, mat)
        ret, o = ex.extract(out)
        assert ret == 0, f"{name}/{out}"
        return np.array(o).copy()


def text_embed(ids):
    return run1("talker_text_embed", [("in0", to_mat_i32(np.asarray(ids, dtype=np.int32)))])


def codec_embed(ids):
    return run1("talker_codec_embed", [("in0", to_mat_i32(np.asarray(ids, dtype=np.int32)))])


def rope_tables(positions, theta):
    inv = 1.0 / (theta ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None]
    emb = np.concatenate([ang, ang], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


class KvNet:
    def __init__(self, name, n_layers, theta):
        self.name, self.n, self.theta = name, n_layers, theta
        self.caches = None
        self.past = 0

    def reset(self):
        self.caches, self.past = None, 0

    def forward(self, embeds):  # embeds (T,1024) at positions past..past+T-1
        T = embeds.shape[0]
        cos, sin = rope_tables(np.arange(self.past, self.past + T), self.theta)
        if self.past == 0:
            mask = np.triu(np.full((1, T, T), -np.inf, dtype=np.float32), 1)
        else:
            mask = np.zeros((1, T, self.past + T), dtype=np.float32)
        with nets[self.name].create_extractor() as ex:
            ex.input("in0", to_mat(embeds))
            ex.input("in1", to_mat(mask))
            ex.input("in2", to_mat(cos))
            ex.input("in3", to_mat(sin))
            if self.caches is not None:
                for i in range(self.n):
                    ex.input(f"cache_k_in_{i}", to_mat(self.caches[i][0]))
                    ex.input(f"cache_v_in_{i}", to_mat(self.caches[i][1]))
            ret, o = ex.extract("out0")
            assert ret == 0
            hidden = np.array(o).copy()
            new_caches = {}
            for i in range(self.n):
                rk, mk = ex.extract(f"cache_k_out_{i}")
                rv, mv = ex.extract(f"cache_v_out_{i}")
                assert rk == 0 and rv == 0
                new_caches[i] = (np.array(mk).copy(), np.array(mv).copy())
        self.caches = new_caches
        self.past += T
        return hidden


talker = KvNet("talker_decoder", N_LAYERS, TC["rope_theta"])
pred_theta = TC["code_predictor_config"]["rope_theta"]

# ---------------- x-vector (torch, fp32 — export in M4) ----------------
import soundfile as sf

wav, sr = sf.read(os.path.join(W, meta["ref_wav"]))
assert sr == 24000
spk_cfg = S.configuration.Qwen3TTSSpeakerEncoderConfig(**cfg_all["speaker_encoder_config"])
spk = S.modeling.Qwen3TTSSpeakerEncoder(spk_cfg).eval()
spk.load_state_dict(common.load_weights("speaker_encoder."), strict=True)
with torch.no_grad():
    mels = S.modeling.mel_spectrogram(
        torch.from_numpy(wav).float().unsqueeze(0), n_fft=1024, num_mels=128,
        sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000,
    ).transpose(1, 2)
    xvec = spk(mels)[0].numpy()  # (1024,)
print("x-vector:", xvec.shape)

# ---------------- tokenizer + prompt ids ----------------
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
text = f"<|im_start|>assistant\n{meta['text']}<|im_end|>\n<|im_start|>assistant\n"
input_id = tok(text, return_tensors="np")["input_ids"][0]
print("prompt ids:", len(input_id))

# ---------------- prompt embedding construction (x-vector clone, streaming) ----------------
lang_id = TC["codec_language_id"][meta["language"].lower()]
tts_bos, tts_eos, tts_pad = cfg_all["tts_bos_token_id"], cfg_all["tts_eos_token_id"], cfg_all["tts_pad_token_id"]

bep = text_embed([tts_bos, tts_eos, tts_pad])
tts_bos_e, tts_eos_e, tts_pad_e = bep[0], bep[1], bep[2]

prefill_list = [TC["codec_think_id"], TC["codec_think_bos_id"], lang_id, TC["codec_think_eos_id"]]
emb0 = codec_embed(prefill_list)                                   # (4,1024)
emb1 = codec_embed([TC["codec_pad_id"], TC["codec_bos_id"]])       # (2,1024)
codec_emb = np.concatenate([emb0, xvec[None], emb1], axis=0)       # (7,1024)

role_e = text_embed(input_id[:3])                                  # (3,1024)
text_ch = np.concatenate([np.tile(tts_pad_e, (codec_emb.shape[0] - 2, 1)), tts_bos_e[None]], axis=0)
body = text_ch + codec_emb[:-1]                                    # (6,1024)
first_text = text_embed(input_id[3:4]) + codec_emb[-1:]            # (1,1024)
prompt = np.concatenate([role_e, body, first_text], axis=0)        # (10,1024)

trailing = np.concatenate([text_embed(input_id[4:-5]), tts_eos_e[None]], axis=0)
print("prompt:", prompt.shape, "| trailing:", trailing.shape)


# ---------------- samplers ----------------
def greedy_code0(logits, history, step):
    logits = logits.astype(np.float64).copy()
    for t in set(history):                      # HF repetition penalty
        logits[t] = logits[t] / 1.05 if logits[t] > 0 else logits[t] * 1.05
    if step < 2:                                # min_new_tokens=2
        logits[EOS] = -np.inf
    logits[SUPPRESS] = -np.inf
    return int(np.argmax(logits))


def predictor_frame(past_hidden, code0):
    """past_hidden (1024,), code0 int -> codes[1..15] via ncnn predictor."""
    emb_c0 = codec_embed([code0])[0]
    pred = KvNet("predictor_decoder", N_PRED_LAYERS, pred_theta)
    h = pred.forward(np.stack([past_hidden, emb_c0]))              # (2,1024)
    codes = []
    last_h = h[-1]
    for s in range(15):
        logits = run1("predictor_heads", [("in0", to_mat(last_h[None]))], out=f"out{s}")[0]
        t = int(np.argmax(logits))
        codes.append(t)
        if s == 14:
            break
        emb = run1("predictor_embeds", [("in0", to_mat_i32(np.array([t + s * 2048])))])
        last_h = pred.forward(emb)[-1]
    return codes


def frame_sum_embed(code0, codes_rest):
    e = codec_embed([code0])[0].copy()
    ids = np.array([c + i * 2048 for i, c in enumerate(codes_rest)], dtype=np.int32)
    e += run1("predictor_embeds", [("in0", to_mat_i32(ids))]).sum(axis=0)
    return e


# ---------------- AR loop ----------------
hidden = talker.forward(prompt)
past_hidden = hidden[-1]
logits = run1("talker_codec_head", [("in0", to_mat(past_hidden[None]))])[0]

frames = []
history = []
for step in range(len(g_codes) + 8):
    code0 = greedy_code0(logits, history, step)
    history.append(code0)
    if code0 == EOS:
        print(f"eos at step {step}")
        break
    rest = predictor_frame(past_hidden, code0)
    frames.append([code0] + rest)
    nxt = frame_sum_embed(code0, rest)
    nxt = nxt + (trailing[step] if step < len(trailing) else tts_pad_e)
    hidden = talker.forward(nxt[None])
    past_hidden = hidden[-1]
    logits = run1("talker_codec_head", [("in0", to_mat(past_hidden[None]))])[0]
    if step < 3 or (step + 1) % 10 == 0:
        match = "=" if step < len(g_codes) and frames[-1] == g_codes[step].tolist() else "≠"
        print(f"frame {step}: code0 {code0} {match}")

frames = np.array(frames)
print(f"generated {frames.shape} | golden {g_codes.shape}")
n = min(len(frames), len(g_codes))
tok_match = (frames[:n] == g_codes[:n]).mean()
exact = len(frames) == len(g_codes) and tok_match == 1.0
print(f"token match: {tok_match * 100:.2f}% | length match: {len(frames)} vs {len(g_codes)}")

# ---------------- codec decode ----------------
T = len(frames)
mask = np.full((1, T, T), -np.inf, dtype=np.float32)
for i in range(T):
    mask[0, i, max(0, i - 71):i + 1] = 0.0
inv = 1.0 / (10000 ** (np.arange(0, 64, 2, dtype=np.float64) / 64))
ang = np.arange(T, dtype=np.float64)[:, None] * inv[None]
c_cos = np.cos(np.concatenate([ang, ang], 1)).astype(np.float32)
c_sin = np.sin(np.concatenate([ang, ang], 1)).astype(np.float32)
off = np.arange(15, dtype=np.int32)[:, None] * 2048
wav_out = run1("codec_decoder", [
    ("in0", to_mat_i32(frames[:, 0][None])),
    ("in1", to_mat_i32((frames[:, 1:].T.astype(np.int32) + off).reshape(-1))),
    ("in2", to_mat(mask)),
    ("in3", to_mat(c_cos)),
    ("in4", to_mat(c_sin)),
]).reshape(-1)

m = min(len(wav_out), len(g_wav))
noise = np.mean((wav_out[:m] - g_wav[:m]) ** 2)
snr = 10 * np.log10(np.mean(g_wav[:m] ** 2) / noise) if noise > 0 else float("inf")
print(f"wav: {len(wav_out)} vs {len(g_wav)} | SNR {snr:.1f} dB")

import soundfile as sf2
sf2.write(os.path.join(W, "ref_output/ncnn_e2e_greedy.wav"), wav_out, 24000)

ok = exact and snr > 40
print("E2E GREEDY PARITY:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
