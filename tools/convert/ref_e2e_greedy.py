"""M0: PyTorch fp32 + greedy end-to-end reference for Qwen3-TTS-12Hz-0.6B-Base.

Runs x-vector-only voice clone (simplest deterministic path), captures the
codec token matrix (T,16) and the output wav. These are the golden artifacts
the ncnn port must reproduce token-for-token (codes) / SNR>40dB (wav).
"""
import json
import time

import numpy as np
import soundfile as sf
import torch

import shim_qwen_tts as S

MODEL_DIR = "models/Qwen3-TTS-12Hz-0.6B-Base"
REF_WAV = "ref_clone_2.wav"
REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")
SYN_TEXT = "你好，这是 ncnn 移植的对齐测试。"
LANG = "Chinese"

torch.manual_seed(0)

t0 = time.time()
tts = S.Qwen3TTSModel.from_pretrained(
    MODEL_DIR,
    device_map="cpu",
    dtype=torch.float32,
    attn_implementation="sdpa",
)
print(f"model loaded in {time.time()-t0:.1f}s")

captured = {}
orig_generate = tts.model.generate

def capture_generate(*args, **kwargs):
    out = orig_generate(*args, **kwargs)
    captured["codes"] = [c.detach().cpu() for c in out[0]]
    captured["hiddens"] = [h.detach().cpu() for h in out[1]]
    return out

tts.model.generate = capture_generate

t0 = time.time()
wavs, sr = tts.generate_voice_clone(
    text=SYN_TEXT,
    language=LANG,
    ref_audio=REF_WAV,
    ref_text=REF_TEXT,
    x_vector_only_mode=True,
    do_sample=False,
    subtalker_dosample=False,
    max_new_tokens=512,
)
dt = time.time() - t0

codes = captured["codes"][0]  # (T, 16)
print(f"generated in {dt:.1f}s | frames: {codes.shape} | wav: {len(wavs[0])} samples @ {sr}Hz "
      f"({len(wavs[0])/sr:.2f}s) | RTF: {dt/(len(wavs[0])/sr):.1f}")

sf.write("ref_output/greedy_base_xvec.wav", wavs[0], sr)
np.savez(
    "ref_output/greedy_base_xvec.npz",
    codes=codes.numpy(),
    wav=wavs[0],
    sr=sr,
)
meta = {
    "model": MODEL_DIR,
    "mode": "x_vector_only clone",
    "dtype": "float32",
    "device": "cpu",
    "greedy": True,
    "text": SYN_TEXT,
    "language": LANG,
    "ref_wav": REF_WAV,
    "ref_text": REF_TEXT,
    "max_new_tokens": 512,
    "frames": list(codes.shape),
    "gen_seconds": round(dt, 1),
}
json.dump(meta, open("ref_output/greedy_base_xvec.json", "w"), indent=2, ensure_ascii=False)
print("saved ref_output/greedy_base_xvec.{wav,npz,json}")
print("codes[:3]:", codes[:3].tolist())
