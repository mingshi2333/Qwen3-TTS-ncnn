"""PyTorch fp32 greedy golden reference for ICL voice-clone mode (M4.3).

Dumps golden codes/wav + everything the C++ test needs (token ids, ref codes,
x-vector) to Qwen3-TTS-ncnn/tests/data/.
"""
import json
import os

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, "export")
import common
from common import S

W = common.WORK_DIR
OUT = os.path.join(W, "ref_output")
CPP = "/home/mingshi/Project/AI/Qwen3-TTS-ncnn/tests/data"

MODEL_DIR = os.path.join(W, "models/Qwen3-TTS-12Hz-0.6B-Base")
REF_WAV = os.path.join(W, "ref_clone_2.wav")
REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")
SYN_TEXT = "你好，这是 ncnn 移植的对齐测试。"

REUSE = os.path.exists(os.path.join(OUT, "greedy_base_icl.npz"))
torch.manual_seed(0)
tts = S.Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:0", dtype=torch.float32,
                                      attn_implementation="sdpa")

captured = {}
orig = tts.model.generate
def cap(*a, **k):
    out = orig(*a, **k)
    captured["codes"] = [c.detach().cpu() for c in out[0]]
    return out
tts.model.generate = cap

if not REUSE:
    wavs, sr = tts.generate_voice_clone(
        text=SYN_TEXT, language="Chinese", ref_audio=REF_WAV, ref_text=REF_TEXT,
        x_vector_only_mode=False,   # ICL
        do_sample=False, subtalker_dosample=False, max_new_tokens=512,
    )
    codes = captured["codes"][0].numpy()
    print("golden ICL codes:", codes.shape, "| wav:", len(wavs[0]))
    sf.write(os.path.join(OUT, "greedy_base_icl.wav"), wavs[0], sr)
    np.savez(os.path.join(OUT, "greedy_base_icl.npz"), codes=codes, wav=wavs[0], sr=sr)
else:
    codes = np.load(os.path.join(OUT, "greedy_base_icl.npz"))["codes"]
    print("reusing golden:", codes.shape)

# inputs for C++: ref codes + x-vector + token ids
enc = tts.model.speech_tokenizer.encode(REF_WAV)
ref_codes = enc.audio_codes[0].cpu().numpy().astype(np.int32)  # (T_ref, 16)
wav24, _ = sf.read(REF_WAV)
mels = S.modeling.mel_spectrogram(torch.from_numpy(wav24).float().unsqueeze(0), n_fft=1024,
                                  num_mels=128, sampling_rate=24000, hop_size=256,
                                  win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
with torch.no_grad():
    xvec = tts.model.speaker_encoder(mels.to(tts.model.device))[0].cpu().numpy().astype(np.float32)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
text_ids = tok(f"<|im_start|>assistant\n{SYN_TEXT}<|im_end|>\n<|im_start|>assistant\n",
               return_tensors="np")["input_ids"][0].astype(np.int32)
ref_ids = tok(f"<|im_start|>assistant\n{REF_TEXT}<|im_end|>\n",
              return_tensors="np")["input_ids"][0].astype(np.int32)

np.save(os.path.join(CPP, "icl_text_ids.npy"), text_ids)
np.save(os.path.join(CPP, "icl_ref_ids.npy"), ref_ids)
np.save(os.path.join(CPP, "icl_ref_codes.npy"), ref_codes)
np.save(os.path.join(CPP, "icl_xvec.npy"), xvec)
np.save(os.path.join(CPP, "icl_codes.npy"), codes.astype(np.int32))
print(f"dumped: text_ids {len(text_ids)}, ref_ids {len(ref_ids)}, ref_codes {ref_codes.shape}")
