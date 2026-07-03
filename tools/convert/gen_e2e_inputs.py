"""Dump tokenizer + x-vector outputs for the C++ E2E acceptance test."""
import json
import os

import numpy as np
import soundfile as sf
import torch

import common
from common import S

W = common.WORK_DIR
OUT = "/home/mingshi/Project/AI/Qwen3-TTS-ncnn/tests/data"
os.makedirs(OUT, exist_ok=True)

meta = json.load(open(os.path.join(W, "ref_output/greedy_base_xvec.json")))
golden = np.load(os.path.join(W, "ref_output/greedy_base_xvec.npz"))

cfg_all = common.load_config()
wav, sr = sf.read(os.path.join(W, meta["ref_wav"]))
assert sr == 24000
spk = S.modeling.Qwen3TTSSpeakerEncoder(
    S.configuration.Qwen3TTSSpeakerEncoderConfig(**cfg_all["speaker_encoder_config"])).eval()
spk.load_state_dict(common.load_weights("speaker_encoder."), strict=True)
with torch.no_grad():
    mels = S.modeling.mel_spectrogram(
        torch.from_numpy(wav).float().unsqueeze(0), n_fft=1024, num_mels=128,
        sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
    xvec = spk(mels)[0].numpy().astype(np.float32)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(W, "models/Qwen3-TTS-12Hz-0.6B-Base"))
text = f"<|im_start|>assistant\n{meta['text']}<|im_end|>\n<|im_start|>assistant\n"
ids = tok(text, return_tensors="np")["input_ids"][0].astype(np.int32)

np.save(os.path.join(OUT, "e2e_text_ids.npy"), ids)
np.save(os.path.join(OUT, "e2e_xvec.npy"), xvec)
np.save(os.path.join(OUT, "e2e_codes.npy"), golden["codes"].astype(np.int32))
lang_id = cfg_all["talker_config"]["codec_language_id"][meta["language"].lower()]
print("ids:", len(ids), "| lang_id:", lang_id, "| golden:", golden["codes"].shape)
