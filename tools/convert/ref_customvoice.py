"""Same-domain (CPU fp32) golden for CustomVoice built-in speaker mode."""
import sys, os
sys.path.insert(0, "export")
os.environ["Q3TTS_MODEL_DIR"] = os.path.abspath("models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
import numpy as np, torch, soundfile as sf
import common
from common import S

SYN_TEXT = "你好，这是内置音色的对齐测试。"
SPEAKER = "serena"
OUT = "/home/mingshi/Project/AI/Qwen3-TTS-ncnn/tests/data"

torch.manual_seed(0)
tts = S.Qwen3TTSModel.from_pretrained(common.MODEL_DIR, device_map="cpu",
                                      dtype=torch.float32, attn_implementation="sdpa")
captured = {}
orig = tts.model.generate
def cap(*a, **k):
    out = orig(*a, **k); captured["codes"] = [c.detach().cpu() for c in out[0]]; return out
tts.model.generate = cap

wavs, sr = tts.generate_custom_voice(text=SYN_TEXT, speaker=SPEAKER, language="Chinese",
                                     do_sample=False, subtalker_dosample=False, max_new_tokens=200)
codes = captured["codes"][0].numpy().astype(np.int32)
print("golden:", codes.shape, "| wav:", len(wavs[0]))
sf.write("ref_output/greedy_customvoice.wav", wavs[0], sr)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(common.MODEL_DIR)
ids = tok(f"<|im_start|>assistant\n{SYN_TEXT}<|im_end|>\n<|im_start|>assistant\n",
          return_tensors="np")["input_ids"][0].astype(np.int32)
cfg = common.load_config()["talker_config"]
np.save(f"{OUT}/cv_text_ids.npy", np.ascontiguousarray(ids))
np.save(f"{OUT}/cv_codes.npy", np.ascontiguousarray(codes))
print("spk_id:", cfg["spk_id"][SPEAKER], "| lang:", cfg["codec_language_id"]["chinese"], "| ids:", len(ids))
