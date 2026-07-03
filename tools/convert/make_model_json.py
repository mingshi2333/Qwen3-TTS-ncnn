"""Generate the model.json manifest from a checkpoint's config.json.

The same schema is shared with the planned ncnn_llm PR (docs/ncnn_llm_pr_plan.md).
Usage: Q3TTS_MODEL_DIR=<checkpoint> python make_model_json.py <out_dir>
"""
import json
import os
import sys

import common

out_dir = sys.argv[1] if len(sys.argv) > 1 else common.OUT_DIR
cfg = common.load_config()
tc = cfg["talker_config"]

manifest = {
    "model_type": "qwen3_tts",
    "tts_model_type": cfg.get("tts_model_type", "base"),
    "setting": {
        "rope": {"type": "rope", "theta": tc["rope_theta"], "head_dim": tc["head_dim"]},
        "codec": {"window": 72, "theta": 1e4, "head_dim": 64,
                  "upsample": 1920, "sample_rate": 24000},
    },
    "ids": {
        "codec_eos": tc["codec_eos_token_id"], "codec_pad": tc["codec_pad_id"],
        "codec_bos": tc["codec_bos_id"], "codec_think": tc["codec_think_id"],
        "codec_nothink": tc["codec_nothink_id"], "codec_think_bos": tc["codec_think_bos_id"],
        "codec_think_eos": tc["codec_think_eos_id"],
        "tts_bos": cfg["tts_bos_token_id"], "tts_eos": cfg["tts_eos_token_id"],
        "tts_pad": cfg["tts_pad_token_id"],
    },
    "arch": {
        "talker_layers": tc["num_hidden_layers"],
        "pred_layers": tc["code_predictor_config"]["num_hidden_layers"],
        "hidden": tc["hidden_size"], "codec_vocab": tc["vocab_size"],
        "pred_vocab": tc["code_predictor_config"]["vocab_size"],
        "num_code_groups": tc["num_code_groups"],
        "pred_rope_theta": tc["code_predictor_config"]["rope_theta"],
    },
    "generation": {"max_new_tokens": 8192, "temperature": 0.9, "top_k": 50, "top_p": 1.0,
                   "repetition_penalty": 1.05, "min_new_tokens": 2},
    "language_ids": {k: v for k, v in tc["codec_language_id"].items()},
    "spk_id": {k: v for k, v in tc.get("spk_id", {}).items()},
}

os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "model.json")
json.dump(manifest, open(path, "w"), indent=2, ensure_ascii=False)
print(f"{path}: {len(manifest['spk_id'])} speakers, {len(manifest['language_ids'])} languages")
