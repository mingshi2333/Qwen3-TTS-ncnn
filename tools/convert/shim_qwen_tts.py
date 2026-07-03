"""Import qwen_tts 12hz modules without the 25hz dependency chain (torchaudio/sox/whisper).

Usage:
    import shim_qwen_tts
    mod = shim_qwen_tts.modeling            # qwen_tts.core.models.modeling_qwen3_tts
    cfg = shim_qwen_tts.configuration       # qwen_tts.core.models.configuration_qwen3_tts
    tok = shim_qwen_tts.tokenizer_v2        # 12hz codec modeling
"""
import importlib
import os
import sys
import types

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-TTS", "qwen_tts")

for _name, _path in [
    ("qwen_tts", _ROOT),
    ("qwen_tts.core", os.path.join(_ROOT, "core")),
    ("qwen_tts.core.models", os.path.join(_ROOT, "core", "models")),
    ("qwen_tts.core.tokenizer_12hz", os.path.join(_ROOT, "core", "tokenizer_12hz")),
    ("qwen_tts.inference", os.path.join(_ROOT, "inference")),
]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [_path]
        sys.modules[_name] = _m

# real 12hz (v2) modules
_tok_cfg_v2 = importlib.import_module("qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2")
tokenizer_v2 = importlib.import_module("qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2")

_core = sys.modules["qwen_tts.core"]
_core.Qwen3TTSTokenizerV2Config = _tok_cfg_v2.Qwen3TTSTokenizerV2Config
_core.Qwen3TTSTokenizerV2Model = tokenizer_v2.Qwen3TTSTokenizerV2Model

# 25hz (v1) stubs — never instantiated on the 12hz path
class _V1Stub:
    def __init__(self, *a, **k):
        raise RuntimeError("25hz tokenizer path is stubbed out by shim_qwen_tts")

_core.Qwen3TTSTokenizerV1Config = type("Qwen3TTSTokenizerV1Config", (_V1Stub,), {})
_core.Qwen3TTSTokenizerV1Model = type("Qwen3TTSTokenizerV1Model", (_V1Stub,), {})

configuration = importlib.import_module("qwen_tts.core.models.configuration_qwen3_tts")
modeling = importlib.import_module("qwen_tts.core.models.modeling_qwen3_tts")
inference_tokenizer = importlib.import_module("qwen_tts.inference.qwen3_tts_tokenizer")
processing = importlib.import_module("qwen_tts.core.models.processing_qwen3_tts")

# populate the dummy qwen_tts.core.models package so `from ..core.models import X` works
_models_pkg = sys.modules["qwen_tts.core.models"]
_models_pkg.Qwen3TTSConfig = configuration.Qwen3TTSConfig
_models_pkg.Qwen3TTSForConditionalGeneration = modeling.Qwen3TTSForConditionalGeneration
_models_pkg.Qwen3TTSProcessor = processing.Qwen3TTSProcessor
_core.Qwen3TTSTokenizerV2Model = tokenizer_v2.Qwen3TTSTokenizerV2Model

inference_model = importlib.import_module("qwen_tts.inference.qwen3_tts_model")
Qwen3TTSModel = inference_model.Qwen3TTSModel
