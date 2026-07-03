# 向 futz12/ncnn_llm 提 PR 的适配方案

目标：让 TTS 支持以**纯增量**方式进入 ncnn_llm —— 不改动其任何现有文件的行为，
维护者审阅面最小。

## ncnn_llm 现有约定（PR 必须遵守）

- 模型族 = 一个类：`ncnn_llm_gpt(model_path, use_vulkan=false, num_threads=0, vulkan_device=0)`；
  nets 用 `std::shared_ptr<ncnn::Net>`；生成状态在 `*_ctx` 对象里
  （`KVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>` + `cur_token/position_id`，支持 `clone()`）。
- 配置来自 `model_path/model.json`，nlohmann::json 解析，schema 风格：
  `setting.rope{type,theta,...}`、特殊 id 顶层键、缺键抛 runtime_error。
- blob 命名 `in0.. / out0 / cache_k{i}...`（与本仓库一致，导出产物可直接复用）。
- int 输入用 `mat_from_int_vector`（int32 位模式，与本仓库 `ids_mat` 相同语义）。
- 采样在 `GenerateConfig`（max_new_tokens/temperature/top_p/top_k/repetition_penalty/do_sample）。
- 构建：xmake.lua（CMake PR 未合入，PR 里只加 xmake target）。

## PR 内容（全部为新增文件 + xmake 两行）

| 新增文件 | 来源（本仓库） | 适配动作 |
|---|---|---|
| `src/ncnn_llm_tts.{h,cpp}` | `src/tts_pipeline.*` + `net_utils.*` | 重排为 `ncnn_llm_tts` 类：构造签名对齐；`KVCache` 换用其 typedef；配置从 model.json 读（nlohmann）；`ncnn_llm_tts_ctx` 持 talker cache + generation_step + code0 历史，实现 clone() |
| `src/utils/tokenizer/bpe_gpt2_pretokenizer.{h,cpp}` | `src/qwen_bpe.*` | 作为其 BpeTokenizer 的**子类/伴生**提供 GPT-2 预分词（其 bbpe 缺此能力，Qwen2 词表必需）；不改其现有 tokenizer 文件 |
| `src/utils/tts_dsp.h` | `src/mel.h` + `split_rvq.h` + `wav_io.h` | 头文件合并，命名空间对齐 |
| `examples/tts_main.cpp` | `examples/tts_cli.cpp` | 参数风格向 llm_ncnn_run 靠拢 |
| `export/qwen3_tts/*.py` | `tools/convert/*` | 目录名对齐其 `export/` 惯例 |
| xmake.lua | — | 追加 tts target（约 6 行） |

不动的部分：其 sampling.cpp（我们的采样器语义独立成 `ncnn_llm_tts` 内部实现，
不去改其签名——suppress-token/播种是 TTS 特有需求）；其 rope_embed（talker 的
rope 表由 tts 模块自建，halfdim 约定）。

## model.json（两仓库共用同一 schema）

```json
{
  "model_type": "qwen3_tts",
  "setting": {
    "rope":  { "type": "rope", "theta": 1e6, "head_dim": 128 },
    "codec": { "window": 72, "theta": 1e4, "head_dim": 64, "upsample": 1920, "sample_rate": 24000 }
  },
  "ids": { "codec_eos": 2150, "codec_pad": 2148, "codec_bos": 2149,
           "codec_think": 2154, "codec_nothink": 2155, "codec_think_bos": 2156, "codec_think_eos": 2157,
           "tts_bos": 151672, "tts_eos": 151673, "tts_pad": 151671 },
  "arch": { "talker_layers": 28, "pred_layers": 5, "hidden": 1024,
            "codec_vocab": 3072, "pred_vocab": 2048, "num_code_groups": 16 },
  "generation": { "max_new_tokens": 8192, "temperature": 0.9, "top_k": 50, "top_p": 1.0,
                  "repetition_penalty": 1.05, "min_new_tokens": 2 },
  "language_ids": { "chinese": 2055, "english": 2050 },
  "spk_id": { "serena": 3066, "vivian": 3065 },
  "files": { "talker_decoder": "talker_decoder.ncnn", "...": "..." }
}
```

本仓库侧先行动作（已排期）：
1. `tools/convert/make_model_json.py` 从 checkpoint config 生成上述清单（Base 与 CustomVoice 各一份）；
2. C++ 侧 `tts_config.h` 的硬编码改为从 model.json 读取（本仓库用 60 行扁平解析器免依赖；
   PR 版直接用其 nlohmann）——保证两边行为同源。

## 提交拆分建议（利于审阅）

1. PR-1：`bpe_gpt2_pretokenizer`（独立价值：任何 Qwen2 词表模型都需要）+ 单测；
2. PR-2：`ncnn_llm_tts` 运行时 + export 脚本 + tts_main + model.json 文档；
3. issue：pnnx 四个转换问题（P2/P5/P7/P10）与 SDPA 3D mask 一致性（P15），附最小复现。
