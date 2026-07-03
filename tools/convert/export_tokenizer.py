"""Export Qwen2 tokenizer to a flat file for the C++ runtime + build a parity test set.

tokenizer.txt format (all strings are byte-level-BPE unicode forms — no raw
spaces/newlines can appear, so line-based is safe):
    V <n_vocab>
    <n_vocab lines: token string, line index == id>
    M <n_merges>
    <n_merges lines: "left right">
    A <n_added>
    <n_added lines: "<id> <content>">   (content has no spaces in Qwen specials)

tokenizer_test.bin: uint32 count, then per case:
    uint32 text_len, text bytes (utf-8), uint32 n_ids, int32 ids...
"""
import json
import os
import struct

import numpy as np
from transformers import AutoTokenizer

import common

W = common.WORK_DIR
MODEL_DIR = os.path.join(W, "models/Qwen3-TTS-12Hz-0.6B-Base")
OUT_DIR = "/home/mingshi/Project/AI/Qwen3-TTS-ncnn/models"
TEST_DIR = "/home/mingshi/Project/AI/Qwen3-TTS-ncnn/tests/data"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

vocab = json.load(open(os.path.join(MODEL_DIR, "vocab.json")))
merges = [l.rstrip("\n") for l in open(os.path.join(MODEL_DIR, "merges.txt"), encoding="utf-8")]
if merges and merges[0].startswith("#"):
    merges = merges[1:]
tc = json.load(open(os.path.join(MODEL_DIR, "tokenizer_config.json")))
added = {int(k): v["content"] for k, v in tc["added_tokens_decoder"].items()}

n_vocab = max(vocab.values()) + 1
by_id = [""] * n_vocab
for tok, i in vocab.items():
    by_id[i] = tok

with open(os.path.join(OUT_DIR, "tokenizer.txt"), "w", encoding="utf-8") as f:
    f.write(f"V {n_vocab}\n")
    for t in by_id:
        f.write(t + "\n")
    f.write(f"M {len(merges)}\n")
    for m in merges:
        f.write(m + "\n")
    f.write(f"A {len(added)}\n")
    for i in sorted(added):
        f.write(f"{i} {added[i]}\n")
print(f"tokenizer.txt: {n_vocab} vocab, {len(merges)} merges, {len(added)} added")

# ---- parity test set ----
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

cases = [
    "你好，这是 ncnn 移植的对齐测试。",
    "<|im_start|>assistant\n你好，世界！<|im_end|>\n<|im_start|>assistant\n",
    "Hello world! It's a test. We're testing don't-cases and I'll check.",
    "混合 mixed 文本 with 123 numbers 和标点……——？！",
    "  leading spaces and   multiple   spaces  ",
    "line\nbreaks\r\nand\ttabs",
    "ＦＵＬＬｗｉｄｔｈ１２３ and ①②③ Ⅷ",
    "email@test.com https://github.com/Tencent/ncnn #hashtag @user",
    "日本語のテキストとカタカナ、ひらがな。한국어 텍스트도 있습니다.",
    "🎉 emoji 😀 test 🚀",
    "C++ template<typename T> void f(std::vector<T>& v);",
    "价格是¥3,999.00（约$550）",
    "Le café est très bon. Übung macht den Meister. Привет мир!",
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE",
    "a'b c'd e's f'll",
    "",
    " ",
    "。",
    "词",
]
# programmatic mixed corpus
rng = np.random.RandomState(7)
zh = "的一是了我不人在他有这上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
en = "the quick brown fox jumps over lazy dog while testing tokenizer parity across languages and scripts".split()
for _ in range(120):
    parts = []
    for _ in range(rng.randint(1, 6)):
        r = rng.rand()
        if r < 0.4:
            parts.append("".join(rng.choice(list(zh), rng.randint(2, 12))))
        elif r < 0.7:
            parts.append(" ".join(rng.choice(en, rng.randint(1, 5))))
        elif r < 0.85:
            parts.append(str(rng.randint(0, 99999)))
        else:
            parts.append(rng.choice(list("，。！？、：；……——()（）\"\"『』<>[]{}#@%&*+-=~")))
    cases.append(rng.choice(["", " ", "，", ". "]).join(parts))

with open(os.path.join(TEST_DIR, "tokenizer_test.bin"), "wb") as f:
    f.write(struct.pack("<I", len(cases)))
    for text in cases:
        b = text.encode("utf-8")
        ids = tok(text)["input_ids"]
        f.write(struct.pack("<I", len(b)))
        f.write(b)
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack(f"<{len(ids)}i", *ids))
print(f"tokenizer_test.bin: {len(cases)} cases")
