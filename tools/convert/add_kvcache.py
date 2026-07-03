"""Patch a pnnx-generated .ncnn.param: add kv-cache I/O to every SDPA layer and
make Gemm layers dynamic-shape. Adapted from ncnn docs/developer-guide/kvcache.md.

Usage: python add_kvcache.py model.ncnn.param
Writes the patched file in place, keeping a .nokv backup.
"""
import re
import shutil
import sys


def add_kv_cache_to_ncnn_param(filename):
    shutil.copyfile(filename, filename + ".nokv")
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_line_index = 1
    header_parts = lines[header_line_index].strip().split()
    original_layer_count = int(header_parts[0])
    original_blob_count = int(header_parts[1])

    attention_indices = [
        i for i, line in enumerate(lines)
        if line.strip().startswith("MultiHeadAttention") or line.strip().startswith("SDPA")
    ]
    if not attention_indices:
        print("no attention layers found, nothing to do")
        return 0

    for i, line_index in enumerate(attention_indices):
        parts = lines[line_index].strip().split()
        layer_type, layer_name, input_count_str, output_count_str = parts[:4]
        input_count, output_count = int(input_count_str), int(output_count_str)

        blob_and_params = parts[4:]
        inputs = blob_and_params[:input_count]
        outputs = blob_and_params[input_count:input_count + output_count]
        params = blob_and_params[input_count + output_count:]

        inputs.extend([f"cache_k_in_{i}", f"cache_v_in_{i}"])
        outputs.extend([f"cache_k_out_{i}", f"cache_v_out_{i}"])
        params.append("7=1")

        lines[line_index] = " ".join([
            f"{layer_type:<24}", f"{layer_name:<24}",
            str(input_count + 2), str(output_count + 2),
            *inputs, *outputs, *params,
        ]) + "\n"

    new_layer_count = original_layer_count + 1
    new_blob_count = original_blob_count + len(attention_indices) * 4
    lines[header_line_index] = f"{new_layer_count} {new_blob_count}\n"

    insert_pos = header_line_index + 1
    while insert_pos < len(lines) and lines[insert_pos].strip().startswith("Input"):
        insert_pos += 1

    cache_blob_names = [
        name for i in range(len(attention_indices))
        for name in (f"cache_k_in_{i}", f"cache_v_in_{i}")
    ]
    lines.insert(
        insert_pos,
        f"{'Input':<24} {'kv_cache_in':<24} 0 {len(cache_blob_names)} {' '.join(cache_blob_names)}\n",
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return len(attention_indices)


def update_gemm_params(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n = 0
    out = []
    for line in lines:
        if line.strip().startswith("Gemm") and re.search(r"\b7=1\b", line):
            line = re.sub(r"(\b7=)1\b", r"\g<1>0", line)
            n += 1
        out.append(line)
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(out)
    return n


if __name__ == "__main__":
    path = sys.argv[1]
    n_attn = add_kv_cache_to_ncnn_param(path)
    n_gemm = update_gemm_params(path)
    print(f"patched {path}: {n_attn} SDPA layers cache-enabled, {n_gemm} Gemm layers made dynamic (backup: .nokv)")
