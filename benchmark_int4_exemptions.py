"""Full WikiText-2 test PPL for INT4 RTN while retaining named tensors in FP32."""
from __future__ import annotations

import json
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_standard_formats import GROUP_SIZE, REVISION, _weight_rows, get_eval_ids, perplexity


MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-1d", action="store_true", help="Keep 1D biases and normalization parameters in FP32")
    parser.add_argument("--keep-embedding", action="store_true", help="Keep the tied token embedding in FP32")
    parser.add_argument("--keep-final-layernorm", action="store_true", help="Keep only final layer norm tensors in FP32")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, torch_dtype=torch.float32)
    if args.keep_final_layernorm and (args.keep_1d or args.keep_embedding):
        parser.error("--keep-final-layernorm cannot be combined with the other exemption flags")
    exempt = set()
    if args.keep_1d:
        exempt.update(name for name, param in model.named_parameters() if param.ndim == 1)
    if args.keep_embedding:
        exempt.add("transformer.wte.weight")
    if args.keep_final_layernorm:
        exempt.update({"transformer.ln_f.weight", "transformer.ln_f.bias"})
    q_count = 0
    scale_count = 0
    for name, param in model.named_parameters():
        if name in exempt:
            continue
        source = param.detach().cpu().float().numpy()
        rows, transposed = _weight_rows(source, name)
        reconstructed_rows = np.empty_like(rows, dtype=np.float32)
        for row_idx, row in enumerate(rows):
            for start in range(0, row.size, GROUP_SIZE):
                block = row[start:start + GROUP_SIZE]
                scale = float(np.max(np.abs(block))) / 7.0
                scale16 = np.float16(scale if scale else 1.0)
                scale_count += 1
                q = np.clip(np.rint(block / float(scale16)), -7, 7).astype(np.int8)
                reconstructed_rows[row_idx, start:start + block.size] = q.astype(np.float32) * float(scale16)
        reconstructed = reconstructed_rows.T.reshape(source.shape) if transposed else reconstructed_rows.reshape(source.shape)
        param.data.copy_(torch.from_numpy(reconstructed).to(param.device, param.dtype))
        q_count += int(source.size)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    ids = get_eval_ids(tokenizer)
    score = perplexity(model, ids)
    if args.keep_final_layernorm:
        label, description = "lnfp32", "retain final layer norm tensors in FP32"
    elif args.keep_1d and args.keep_embedding:
        label, description = "1d-embedding-fp32", "retain 1D parameters and tied embedding in FP32"
    elif args.keep_1d:
        label, description = "1d-fp32", "retain 1D parameters in FP32"
    elif args.keep_embedding:
        label, description = "embedding-fp32", "retain tied embedding in FP32"
    else:
        label, description = "all-rtn", "quantize every parameter"
    total = q_count + sum(p.numel() for n, p in model.named_parameters() if n in exempt)
    estimated_rate = (4 * q_count + 16 * scale_count + 32 * (total - q_count)) / total
    report = {
        "model": "openai-community/gpt2", "revision": REVISION,
        "method": f"INT4 group-128 symmetric RTN; {description}",
        "keep_1d": args.keep_1d, "keep_embedding": args.keep_embedding,
        "keep_final_layernorm": args.keep_final_layernorm,
        "exempt_tensors": sorted(exempt), "quantized_parameters": q_count,
        "fp32_parameters": sum(p.numel() for n, p in model.named_parameters() if n in exempt),
        "scale_count": scale_count, "estimated_bits_per_weight": estimated_rate,
        "storage_estimate_note": "Estimated payload from 4-bit packed codes, FP16 group scales, and FP32 exempt tensors; not measured from a serialized artifact.",
        "tokens": int(ids.numel()), "sequence_length": 1024, "perplexity": score,
    }
    out = Path("research-results/standard-format-comparison") / f"int4-{label}-ablation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
