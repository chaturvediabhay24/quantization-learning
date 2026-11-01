"""Report per-tensor distortion and scale diagnostics for GPT-2 INT4 group-128 RTN."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from benchmark_standard_formats import GROUP_SIZE, _weight_rows


MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, torch_dtype=torch.float32)
    rows = []
    for name, param in model.named_parameters():
        source = param.detach().cpu().float().numpy()
        arranged, transposed = _weight_rows(source, name)
        restored = np.empty_like(arranged, dtype=np.float32)
        scales = []
        zero_scales = 0
        for row_idx, row in enumerate(arranged):
            for start in range(0, row.size, GROUP_SIZE):
                block = row[start:start + GROUP_SIZE]
                scale = float(np.max(np.abs(block))) / 7.0
                scale16 = np.float16(scale if scale else 1.0)
                scales.append(float(scale16))
                zero_scales += int(scale16 == 0)
                q = np.clip(np.rint(block / float(scale16)), -7, 7).astype(np.int8)
                restored[row_idx, start:start + block.size] = q.astype(np.float32) * float(scale16)
        reconstructed = restored.T.reshape(source.shape) if transposed else restored.reshape(source.shape)
        error = source.astype(np.float64) - reconstructed.astype(np.float64)
        rows.append({
            "tensor": name, "shape": list(source.shape), "parameters": int(source.size),
            "source_min": float(source.min()), "source_max": float(source.max()),
            "mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "max_abs_error": float(np.abs(error).max(initial=0.0)),
            "scale_count": len(scales), "zero_fp16_scales": zero_scales,
            "min_nonzero_scale": min((x for x in scales if x > 0), default=None),
            "max_scale": max(scales, default=None),
        })
    total = sum(item["parameters"] for item in rows)
    total_squared_error = sum(item["parameters"] * item["mse"] for item in rows)
    report = {
        "model": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "method": "symmetric per-output-channel group-128 RTN; FP16 scales; all named parameters",
        "total_parameters": total,
        "zero_fp16_scales": sum(item["zero_fp16_scales"] for item in rows),
        "tensors_sorted_by_mse": sorted(rows, key=lambda x: x["mse"], reverse=True),
        "tensors_sorted_by_squared_error_contribution": sorted(
            [{**item, "squared_error_contribution": item["parameters"] * item["mse"],
              "share_of_total_squared_error": item["parameters"] * item["mse"] / total_squared_error}
             for item in rows], key=lambda x: x["squared_error_contribution"], reverse=True
        ),
        "total_squared_error": total_squared_error,
    }
    out = Path("research-results/standard-format-comparison/int4-layer-diagnostics.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out), "parameters": total, "tensors": len(rows),
                      "zero_fp16_scales": report["zero_fp16_scales"],
                      "top_mse": report["tensors_sorted_by_mse"][:8]}, indent=2))


if __name__ == "__main__":
    main()
