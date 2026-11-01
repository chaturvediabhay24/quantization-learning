"""Add FP32-relative weight MAE/MSE for existing Huffman bin-width results."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION
WIDTHS = [0.0003, 0.001, 0.002, 0.004, 0.008, 0.012]


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
                                                 torch_dtype=torch.float32)
    sums = {w: [0.0, 0.0, 0.0, 0] for w in WIDTHS}
    fp16_storage_sums = [0.0, 0.0, 0.0, 0]
    for _, param in model.named_parameters():
        values = param.detach().cpu().float().numpy().reshape(-1)
        for width in WIDTHS:
            bins = np.floor(values / width).astype(np.int32)
            reconstructed = (bins.astype(np.float32) + 0.5) * width
            error = values.astype(np.float64) - reconstructed.astype(np.float64)
            sums[width][0] += float(np.abs(error).sum())
            sums[width][1] += float(np.dot(error, error))
            sums[width][2] = max(sums[width][2], float(np.abs(error).max(initial=0.0)))
            sums[width][3] += values.size
            if width == 0.012:
                half_values = reconstructed.astype(np.float16).astype(np.float32)
                half_error = values.astype(np.float64) - half_values.astype(np.float64)
                fp16_storage_sums[0] += float(np.abs(half_error).sum())
                fp16_storage_sums[1] += float(np.dot(half_error, half_error))
                fp16_storage_sums[2] = max(fp16_storage_sums[2], float(np.abs(half_error).max(initial=0.0)))
                fp16_storage_sums[3] += values.size
    result_path = Path("research-results/gpt2-huffman-sweep/results.json")
    data = json.loads(result_path.read_text())
    for row in data["results"]:
        width = float(row["bin_width"])
        absolute, squared, maximum, count = sums[width]
        row["weight_mae_vs_fp32"] = absolute / count
        row["weight_mse_vs_fp32"] = squared / count
        row["weight_max_abs_error_vs_fp32"] = maximum
        if width == 0.012:
            row["weight_mae_fp16_storage_vs_fp32"] = fp16_storage_sums[0] / fp16_storage_sums[3]
            row["weight_mse_fp16_storage_vs_fp32"] = fp16_storage_sums[1] / fp16_storage_sums[3]
            row["weight_max_abs_error_fp16_storage_vs_fp32"] = fp16_storage_sums[2]
    result_path.write_text(json.dumps(data, indent=2) + "\n")
    metrics = {str(w): {"mae_vs_fp32": sums[w][0] / sums[w][3],
                        "mse_vs_fp32": sums[w][1] / sums[w][3],
                        "max_abs_error_vs_fp32": sums[w][2],
                        "weights": sums[w][3]} for w in WIDTHS}
    metrics["0.012"]["mae_fp16_storage_vs_fp32"] = fp16_storage_sums[0] / fp16_storage_sums[3]
    metrics["0.012"]["mse_fp16_storage_vs_fp32"] = fp16_storage_sums[1] / fp16_storage_sums[3]
    metrics["0.012"]["max_abs_error_fp16_storage_vs_fp32"] = fp16_storage_sums[2]
    Path("research-results/gpt2-huffman-sweep/weight-distortion.json").write_text(
        json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
