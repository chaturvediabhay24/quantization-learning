"""Measure dequantized GPTQ checkpoint errors against the pinned GPT-2 weights."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_load


MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPTQ_PATH = Path("research-results/calibrated-baselines/gpt2-gptq-int4-g128-mlx")
OUTPUT = Path("research-results/calibrated-baselines/gpt2-gptq-int4-g128-mlx-distortion.json")
PARAMETERS = 124_439_808


def save(report: dict) -> None:
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    started = time.perf_counter()
    report = {
        "model": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "method": "GPTQ W4 group 128, dequantized MLX checkpoint",
        "reference": "pinned source checkpoint, MLX FP32",
        "status": "loading_reference",
        "weights_expected": PARAMETERS,
    }
    save(report)
    try:
        reference, _ = mlx_load(str(MODEL_PATH))
        ref_params = dict(tree_flatten(reference.parameters()))
        report.update({"status": "loading_gptq", "reference_tensors": len(ref_params)})
        save(report)
        quantized, _ = mlx_load(str(GPTQ_PATH))
        quant_params = dict(tree_flatten(quantized.parameters()))
        quantized_modules_by_weight = {
            f"{name}.weight": module
            for name, module in quantized.named_modules()
            if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        }
        report.update({"status": "measuring", "gptq_tensors": len(quant_params),
                       "quantized_modules": len(quantized_modules_by_weight),
                       "tensors_done": 0})
        save(report)

        total_abs = 0.0
        total_sq = 0.0
        total_max = 0.0
        total_count = 0
        q_abs = 0.0
        q_sq = 0.0
        q_max = 0.0
        q_count = 0
        per_tensor = {}
        names = list(ref_params)
        for index, name in enumerate(names, start=1):
            ref = ref_params[name].astype(mx.float32)
            if name in quantized_modules_by_weight:
                layer = quantized_modules_by_weight[name]
                qhat = mx.dequantize(
                    layer["weight"], layer["scales"], layer.get("biases"),
                    group_size=layer.group_size, bits=layer.bits,
                    mode=layer.mode, dtype=mx.float32,
                )
            else:
                if name not in quant_params:
                    raise KeyError(f"GPTQ checkpoint has no corresponding tensor: {name}")
                qhat = quant_params[name].astype(mx.float32)
            if tuple(qhat.shape) != tuple(ref.shape):
                raise ValueError(f"Shape mismatch for {name}: reference {ref.shape}, GPTQ {qhat.shape}")
            err = mx.abs(qhat - ref)
            n = math.prod(ref.shape)
            abs_sum = float(mx.sum(err).item())
            sq_sum = float(mx.sum(err * err).item())
            max_err = float(mx.max(err).item())
            per_tensor[name] = {"count": n, "mae": abs_sum / n, "mse": sq_sum / n,
                                "max_abs_error": max_err,
                                "quantized_4bit": name in quantized_modules_by_weight}
            total_abs += abs_sum
            total_sq += sq_sum
            total_max = max(total_max, max_err)
            total_count += n
            if name in quantized_modules_by_weight:
                q_abs += abs_sum
                q_sq += sq_sum
                q_max = max(q_max, max_err)
                q_count += n
            if index % 16 == 0 or index == len(names):
                report.update({"status": "measuring", "tensors_done": index,
                               "tensors_total": len(names), "weights_done": total_count,
                               "partial_mae": total_abs / total_count,
                               "partial_mse": total_sq / total_count})
                save(report)

        if total_count != PARAMETERS:
            raise ValueError(f"Expected {PARAMETERS} unique parameters, found {total_count}")
        report.update({
            "status": "complete", "weights": total_count,
            "overall_mae": total_abs / total_count,
            "overall_mse": total_sq / total_count,
            "overall_rmse": math.sqrt(total_sq / total_count),
            "overall_max_abs_error": total_max,
            "quantized_4bit_weight_count": q_count,
            "quantized_4bit_weight_mae": q_abs / q_count,
            "quantized_4bit_weight_mse": q_sq / q_count,
            "quantized_4bit_weight_rmse": math.sqrt(q_sq / q_count),
            "quantized_4bit_weight_max_abs_error": q_max,
            "per_tensor": per_tensor,
            "elapsed_seconds": time.perf_counter() - started,
            "note": "Packed 4-bit linear and embedding weights are dequantized with MLX's affine reconstruction; retained FP16/nonquantized tensors are compared in their stored dtype. Tied weights counted once.",
        })
        save(report)
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__,
                       "error": str(exc), "elapsed_seconds": time.perf_counter() - started})
        save(report)
        raise


if __name__ == "__main__":
    main()
