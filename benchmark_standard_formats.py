"""Compare GPT-2 weight storage formats by WikiText-2 perplexity and actual size.

INT8 uses symmetric per-output-channel rounding (one FP16 scale per row).
INT4 uses symmetric round-to-nearest in groups of 128 (one FP16 scale/group).
FP16 and BF16 store cast weights. These are format baselines, not GPTQ/AWQ.
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATASET = "Salesforce/wikitext / wikitext-2-raw-v1 / test"
TOKENS = 1024
GROUP_SIZE = 128


def get_eval_ids(tokenizer, limit: int | None = None) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(line for line in dataset["text"] if line.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids[:limit] if limit else ids


def perplexity(model, ids: torch.Tensor, seq_len: int = TOKENS) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, seq_len):
            end = min(start + seq_len, ids.numel() - 1)
            inputs = ids[start:end].unsqueeze(0)
            labels = ids[start + 1:end + 1].unsqueeze(0)
            logits = model(inputs).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
            )
            total_loss += float(loss)
            total_count += labels.numel()
    return math.exp(total_loss / total_count)


def _pack_int4(q: np.ndarray) -> bytes:
    nibble = (q.astype(np.int16) & 0xF).astype(np.uint8).reshape(-1)
    if nibble.size % 2:
        nibble = np.pad(nibble, (0, 1))
    return (nibble[0::2] | (nibble[1::2] << 4)).tobytes()


def _weight_rows(source: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    """Arrange each output channel as a row; GPT-2 Conv1D stores [input, output]."""
    if source.ndim == 2 and name.startswith("transformer.h."):
        return source.T.reshape(source.shape[1], -1), True
    if source.ndim > 1:
        return source.reshape(source.shape[0], -1), False
    return source.reshape(1, -1), False


def convert_format(model, fmt: str) -> tuple[bytearray, bytearray, dict, int, dict]:
    weight_bytes = bytearray()
    scale_bytes = bytearray()
    tensor_meta = {}
    parameter_count = 0
    absolute_error = 0.0
    squared_error = 0.0
    max_error = 0.0
    for name, param in model.named_parameters():
        source = param.detach().cpu().float().numpy()
        shape = list(param.shape)
        count = int(source.size)
        parameter_count += count
        if fmt == "fp16":
            stored = source.astype(np.float16)
            reconstructed = stored.astype(np.float32)
            weight_bytes.extend(stored.tobytes())
            method = "IEEE float16 cast"
        elif fmt == "bf16":
            packed = param.detach().to(torch.bfloat16).contiguous()
            reconstructed = packed.float().numpy()
            weight_bytes.extend(packed.view(torch.uint8).numpy().tobytes())
            method = "bfloat16 cast"
        elif fmt == "int8-rowwise":
            rows, transposed = _weight_rows(source, name)
            scale = np.max(np.abs(rows), axis=1) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.rint(rows / scale[:, None]), -127, 127).astype(np.int8)
            stored_scale = scale.astype(np.float16)
            reconstructed_rows = q.astype(np.float32) * stored_scale.astype(np.float32)[:, None]
            reconstructed = (reconstructed_rows.T.reshape(source.shape) if transposed
                             else reconstructed_rows.reshape(source.shape))
            weight_bytes.extend(q.tobytes())
            scale_bytes.extend(stored_scale.tobytes())
            method = "symmetric per-output-channel round-to-nearest; FP16 scales"
        elif fmt == "int4-g128":
            rows, transposed = _weight_rows(source, name)
            q = np.empty(rows.size, dtype=np.int8).reshape(rows.shape)
            scales = []
            reconstructed_rows = np.empty_like(rows, dtype=np.float32)
            for row_idx, row in enumerate(rows):
                for start in range(0, row.size, GROUP_SIZE):
                    block = row[start:start + GROUP_SIZE]
                    scale = float(np.max(np.abs(block))) / 7.0
                    if scale == 0:
                        scale = 1.0
                    scale16 = np.float16(scale)
                    scales.append(scale16)
                    q[row_idx, start:start + block.size] = np.clip(
                        np.rint(block / float(scale16)), -7, 7
                    ).astype(np.int8)
                    reconstructed_rows[row_idx, start:start + block.size] = (
                        q[row_idx, start:start + block.size].astype(np.float32) * float(scale16)
                    )
            stored_scales = np.asarray(scales, dtype=np.float16)
            reconstructed = (reconstructed_rows.T.reshape(source.shape) if transposed
                             else reconstructed_rows.reshape(source.shape))
            weight_bytes.extend(_pack_int4(q.reshape(-1)))
            scale_bytes.extend(stored_scales.tobytes())
            method = f"symmetric output-channel round-to-nearest, groups of {GROUP_SIZE}; FP16 scales"
        else:
            raise ValueError(fmt)
        error = source.reshape(-1).astype(np.float64) - reconstructed.reshape(-1).astype(np.float64)
        absolute_error += float(np.abs(error).sum())
        squared_error += float(np.dot(error, error))
        max_error = max(max_error, float(np.abs(error).max(initial=0.0)))
        with torch.no_grad():
            param.copy_(torch.from_numpy(reconstructed).to(dtype=param.dtype))
        tensor_meta[name] = {"shape": shape, "count": count}
    distortion = {"mae": absolute_error / parameter_count,
                  "mse": squared_error / parameter_count,
                  "max_abs_error": max_error}
    return weight_bytes, scale_bytes, tensor_meta, parameter_count, distortion


def write_artifact(output: Path, fmt: str, model, weights: bytearray,
                   scales: bytearray, tensor_meta: dict, count: int,
                   ppl: float, baseline: float, distortion: dict) -> dict:
    manifest = {
        "model": MODEL, "revision": REVISION, "format": fmt,
        "method": {"fp16": "IEEE float16 cast", "bf16": "bfloat16 cast",
                   "int8-rowwise": "symmetric per-output-channel round-to-nearest; FP16 scales",
                   "int4-g128": f"symmetric output-channel round-to-nearest, groups of {GROUP_SIZE}; FP16 scales"}[fmt],
        "dataset": DATASET, "sequence_length": TOKENS, "weights": count,
        "tensor_order": list(tensor_meta), "tensors": tensor_meta,
        "original_perplexity": baseline, "quantized_perplexity": ppl,
        "weight_distortion_vs_fp32": distortion,
    }
    out_path = output / f"gpt2-{fmt}.zip"
    weights_path = output / f".{fmt}-weights.tmp"
    scales_path = output / f".{fmt}-scales.tmp"
    manifest_path = output / f".{fmt}-manifest.tmp"
    weights_path.write_bytes(weights)
    scales_path.write_bytes(scales)
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    config = model.config.to_json_string().encode()
    config_path = output / f".{fmt}-config.tmp"
    config_path.write_bytes(config)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(weights_path, "weights.bin")
        if scales:
            archive.write(scales_path, "scales-fp16.bin")
        archive.write(manifest_path, "manifest.json")
        archive.write(config_path, "config.json")
    for path in (weights_path, scales_path, manifest_path, config_path):
        path.unlink(missing_ok=True)
    actual_bytes = out_path.stat().st_size
    return {
        "format": fmt, "perplexity": ppl, "baseline_perplexity": baseline,
        "delta_perplexity": ppl - baseline, "artifact_bytes": actual_bytes,
        "effective_bits_per_weight": actual_bytes * 8 / count,
        "weight_mae_vs_fp32": distortion["mae"],
        "weight_mse_vs_fp32": distortion["mse"],
        "weight_max_abs_error_vs_fp32": distortion["max_abs_error"],
        "nominal_weight_bits_per_weight": {"fp16": 16, "bf16": 16,
                                           "int8-rowwise": 8,
                                           "int4-g128": 4}[fmt],
        "scale_bytes": len(scales), "occupied_weights": count,
        "artifact": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("research-results/standard-format-comparison"))
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--baseline-perplexity", type=float, default=None,
                        help="Reuse a previously measured baseline on the identical model and test protocol")
    parser.add_argument("--formats", nargs="+", default=["fp16", "bf16", "int8-rowwise", "int4-g128"],
                        choices=["fp16", "bf16", "int8-rowwise", "int4-g128"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ids = get_eval_ids(tokenizer, args.max_tokens)
    if args.baseline_perplexity is None:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                     torch_dtype=torch.float32)
        baseline = perplexity(model, ids)
        del model
    else:
        baseline = args.baseline_perplexity
    print(json.dumps({"format": "fp32-baseline", "perplexity": baseline,
                      "tokens": int(ids.numel()), "reused": args.baseline_perplexity is not None}), flush=True)
    rows = []
    for fmt in args.formats:
        variant = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                       torch_dtype=torch.float32)
        weights, scales, metadata, count, distortion = convert_format(variant, fmt)
        ppl = perplexity(variant, ids)
        row = write_artifact(args.output, fmt, variant, weights, scales,
                             metadata, count, ppl, baseline, distortion)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del variant, weights, scales
    result = {"model": MODEL, "revision": REVISION, "dataset": DATASET,
              "tokens": int(ids.numel()), "sequence_length": TOKENS,
              "baseline_perplexity": baseline,
              "quantization_note": "INT8 and INT4 are explicit RTN format baselines, not GPTQ/AWQ.",
              "results": rows}
    results_path = args.output / "results.json"
    if results_path.exists():
        previous = json.loads(results_path.read_text())
        if previous.get("model") == MODEL and previous.get("revision") == REVISION:
            merged = {row["format"]: row for row in previous.get("results", [])}
            merged.update({row["format"]: row for row in rows})
            result["results"] = [merged[key] for key in sorted(merged)]
    results_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
