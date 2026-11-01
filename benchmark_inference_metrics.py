"""Measure fixed-workload GPT-2 inference latency for saved experiment variants."""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_standard_formats import REVISION, convert_format


MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION
WIDTHS = [0.0003, 0.001, 0.002, 0.004, 0.008]
PROMPT_TOKENS = 128
GENERATED_TOKENS = 64
REPEATS = 3


def prompt_ids(tokenizer) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(line for line in dataset["text"] if line.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids[:PROMPT_TOKENS].unsqueeze(0)


def run_request(model, input_ids: torch.Tensor, generated_tokens: int) -> dict:
    model.eval()
    with torch.inference_mode():
        start = time.perf_counter()
        outputs = model(input_ids, use_cache=True)
        prefill_seconds = time.perf_counter() - start
        cache = outputs.past_key_values
        token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ttft_seconds = time.perf_counter() - start
        decode_start = time.perf_counter()
        for _ in range(generated_tokens - 1):
            outputs = model(token, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_seconds = time.perf_counter() - decode_start
    return {"prefill_seconds": prefill_seconds, "ttft_seconds": ttft_seconds,
            "decode_seconds": decode_seconds,
            "end_to_end_seconds": ttft_seconds + decode_seconds}


def profile(model, ids: torch.Tensor) -> dict:
    run_request(model, ids, 4)  # warm-up, excluded from reported samples
    samples = [run_request(model, ids, GENERATED_TOKENS) for _ in range(REPEATS)]
    med = {key: statistics.median(sample[key] for sample in samples)
           for key in samples[0]}
    decode_tokens = GENERATED_TOKENS - 1
    return {
        "ttft_ms": med["ttft_seconds"] * 1000,
        "prefill_ms": med["prefill_seconds"] * 1000,
        "prefill_tokens_per_second": PROMPT_TOKENS / med["prefill_seconds"],
        "decode_ms_per_token": med["decode_seconds"] * 1000 / decode_tokens,
        "decode_tokens_per_second": decode_tokens / med["decode_seconds"],
        "end_to_end_64_tokens_ms": med["end_to_end_seconds"] * 1000,
        "samples": REPEATS,
    }


def quantize_bins(model, width: float) -> None:
    for _, param in model.named_parameters():
        values = param.detach().cpu().float().numpy()
        bins = np.floor(values / width).astype(np.int32)
        reconstructed = (bins.astype(np.float32) + 0.5) * width
        with torch.no_grad():
            param.copy_(torch.from_numpy(reconstructed).to(dtype=param.dtype))


def main():
    output = Path("research-results/inference-metrics")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    ids = prompt_ids(tokenizer)
    rows = []
    variants = [
        ("FP32 reference", "fp32", None),
        ("FP16", "fp16", None),
        ("BF16", "bf16", None),
        ("INT8 per-channel RTN", "int8-rowwise", None),
        ("INT4 group-128 RTN", "int4-g128", None),
        *((f"Huffman bins {width:g}", "huffman", width) for width in WIDTHS),
    ]
    for label, kind, parameter in variants:
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, torch_dtype=torch.float32
        )
        setup_load = time.perf_counter() - started
        execution_mode = "FP32"
        if kind in {"fp16", "bf16", "int8-rowwise", "int4-g128"}:
            conversion_start = time.perf_counter()
            if kind in {"fp16", "bf16", "int8-rowwise", "int4-g128"}:
                weights, scales, metadata, count, distortion = convert_format(model, kind)
                del weights, scales, metadata, count, distortion
            if kind in {"fp16", "bf16"}:
                target_dtype = torch.float16 if kind == "fp16" else torch.bfloat16
                try:
                    model.to(dtype=target_dtype)
                    execution_mode = kind
                except Exception:
                    model.to(dtype=torch.float32)
                    execution_mode = "FP32 reconstructed (low-precision op unsupported)"
            setup_conversion = time.perf_counter() - conversion_start
        elif kind == "huffman":
            conversion_start = time.perf_counter()
            quantize_bins(model, float(parameter))
            setup_conversion = time.perf_counter() - conversion_start
            execution_mode = "FP32 reconstructed; Huffman stream not decoded at runtime"
        else:
            setup_conversion = 0.0
        try:
            timing = profile(model, ids)
            error = None
        except Exception as exc:
            timing = None
            error = f"{type(exc).__name__}: {exc}"
        row = {"variant": label, "execution_mode": execution_mode,
               "model_load_seconds": setup_load,
               "variant_setup_seconds": setup_conversion,
               "timing": timing, "error": error}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    result = {
        "model": "openai-community/gpt2", "revision": REVISION,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "cpu_count": __import__("os").cpu_count(),
                     "mps_available": bool(torch.backends.mps.is_available())},
        "workload": {"prompt_tokens": PROMPT_TOKENS,
                     "generated_tokens": GENERATED_TOKENS,
                     "warmup_requests": 1, "measured_requests": REPEATS,
                     "aggregation": "median", "selection": "greedy argmax",
                     "timing_scope": "model-side; excludes tokenization and queueing"},
        "results": rows,
        "note": "INT8/INT4 and Huffman variants are reconstructed to FP32 for execution; these timings do not represent packed-kernel inference.",
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
