"""Measure generation latency for dense, Huffman-midpoint, and GPTQ GPT-2 in MLX."""
from __future__ import annotations

import json
import statistics
import time
import zipfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load as mlx_load, stream_generate

from validate_huffman_codec import decode_stream


REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION
GPTQ_PATH = Path("research-results/calibrated-baselines/gpt2-gptq-int4-g128-mlx")
HUFFMAN_ARTIFACT = Path("research-results/gpt2-huffman-sweep/gpt2-huffman-0.012.zip")
OUTPUT = Path("research-results/calibrated-baselines/gpt2-huffman-vs-gptq-generation-mlx.json")
PROMPT_TOKENS = 128
NEW_TOKENS = 64
REPEATS = 3


def save(report: dict) -> None:
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


def make_prompt(tokenizer) -> list[int]:
    text = ("The study of language models shows that careful measurement matters. " * 80)
    token_ids = tokenizer.encode(text)
    if len(token_ids) < PROMPT_TOKENS:
        raise ValueError("Prompt template did not produce enough tokens")
    return [int(token) for token in token_ids[:PROMPT_TOKENS]]


def greedy(logits):
    return mx.argmax(logits, axis=-1)


def one_generation(model, tokenizer, prompt_ids: list[int]) -> dict:
    started = time.perf_counter()
    stream = stream_generate(model, tokenizer, prompt_ids, max_tokens=NEW_TOKENS, sampler=greedy)
    first = next(stream)
    first_yield = time.perf_counter()
    final = first
    for response in stream:
        final = response
    finished = time.perf_counter()
    tokens = int(final.generation_tokens)
    prompt_tps = float(first.prompt_tps)
    prompt_time = len(prompt_ids) / prompt_tps
    decode_time = max(0.0, finished - first_yield)
    decode_tokens = max(0, tokens - 1)
    return {
        "prompt_tokens": len(prompt_ids), "generated_tokens": tokens,
        "ttft_ms": (first_yield - started) * 1000,
        "prompt_processing_ms": prompt_time * 1000,
        "prompt_tokens_per_second": prompt_tps,
        "decode_ms_after_first_token": decode_time * 1000,
        "decode_tokens_per_second": decode_tokens / decode_time if decode_time else None,
        "end_to_end_ms": (finished - started) * 1000,
        "finish_reason": final.finish_reason,
    }


def summarize_model(model, tokenizer, prompt_ids: list[int], report: dict, key: str) -> dict:
    report.update({"status": f"{key}_warmup", "current_model": key})
    save(report)
    warmup = one_generation(model, tokenizer, prompt_ids)
    report.setdefault("warmups", {})[key] = warmup
    save(report)
    runs = []
    for i in range(REPEATS):
        report.update({"status": f"{key}_trial_{i + 1}_of_{REPEATS}", "current_model": key,
                       "trials_done": i})
        save(report)
        runs.append(one_generation(model, tokenizer, prompt_ids))
        report.setdefault("runs", {})[key] = runs
        save(report)
    metrics = {}
    for field in ("ttft_ms", "prompt_processing_ms", "prompt_tokens_per_second",
                  "decode_ms_after_first_token", "decode_tokens_per_second", "end_to_end_ms"):
        values = [run[field] for run in runs if run[field] is not None]
        metrics[field + "_median"] = statistics.median(values)
        metrics[field + "_runs"] = values
    report.setdefault("summary", {})[key] = metrics
    report["status"] = f"{key}_complete"
    save(report)
    return metrics


def main() -> None:
    started = time.perf_counter()
    report = {
        "model": "openai-community/gpt2", "revision": REVISION,
        "status": "loading_dense", "runtime": "MLX/Metal",
        "prompt_tokens": PROMPT_TOKENS, "max_new_tokens": NEW_TOKENS,
        "measured_repeats": REPEATS, "warmups_per_model": 1,
        "sampling": "greedy argmax", "tokenization_in_timing": False,
        "decode_definition": "elapsed from first generated-token yield to stream completion; throughput uses remaining tokens",
        "prefill_definition": "prompt_tokens / MLX stream_generate prompt_tps; includes first-token work per MLX timing",
    }
    save(report)
    try:
        model, tokenizer = mlx_load(str(MODEL_PATH))
        prompt_ids = make_prompt(tokenizer)
        report["prompt_token_ids"] = prompt_ids
        summarize_model(model, tokenizer, prompt_ids, report, "dense_fp32")

        report.update({"status": "decoding_huffman", "current_model": "huffman_fp16"})
        save(report)
        with zipfile.ZipFile(HUFFMAN_ARTIFACT) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            stream = archive.read("stream.bin")
        lengths = {int(k): int(v) for k, v in manifest["huffman_code_lengths"].items()}
        count = sum(int(t["count"]) for t in manifest["tensors"].values())
        t0 = time.perf_counter()
        bins = decode_stream(stream, lengths, count, int(manifest["huffman_payload_bits"]), int(manifest["padding_bits"]))
        offset = 0
        names = manifest["tensor_order"]
        for name in names:
            meta = manifest["tensors"][name]
            n = int(meta["count"])
            values = (bins[offset:offset + n].astype(np.float32) + 0.5) * float(manifest["bin_width"])
            mlx_name = name.removeprefix("transformer.")
            weights = model.sanitize({mlx_name: mx.array(values.reshape(meta["shape"]), dtype=mx.float16)})
            model.load_weights(list(weights.items()), strict=False)
            offset += n
        if offset != count:
            raise ValueError("Decoded weight count differs from Huffman manifest")
        mx.eval(model.parameters())
        report["huffman_decode_install_seconds"] = time.perf_counter() - t0
        report["status"] = "huffman_weights_ready"
        save(report)
        summarize_model(model, tokenizer, prompt_ids, report, "huffman_fp16")

        report.update({"status": "loading_gptq", "current_model": "gptq_w4_g128"})
        save(report)
        t0 = time.perf_counter()
        gptq_model, gptq_tokenizer = mlx_load(str(GPTQ_PATH))
        report["gptq_checkpoint_load_seconds"] = time.perf_counter() - t0
        # Reuse the same explicit token IDs; the two model tokenizers are the same.
        summarize_model(gptq_model, gptq_tokenizer, prompt_ids, report, "gptq_w4_g128")
        report.update({"status": "complete", "total_seconds": time.perf_counter() - started})
        save(report)
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__,
                       "error": str(exc), "total_seconds": time.perf_counter() - started})
        save(report)
        raise


if __name__ == "__main__":
    main()
