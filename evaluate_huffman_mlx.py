"""Evaluate a decoded Huffman-bin model through the same MLX path as GPTQ."""
from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx_lm import load as mlx_load
from transformers import AutoTokenizer

from validate_huffman_codec import decode_stream


MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION
ARTIFACT = Path("research-results/gpt2-huffman-sweep/gpt2-huffman-0.012.zip")
OUTPUT = Path("research-results/calibrated-baselines/gpt2-huffman-0.012-mlx-eval.json")
SEQ_LEN = 1024


def save_report(report: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


def get_ids(tokenizer) -> np.ndarray:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(line for line in ds["text"] if line.strip())
    return tokenizer(text, return_tensors="np").input_ids[0].astype(np.int32)


def evaluate(model, ids: np.ndarray, report: dict) -> tuple[float, int]:
    loss_sum = 0.0
    scored = 0
    vocab = int(model.args.vocab_size)
    n_chunks = math.ceil((ids.size - 1) / SEQ_LEN)
    for chunk, start in enumerate(range(0, ids.size - 1, SEQ_LEN), start=1):
        end = min(start + SEQ_LEN, ids.size - 1)
        inp = mx.array(ids[start:end][None, :])
        labels = mx.array(ids[start + 1:end + 1])
        logits = model(inp).astype(mx.float32).reshape(-1, vocab)
        loss = nn.losses.cross_entropy(logits, labels, reduction="sum")
        mx.eval(loss)
        loss_sum += float(loss.item())
        scored += int(labels.size)
        if chunk % 32 == 0 or chunk == n_chunks:
            report.update({"evaluation_chunks_done": chunk, "evaluation_chunks_total": n_chunks,
                           "predictions_scored": scored, "partial_perplexity": math.exp(loss_sum / scored)})
            save_report(report)
    return math.exp(loss_sum / scored), scored


def main() -> None:
    started_all = time.perf_counter()
    report = {
        "artifact": str(ARTIFACT), "model": MODEL_ID, "revision": REVISION,
        "status": "starting", "runtime": "MLX/Metal", "weight_dtype": "float16",
        "dataset": "WikiText-2 raw test", "sequence_length": SEQ_LEN,
    }
    save_report(report)
    try:
        report["status"] = "decoding"
        save_report(report)
        with zipfile.ZipFile(ARTIFACT) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            stream = archive.read("stream.bin")
        lengths = {int(k): int(v) for k, v in manifest["huffman_code_lengths"].items()}
        count = sum(int(t["count"]) for t in manifest["tensors"].values())
        decode_started = time.perf_counter()
        bins = decode_stream(stream, lengths, count, int(manifest["huffman_payload_bits"]), int(manifest["padding_bits"]))
        if Counter(map(int, bins)) != Counter({int(k): int(v) for k, v in manifest["symbol_frequencies"].items()}):
            raise ValueError("Decoded histogram does not match manifest")
        report.update({"status": "decoded", "weights_decoded": count,
                       "decode_seconds": time.perf_counter() - decode_started,
                       "codec_round_trip_bin_ids": "passed"})
        save_report(report)

        report["status"] = "loading_base_model"
        save_report(report)
        model, _ = mlx_load(str(MODEL_PATH))
        offset = 0
        weight_started = time.perf_counter()
        tensor_order = manifest["tensor_order"]
        for i, name in enumerate(tensor_order, start=1):
            meta = manifest["tensors"][name]
            n = int(meta["count"])
            midpoint = (bins[offset:offset + n].astype(np.float32) + 0.5) * float(manifest["bin_width"])
            # The archive uses Transformers names; MLX GPT-2 stores the same
            # parameters under the `model.` prefix and transposes Conv1D matrices.
            mlx_name = name.removeprefix("transformer.")
            weight = mx.array(midpoint.reshape(meta["shape"]), dtype=mx.float16)
            converted = model.sanitize({mlx_name: weight})
            model.load_weights(list(converted.items()), strict=False)
            offset += n
            if i % 16 == 0 or i == len(tensor_order):
                report.update({"status": "installing_decoded_weights", "tensors_installed": i,
                               "tensors_total": len(tensor_order), "weights_installed": offset})
                save_report(report)
        if offset != count:
            raise ValueError("Decoded parameter count differs from manifest")
        mx.eval(model.parameters())
        report.update({"status": "weights_ready", "weights_installed": offset,
                       "weight_install_seconds": time.perf_counter() - weight_started})
        save_report(report)
        del bins

        report["status"] = "loading_dataset"
        save_report(report)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        ids = get_ids(tokenizer)
        report.update({"tokens": int(ids.size), "evaluation_token_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                       "status": "evaluating"})
        save_report(report)
        eval_started = time.perf_counter()
        ppl, predictions = evaluate(model, ids, report)
        report.update({"status": "complete", "predictions_scored": predictions,
                       "perplexity": ppl, "evaluation_seconds": time.perf_counter() - eval_started,
                       "total_seconds": time.perf_counter() - started_all})
        save_report(report)
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                       "total_seconds": time.perf_counter() - started_all})
        save_report(report)
        raise


if __name__ == "__main__":
    main()
