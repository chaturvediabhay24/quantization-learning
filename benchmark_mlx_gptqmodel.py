"""Apple-silicon GPTQ/AWQ calibration and WikiText-2 perplexity benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel


MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION
ROOT = Path("research-results/calibrated-baselines")
PARAMETERS = 124_439_808
SEQ_LEN = 1024


def wiki_ids(tokenizer, split: str) -> np.ndarray:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(line for line in ds["text"] if line.strip())
    return tokenizer(text, return_tensors="np").input_ids[0].astype(np.int32)


def perplexity(model, ids: np.ndarray) -> tuple[float, int]:
    total_loss = 0.0
    total_count = 0
    vocab_size = int(model.args.vocab_size)
    for start in range(0, ids.size - 1, SEQ_LEN):
        end = min(start + SEQ_LEN, ids.size - 1)
        input_ids = mx.array(ids[start:end][None, :])
        labels = mx.array(ids[start + 1:end + 1].astype(np.int32))
        logits = model(input_ids).astype(mx.float32).reshape(-1, vocab_size)
        loss = nn.losses.cross_entropy(logits, labels, reduction="sum")
        mx.eval(loss)
        total_loss += float(loss.item())
        total_count += int(labels.size)
    return math.exp(total_loss / total_count), total_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["gptq", "awq"], default="gptq")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_name = f"gpt2-{args.method}-int{args.bits}-g{args.group_size}-mlx"
    output_dir = ROOT / run_name
    report_path = ROOT / f"{run_name}-results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": MODEL_ID, "revision": REVISION, "method": args.method.upper(),
        "format": "GPTQModel native MLX-LM quantized checkpoint",
        "bits": args.bits, "group_size": args.group_size,
        "device": "Apple Silicon MLX/Metal",
        "calibration_dataset": "WikiText-2 raw train split",
        "calibration_sequence_length": SEQ_LEN,
        "calibration_samples": args.calibration_samples,
        "calibration_seed": args.seed, "status": "initializing",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        train_ids = wiki_ids(tokenizer, "train")
        if train_ids.size <= SEQ_LEN:
            raise ValueError("Training split too short for calibration windows")
        rng = np.random.default_rng(args.seed)
        starts = rng.choice(train_ids.size - SEQ_LEN, size=args.calibration_samples, replace=False)
        calib = np.stack([train_ids[s:s + SEQ_LEN] for s in starts]).astype(np.int32)
        report.update({
            "calibration_token_count": int(calib.size),
            "calibration_token_ids_sha256": hashlib.sha256(calib.tobytes()).hexdigest(),
            "status": "quantizing",
        })
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        start = time.perf_counter()
        GPTQModel.quantize_mlx(
            str(MODEL_PATH), str(output_dir), method=args.method, bits=args.bits,
            group_size=args.group_size, num_samples=args.calibration_samples,
            sequence_length=SEQ_LEN, calibration_data=mx.array(calib), seed=args.seed,
        )
        report["quantize_and_save_seconds"] = time.perf_counter() - start
        report["status"] = "evaluating"
        report_path.write_text(json.dumps(report, indent=2) + "\n")

        from mlx_lm import load as mlx_load
        model, mlx_tokenizer = mlx_load(str(output_dir))
        test_ids = wiki_ids(tokenizer, "test")
        report["test_tokens"] = int(test_ids.size)
        report["evaluation_sequence_length"] = SEQ_LEN
        report["evaluation_token_ids_sha256"] = hashlib.sha256(test_ids.tobytes()).hexdigest()
        start = time.perf_counter()
        score, prediction_count = perplexity(model, test_ids)
        report["perplexity"] = score
        report["predictions_scored"] = prediction_count
        report["evaluation_seconds"] = time.perf_counter() - start
        files = [p for p in output_dir.rglob("*") if p.is_file()]
        report["artifact_bytes"] = sum(p.stat().st_size for p in files)
        report["effective_bits_per_weight"] = report["artifact_bytes"] * 8 / PARAMETERS
        report["artifact_files"] = [{"path": p.relative_to(output_dir).as_posix(), "bytes": p.stat().st_size}
                                     for p in files]
        report["status"] = "complete"
    except Exception as exc:
        report.update({"status": "failed", "error": repr(exc)})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
