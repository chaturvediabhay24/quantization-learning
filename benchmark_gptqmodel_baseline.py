"""Calibrate, save, and evaluate a GPTQModel 4-bit GPT-2 baseline on WikiText-2."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import AWQConfig, GPTQConfig


MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_PATH = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION
ROOT = Path("research-results/calibrated-baselines")


def wiki_ids(tokenizer, split: str) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row for row in ds["text"] if row.strip())
    return tokenizer(text, return_tensors="pt").input_ids[0]


def calibration_texts(tokenizer, seed: int, samples: int, seq_len: int = 1024) -> list[str]:
    ids = wiki_ids(tokenizer, "train").numpy()
    if ids.size <= seq_len:
        raise ValueError("WikiText-2 training split is too short for calibration")
    rng = np.random.default_rng(seed)
    starts = rng.choice(ids.size - seq_len, size=samples, replace=False)
    return [tokenizer.decode(ids[s:s + seq_len], skip_special_tokens=False,
                             clean_up_tokenization_spaces=False) for s in starts]


def evaluate(model, ids: torch.Tensor, device: torch.device, seq_len: int = 1024) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    ids = ids.to(device)
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, seq_len):
            end = min(start + seq_len, ids.numel() - 1)
            inputs = ids[start:end].unsqueeze(0)
            labels = ids[start + 1:end + 1].unsqueeze(0)
            logits = model(input_ids=inputs).logits
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
            )
            total_loss += float(loss)
            total_count += labels.numel()
    return math.exp(total_loss / total_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["gptq", "awq"], default="gptq")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    name = f"gpt2-{args.method}-int{args.bits}-g{args.group_size}"
    output_dir = ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / f"{name}-results.json"
    report = {
        "model": MODEL_ID, "revision": REVISION, "method": args.method,
        "bits": args.bits, "group_size": args.group_size,
        "calibration_dataset": "wikitext-2-raw-v1 train split",
        "calibration_samples": args.calibration_samples,
        "calibration_sequence_length": 1024, "calibration_seed": args.seed,
        "device": args.device, "status": "initializing",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    calibration = calibration_texts(tokenizer, args.seed, args.calibration_samples)
    if args.method == "gptq":
        config = GPTQConfig(bits=args.bits, group_size=args.group_size, sym=True, desc_act=False)
        backend = BACKEND.TORCH
    else:
        config = AWQConfig(bits=args.bits, group_size=args.group_size, sym=False)
        backend = BACKEND.AWQ_TORCH

    start = time.perf_counter()
    report["status"] = "loading_model"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        model = GPTQModel.load(str(MODEL_PATH), config, device=args.device, dtype=torch.float16)
    except Exception as exc:
        report.update({"status": "failed_loading_model", "error": repr(exc)})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["load_seconds"] = time.perf_counter() - start
    report["status"] = "quantizing"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    start = time.perf_counter()
    try:
        model.quantize(calibration, batch_size=1, backend=backend)
    except Exception as exc:
        report.update({"status": "failed_quantization", "error": repr(exc)})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["quantization_seconds"] = time.perf_counter() - start
    report["status"] = "saving"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    start = time.perf_counter()
    try:
        model.save(output_dir)
    except Exception as exc:
        report.update({"status": "failed_saving", "error": repr(exc)})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["save_seconds"] = time.perf_counter() - start
    del calibration

    eval_ids = wiki_ids(tokenizer, "test")
    underlying = model.model
    report["evaluation_tokens"] = int(eval_ids.numel())
    report["evaluation_sequence_length"] = 1024
    report["evaluation_dtype"] = "float16 activations / native GPTQModel quantized path"
    start = time.perf_counter()
    try:
        report["perplexity"] = evaluate(underlying, eval_ids, torch.device(args.device))
    except Exception as exc:
        report.update({"status": "failed_evaluation", "error": repr(exc)})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["evaluation_seconds"] = time.perf_counter() - start
    artifact_files = [p for p in output_dir.rglob("*") if p.is_file()]
    report["artifact_bytes"] = sum(p.stat().st_size for p in artifact_files)
    report["artifact_files"] = [{"path": p.relative_to(output_dir).as_posix(), "bytes": p.stat().st_size}
                                 for p in artifact_files]
    report["effective_bits_per_weight"] = report["artifact_bytes"] * 8 / 124_439_808
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
