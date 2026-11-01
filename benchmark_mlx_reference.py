"""Evaluate the dense GPT-2 reference with the same MLX and WikiText protocol."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from mlx_lm import load as mlx_load

from benchmark_mlx_gptqmodel import wiki_ids, perplexity, MODEL_PATH, MODEL_ID, REVISION, ROOT
from transformers import AutoTokenizer


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    ids = wiki_ids(tokenizer, "test")
    model, _ = mlx_load(str(MODEL_PATH))
    started = time.perf_counter()
    score, predictions = perplexity(model, ids)
    elapsed = time.perf_counter() - started
    report = {
        "model": MODEL_ID, "revision": REVISION, "method": "dense FP32-source weights loaded in MLX",
        "evaluation_dataset": "WikiText-2 raw test split", "tokens": int(ids.size),
        "predictions_scored": predictions, "sequence_length": 1024,
        "evaluation_token_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "perplexity": score, "evaluation_seconds": elapsed,
        "runtime": "MLX/Metal", "status": "complete",
    }
    out = ROOT / "gpt2-mlx-dense-reference-results.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
