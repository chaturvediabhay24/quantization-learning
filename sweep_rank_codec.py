"""Sweep rank-function slice sizes and coefficient precisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rank_function_codec import evaluate_tensor, learned_float_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--tensor", default="transformer.wte.weight")
    parser.add_argument("--function", choices=["line", "normal-quantile"], default="normal-quantile")
    parser.add_argument("--slice-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tensor = learned_float_tensors(state)[args.tensor]
    print(f"tensor: {args.tensor} | function: {args.function}")
    print(f"{'slice':>6} {'coefficients':>14} {'MAE':>12} {'RMSE':>12} {'max error':>12} {'bits/w':>8} {'ratio':>8}")
    for size in args.slice_sizes:
        if size < 2:
            parser.error("all slice sizes must be at least 2")
        for label, dtype in [("float32", torch.float32), ("float16", torch.float16)]:
            result = evaluate_tensor(tensor, size, args.function, dtype)
            print(
                f"{size:6d} {label:>14} {result.mae:12.8f} {result.rmse:12.8f} "
                f"{result.max_abs_error:12.8f} {result.bits_per_weight:8.4f} "
                f"{result.compression_ratio:7.3f}x"
            )


if __name__ == "__main__":
    main()
