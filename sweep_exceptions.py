"""Measure exact float32 exception storage for rank-normal reconstruction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from rank_function_codec import evaluate_tensor, learned_float_tensors, reconstruct_slice


def reconstruct(values: torch.Tensor, slice_size: int) -> torch.Tensor:
    flat = values.detach().flatten().to(torch.float32)
    restored = torch.empty_like(flat)
    for start in range(0, flat.numel(), slice_size):
        end = min(start + slice_size, flat.numel())
        restored[start:end], _ = reconstruct_slice(
            flat[start:end], "normal-quantile", torch.float16
        )
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--tensor", default="transformer.wte.weight")
    parser.add_argument("--slice-size", type=int, default=256)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02])
    args = parser.parse_args()
    if args.slice_size < 2:
        parser.error("--slice-size must be at least 2")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    original = learned_float_tensors(state)[args.tensor].detach().flatten().to(torch.float32)
    baseline = reconstruct(original, args.slice_size)
    base_metrics = evaluate_tensor(original, args.slice_size, "normal-quantile", torch.float16)
    position_bits = math.ceil(math.log2(args.slice_size))
    exception_bits = position_bits + 32  # within-block location plus original float32 value

    print(f"tensor: {args.tensor} | slice size: {args.slice_size}")
    print(
        f"base codec: {base_metrics.bits_per_weight:.4f} bits/weight | "
        f"exact exception: {position_bits}-bit position + float32 value = {exception_bits} bits"
    )
    print(f"{'threshold':>10} {'exceptions':>12} {'share':>9} {'MAE':>12} {'RMSE':>12} {'max error':>12} {'bits/w':>8}")
    for threshold in args.thresholds:
        error = (baseline - original).abs()
        exception_mask = error > threshold
        restored = torch.where(exception_mask, original, baseline)
        final_error = restored - original
        total_bits = base_metrics.compressed_bits + exception_mask.sum().item() * exception_bits
        print(
            f"{threshold:10.4f} {exception_mask.sum().item():12,} "
            f"{exception_mask.float().mean().item():8.3%} "
            f"{final_error.abs().mean().item():12.8f} "
            f"{final_error.square().mean().sqrt().item():12.8f} "
            f"{final_error.abs().max().item():12.8f} "
            f"{total_bits / original.numel():8.4f}"
        )


if __name__ == "__main__":
    main()
