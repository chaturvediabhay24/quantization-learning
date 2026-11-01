"""Compare normal-codebook exceptions with matched 256-block quantizers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from compare_codecs import (
    NF4_LEVELS,
    UNIFORM_INT4_LEVELS,
    UNIFORM_INT5_LEVELS,
    UNIFORM_INT6_LEVELS,
    UNIFORM_INT7_LEVELS,
    quantize_blockwise,
)
from rank_function_codec import learned_float_tensors
from sweep_codebook_exceptions import reconstruct


def print_row(name: str, error: torch.Tensor, bits: float) -> None:
    print(
        f"{name:31} {error.abs().mean().item():12.8f} "
        f"{error.square().mean().sqrt().item():12.8f} "
        f"{error.abs().max().item():12.8f} {bits:8.4f}"
    )


def reconstruct_blockwise(values: torch.Tensor, levels: torch.Tensor, block_size: int) -> torch.Tensor:
    flat = values.detach().flatten().to(torch.float32)
    output = torch.empty_like(flat)
    levels = levels.to(flat.device)
    for start in range(0, flat.numel(), block_size):
        end = min(start + block_size, flat.numel())
        block = flat[start:end]
        scale = block.abs().max()
        if scale == 0:
            output[start:end] = 0
        else:
            codes = (block[:, None] / scale - levels[None, :]).abs().argmin(dim=1)
            output[start:end] = levels[codes] * scale
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--tensor", default="transformer.wte.weight")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=32)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0025, 0.005, 0.01])
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    original = learned_float_tensors(state)[args.tensor].detach().flatten().to(torch.float32)
    levels = {
        "uniform INT4": (UNIFORM_INT4_LEVELS, 4),
        "NF4": (NF4_LEVELS, 4),
        "uniform INT5": (UNIFORM_INT5_LEVELS, 5),
        "uniform INT6": (UNIFORM_INT6_LEVELS, 6),
        "uniform INT7": (UNIFORM_INT7_LEVELS, 7),
    }
    print(f"tensor: {args.tensor} | common block size: {args.block_size}")
    print(f"{'method':31} {'MAE':>12} {'RMSE':>12} {'max error':>12} {'bits/w':>8}")
    for name, (codebook, bits) in levels.items():
        result = quantize_blockwise(original, codebook, args.block_size, bits)
        print(f"{name:31} {result.mae:12.8f} {result.rmse:12.8f} {result.max_abs_error:12.8f} {result.bits_per_weight:8.4f}")

    base = reconstruct(original, args.block_size, args.levels)
    int5_reconstructed = reconstruct_blockwise(original, UNIFORM_INT5_LEVELS, args.block_size)
    code_bits = math.ceil(math.log2(args.levels))
    blocks = math.ceil(original.numel() / args.block_size)
    base_bits = original.numel() * code_bits + blocks * 32
    exception_bits = math.ceil(math.log2(args.block_size)) + 32
    for threshold in args.thresholds:
        mask = (base - original).abs() > threshold
        reconstructed = torch.where(mask, original, base)
        total_bits = base_bits + mask.sum().item() * exception_bits
        print_row(
            f"ours normal32 + exceptions {threshold:g}",
            reconstructed - original,
            total_bits / original.numel(),
        )
        # Give INT5 the same exception count and select its worst errors, which
        # is the strongest exact-exception baseline at this storage budget.
        int5_errors = (int5_reconstructed - original).abs()
        count = mask.sum().item()
        keep = torch.zeros_like(mask)
        if count:
            keep[torch.topk(int5_errors, count).indices] = True
        int5_with_exceptions = torch.where(keep, original, int5_reconstructed)
        print_row(
            f"INT5 + same {count:,} exceptions",
            int5_with_exceptions - original,
            total_bits / original.numel(),
        )


if __name__ == "__main__":
    main()
