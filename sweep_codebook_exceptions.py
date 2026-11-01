"""Test exact float32 exceptions on the function-generated normal codebook."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from compare_cluster_codecs import normal_codebook
from rank_function_codec import learned_float_tensors


def reconstruct(values: torch.Tensor, block_size: int, levels: int) -> torch.Tensor:
    flat = values.detach().flatten().to(torch.float32)
    output = torch.empty_like(flat)
    for start in range(0, flat.numel(), block_size):
        end = min(start + block_size, flat.numel())
        output[start:end] = normal_codebook(flat[start:end], levels)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--tensor", default="transformer.wte.weight")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=32)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02])
    args = parser.parse_args()
    if args.levels < 2 or args.levels & (args.levels - 1):
        parser.error("--levels must be a power of two greater than one")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    original = learned_float_tensors(state)[args.tensor].detach().flatten().to(torch.float32)
    base = reconstruct(original, args.block_size, args.levels)
    code_bits = math.ceil(math.log2(args.levels))
    blocks = math.ceil(original.numel() / args.block_size)
    base_bits = original.numel() * code_bits + blocks * 32  # float16 scale + location
    position_bits = math.ceil(math.log2(args.block_size))
    exception_bits = position_bits + 32

    print(f"tensor: {args.tensor} | blocks: {args.block_size} | normal levels: {args.levels}")
    print(
        f"base: {base_bits / original.numel():.4f} bits/weight | "
        f"exception: {position_bits}-bit position + float32 = {exception_bits} bits"
    )
    print(f"{'threshold':>10} {'exceptions':>12} {'share':>9} {'MAE':>12} {'RMSE':>12} {'max error':>12} {'bits/w':>8}")
    for threshold in args.thresholds:
        mask = (base - original).abs() > threshold
        restored = torch.where(mask, original, base)
        error = restored - original
        total_bits = base_bits + mask.sum().item() * exception_bits
        print(
            f"{threshold:10.4f} {mask.sum().item():12,} {mask.float().mean().item():8.3%} "
            f"{error.abs().mean().item():12.8f} {error.square().mean().sqrt().item():12.8f} "
            f"{error.abs().max().item():12.8f} {total_bits / original.numel():8.4f}"
        )


if __name__ == "__main__":
    main()
