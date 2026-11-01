"""Compare nearby-value mean clustering with a function-generated codebook.

Both methods encode each weight with a shared cluster ID instead of a unique
sorted rank. Mean clustering learns a value for every cluster; the normal
codebook learns just a float16 scale and location per block.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from compare_codecs import NF4_LEVELS, UNIFORM_INT4_LEVELS, UNIFORM_INT5_LEVELS, quantize_blockwise
from rank_function_codec import learned_float_tensors


@dataclass
class Result:
    mae: float
    rmse: float
    maximum: float
    bits_per_weight: float


def normal_levels(count: int, device: torch.device) -> torch.Tensor:
    p = (torch.arange(count, dtype=torch.float64, device=device) + 0.5) / count
    return torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device)) * torch.erfinv(2 * p - 1)


def nearest_codes(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (values[:, None] - centers[None, :]).abs().argmin(dim=1)


def fit_normal_parameters(block: torch.Tensor, levels: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a normal-quantile grid and return its levels, scale, and location."""
    z = normal_levels(levels, block.device)
    scale = block.to(torch.float64).std() / z.std()
    location = block.to(torch.float64).mean()
    x = block.to(torch.float64)
    for _ in range(30):
        codes = nearest_codes(x, scale * z + location)
        chosen = z[codes]
        centered = chosen - chosen.mean()
        next_scale = (centered * (x - x.mean())).sum() / centered.square().sum()
        next_location = x.mean() - next_scale * chosen.mean()
        if torch.allclose(next_scale, scale) and torch.allclose(next_location, location):
            break
        scale, location = next_scale, next_location
    return z, scale, location


def normal_codebook(block: torch.Tensor, levels: int) -> torch.Tensor:
    """Fit float16 scale/location for a fixed normal-quantile codebook."""
    z, scale, location = fit_normal_parameters(block, levels)
    x = block.to(torch.float64)
    # Simulate the stored float16 parameters before final coding and decoding.
    scale = scale.to(torch.float16).to(torch.float64)
    location = location.to(torch.float16).to(torch.float64)
    codes = nearest_codes(x, scale * z + location)
    return (scale * z[codes] + location).to(torch.float32)


def mean_codebook(block: torch.Tensor, levels: int) -> torch.Tensor:
    """One-dimensional Lloyd-Max / k-means clustering, storing float16 means."""
    x = block.to(torch.float32)
    if x.numel() <= levels:
        return x.clone()
    z, scale, location = fit_normal_parameters(x, levels)
    # Start from the already-fitted normal grid. Lloyd updates can only reduce
    # squared reconstruction error before the final float16 table rounding.
    centers = (scale * z + location).to(torch.float32)
    for _ in range(100):
        codes = nearest_codes(x, centers)
        updated = centers.clone()
        for code in range(levels):
            selected = x[codes == code]
            if selected.numel():
                updated[code] = selected.mean()
        if torch.allclose(updated, centers):
            break
        centers = updated
    # Simulate storing the learned mean table as float16.
    centers = centers.to(torch.float16).to(torch.float32)
    return centers[nearest_codes(x, centers)]


def evaluate(values: torch.Tensor, block_size: int, levels: int, method: str) -> Result:
    flat = values.detach().flatten().to(torch.float32)
    reconstructed = torch.empty_like(flat)
    for start in range(0, flat.numel(), block_size):
        end = min(start + block_size, flat.numel())
        block = flat[start:end]
        reconstructed[start:end] = (
            normal_codebook(block, levels) if method == "normal" else mean_codebook(block, levels)
        )
    error = reconstructed - flat
    code_bits = math.ceil(math.log2(levels))
    blocks = math.ceil(flat.numel() / block_size)
    parameter_bits = blocks * (32 if method == "normal" else levels * 16)
    return Result(
        mae=error.abs().mean().item(),
        rmse=error.square().mean().sqrt().item(),
        maximum=error.abs().max().item(),
        bits_per_weight=(flat.numel() * code_bits + parameter_bits) / flat.numel(),
    )


def print_row(name: str, result: Result) -> None:
    print(f"{name:27} {result.mae:12.8f} {result.rmse:12.8f} {result.maximum:12.8f} {result.bits_per_weight:8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--tensor", default="transformer.wte.weight")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--levels", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    values = learned_float_tensors(state)[args.tensor]
    print(f"tensor: {args.tensor} | block size: {args.block_size}")
    print(f"{'method':27} {'MAE':>12} {'RMSE':>12} {'max error':>12} {'bits/w':>8}")
    int4 = quantize_blockwise(values, UNIFORM_INT4_LEVELS, args.block_size, 4)
    nf4 = quantize_blockwise(values, NF4_LEVELS, args.block_size, 4)
    int5 = quantize_blockwise(values, UNIFORM_INT5_LEVELS, args.block_size, 5)
    print_row("uniform INT4", Result(int4.mae, int4.rmse, int4.max_abs_error, int4.bits_per_weight))
    print_row("NF4", Result(nf4.mae, nf4.rmse, nf4.max_abs_error, nf4.bits_per_weight))
    print_row("uniform INT5", Result(int5.mae, int5.rmse, int5.max_abs_error, int5.bits_per_weight))
    for levels in args.levels:
        if levels < 2 or levels & (levels - 1):
            parser.error("--levels values must be powers of two greater than one")
        print_row(f"normal codebook ({levels})", evaluate(values, args.block_size, levels, "normal"))
        print_row(f"mean codebook ({levels})", evaluate(values, args.block_size, levels, "mean"))


if __name__ == "__main__":
    main()
