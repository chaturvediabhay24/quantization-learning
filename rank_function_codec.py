"""Evaluate rank-and-function reconstruction for a PyTorch checkpoint.

For every contiguous slice of a flattened tensor, this experiment:
1. sorts the values;
2. fits ``f(rank) = slope * rank + intercept`` to those sorted values;
3. stores the rank of each original position conceptually as a permutation; and
4. reconstructs each value with ``f(original_rank)``.

The code measures numerical error and the exact *minimum* fixed-length bit count
for one permutation per slice: ceil(log2(slice_length!)). It does not write a
compressed model file yet.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ReconstructionMetrics:
    values: int
    slices: int
    mae: float
    rmse: float
    max_abs_error: float
    original_bits: int
    rank_bits: int
    coefficient_bits: int

    @property
    def compressed_bits(self) -> int:
        return self.rank_bits + self.coefficient_bits

    @property
    def bits_per_weight(self) -> float:
        return self.compressed_bits / self.values

    @property
    def compression_ratio(self) -> float:
        return self.original_bits / self.compressed_bits


def permutation_bits(length: int) -> int:
    """Minimum fixed-length bits needed to identify one of length! permutations."""
    return math.ceil(math.log2(math.factorial(length))) if length > 1 else 0


def reconstruct_slice(
    values: torch.Tensor,
    function: str = "line",
    coefficient_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    """Reconstruct one vector from sorted ranks with a two-parameter function."""
    length = values.numel()
    sorted_values, original_positions = torch.sort(values)
    ranks = torch.empty(length, dtype=torch.long, device=values.device)
    ranks[original_positions] = torch.arange(length, device=values.device)

    if function == "line":
        x = torch.arange(length, dtype=torch.float64, device=values.device)
    elif function == "normal-quantile":
        probabilities = (torch.arange(length, dtype=torch.float64, device=values.device) + 0.5) / length
        x = torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=values.device)) * torch.erfinv(
            2 * probabilities - 1
        )
    else:
        raise ValueError(f"unknown function: {function}")
    y = sorted_values.to(torch.float64)
    x_centered = x - x.mean()
    slope = (x_centered * (y - y.mean())).sum() / (x_centered.square().sum())
    intercept = y.mean() - slope * x.mean()
    # Simulate writing and reloading the function parameters at the selected precision.
    slope = slope.to(coefficient_dtype).to(torch.float64)
    intercept = intercept.to(coefficient_dtype).to(torch.float64)
    reconstructed = (slope * x[ranks] + intercept).to(values.dtype)
    return reconstructed, permutation_bits(length)


def evaluate_tensor(
    values: torch.Tensor,
    slice_size: int,
    function: str = "line",
    coefficient_dtype: torch.dtype = torch.float32,
) -> ReconstructionMetrics:
    flat = values.detach().flatten().to(torch.float32)
    reconstructed = torch.empty_like(flat)
    rank_bits = 0
    slices = 0
    for start in range(0, flat.numel(), slice_size):
        end = min(start + slice_size, flat.numel())
        restored, bits = reconstruct_slice(flat[start:end], function, coefficient_dtype)
        reconstructed[start:end] = restored
        rank_bits += bits
        slices += 1

    error = reconstructed - flat
    coefficient_bits = slices * 2 * torch.finfo(coefficient_dtype).bits
    return ReconstructionMetrics(
        values=flat.numel(),
        slices=slices,
        mae=error.abs().mean().item(),
        rmse=error.square().mean().sqrt().item(),
        max_abs_error=error.abs().max().item(),
        original_bits=flat.numel() * 32,
        rank_bits=rank_bits,
        coefficient_bits=coefficient_bits,
    )


def print_metrics(name: str, metrics: ReconstructionMetrics) -> None:
    print(f"{name}")
    print(f"  values:              {metrics.values:,}")
    print(f"  slices:              {metrics.slices:,}")
    print(f"  MAE:                 {metrics.mae:.8f}")
    print(f"  RMSE:                {metrics.rmse:.8f}")
    print(f"  maximum abs. error:  {metrics.max_abs_error:.8f}")
    print(f"  rank bits:           {metrics.rank_bits:,}")
    print(f"  coefficient bits:    {metrics.coefficient_bits:,}")
    print(f"  total bits:          {metrics.compressed_bits:,}")
    print(f"  bits per weight:     {metrics.bits_per_weight:.4f}")
    print(f"  compression ratio:   {metrics.compression_ratio:.3f}x vs float32")


def learned_float_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Exclude GPT-2 causal-mask buffers and scalar mask constants."""
    return {
        name: value
        for name, value in state.items()
        if value.is_floating_point()
        and "attn.bias" not in name
        and "masked_bias" not in name
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/tiny-gpt2/pytorch_model.bin"),
    )
    parser.add_argument("--slice-size", type=int, default=16)
    parser.add_argument("--function", choices=["line", "normal-quantile"], default="line")
    parser.add_argument("--coefficient-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--tensor",
        default="transformer.wte.weight",
        help="Tensor name, or 'all' for all learned floating-point tensors.",
    )
    args = parser.parse_args()
    if args.slice_size < 2:
        parser.error("--slice-size must be at least 2")
    coefficient_dtype = getattr(torch, args.coefficient_dtype)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tensors = learned_float_tensors(state)
    selected = tensors if args.tensor == "all" else {args.tensor: tensors[args.tensor]}
    if args.function == "line":
        description = "f(rank) = slope * rank + intercept"
    else:
        description = "f(rank) = scale * normal_quantile((rank + 0.5) / slice_size) + location"
    print(f"slice size: {args.slice_size} | {description} | coefficients: {args.coefficient_dtype}")
    print(f"full-slice permutation cost: {permutation_bits(args.slice_size)} bits")

    totals = [0, 0, 0, 0, 0.0, 0.0, 0.0]
    for name, value in selected.items():
        metrics = evaluate_tensor(value, args.slice_size, args.function, coefficient_dtype)
        print_metrics(name, metrics)
        totals[0] += metrics.values
        totals[1] += metrics.slices
        totals[2] += metrics.original_bits
        totals[3] += metrics.rank_bits
        totals[4] += metrics.coefficient_bits
        totals[5] += metrics.mae * metrics.values
        totals[6] += metrics.rmse**2 * metrics.values

    if len(selected) > 1:
        values, slices, original, rank, coeff, weighted_mae, weighted_mse = totals
        combined = ReconstructionMetrics(
            values=values,
            slices=slices,
            mae=weighted_mae / values,
            rmse=(weighted_mse / values) ** 0.5,
            max_abs_error=float("nan"),
            original_bits=original,
            rank_bits=rank,
            coefficient_bits=int(coeff),
        )
        print_metrics("ALL LEARNED FLOAT TENSORS", combined)


if __name__ == "__main__":
    main()
