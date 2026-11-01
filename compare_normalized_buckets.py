"""Compare normalization and learned buckets for floating-point weights.

This experiment treats a bucket as a shared reconstruction value.  It reports
the resulting quantization error and storage estimate, including the bucket
table and normalization parameters.  A zero-aware variant reserves one bucket
at exactly zero, which makes the effect on sparsity visible.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from compare_cluster_codecs import mean_codebook, normal_codebook
from compare_codecs import NF4_LEVELS, UNIFORM_INT4_LEVELS, UNIFORM_INT5_LEVELS
from rank_function_codec import learned_float_tensors


@dataclass
class Result:
    rmse: float
    maximum: float
    bits_per_weight: float
    zero_fraction: float
    unique_reconstructions: int
    elapsed_ms: float


def result(original: torch.Tensor, reconstructed: torch.Tensor, bits: int, elapsed_ms: float) -> Result:
    error = reconstructed - original
    return Result(
        rmse=error.square().mean().sqrt().item(),
        maximum=error.abs().max().item(),
        bits_per_weight=bits / original.numel(),
        zero_fraction=(reconstructed == 0).float().mean().item(),
        unique_reconstructions=reconstructed.unique().numel(),
        elapsed_ms=elapsed_ms,
    )


def nearest_codes(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (values[:, None] - centers[None, :]).abs().argmin(dim=1)


def fit_kmeans(
    values: torch.Tensor,
    buckets: int,
    force_zero: bool = False,
    forced_center: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit scalar buckets, starting from quantiles and simulating fp16 storage."""
    x = values.to(torch.float32)
    if force_zero:
        nonzero = x[x != 0]
        learned = buckets - 1
        if learned == 0 or nonzero.numel() == 0:
            centers = torch.zeros(buckets, dtype=x.dtype, device=x.device)
        else:
            positions = (torch.arange(learned, device=x.device, dtype=torch.float32) + 0.5) / learned
            centers = torch.cat((torch.tensor([forced_center], device=x.device), torch.quantile(nonzero, positions)))
    else:
        positions = (torch.arange(buckets, device=x.device, dtype=torch.float32) + 0.5) / buckets
        centers = torch.quantile(x, positions)

    for _ in range(60):
        codes = nearest_codes(x, centers)
        updated = centers.clone()
        for code in range(buckets):
            selected = x[codes == code]
            if selected.numel():
                if force_zero and code == 0:
                    updated[code] = forced_center
                else:
                    updated[code] = selected.mean()
        if torch.allclose(updated, centers, rtol=1e-5, atol=1e-7):
            centers = updated
            break
        centers = updated
    centers = centers.to(torch.float16).to(torch.float32)
    return centers, nearest_codes(x, centers)


def global_bucket(values: torch.Tensor, buckets: int, *, normalize: bool, force_zero: bool) -> tuple[torch.Tensor, int, float]:
    x = values
    parameter_bits = 0
    if normalize:
        location = x.mean()
        scale = x.std()
        normalized = (x - location) / scale
        forced_center = (-location / scale).item() if force_zero else 0.0
        centers, codes = fit_kmeans(normalized, buckets, force_zero=force_zero, forced_center=forced_center)
        reconstructed = centers[codes] * scale + location
        if force_zero:
            reconstructed[codes == 0] = 0
        # Store location and scale as fp32. Bucket centers are fp16.
        parameter_bits += 2 * 32
    else:
        centers, codes = fit_kmeans(x, buckets, force_zero=force_zero)
        reconstructed = centers[codes]
    parameter_bits += buckets * 16
    return reconstructed, parameter_bits, float(codes.unique().numel())


def block_reconstruct(values: torch.Tensor, levels: torch.Tensor, block_size: int) -> torch.Tensor:
    reconstructed = torch.empty_like(values)
    levels = levels.to(values.device)
    for start in range(0, values.numel(), block_size):
        end = min(start + block_size, values.numel())
        block = values[start:end]
        scale = block.abs().max()
        if scale == 0:
            reconstructed[start:end] = 0
        else:
            codes = (block[:, None] / scale - levels[None, :]).abs().argmin(dim=1)
            reconstructed[start:end] = levels[codes] * scale
    return reconstructed


def print_row(name: str, r: Result) -> None:
    print(
        f"{name:34} {r.rmse:12.8f} {r.maximum:12.8f} {r.bits_per_weight:9.4f} "
        f"{100*r.zero_fraction:8.3f}% {r.unique_reconstructions:8d} {r.elapsed_ms:9.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--tensor", default="transformer.wte.weight")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--buckets", type=int, default=16)
    args = parser.parse_args()
    if args.block_size < 1 or args.buckets < 2:
        parser.error("block size must be positive and buckets must be at least two")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tensors = learned_float_tensors(state)
    if args.tensor not in tensors:
        parser.error(f"unknown tensor {args.tensor!r}; choose from: {', '.join(tensors)}")
    values = tensors[args.tensor].detach().flatten().to(torch.float32)
    n = values.numel()
    blocks = math.ceil(n / args.block_size)
    code_bits = math.ceil(math.log2(args.buckets))

    print(f"tensor: {args.tensor} | shape: {tuple(tensors[args.tensor].shape)} | values: {n:,}")
    print(f"block size: {args.block_size} | buckets: {args.buckets}")
    print(f"input exact zeros: {100*(values == 0).float().mean().item():.3f}% | input |x| < 1e-3: {100*(values.abs() < 1e-3).float().mean().item():.3f}%")
    print(f"{'method':34} {'RMSE':>12} {'max error':>12} {'bits/w':>9} {'zero':>9} {'unique':>8} {'time ms':>9}")

    started = time.perf_counter()
    reconstructed = block_reconstruct(values, UNIFORM_INT4_LEVELS, args.block_size)
    print_row("uniform INT4 / block absmax", result(values, reconstructed, n * 4 + blocks * 32, (time.perf_counter() - started) * 1000))
    started = time.perf_counter()
    reconstructed = block_reconstruct(values, NF4_LEVELS, args.block_size)
    print_row("NF4 / block absmax", result(values, reconstructed, n * 4 + blocks * 32, (time.perf_counter() - started) * 1000))
    started = time.perf_counter()
    reconstructed = block_reconstruct(values, UNIFORM_INT5_LEVELS, args.block_size)
    print_row("uniform INT5 / block absmax", result(values, reconstructed, n * 5 + blocks * 32, (time.perf_counter() - started) * 1000))

    started = time.perf_counter()
    reconstructed = torch.empty_like(values)
    for start in range(0, n, args.block_size):
        end = min(start + args.block_size, n)
        reconstructed[start:end] = normal_codebook(values[start:end], args.buckets)
    bits = n * code_bits + blocks * 32
    print_row("normalized normal buckets / block", result(values, reconstructed, bits, (time.perf_counter() - started) * 1000))

    started = time.perf_counter()
    reconstructed = torch.empty_like(values)
    for start in range(0, n, args.block_size):
        end = min(start + args.block_size, n)
        reconstructed[start:end] = mean_codebook(values[start:end], args.buckets)
    bits = n * code_bits + blocks * args.buckets * 16
    print_row("learned buckets / block", result(values, reconstructed, bits, (time.perf_counter() - started) * 1000))

    for normalize, force_zero, label in [
        (False, False, "learned buckets / global"),
        (True, False, "z-score + learned buckets / global"),
        (True, True, "z-score + zero bucket / global"),
    ]:
        started = time.perf_counter()
        reconstructed, table_bits, _ = global_bucket(values, args.buckets, normalize=normalize, force_zero=force_zero)
        bits = n * code_bits + table_bits
        print_row(label, result(values, reconstructed, bits, (time.perf_counter() - started) * 1000))


if __name__ == "__main__":
    main()
