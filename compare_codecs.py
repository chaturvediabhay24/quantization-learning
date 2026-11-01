"""Compare rank-function reconstruction with blockwise INT4 and NF4.

All methods use the same contiguous block size. Reported storage includes one
float32 block scale for INT4/NF4 and two float32 fitted coefficients for the
rank-function method. It is a weight-reconstruction benchmark, not a language
quality benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from rank_function_codec import evaluate_tensor, learned_float_tensors


# The 16-level NormalFloat (NF4) codebook used by bitsandbytes / QLoRA.
NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928,
        -0.52507305,
        -0.3949175,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.0,
        0.0795803,
        0.1609302,
        0.2461123,
        0.33791524,
        0.44070983,
        0.562617,
        0.72295684,
        1.0,
    ]
)
UNIFORM_INT4_LEVELS = torch.linspace(-1.0, 1.0, 16)
UNIFORM_INT5_LEVELS = torch.linspace(-1.0, 1.0, 32)
UNIFORM_INT6_LEVELS = torch.linspace(-1.0, 1.0, 64)
UNIFORM_INT7_LEVELS = torch.linspace(-1.0, 1.0, 128)


@dataclass
class Metrics:
    mae: float
    rmse: float
    max_abs_error: float
    bits: int
    values: int

    @property
    def bits_per_weight(self) -> float:
        return self.bits / self.values

    @property
    def ratio_vs_fp32(self) -> float:
        return 32 / self.bits_per_weight


def quantize_blockwise(
    values: torch.Tensor, levels: torch.Tensor, block_size: int, code_bits: int
) -> Metrics:
    flat = values.detach().flatten().to(torch.float32)
    restored = torch.empty_like(flat)
    levels = levels.to(flat.device)
    blocks = 0
    for start in range(0, flat.numel(), block_size):
        end = min(start + block_size, flat.numel())
        block = flat[start:end]
        scale = block.abs().max()
        if scale == 0:
            restored[start:end] = 0
        else:
            normalized = block / scale
            codes = (normalized[:, None] - levels[None, :]).abs().argmin(dim=1)
            restored[start:end] = levels[codes] * scale
        blocks += 1

    error = restored - flat
    # Fixed-size codes plus one float32 absmax scale per block.
    bits = flat.numel() * code_bits + blocks * 32
    return Metrics(
        mae=error.abs().mean().item(),
        rmse=error.square().mean().sqrt().item(),
        max_abs_error=error.abs().max().item(),
        bits=bits,
        values=flat.numel(),
    )


def rank_function_metrics(values: torch.Tensor, block_size: int) -> Metrics:
    result = evaluate_tensor(values, block_size)
    return Metrics(
        mae=result.mae,
        rmse=result.rmse,
        max_abs_error=result.max_abs_error,
        bits=result.compressed_bits,
        values=result.values,
    )


def normal_quantile_metrics(
    values: torch.Tensor, block_size: int, coefficient_dtype: torch.dtype
) -> Metrics:
    result = evaluate_tensor(values, block_size, "normal-quantile", coefficient_dtype)
    return Metrics(
        mae=result.mae,
        rmse=result.rmse,
        max_abs_error=result.max_abs_error,
        bits=result.compressed_bits,
        values=result.values,
    )


def print_row(name: str, method: str, result: Metrics) -> None:
    print(
        f"{name:31} {method:18} {result.mae:12.8f} {result.rmse:12.8f} "
        f"{result.max_abs_error:12.8f} {result.bits_per_weight:8.4f} "
        f"{result.ratio_vs_fp32:7.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tensor", default="transformer.wte.weight", help="Tensor name or 'all'.")
    parser.add_argument(
        "--rank-coefficient-dtype", choices=["float32", "float16"], default="float32"
    )
    args = parser.parse_args()
    if args.block_size < 2:
        parser.error("--block-size must be at least 2")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    available = learned_float_tensors(state)
    tensors = available if args.tensor == "all" else {args.tensor: available[args.tensor]}
    rank_coefficient_dtype = getattr(torch, args.rank_coefficient_dtype)
    print(
        f"block size: {args.block_size}; INT4/NF4 scales: float32; "
        f"rank-function coefficients: {args.rank_coefficient_dtype}"
    )
    print(f"{'tensor':31} {'method':18} {'MAE':>12} {'RMSE':>12} {'max error':>12} {'bits/w':>8} {'ratio':>8}")

    for name, values in tensors.items():
        print_row(name, "rank + line", rank_function_metrics(values, args.block_size))
        print_row(
            name,
            "rank + normal",
            normal_quantile_metrics(values, args.block_size, rank_coefficient_dtype),
        )
        print_row(name, "uniform INT4", quantize_blockwise(values, UNIFORM_INT4_LEVELS, args.block_size, 4))
        print_row(name, "NF4", quantize_blockwise(values, NF4_LEVELS, args.block_size, 4))
        print_row(name, "uniform INT5", quantize_blockwise(values, UNIFORM_INT5_LEVELS, args.block_size, 5))
        print_row(name, "uniform INT6", quantize_blockwise(values, UNIFORM_INT6_LEVELS, args.block_size, 6))
        print_row(name, "uniform INT7", quantize_blockwise(values, UNIFORM_INT7_LEVELS, args.block_size, 7))


if __name__ == "__main__":
    main()
