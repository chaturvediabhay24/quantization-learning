"""Compare full Huffman coding with a frequent-symbol escape codec.

The input is first quantized into scalar IDs.  A top-K codec Huffman-codes only
the most frequent IDs and replaces every other ID with one ESC symbol.  The
lossless variant writes the original fixed-width ID after ESC.  The optional
collapsed variant maps ESC to one fallback reconstruction value and therefore
adds quantization error.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from compare_codecs import NF4_LEVELS, UNIFORM_INT4_LEVELS, UNIFORM_INT5_LEVELS
from compare_entropy_codecs import huffman_lengths
from rank_function_codec import learned_float_tensors


@dataclass
class Quantized:
    ids: list[int]
    scales: torch.Tensor
    reconstructed: torch.Tensor
    levels: torch.Tensor
    code_bits: int


@dataclass
class Result:
    bits: int
    rmse: float
    rare_fraction: float
    metadata_bits: int
    payload_bits: int

    def bits_per_weight(self, values: int) -> float:
        return self.bits / values


def quantize(values: torch.Tensor, levels: torch.Tensor, block_size: int, code_bits: int) -> Quantized:
    flat = values.detach().flatten().to(torch.float32)
    levels = levels.to(flat.device)
    reconstructed = torch.empty_like(flat)
    ids: list[int] = []
    scales: list[float] = []
    for start in range(0, flat.numel(), block_size):
        end = min(start + block_size, flat.numel())
        block = flat[start:end]
        scale = block.abs().max()
        if scale == 0:
            codes = torch.zeros(block.numel(), dtype=torch.long, device=flat.device)
            reconstructed[start:end] = 0
        else:
            codes = (block[:, None] / scale - levels[None, :]).abs().argmin(dim=1)
            reconstructed[start:end] = levels[codes] * scale
        ids.extend(codes.tolist())
        scales.append(float(scale))
    return Quantized(ids, torch.tensor(scales), reconstructed, levels.cpu(), code_bits)


def full_huffman(ids: list[int], levels: int, values: int, scale_bits: int) -> Result:
    lengths = huffman_lengths(ids)
    frequencies = Counter(ids)
    payload = sum(frequencies[symbol] * lengths[symbol] for symbol in frequencies)
    metadata = levels * 8 + scale_bits
    return Result(payload + metadata, 0.0, 0.0, metadata, payload)


def top_huffman(
    ids: list[int], levels: int, values: int, scale_bits: int, top_k: int, *, collapse: bool, fallback_id: int
) -> Result:
    frequencies = Counter(ids)
    common = {symbol for symbol, _ in frequencies.most_common(min(top_k, len(frequencies)))}
    rare_count = sum(count for symbol, count in frequencies.items() if symbol not in common)
    escape = levels
    mapped = [symbol if symbol in common else escape for symbol in ids]
    mapped_frequencies = Counter(mapped)
    lengths = huffman_lengths(mapped)
    payload = sum(mapped_frequencies[symbol] * lengths[symbol] for symbol in mapped_frequencies)
    if not collapse:
        payload += rare_count * math.ceil(math.log2(levels))

    # Store the common ID list, one canonical Huffman code length per common ID
    # plus ESC, and the quantizer's block scales.  The 32-bit field represents
    # the top-K/format metadata; this keeps tiny tensors from looking free.
    symbol_width = max(1, math.ceil(math.log2(levels)))
    metadata = scale_bits + len(common) * symbol_width + (len(common) + 1) * 8 + 32
    return Result(
        bits=payload + metadata,
        rmse=0.0,
        rare_fraction=rare_count / values,
        metadata_bits=metadata,
        payload_bits=payload,
    )


def collapsed_reconstruction_with_block_size(
    quantized: Quantized, block_size: int, top_k: int, fallback_id: int
) -> tuple[torch.Tensor, float]:
    frequencies = Counter(quantized.ids)
    common = {symbol for symbol, _ in frequencies.most_common(min(top_k, len(frequencies)))}
    reconstructed = torch.empty_like(quantized.reconstructed)
    cursor = 0
    rare = 0
    for scale in quantized.scales:
        end = min(cursor + block_size, reconstructed.numel())
        for index in range(cursor, end):
            symbol = quantized.ids[index]
            if symbol in common:
                reconstructed[index] = quantized.levels[symbol] * scale
            else:
                reconstructed[index] = quantized.levels[fallback_id] * scale
                rare += 1
        cursor = end
    return reconstructed, rare / len(quantized.ids)


def print_row(name: str, result: Result, values: int, rmse: float) -> None:
    print(
        f"{name:32} {result.bits_per_weight(values):9.4f} {rmse:12.8f} "
        f"{100*result.rare_fraction:8.3f}% {result.payload_bits / values:9.4f} {result.metadata_bits / 8:10.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--tensor", default="transformer.wte.weight")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, nargs="+", default=[4, 8, 12])
    args = parser.parse_args()
    if args.block_size < 1 or any(k < 1 for k in args.top_k):
        parser.error("block size and top-k values must be positive")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tensors = learned_float_tensors(state)
    if args.tensor not in tensors:
        parser.error(f"unknown tensor {args.tensor!r}; choose from: {', '.join(tensors)}")
    values = tensors[args.tensor].detach().flatten().to(torch.float32)
    n = values.numel()
    blocks = math.ceil(n / args.block_size)
    scale_bits = blocks * 32
    print(f"tensor: {args.tensor} | values: {n:,} | block size: {args.block_size}")
    print(f"{'method':32} {'bits/w':>9} {'RMSE':>12} {'rare':>9} {'payload/w':>9} {'metadata B':>10}")

    methods = [
        ("uniform INT4", UNIFORM_INT4_LEVELS, 4),
        ("NF4", NF4_LEVELS, 4),
        ("uniform INT5", UNIFORM_INT5_LEVELS, 5),
    ]
    for name, levels, code_bits in methods:
        quantized = quantize(values, levels, args.block_size, code_bits)
        rmse = (quantized.reconstructed - values).square().mean().sqrt().item()
        fixed = Result(n * code_bits + scale_bits, rmse, 0.0, scale_bits, n * code_bits)
        print_row(f"{name} fixed", fixed, n, rmse)
        full = full_huffman(quantized.ids, len(levels), n, scale_bits)
        print_row(f"{name} full Huffman", full, n, rmse)
        fallback_id = int((levels.abs()).argmin().item())
        for top_k in args.top_k:
            escaped = top_huffman(quantized.ids, len(levels), n, scale_bits, top_k, collapse=False, fallback_id=fallback_id)
            print_row(f"{name} top-{top_k} + ESC", escaped, n, rmse)
            collapsed, rare_fraction = collapsed_reconstruction_with_block_size(quantized, args.block_size, top_k, fallback_id)
            collapsed_result = top_huffman(quantized.ids, len(levels), n, scale_bits, top_k, collapse=True, fallback_id=fallback_id)
            collapsed_result.rare_fraction = rare_fraction
            collapsed_rmse = (collapsed - values).square().mean().sqrt().item()
            print_row(f"{name} top-{top_k} collapsed", collapsed_result, n, collapsed_rmse)


if __name__ == "__main__":
    main()
