"""Small experiment: quantization versus Huffman and byte-pair coding.

This is a storage experiment, not an inference kernel.  Quantized values are
represented by IDs plus one fp32 scale per block.  Huffman and BPE are then
applied to the ID stream.  The raw-fp32 rows apply the same lossless coders to
the little-endian fp32 bytes, so their reconstruction is exact.

The reported sizes include estimated canonical-Huffman code-length metadata,
the BPE pair table, and quantizer scales.  They are therefore more useful than
payload entropy alone, but still do not describe a particular file format.
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from compare_codecs import NF4_LEVELS, UNIFORM_INT4_LEVELS, UNIFORM_INT5_LEVELS
from rank_function_codec import learned_float_tensors


@dataclass
class Quantized:
    ids: list[int]
    reconstructed: torch.Tensor
    scales: int
    levels: int
    code_bits: int


@dataclass
class Encoded:
    bits: int
    payload_bits: int
    metadata_bits: int
    symbols: int
    merges: int

    @property
    def bits_per_weight(self) -> float:
        return self.bits / self.values

    values: int = 1
    encode_ms: float = 0.0


def huffman_lengths(symbols: Sequence[int]) -> dict[int, int]:
    """Return optimal binary Huffman code lengths for a finite symbol stream."""
    frequencies = Counter(symbols)
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        return {next(iter(frequencies)): 1}

    serial = 0
    heap: list[tuple[int, int, object]] = []
    for symbol, frequency in frequencies.items():
        heap.append((frequency, serial, symbol))
        serial += 1
    heapq.heapify(heap)
    while len(heap) > 1:
        left_frequency, _, left = heapq.heappop(heap)
        right_frequency, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_frequency + right_frequency, serial, (left, right)))
        serial += 1

    lengths: dict[int, int] = {}

    def visit(node: object, depth: int) -> None:
        if isinstance(node, int):
            lengths[node] = depth
            return
        left, right = node
        visit(left, depth + 1)
        visit(right, depth + 1)

    visit(heap[0][2], 0)
    return lengths


def bpe_encode(sequence: Iterable[int], base_alphabet: int, max_merges: int, min_count: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Greedily replace frequent, non-overlapping adjacent pairs."""
    tokens = list(sequence)
    merges: list[tuple[int, int]] = []
    for _ in range(max_merges):
        if len(tokens) < 2:
            break
        counts = Counter(zip(tokens, tokens[1:]))
        (left, right), count = counts.most_common(1)[0]
        if count < min_count:
            break
        new_symbol = base_alphabet + len(merges)
        replaced: list[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
                replaced.append(new_symbol)
                index += 2
            else:
                replaced.append(tokens[index])
                index += 1
        if len(replaced) >= len(tokens):
            break
        tokens = replaced
        merges.append((left, right))
    return tokens, merges


def entropy_encode(
    symbols: Sequence[int],
    base_alphabet: int,
    values: int,
    *,
    bpe_merges: int = 0,
    min_pair_count: int = 3,
    extra_metadata_bits: int = 0,
) -> Encoded:
    """Estimate a canonical-Huffman bitstream, optionally after BPE."""
    started = time.perf_counter()
    final_symbols, merges = bpe_encode(symbols, base_alphabet, bpe_merges, min_pair_count)
    lengths = huffman_lengths(final_symbols)
    frequencies = Counter(final_symbols)
    payload_bits = sum(frequencies[symbol] * lengths[symbol] for symbol in frequencies)

    # A canonical decoder can reconstruct codes from one byte-length per
    # possible symbol.  The BPE table stores two symbol IDs per merge.  This is
    # deliberately explicit so small tensors do not get an unrealistic payload
    # advantage from omitting their dictionaries.
    alphabet_size = base_alphabet + len(merges)
    huffman_table_bits = alphabet_size * 8
    symbol_width = max(1, math.ceil(math.log2(alphabet_size)))
    bpe_table_bits = len(merges) * 2 * symbol_width + 32 if merges else 0
    metadata_bits = huffman_table_bits + bpe_table_bits + extra_metadata_bits
    return Encoded(
        bits=payload_bits + metadata_bits,
        payload_bits=payload_bits,
        metadata_bits=metadata_bits,
        symbols=len(final_symbols),
        merges=len(merges),
        values=values,
        encode_ms=(time.perf_counter() - started) * 1000,
    )


def quantize(values: torch.Tensor, levels: torch.Tensor, block_size: int, code_bits: int) -> Quantized:
    flat = values.detach().flatten().to(torch.float32)
    restored = torch.empty_like(flat)
    ids: list[int] = []
    block_count = 0
    levels = levels.to(flat.device)
    for start in range(0, flat.numel(), block_size):
        end = min(start + block_size, flat.numel())
        block = flat[start:end]
        scale = block.abs().max()
        if scale == 0:
            codes = torch.zeros(block.numel(), dtype=torch.long, device=flat.device)
            restored[start:end] = 0
        else:
            normalized = block / scale
            codes = (normalized[:, None] - levels[None, :]).abs().argmin(dim=1)
            restored[start:end] = levels[codes] * scale
        ids.extend(codes.tolist())
        block_count += 1
    return Quantized(ids, restored, block_count, len(levels), code_bits)


def raw_bytes(values: torch.Tensor) -> list[int]:
    # Numpy is intentionally avoided: Tensor.numpy().tobytes() is available
    # with the project's PyTorch dependency and preserves fp32 byte order.
    return list(values.detach().flatten().to(torch.float32).contiguous().numpy().tobytes())


def print_result(name: str, encoded: Encoded, error: float | None) -> None:
    error_text = "exact" if error is None else f"{error:.6g}"
    print(
        f"{name:31} {encoded.bits_per_weight:9.4f} {error_text:>12} "
        f"{encoded.payload_bits / encoded.values:9.4f} {encoded.metadata_bits / 8:10.1f} "
        f"{encoded.merges:7d} {encoded.encode_ms:9.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tiny-gpt2/pytorch_model.bin"))
    parser.add_argument("--tensor", default="transformer.wte.weight")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--bpe-merges", type=int, default=16)
    parser.add_argument("--min-pair-count", type=int, default=3)
    args = parser.parse_args()
    if args.block_size < 1 or args.bpe_merges < 0 or args.min_pair_count < 2:
        parser.error("block size and BPE merge count must be non-negative; min pair count must be at least 2")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tensors = learned_float_tensors(state)
    if args.tensor not in tensors:
        parser.error(f"unknown tensor {args.tensor!r}; choose from: {', '.join(tensors)}")
    values = tensors[args.tensor].detach().flatten().to(torch.float32)
    n = values.numel()
    print(f"tensor: {args.tensor} | shape: {tuple(tensors[args.tensor].shape)} | values: {n:,}")
    print(f"block size: {args.block_size} | BPE merges: {args.bpe_merges} | minimum pair count: {args.min_pair_count}")
    print(f"{'method':31} {'bits/w':>9} {'RMSE':>12} {'payload/w':>9} {'metadata B':>10} {'merges':>7} {'encode ms':>9}")

    raw_started = time.perf_counter()
    raw = raw_bytes(values)
    raw_bytes_ms = (time.perf_counter() - raw_started) * 1000
    print_result("FP32 raw", Encoded(n * 32, n * 32, 0, 256, 0, n, raw_bytes_ms), None)
    raw_huffman = entropy_encode(raw, 256, n)
    raw_huffman.encode_ms += raw_bytes_ms
    print_result("FP32 + Huffman", raw_huffman, None)
    raw_bpe = entropy_encode(raw, 256, n, bpe_merges=args.bpe_merges, min_pair_count=args.min_pair_count)
    raw_bpe.encode_ms += raw_bytes_ms
    print_result("FP32 + BPE + Huffman", raw_bpe, None)

    methods = [
        ("uniform INT4", UNIFORM_INT4_LEVELS, 4),
        ("NF4", NF4_LEVELS, 4),
        ("uniform INT5", UNIFORM_INT5_LEVELS, 5),
    ]
    for name, levels, code_bits in methods:
        quantize_started = time.perf_counter()
        quantized = quantize(values, levels, args.block_size, code_bits)
        quantize_ms = (time.perf_counter() - quantize_started) * 1000
        error = (quantized.reconstructed - values).square().mean().sqrt().item()
        quantizer_bits = quantized.scales * 32
        fixed = Encoded(n * code_bits + quantizer_bits, n * code_bits, quantizer_bits, quantized.levels, 0, n, quantize_ms)
        print_result(f"{name} fixed", fixed, error)
        huffman = entropy_encode(
            quantized.ids,
            quantized.levels,
            n,
            extra_metadata_bits=quantizer_bits,
        )
        huffman.encode_ms += quantize_ms
        print_result(f"{name} + Huffman", huffman, error)
        bpe = entropy_encode(
            quantized.ids,
            quantized.levels,
            n,
            bpe_merges=args.bpe_merges,
            min_pair_count=args.min_pair_count,
            extra_metadata_bits=quantizer_bits,
        )
        bpe.encode_ms += quantize_ms
        print_result(f"{name} + BPE + Huffman", bpe, error)


if __name__ == "__main__":
    main()
