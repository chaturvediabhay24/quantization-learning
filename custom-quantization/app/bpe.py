"""Byte-pair merges over a quantized weight-symbol sequence."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _huffman_codes(frequencies: dict[int, int]) -> dict[int, str]:
    """Build deterministic canonical Huffman codes for integer symbols."""
    import heapq

    if not frequencies:
        return {}
    if len(frequencies) == 1:
        return {next(iter(frequencies)): "0"}
    heap: list[tuple[int, int, int | tuple[Any, Any]]] = []
    serial = 0
    for symbol, count in sorted(frequencies.items()):
        heapq.heappush(heap, (count, serial, symbol))
        serial += 1
    while len(heap) > 1:
        left_count, _, left = heapq.heappop(heap)
        right_count, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_count + right_count, serial, (left, right)))
        serial += 1
    lengths: dict[int, int] = {}

    def visit(node: int | tuple[Any, Any], depth: int) -> None:
        if isinstance(node, int):
            lengths[node] = depth
        else:
            visit(node[0], depth + 1)
            visit(node[1], depth + 1)

    visit(heap[0][2], 0)
    ordered = sorted(lengths.items(), key=lambda item: (item[1], item[0]))
    codes: dict[int, str] = {}
    code = 0
    previous_length = 0
    for symbol, length in ordered:
        code <<= length - previous_length
        codes[symbol] = format(code, f"0{length}b")
        code += 1
        previous_length = length
    return codes


def analyze_bpe_huffman(
    weight_sequences: list[np.ndarray],
    tensor_names: list[str],
    bin_width: float,
    merge_count: int,
    min_pair_frequency: int = 2,
) -> dict[str, Any]:
    """Quantize each tensor, learn BPE merges, and Huffman-code the final stream."""
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive")
    if merge_count < 0:
        raise ValueError("merge_count must be non-negative")
    if min_pair_frequency < 2:
        raise ValueError("min_pair_frequency must be at least 2")
    if len(weight_sequences) != len(tensor_names):
        raise ValueError("Each weight sequence must have a tensor name")

    arrays = [np.asarray(values, dtype=np.float64).reshape(-1) for values in weight_sequences]
    if not arrays or any(array.size == 0 for array in arrays):
        raise ValueError("At least one non-empty tensor sequence is required")
    values = np.concatenate(arrays)
    index_sequences = [np.floor(array / bin_width).astype(np.int64).tolist() for array in arrays]
    original_counts: Counter[int] = Counter()
    for sequence in index_sequences:
        original_counts.update(sequence)

    next_symbol = max(original_counts) + 1
    merges: list[dict[str, int]] = []
    for _ in range(merge_count):
        pair_counts: Counter[tuple[int, int]] = Counter()
        for sequence in index_sequences:
            pair_counts.update(zip(sequence, sequence[1:]))
        if not pair_counts:
            break
        pair, pair_frequency = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        if pair_frequency < min_pair_frequency:
            break
        left, right = pair
        new_symbol = next_symbol
        next_symbol += 1
        merged_occurrences = 0
        updated_sequences: list[list[int]] = []
        for sequence in index_sequences:
            merged: list[int] = []
            position = 0
            while position < len(sequence):
                if position + 1 < len(sequence) and sequence[position] == left and sequence[position + 1] == right:
                    merged.append(new_symbol)
                    merged_occurrences += 1
                    position += 2
                else:
                    merged.append(sequence[position])
                    position += 1
            updated_sequences.append(merged)
        index_sequences = updated_sequences
        merges.append({
            "rank": len(merges) + 1,
            "left_symbol": int(left),
            "right_symbol": int(right),
            "new_symbol": int(new_symbol),
            "pair_frequency": int(pair_frequency),
            "merged_occurrences": int(merged_occurrences),
        })

    final_counts: Counter[int] = Counter()
    for sequence in index_sequences:
        final_counts.update(sequence)
    codes = _huffman_codes(dict(final_counts))
    code_lengths = {symbol: len(code) for symbol, code in codes.items()}
    payload_bits = sum(final_counts[symbol] * code_lengths[symbol] for symbol in final_counts)

    original_indices = np.floor(values / bin_width).astype(np.int64)
    reconstructed = (original_indices.astype(np.float64) + 0.5) * bin_width
    errors = values - reconstructed
    absolute_errors = np.abs(errors)
    weight_count = int(values.size)
    huffman_baseline_codes = _huffman_codes(dict(original_counts))
    baseline_payload_bits = sum(
        count * len(huffman_baseline_codes[symbol]) for symbol, count in original_counts.items()
    )
    symbol_min = min(final_counts)
    symbol_max = max(final_counts)
    symbol_id_width = max(1, int(np.ceil(np.log2(symbol_max - symbol_min + 1))))
    leaf_count = len(final_counts)
    huffman_tree_bits = 2 * leaf_count - 1
    leaf_symbol_bits = leaf_count * symbol_id_width
    framing_bits = 64 + 64 + 1
    merge_rule_bits = 64 + len(merges) * 2 * symbol_id_width
    table_bits = huffman_tree_bits + leaf_symbol_bits + framing_bits + merge_rule_bits

    baseline_min = min(original_counts)
    baseline_max = max(original_counts)
    baseline_symbol_width = max(1, int(np.ceil(np.log2(baseline_max - baseline_min + 1))))
    baseline_leaf_count = len(original_counts)
    baseline_table_bits = (
        2 * baseline_leaf_count - 1
        + baseline_leaf_count * baseline_symbol_width
        + framing_bits
    )
    final_symbols = []
    merge_by_symbol = {merge["new_symbol"]: merge for merge in merges}
    for symbol, frequency in sorted(final_counts.items(), key=lambda item: (-item[1], item[0])):
        merge = merge_by_symbol.get(symbol)
        final_symbols.append({
            "symbol": int(symbol),
            "frequency": int(frequency),
            "huffman_code": codes[symbol],
            "code_length": code_lengths[symbol],
            "merge_left": None if merge is None else int(merge["left_symbol"]),
            "merge_right": None if merge is None else int(merge["right_symbol"]),
            "representation": (
                f"#{merge['left_symbol']} + #{merge['right_symbol']}"
                if merge else f"bin {symbol}"
            ),
        })

    return {
        "bin_width": float(bin_width),
        "merge_count_requested": int(merge_count),
        "min_pair_frequency": int(min_pair_frequency),
        "sequence_order": "each tensor flattened in C (row-major) order; tensor boundaries are preserved",
        "tensor_names": tensor_names,
        "tensor_count": len(tensor_names),
        "weight_count": weight_count,
        "original_occupied_bins": len(original_counts),
        "bpe_merge_count": len(merges),
        "bpe_vocabulary_symbols": len(original_counts) + len(merges),
        "final_symbol_types": len(final_counts),
        "final_sequence_length": sum(map(len, index_sequences)),
        "quantization_error": {
            "mae": float(absolute_errors.mean()),
            "mse": float(np.mean(errors * errors)),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "max_abs_error": float(absolute_errors.max()),
        },
        "huffman": {
            "payload_bits": int(payload_bits),
            "average_bits_per_weight": float(payload_bits / weight_count),
            "table_bits_estimate": int(table_bits),
            "estimated_total_bits": int(payload_bits + table_bits),
            "estimated_total_bytes": int(np.ceil((payload_bits + table_bits) / 8)),
            "baseline_payload_bits": int(baseline_payload_bits),
            "baseline_average_bits_per_weight": float(baseline_payload_bits / weight_count),
            "baseline_table_bits_estimate": int(baseline_table_bits),
            "baseline_estimated_total_bits": int(baseline_payload_bits + baseline_table_bits),
            "payload_bits_saved_vs_baseline": int(baseline_payload_bits - payload_bits),
            "total_bits_saved_vs_baseline": int(
                baseline_payload_bits + baseline_table_bits - payload_bits - table_bits
            ),
            "huffman_tree_bits": int(huffman_tree_bits),
            "leaf_symbol_bits": int(leaf_symbol_bits),
            "bpe_merge_rule_bits": int(merge_rule_bits),
            "symbol_id_width_bits": int(symbol_id_width),
        },
        "original_bins": [
            {"symbol": int(symbol), "frequency": int(frequency)}
            for symbol, frequency in sorted(original_counts.items())
        ],
        "merges": merges,
        "final_symbols": final_symbols,
    }


def merge_weight_symbols(
    values: np.ndarray,
    bin_width: float,
    merge_count: int = 100,
    min_pair_frequency: int = 2,
) -> dict[str, Any]:
    """Quantize weights, then repeatedly merge the most frequent adjacent pair.

    The sequence is the embedding matrix in C (row-major) order. New symbols
    represent ordered pairs of earlier symbols; they are coding symbols rather
    than new scalar-value intervals.
    """
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive")
    if merge_count < 0:
        raise ValueError("merge_count must be non-negative")
    if min_pair_frequency < 2:
        raise ValueError("min_pair_frequency must be at least 2")

    sequence = np.floor(np.asarray(values, dtype=np.float64).reshape(-1) / bin_width).astype(np.int64).tolist()
    initial_counts = Counter(sequence)
    next_symbol = max(initial_counts, default=-1) + 1
    merges: list[dict[str, int]] = []

    for _ in range(merge_count):
        pair_counts = Counter(zip(sequence, sequence[1:]))
        if not pair_counts:
            break
        # Lexicographic tie-breaking makes merge rules reproducible.
        pair, frequency = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        if frequency < min_pair_frequency:
            break

        left, right = pair
        new_symbol = next_symbol
        next_symbol += 1
        merged: list[int] = []
        merged_occurrences = 0
        i = 0
        while i < len(sequence):
            if i + 1 < len(sequence) and sequence[i] == left and sequence[i + 1] == right:
                merged.append(new_symbol)
                merged_occurrences += 1
                i += 2
            else:
                merged.append(sequence[i])
                i += 1
        sequence = merged
        merges.append({
            "rank": len(merges) + 1,
            "left_symbol": int(left),
            "right_symbol": int(right),
            "new_symbol": int(new_symbol),
            "pair_frequency": int(frequency),
            "merged_occurrences": merged_occurrences,
        })

    final_counts = Counter(sequence)
    return {
        "bin_width": float(bin_width),
        "sequence_order": "embedding weights flattened in C (row-major) order",
        "weight_count": len(values),
        "original_occupied_bins": len(initial_counts),
        "merge_count": len(merges),
        "final_symbol_count": len(final_counts),
        "final_sequence_length": len(sequence),
        "min_pair_frequency": min_pair_frequency,
        "merges": merges,
        "original_bins": [
            {
                "symbol": int(symbol),
                "frequency": int(count),
            }
            for symbol, count in sorted(initial_counts.items())
        ],
        "final_symbols": [
            {"symbol": int(symbol), "frequency": int(count)}
            for symbol, count in sorted(final_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
