from __future__ import annotations

import heapq
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .bpe import analyze_bpe_huffman, merge_weight_symbols
from .real_model_analysis import (
    MODEL_REGISTRY,
    create_job,
    get_job,
    list_models,
    run_analysis_job,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = Path(os.environ.get("QUANTIZATION_DATA_FILE", ROOT / "data" / "tiny_gpt2_weights.npz"))
DEFAULT_BIN_WIDTH = 0.00035712
MAX_PREVIEW_BINS = 30
ALL_TENSORS = "__all__"
BIN_WIDTH_SWEEP_POINTS = 17
BIN_WIDTH_SWEEP_SPAN = 4.0


def _load_weights() -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    if not DATA_FILE.is_file():
        raise RuntimeError(
            f"Quantization data not found at {DATA_FILE}. See scripts/extract_checkpoint.py."
        )
    loaded = np.load(DATA_FILE, mmap_mode="r")
    if isinstance(loaded, np.lib.npyio.NpzFile):
        weights = {
            name: np.asarray(loaded[name], dtype=np.float64).reshape(-1)
            for name in loaded.files
            if loaded[name].dtype.kind == "f" and loaded[name].size > 0
        }
        shapes = {name: list(loaded[name].shape) for name in weights}
        loaded.close()
    else:
        if loaded.dtype.kind != "f" or loaded.size == 0:
            raise RuntimeError("The configured dataset must be a non-empty floating-point NumPy array.")
        weights = {"custom tensor": np.asarray(loaded, dtype=np.float64).reshape(-1)}
        shapes = {"custom tensor": list(loaded.shape)}
    if not weights:
        raise RuntimeError("No non-empty floating-point tensors were found in the configured dataset.")
    return weights, shapes


WEIGHTS, WEIGHT_SHAPES = _load_weights()
ALL_VALUES = np.concatenate(list(WEIGHTS.values()))
TOTAL_WEIGHT_COUNT = int(ALL_VALUES.size)


def _huffman_lengths(frequencies: dict[int, int]) -> dict[int, int]:
    """Build deterministic optimal binary Huffman lengths for integer symbols."""
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        return {next(iter(frequencies)): 1}

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
            return
        visit(node[0], depth + 1)
        visit(node[1], depth + 1)

    visit(heap[0][2], 0)
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> dict[int, str]:
    """Return canonical Huffman codewords, useful for the frequency table."""
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


def _unary_codes(lengths: dict[int, int]) -> dict[int, str]:
    """Render unary codewords without constructing very long zero runs."""
    codes: dict[int, str] = {}
    for symbol, length in lengths.items():
        zeros = length - 1
        codes[symbol] = f"0^{zeros}1" if zeros > 32 else "0" * zeros + "1"
    return codes


def _format_interval(index: int, width: float) -> tuple[float, float, float]:
    lower = index * width
    upper = (index + 1) * width
    midpoint = (index + 0.5) * width
    return lower, upper, midpoint


def analyze(
    bin_width: float,
    tensor_name: str = ALL_TENSORS,
    merge_rare_bins: int = 0,
    frequency_order: str = "desc",
    coding_method: str = "huffman",
    max_code_length: int | None = None,
) -> dict[str, Any]:
    """Analyze a scope with Huffman or frequency-ranked unary coding."""
    if coding_method not in {"huffman", "unary"}:
        raise ValueError("coding_method must be 'huffman' or 'unary'")
    if max_code_length is not None and not 1 <= max_code_length <= 64:
        raise ValueError("max_code_length must be between 1 and 64 bits")
    if tensor_name == ALL_TENSORS:
        selected_names = list(WEIGHTS)
        values = ALL_VALUES
    else:
        if tensor_name not in WEIGHTS:
            raise KeyError(tensor_name)
        selected_names = [tensor_name]
        values = WEIGHTS[tensor_name]

    indices_array = np.floor(values / bin_width).astype(np.int64)
    indices, counts_array = np.unique(indices_array, return_counts=True)
    original_frequencies = {int(index): int(count) for index, count in zip(indices, counts_array)}
    original_bin_count = len(original_frequencies)
    effective_merge_count = min(max(0, int(merge_rare_bins)), original_bin_count)
    rare_order = sorted(original_frequencies, key=lambda index: (original_frequencies[index], index))
    rare_indices = rare_order[:effective_merge_count]
    rare_set = set(rare_indices)

    symbol_by_bin = {index: index for index in original_frequencies}
    symbol_array = indices_array.copy()
    reconstruction_by_symbol = {
        index: (index + 0.5) * bin_width for index in original_frequencies
    }


    custom_symbols: set[int] = set()
    merged_symbol: int | None = None
    merged_representative: float | None = None
    merged_value_count = 0
    rare_mask = np.zeros(indices_array.shape, dtype=bool)
    if effective_merge_count:
        merged_symbol = int(indices[-1]) + 1
        merged_value_count = sum(original_frequencies[index] for index in rare_indices)
        rare_mask = np.isin(indices_array, np.asarray(rare_indices, dtype=np.int64))
        merged_representative = float(np.mean(values[rare_mask], dtype=np.float64))
        symbol_array[rare_mask] = merged_symbol
        for index in rare_indices:
            symbol_by_bin[index] = merged_symbol
        reconstruction_by_symbol[merged_symbol] = merged_representative
        custom_symbols.add(merged_symbol)

    overflow_symbol = int(indices[-1]) + (2 if effective_merge_count else 1)
    length_limited_original_bins: set[int] = set()
    overflow_representative: float | None = None
    if coding_method == "huffman":
        for _ in range(original_bin_count + 1):
            current_symbols, current_counts = np.unique(symbol_array, return_counts=True)
            frequencies = {
                int(symbol): int(count) for symbol, count in zip(current_symbols, current_counts)
            }
            lengths = _huffman_lengths(frequencies)
            if max_code_length is None:
                break
            long_symbols = {
                symbol for symbol, length in lengths.items() if length > max_code_length
            }
            if not long_symbols:
                break

            to_merge = set(long_symbols)
            if overflow_symbol in frequencies:
                # Keep every overlength source in the same shared bucket across rounds.
                to_merge.add(overflow_symbol)

            merge_mask = np.isin(symbol_array, np.asarray(sorted(to_merge), dtype=np.int64))
            if not np.any(merge_mask):
                raise ValueError("The Huffman code-length limit could not be applied")
            for original_index, symbol in list(symbol_by_bin.items()):
                if symbol in to_merge:
                    symbol_by_bin[original_index] = overflow_symbol
                    length_limited_original_bins.add(original_index)
            symbol_array[merge_mask] = overflow_symbol
            overflow_representative = float(np.mean(values[merge_mask], dtype=np.float64))
            reconstruction_by_symbol[overflow_symbol] = overflow_representative
            custom_symbols.add(overflow_symbol)
        else:
            raise ValueError("The Huffman code-length limit did not converge")
    elif max_code_length is not None:
        current_symbols, current_counts = np.unique(symbol_array, return_counts=True)
        frequencies_before_limit = {
            int(symbol): int(count) for symbol, count in zip(current_symbols, current_counts)
        }
        if len(frequencies_before_limit) > max_code_length:
            ordered_symbols = sorted(
                frequencies_before_limit,
                key=lambda symbol: (-frequencies_before_limit[symbol], symbol),
            )
            keep_symbols = set(ordered_symbols[: max_code_length - 1])
            to_merge = set(frequencies_before_limit) - keep_symbols
            merge_mask = np.isin(symbol_array, np.asarray(sorted(to_merge), dtype=np.int64))
            for original_index, symbol in list(symbol_by_bin.items()):
                if symbol in to_merge:
                    symbol_by_bin[original_index] = overflow_symbol
                    length_limited_original_bins.add(original_index)
            symbol_array[merge_mask] = overflow_symbol
            overflow_representative = float(np.mean(values[merge_mask], dtype=np.float64))
            reconstruction_by_symbol[overflow_symbol] = overflow_representative
            custom_symbols.add(overflow_symbol)

    final_symbols, final_counts = np.unique(symbol_array, return_counts=True)
    frequencies = {int(symbol): int(count) for symbol, count in zip(final_symbols, final_counts)}
    if coding_method == "huffman":
        lengths = _huffman_lengths(frequencies)
        codes = _canonical_codes(lengths)
    else:
        frequency_rank = sorted(
            frequencies,
            key=lambda symbol: (-frequencies[symbol], symbol),
        )
        lengths = {symbol: rank + 1 for rank, symbol in enumerate(frequency_rank)}
        codes = _unary_codes(lengths)
    if max_code_length is not None and max(lengths.values()) > max_code_length:
        raise ValueError(f"Could not satisfy the requested {coding_method} code-length limit")

    payload_bits = sum(frequencies[index] * lengths[index] for index in frequencies)
    leaf_count = len(frequencies)
    symbol_min = min(frequencies)
    symbol_max = max(frequencies)
    symbol_index_width = max(1, math.ceil(math.log2(symbol_max - symbol_min + 1)))
    # Huffman stores a full binary tree; unary stores a frequency-ranked symbol map.
    tree_bits = 2 * leaf_count - 1 if coding_method == "huffman" else 0
    leaf_table_bits = leaf_count * symbol_index_width
    framing_bits = 64 + 64  # float64 width and int64 minimum index
    coding_method_flag_bits = 1
    custom_symbols.intersection_update(frequencies)
    merged_representative_bits = 96 * len(custom_symbols)
    table_bits = tree_bits + leaf_table_bits + framing_bits + coding_method_flag_bits
    table_bits += merged_representative_bits

    reconstructed = (indices_array.astype(np.float64) + 0.5) * bin_width
    if effective_merge_count:
        reconstructed[rare_mask] = merged_representative
    if overflow_symbol in frequencies:
        reconstructed[symbol_array == overflow_symbol] = overflow_representative
    errors = values - reconstructed
    absolute = np.abs(errors)
    mae = float(absolute.mean())
    rmse = float(np.sqrt(np.mean(errors * errors)))
    max_error = float(absolute.max())
    value_count = int(values.size)
    reference_bits = value_count * 32
    fixed_width_bits_per_symbol = max(1, math.ceil(math.log2(leaf_count)))
    fixed_width_payload = value_count * fixed_width_bits_per_symbol

    bin_count_by_symbol = Counter(symbol_by_bin.values())
    merged_symbols = custom_symbols | {
        symbol for symbol, source_bin_count in bin_count_by_symbol.items() if source_bin_count > 1
    }
    output_positions = {symbol: reconstruction_by_symbol[symbol] for symbol in frequencies}
    output_symbols = sorted(frequencies, key=lambda index: (output_positions[index], index))
    frequency_indices = sorted(
        original_frequencies,
        key=(lambda index: (original_frequencies[index], index))
        if frequency_order == "asc"
        else (lambda index: (-original_frequencies[index], index)),
    )
    frequency_bins = {
        "indices": [],
        "counts": [],
        "code_lengths": [],
        "codes": [],
        "pooled": [],
        "pool_reasons": [],
    }
    for index in frequency_indices:
        coded_symbol = symbol_by_bin[index]
        pooled = coded_symbol != index
        if index in rare_set and index in length_limited_original_bins:
            pool_reason = "least-frequent + code-length limit"
        elif index in rare_set:
            pool_reason = "least-frequent bins"
        elif index in length_limited_original_bins:
            pool_reason = "code-length limit"
        else:
            pool_reason = ""
        count = original_frequencies[index]
        frequency_bins["indices"].append(index)
        frequency_bins["counts"].append(count)
        frequency_bins["code_lengths"].append(lengths[coded_symbol])
        frequency_bins["codes"].append(codes[coded_symbol])
        frequency_bins["pooled"].append(pooled)
        frequency_bins["pool_reasons"].append(pool_reason)

    length_limited_preview = [
        {
            "index": int(index),
            "lower": float(index * bin_width),
            "upper": float((index + 1) * bin_width),
            "count": int(original_frequencies[index]),
        }
        for index in sorted(
            length_limited_original_bins,
            key=lambda index: (original_frequencies[index], index),
        )[:MAX_PREVIEW_BINS]
    ]
    rare_bucket_absorbed = bool(
        effective_merge_count
        and overflow_symbol in frequencies
        and any(symbol_by_bin[index] == overflow_symbol for index in rare_set)
    )
    additional_length_limited_bins = length_limited_original_bins - rare_set
    length_limited_value_count = sum(
        original_frequencies[index] for index in length_limited_original_bins
    )
    additional_length_limited_value_count = sum(
        original_frequencies[index] for index in additional_length_limited_bins
    )
    actual_max_code_length = max(lengths.values())
    entropy_bits_per_weight = -sum(
        (count / value_count) * math.log2(count / value_count)
        for count in frequencies.values()
    )

    full_model = tensor_name == ALL_TENSORS
    selected_shape = None if full_model else WEIGHT_SHAPES[tensor_name]
    return {
        "dataset": {
            "name": "Tiny GPT-2 · full model" if full_model else tensor_name,
            "scope": tensor_name,
            "tensor_count": len(selected_names),
            "tensors": selected_names,
            "shape": selected_shape,
            "values": value_count,
            "source": "Bundled unique learned float tensors from sshleifer/tiny-gpt2; masks and tied aliases are excluded.",
        },
        "bin_width": bin_width,
        "coding_method": coding_method,
        "binning": "floor(value / width), reconstructed at (index + 0.5) × width",
        "metrics": {
            "mae": mae,
            "mse": float(np.mean(errors * errors)),
            "rmse": rmse,
            "max_abs_error": max_error,
            "occupied_bins": leaf_count,
            "original_occupied_bins": original_bin_count,
            "merged_rare_bin_count": effective_merge_count,
            "merged_value_count": merged_value_count,
            "merged_representative": merged_representative,
            "coding_method": coding_method,
            "max_code_length": max_code_length,
            "actual_max_code_length": actual_max_code_length,
            # Legacy aliases retained for clients built against the Huffman-only API.
            "max_huffman_length": max_code_length,
            "actual_max_huffman_length": actual_max_code_length,
            "length_limited_bin_count": len(length_limited_original_bins),
            "length_limited_value_count": length_limited_value_count,
            "additional_length_limited_bin_count": len(additional_length_limited_bins),
            "additional_length_limited_value_count": additional_length_limited_value_count,
            "rare_bucket_absorbed_by_length_limit": rare_bucket_absorbed,
            "custom_bucket_count": len(custom_symbols),
            "mean_code_bits_per_value": payload_bits / value_count,
            "payload_bits_per_weight": payload_bits / value_count,
            "average_code_bits_per_original_weight": payload_bits / value_count,
            "mean_huffman_bits_per_value": payload_bits / value_count,
            "entropy_bits_per_weight": entropy_bits_per_weight,
            "entropy_bits_per_value": entropy_bits_per_weight,
            "coding_payload_bits": payload_bits,
            "huffman_payload_bits": payload_bits,
            "fixed_width_payload_bits": fixed_width_payload,
            "estimated_table_bits": table_bits,
            "estimated_total_bits": payload_bits + table_bits,
            "estimated_total_bytes": math.ceil((payload_bits + table_bits) / 8),
            "fp32_reference_bits": reference_bits,
            "fp32_reference_bytes": value_count * 4,
            "estimated_fp32_reduction_percent": 100 * (1 - (payload_bits + table_bits) / reference_bits),
            "symbol_index_bits": symbol_index_width,
            "table_components_bits": {
                "tree_shape": tree_bits,
                "leaf_bin_ids": leaf_table_bits,
                "width_and_minimum_index_header": framing_bits,
                "coding_method_flag": coding_method_flag_bits,
                "custom_bucket_metadata": merged_representative_bits,
            },
        },
        # Numeric arrays keep responses compact even when a very narrow width
        # creates tens of thousands of occupied bins.
        "histogram": {
            "indices": [int(index) for index in output_symbols],
            "positions": [float(output_positions[index]) for index in output_symbols],
            "frequencies": [int(frequencies[index]) for index in output_symbols],
            "code_lengths": [int(lengths[index]) for index in output_symbols],
            "is_merged": [bool(index in merged_symbols) for index in output_symbols],
            "merged_bin_counts": [bin_count_by_symbol[index] if index in merged_symbols else 0 for index in output_symbols],
            "min_position": float(output_positions[output_symbols[0]]),
            "max_position": float(output_positions[output_symbols[-1]]),
        },
        "frequency_order": frequency_order,
        "frequency_bins": frequency_bins,
        "merged_bin_preview": [
            {
                "index": int(index),
                "lower": float(index * bin_width),
                "upper": float((index + 1) * bin_width),
                "count": int(original_frequencies[index]),
            }
            for index in rare_indices[:MAX_PREVIEW_BINS]
        ],
        "length_limited_bin_preview": length_limited_preview,
        "table_estimate_note": (
            ("Estimated full-tree markers, " if coding_method == "huffman" else "")
            + "fixed-width output-bin IDs, a one-bit coding-method flag, and 16 bytes for the "
            "float64 width and int64 minimum index, plus one float64 representative and "
            "32-bit source-bin count per custom bucket. Excludes tensor-shape metadata, "
            "container headers, and byte padding."
        ),
    }

app = FastAPI(
    title="Custom Quantization Lab",
    description="Analyze midpoint quantization on bundled tensors and real large-model checkpoints without inference.",
    version="0.3.0",
)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/analysis")
def get_analysis(
    bin_width: float = Query(DEFAULT_BIN_WIDTH, ge=1e-9, le=1, description="Fixed bin width from 1e-9 through 1"),
    tensor: str = Query(ALL_TENSORS, description="__all__ or an exact learned tensor name"),
    merge_rare_bins: int = Query(0, ge=0, description="Pool this many least-frequent occupied bins"),
    frequency_order: Literal["asc", "desc"] = Query("desc", description="Frequency rank direction"),
    coding_method: Literal["huffman", "unary"] = Query("huffman", description="Variable-length coding method"),
    max_code_length: int | None = Query(None, ge=1, le=64, description="Maximum code length in bits"),
    max_huffman_length: int | None = Query(None, ge=1, le=64, include_in_schema=False),
) -> dict[str, Any]:
    if not math.isfinite(bin_width):
        raise HTTPException(status_code=422, detail="bin_width must be finite")
    if max_code_length is not None and max_huffman_length is not None:
        raise HTTPException(status_code=422, detail="Use max_code_length; do not send both code-length parameters")
    effective_max_length = max_code_length if max_code_length is not None else max_huffman_length
    try:
        result = analyze(bin_width, tensor, merge_rare_bins, frequency_order, coding_method, effective_max_length)
        if merge_rare_bins or effective_max_length is not None:
            unfiltered = analyze(bin_width, tensor, 0, frequency_order, coding_method, None)
            unfiltered_metrics = unfiltered["metrics"]
        else:
            # With both pooling controls disabled, the current result is already the baseline.
            unfiltered_metrics = result["metrics"]
        result["unfiltered_comparison"] = {
            "merge_rare_bins": 0,
            "max_code_length": None,
            "metrics": unfiltered_metrics,
        }
        return result
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Unknown tensor: {error.args[0]}") from error
    except (OverflowError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Could not analyze this bin width: {error}") from error


@app.get("/api/bin-width-sweep")
def get_bin_width_sweep(
    center_bin_width: float = Query(DEFAULT_BIN_WIDTH, ge=1e-9, le=1),
    tensor: str = Query(ALL_TENSORS, description="__all__ or an exact learned tensor name"),
    merge_rare_bins: int = Query(0, ge=0),
    coding_method: Literal["huffman", "unary"] = Query("huffman"),
    max_code_length: int | None = Query(None, ge=1, le=64),
) -> dict[str, Any]:
    """Compare error, occupied-bin count, and code rate across nearby widths."""
    if not math.isfinite(center_bin_width):
        raise HTTPException(status_code=422, detail="center_bin_width must be finite")
    if tensor != ALL_TENSORS and tensor not in WEIGHTS:
        raise HTTPException(status_code=404, detail=f"Unknown tensor: {tensor}")

    minimum_width = max(1e-9, center_bin_width / BIN_WIDTH_SWEEP_SPAN)
    maximum_width = min(1.0, center_bin_width * BIN_WIDTH_SWEEP_SPAN)
    widths = np.geomspace(minimum_width, maximum_width, BIN_WIDTH_SWEEP_POINTS)
    points = []
    for width in widths:
        try:
            result = analyze(
                float(width), tensor, merge_rare_bins, "desc", coding_method, max_code_length
            )
        except (OverflowError, ValueError) as error:
            raise HTTPException(status_code=422, detail=f"Could not analyze bin width {width:g}: {error}") from error
        metrics = result["metrics"]
        points.append({
            "bin_width": float(width),
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "occupied_input_bins": metrics["original_occupied_bins"],
            "output_symbols": metrics["occupied_bins"],
            "average_code_bits_per_original_weight": metrics["average_code_bits_per_original_weight"],
        })

    return {
        "dataset": result["dataset"],
        "settings": {
            "center_bin_width": center_bin_width,
            "minimum_bin_width": minimum_width,
            "maximum_bin_width": maximum_width,
            "merge_rare_bins": merge_rare_bins,
            "coding_method": coding_method,
            "max_code_length": max_code_length,
            "points": BIN_WIDTH_SWEEP_POINTS,
        },
        "points": points,
    }


@app.get("/api/bpe-analysis")
def get_bpe_analysis(
    bin_width: float = Query(DEFAULT_BIN_WIDTH, gt=0, le=1),
    tensor: str = Query("transformer.wte.weight", description="The token embedding tensor"),
    merges: int = Query(100, ge=0, le=1000, description="Maximum number of BPE merge rules"),
    min_pair_frequency: int = Query(2, ge=2, description="Stop when the most frequent pair occurs fewer times"),
) -> dict[str, Any]:
    """Apply ordered pair merges to the quantized GPT-2 token embedding stream."""
    if tensor != "transformer.wte.weight":
        raise HTTPException(status_code=422, detail="BPE prototype currently supports transformer.wte.weight only")
    if tensor not in WEIGHTS:
        raise HTTPException(status_code=404, detail=f"Unknown tensor: {tensor}")
    try:
        result = merge_weight_symbols(WEIGHTS[tensor], bin_width, merges, min_pair_frequency)
    except (OverflowError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Could not create BPE bins: {error}") from error
    result["dataset"] = {
        "name": "Tiny GPT-2 token embedding",
        "tensor": tensor,
        "shape": WEIGHT_SHAPES[tensor],
    }
    result["original_bins"] = [
        {
            **item,
            "lower": item["symbol"] * bin_width,
            "upper": (item["symbol"] + 1) * bin_width,
            "midpoint": (item["symbol"] + 0.5) * bin_width,
        }
        for item in result["original_bins"]
    ]
    result["final_symbols"] = result["final_symbols"][:100]
    result["final_symbols_returned"] = len(result["final_symbols"])
    return result


@app.get("/api/bpe-huffman-analysis")
def get_bpe_huffman_analysis(
    bin_width: float = Query(DEFAULT_BIN_WIDTH, gt=0, le=1),
    tensor: str = Query(ALL_TENSORS, description="__all__ or an exact learned tensor name"),
    merges: int = Query(100, ge=0, le=1000, description="Number of top-ranked BPE merge rules to apply"),
    min_pair_frequency: int = Query(2, ge=2),
) -> dict[str, Any]:
    """Compare ordinary Huffman with BPE-tokenized Huffman on one scope."""
    if tensor == ALL_TENSORS:
        tensor_names = list(WEIGHTS)
    elif tensor in WEIGHTS:
        tensor_names = [tensor]
    else:
        raise HTTPException(status_code=404, detail=f"Unknown tensor: {tensor}")
    try:
        result = analyze_bpe_huffman(
            [WEIGHTS[name] for name in tensor_names],
            tensor_names,
            bin_width,
            merges,
            min_pair_frequency,
        )
    except (OverflowError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Could not analyze BPE + Huffman: {error}") from error
    result["dataset"] = {
        "name": "Tiny GPT-2 · full model" if tensor == ALL_TENSORS else tensor,
        "scope": tensor,
        "tensor_count": len(tensor_names),
        "shape": None if tensor == ALL_TENSORS else WEIGHT_SHAPES[tensor],
    }
    result["original_bins"] = [
        {
            **item,
            "lower": item["symbol"] * bin_width,
            "upper": (item["symbol"] + 1) * bin_width,
            "midpoint": (item["symbol"] + 0.5) * bin_width,
        }
        for item in result["original_bins"]
    ]
    return result


@app.get("/api/layers")
def get_layers() -> dict[str, Any]:
    """Return the selectable learned tensors and their shapes and sizes."""
    return {
        "dataset": "Tiny GPT-2",
        "tensor_count": len(WEIGHTS),
        "total_values": TOTAL_WEIGHT_COUNT,
        "layers": [
            {"name": name, "shape": WEIGHT_SHAPES[name], "values": int(values.size)}
            for name, values in WEIGHTS.items()
        ],
    }


@app.get("/api/real-models")
def get_real_models() -> dict[str, Any]:
    return {"models": list_models()}


@app.post("/api/real-models/{model_id}/analysis", status_code=202)
def start_real_model_analysis(
    model_id: str,
    background_tasks: BackgroundTasks,
    bin_width: float = Query(DEFAULT_BIN_WIDTH, ge=1e-5, le=1),
    merge_rare_bins: int = Query(0, ge=0, le=100000),
    coding_method: Literal["huffman", "unary"] = Query("huffman"),
    max_code_length: int | None = Query(None, ge=1, le=64),
    tensor: str = Query(ALL_TENSORS),
) -> dict[str, str]:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown real model: {model_id}")
    if not math.isfinite(bin_width):
        raise HTTPException(status_code=422, detail="bin_width must be finite")
    settings = {
        "bin_width": bin_width,
        "merge_rare_bins": merge_rare_bins,
        "coding_method": coding_method,
        "max_code_length": max_code_length,
        "tensor": tensor,
    }
    job_id = create_job(model_id, settings)
    background_tasks.add_task(run_analysis_job, job_id, model_id, settings)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/real-analysis-jobs/{job_id}")
def get_real_analysis_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return job


