from __future__ import annotations

import heapq
import math
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np


MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "qwen3-0.6b": {"name": "Qwen3-0.6B", "repo_id": "Qwen/Qwen3-0.6B", "family": "Qwen3"},
    "qwen3-1.7b": {"name": "Qwen3-1.7B", "repo_id": "Qwen/Qwen3-1.7B", "family": "Qwen3"},
    "qwen3-4b": {
        "name": "Qwen3-4B-Instruct-2507",
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "family": "Qwen3",
    },
}
MODEL_FILE_PATTERNS = ["config.json", "*.safetensors", "*.safetensors.index.json"]
CHUNK_VALUES = 1_000_000
MAX_OCCUPIED_BINS = 150_000

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _set_job(job_id: str, **updates: Any) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].update(updates)


def create_job(model_id: str, settings: dict[str, Any]) -> str:
    job_id = uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "model_id": model_id,
            "status": "queued",
            "stage": "Waiting to start",
            "progress": 0.0,
            "settings": settings,
            "result": None,
            "error": None,
        }
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _huffman_lengths(frequencies: dict[int, int]) -> dict[int, int]:
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
        else:
            visit(node[0], depth + 1)
            visit(node[1], depth + 1)

    visit(heap[0][2], 0)
    return lengths


def _snapshot_for_model(repo_id: str, progress: Callable[[str, float], None]) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    progress("Checking local model caches", 0.02)
    project_cache = Path(__file__).resolve().parents[1] / "model-cache"
    for cache_dir in (project_cache, Path(HF_HUB_CACHE)):
        try:
            snapshot = snapshot_download(
                repo_id=repo_id,
                allow_patterns=MODEL_FILE_PATTERNS,
                local_files_only=True,
                cache_dir=str(cache_dir),
            )
            local_weights = list(Path(snapshot).glob("*.safetensors"))
            if local_weights and all(path.stat().st_size > 1_000_000 for path in local_weights):
                return Path(snapshot)
        except Exception:
            continue

    progress("Downloading checkpoint weights from Hugging Face", 0.04)
    project_cache = Path(__file__).resolve().parents[1] / "model-cache"
    snapshot = snapshot_download(
        repo_id=repo_id,
        allow_patterns=MODEL_FILE_PATTERNS,
        cache_dir=str(project_cache),
    )
    return Path(snapshot)


def _weight_chunks(path: Path, tensor_scope: str):
    """Yield floating weight chunks directly from a safetensors file on CPU."""
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if tensor_scope != "__all__" and name != tensor_scope:
                continue
            tensor_slice = handle.get_slice(name)
            dtype = tensor_slice.get_dtype()
            if not (dtype.startswith("F") or dtype == "BF16"):
                continue
            shape = tuple(tensor_slice.get_shape())
            if not shape:
                yield name, dtype, handle.get_tensor(name).reshape(-1)
                continue
            values_per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
            rows_per_chunk = max(1, CHUNK_VALUES // values_per_row)
            for start in range(0, shape[0], rows_per_chunk):
                stop = min(shape[0], start + rows_per_chunk)
                part = tensor_slice[start:stop]
                if part.numel():
                    yield name, dtype, part


def _frequencies_for_symbols(
    source_frequencies: dict[int, int], symbol_by_bin: dict[int, int]
) -> dict[int, int]:
    output: Counter[int] = Counter()
    for source_bin, count in source_frequencies.items():
        output[symbol_by_bin[source_bin]] += count
    return dict(output)


def analyze_checkpoint(
    model_id: str,
    bin_width: float,
    merge_rare_bins: int,
    coding_method: str,
    max_code_length: int | None,
    tensor_scope: str,
    progress: Callable[[str, float], None],
) -> dict[str, Any]:
    import torch

    model = MODEL_REGISTRY[model_id]
    snapshot = _snapshot_for_model(model["repo_id"], progress)
    weight_files = sorted(snapshot.glob("*.safetensors"))
    weight_files = [path for path in weight_files if path.stat().st_size > 0]
    if not weight_files:
        raise RuntimeError("The downloaded model snapshot has no complete safetensors weights.")

    checkpoint_bytes = sum(path.stat().st_size for path in weight_files)
    from safetensors import safe_open

    tensor_names: list[str] = []
    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                dtype = handle.get_slice(name).get_dtype()
                if dtype.startswith("F") or dtype == "BF16":
                    tensor_names.append(name)
    if tensor_scope != "__all__" and tensor_scope not in tensor_names:
        raise ValueError(f"Unknown tensor scope: {tensor_scope}")

    source_frequencies: Counter[int] = Counter()
    source_sums: Counter[int] = Counter()
    dtype_counts: Counter[str] = Counter()
    value_count = 0
    baseline_absolute_sum = 0.0
    baseline_squared_sum = 0.0
    baseline_max_error = 0.0
    def visit_pass(pass_number: int, pass_total: int, visit_chunk: Callable[..., None]) -> None:
        for file_number, weight_file in enumerate(weight_files, start=1):
            file_size = weight_file.stat().st_size
            pass_base, pass_span = (0.05, 0.43) if pass_number == 1 else (0.53, 0.40)
            progress(
                f"Reading checkpoint · pass {pass_number}/{pass_total} · file {file_number}/{len(weight_files)} · {weight_file.name}",
                pass_base + pass_span * (file_number - 1) / len(weight_files),
            )
            processed_in_file = 0
            for tensor_name, dtype, tensor_chunk in _weight_chunks(weight_file, tensor_scope):
                processed_in_file += tensor_chunk.numel() * tensor_chunk.element_size()
                values = tensor_chunk.to(dtype=torch.float32).contiguous().numpy().astype(np.float64).reshape(-1)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite source value in tensor {tensor_name}")
                visit_chunk(tensor_name, dtype, values)
                progress(
                    f"Reading checkpoint · pass {pass_number}/{pass_total} · {tensor_name}",
                    min(0.94, pass_base + pass_span * (file_number - 1 + min(1.0, processed_in_file / max(file_size, 1))) / len(weight_files)),
                )

    def collect_histogram(tensor_name: str, dtype: str, values: np.ndarray) -> None:
        nonlocal value_count, baseline_absolute_sum, baseline_squared_sum, baseline_max_error
        dtype_counts[dtype] += int(values.size)
        value_count += int(values.size)
        indices = np.floor(values / bin_width).astype(np.int64)
        unique_bins, inverse, counts = np.unique(indices, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=values)
        for source_bin, count, value_sum in zip(unique_bins, counts, sums):
            key = int(source_bin)
            source_frequencies[key] += int(count)
            source_sums[key] += float(value_sum)
        reconstruction = (indices.astype(np.float64) + 0.5) * bin_width
        errors = values - reconstruction
        baseline_absolute_sum += float(np.abs(errors).sum())
        baseline_squared_sum += float(np.dot(errors, errors))
        baseline_max_error = max(baseline_max_error, float(np.abs(errors).max(initial=0.0)))
        if len(source_frequencies) > MAX_OCCUPIED_BINS:
            raise ValueError(
                f"This width creates more than {MAX_OCCUPIED_BINS:,} occupied bins. Increase the bin width."
            )

    # Pass one measures actual checkpoint values and builds the exact global bin histogram.
    visit_pass(1, 1, collect_histogram)
    if value_count == 0:
        raise ValueError("The selected scope contains no floating-point weights.")

    original_bins = sorted(source_frequencies)
    symbol_by_bin = {source_bin: source_bin for source_bin in original_bins}
    reconstruction_by_symbol = {
        source_bin: (source_bin + 0.5) * bin_width for source_bin in original_bins
    }
    custom_symbols: set[int] = set()
    effective_merge_count = min(merge_rare_bins, len(original_bins))
    rare_bins = set(
        sorted(original_bins, key=lambda source_bin: (source_frequencies[source_bin], source_bin))[
            :effective_merge_count
        ]
    )
    max_input_bin = original_bins[-1]
    merged_symbol = max_input_bin + 1
    if rare_bins:
        merged_count = sum(source_frequencies[source_bin] for source_bin in rare_bins)
        merged_sum = sum(source_sums[source_bin] for source_bin in rare_bins)
        reconstruction_by_symbol[merged_symbol] = merged_sum / merged_count
        custom_symbols.add(merged_symbol)
        for source_bin in rare_bins:
            symbol_by_bin[source_bin] = merged_symbol

    overflow_symbol = max_input_bin + (2 if rare_bins else 1)
    length_limited_bins: set[int] = set()
    overflow_representative: float | None = None
    if coding_method == "huffman":
        for _ in range(len(original_bins) + 1):
            current_frequencies = _frequencies_for_symbols(source_frequencies, symbol_by_bin)
            current_lengths = _huffman_lengths(current_frequencies)
            if max_code_length is None:
                break
            to_merge = {
                symbol for symbol, length in current_lengths.items() if length > max_code_length
            }
            if not to_merge:
                break
            if overflow_symbol in current_frequencies:
                to_merge.add(overflow_symbol)
            source_bins_to_merge = [
                source_bin for source_bin, symbol in symbol_by_bin.items() if symbol in to_merge
            ]
            merged_count = sum(source_frequencies[source_bin] for source_bin in source_bins_to_merge)
            merged_sum = sum(source_sums[source_bin] for source_bin in source_bins_to_merge)
            overflow_representative = merged_sum / merged_count
            for source_bin in source_bins_to_merge:
                symbol_by_bin[source_bin] = overflow_symbol
                length_limited_bins.add(source_bin)
            reconstruction_by_symbol[overflow_symbol] = overflow_representative
            custom_symbols.add(overflow_symbol)
        else:
            raise ValueError("The Huffman code-length limit did not converge.")
    elif max_code_length is not None:
        current_frequencies = _frequencies_for_symbols(source_frequencies, symbol_by_bin)
        if len(current_frequencies) > max_code_length:
            ranked = sorted(
                current_frequencies,
                key=lambda symbol: (-current_frequencies[symbol], symbol),
            )
            keep_symbols = set(ranked[:max_code_length - 1])
            to_merge = set(current_frequencies) - keep_symbols
            source_bins_to_merge = [
                source_bin for source_bin, symbol in symbol_by_bin.items() if symbol in to_merge
            ]
            merged_count = sum(source_frequencies[source_bin] for source_bin in source_bins_to_merge)
            merged_sum = sum(source_sums[source_bin] for source_bin in source_bins_to_merge)
            overflow_representative = merged_sum / merged_count
            for source_bin in source_bins_to_merge:
                symbol_by_bin[source_bin] = overflow_symbol
                length_limited_bins.add(source_bin)
            reconstruction_by_symbol[overflow_symbol] = overflow_representative
            custom_symbols.add(overflow_symbol)

    output_frequencies = _frequencies_for_symbols(source_frequencies, symbol_by_bin)
    if coding_method == "huffman":
        lengths = _huffman_lengths(output_frequencies)
        tree_bits = 2 * len(output_frequencies) - 1
    else:
        ranked = sorted(
            output_frequencies,
            key=lambda symbol: (-output_frequencies[symbol], symbol),
        )
        lengths = {symbol: rank + 1 for rank, symbol in enumerate(ranked)}
        tree_bits = 0
    if max_code_length is not None and max(lengths.values()) > max_code_length:
        raise ValueError(f"Could not satisfy the {coding_method} code-length limit.")

    pooled = bool(custom_symbols)
    if pooled:
        reconstruction_by_bin = np.asarray(
            [reconstruction_by_symbol[symbol_by_bin[source_bin]] for source_bin in original_bins],
            dtype=np.float64,
        )
        sorted_bins = np.asarray(original_bins, dtype=np.int64)
        second_absolute_sum = 0.0
        second_squared_sum = 0.0
        second_max_error = 0.0
        second_value_count = 0

        def collect_pooled_error(_tensor_name: str, _dtype: str, values: np.ndarray) -> None:
            nonlocal second_absolute_sum, second_squared_sum, second_max_error, second_value_count
            indices = np.floor(values / bin_width).astype(np.int64)
            positions = np.searchsorted(sorted_bins, indices)
            reconstruction = reconstruction_by_bin[positions]
            errors = values - reconstruction
            second_absolute_sum += float(np.abs(errors).sum())
            second_squared_sum += float(np.dot(errors, errors))
            second_max_error = max(second_max_error, float(np.abs(errors).max(initial=0.0)))
            second_value_count += int(values.size)

        progress("Applying pooled reconstructions and measuring exact error", 0.52)
        visit_pass(2, 2, collect_pooled_error)
        absolute_sum = second_absolute_sum
        squared_sum = second_squared_sum
        max_error = second_max_error
        measured_value_count = second_value_count
    else:
        absolute_sum = baseline_absolute_sum
        squared_sum = baseline_squared_sum
        max_error = baseline_max_error
        measured_value_count = value_count

    if measured_value_count != value_count:
        raise RuntimeError("The streamed checkpoint passes did not process the same number of weights.")
    mae = absolute_sum / value_count
    mse = squared_sum / value_count
    rmse = math.sqrt(mse)
    sample_payload_bits = sum(
        output_frequencies[symbol] * lengths[symbol] for symbol in output_frequencies
    )
    average_code_bits = sample_payload_bits / value_count
    entropy_bits = -sum(
        (count / value_count) * math.log2(count / value_count)
        for count in output_frequencies.values()
    )
    symbol_min = min(output_frequencies)
    symbol_max = max(output_frequencies)
    symbol_index_width = max(1, math.ceil(math.log2(symbol_max - symbol_min + 1)))
    custom_symbols.intersection_update(output_frequencies)
    table_bits = tree_bits + len(output_frequencies) * symbol_index_width + 129 + 96 * len(custom_symbols)
    fixed_width_bits = max(1, math.ceil(math.log2(len(output_frequencies))))
    total_payload_bits = sample_payload_bits
    total_table_bits = table_bits
    fp32_reference_bytes = value_count * 4
    source_tensor_bytes = sum(
        count * {"F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}.get(dtype, 4)
        for dtype, count in dtype_counts.items()
    )
    comparison_bytes = checkpoint_bytes if tensor_scope == "__all__" else source_tensor_bytes
    estimated_total_bytes = math.ceil((total_payload_bits + total_table_bits) / 8)
    top_bins = sorted(
        (
            {
                "index": int(source_bin),
                "frequency": int(source_frequencies[source_bin]),
                "code_length": int(lengths[symbol_by_bin[source_bin]]),
                "pooled": symbol_by_bin[source_bin] != source_bin,
            }
            for source_bin in original_bins
        ),
        key=lambda item: (-item["frequency"], item["index"]),
    )[:30]
    progress("Analysis complete", 1.0)
    return {
        "model": {
            **model,
            "revision": snapshot.name,
            "checkpoint_files": len(weight_files),
            "checkpoint_bytes": checkpoint_bytes,
            "selected_tensor": tensor_scope,
            "tensor_count": len(tensor_names) if tensor_scope == "__all__" else 1,
            "available_tensors": tensor_names,
            "weights_analyzed": value_count,
            "source_dtypes": dict(dtype_counts),
        },
        "settings": {
            "bin_width": bin_width,
            "coding_method": coding_method,
            "merge_rare_bins": effective_merge_count,
            "max_code_length": max_code_length,
        },
        "metrics": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "max_abs_error": max_error,
            "occupied_bins": len(output_frequencies),
            "original_occupied_bins": len(original_bins),
            "merged_rare_bin_count": effective_merge_count,
            "merged_rare_value_count": sum(source_frequencies[b] for b in rare_bins),
            "length_limited_bin_count": len(length_limited_bins),
            "length_limited_value_count": sum(source_frequencies[b] for b in length_limited_bins),
            "average_code_bits_per_weight": average_code_bits,
            "entropy_bits_per_weight": entropy_bits,
            "actual_max_code_length": max(lengths.values()),
            "coding_payload_bits": total_payload_bits,
            "estimated_table_bits": total_table_bits,
            "estimated_total_bytes": estimated_total_bytes,
            "fixed_width_bits_per_weight": fixed_width_bits,
            "fixed_width_payload_bytes": math.ceil(value_count * fixed_width_bits / 8),
            "fp32_reference_bytes": fp32_reference_bytes,
            "source_checkpoint_bytes": comparison_bytes,
            "estimated_reduction_vs_fp32_percent": 100 * (1 - estimated_total_bytes / fp32_reference_bytes),
            "estimated_reduction_vs_source_percent": 100 * (1 - estimated_total_bytes / comparison_bytes),
        },
        "top_bins": top_bins,
        "method_note": (
            "Metrics are computed from every selected floating-point tensor in the real safetensors "
            "checkpoint. The checkpoint is read in bounded CPU chunks; no model object, forward pass, "
            "or inference is used. The compressed size is an estimate for the selected global codebook, "
            "not a written Huffman bitstream."
        ),
    }


def list_models() -> list[dict[str, Any]]:
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    models = []
    for model_id, model in MODEL_REGISTRY.items():
        available = False
        cached_path: Path | None = None
        for cache_dir in (Path(__file__).resolve().parents[1] / "model-cache", Path(HF_HUB_CACHE)):
            try:
                snapshot = snapshot_download(
                    repo_id=model["repo_id"],
                    allow_patterns=MODEL_FILE_PATTERNS,
                    local_files_only=True,
                    cache_dir=str(cache_dir),
                )
                candidate = Path(snapshot)
                weights = list(candidate.glob("*.safetensors"))
                available = bool(weights) and all(path.stat().st_size > 1_000_000 for path in weights)
                if available:
                    cached_path = candidate
                    break
            except Exception:
                continue
        models.append({
            "id": model_id,
            **model,
            "available_locally": available,
            "cached_path": str(cached_path) if cached_path else None,
        })
    return models


def run_analysis_job(
    job_id: str,
    model_id: str,
    settings: dict[str, Any],
) -> None:
    def update(stage: str, progress: float) -> None:
        _set_job(job_id, status="running", stage=stage, progress=progress)

    try:
        result = analyze_checkpoint(
            model_id=model_id,
            bin_width=settings["bin_width"],
            merge_rare_bins=settings["merge_rare_bins"],
            coding_method=settings["coding_method"],
            max_code_length=settings["max_code_length"],
            tensor_scope=settings["tensor"],
            progress=update,
        )
        _set_job(job_id, status="complete", stage="Analysis complete", progress=1.0, result=result)
    except Exception as error:
        _set_job(job_id, status="failed", stage="Analysis failed", error=str(error))
