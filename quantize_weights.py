"""Compare symmetric low-bit fake quantization over a PyTorch checkpoint.

The script rounds each learned floating-point parameter to a blockwise integer
grid, reconstructs it in float32, and reports reconstruction error and the
estimated code-plus-scale storage. It does not write a quantized checkpoint or
benchmark inference kernels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pickle
import struct


class StorageType:
    def __init__(self, name: str):
        self.name = name


class StorageRef:
    def __init__(self, dtype: str, key: str, length: int):
        self.dtype = dtype
        self.key = key
        self.length = length


class TensorRef:
    def __init__(self, storage: StorageRef, offset: int, shape: tuple[int, ...], stride: tuple[int, ...]):
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.stride = stride


class CheckpointUnpickler(pickle.Unpickler):
    """Read tensor metadata from PyTorch's legacy checkpoint format."""

    def persistent_load(self, pid: tuple[object, ...]) -> StorageRef:
        if pid[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent object: {pid[0]}")
        storage_type, key, _location, length = pid[1:5]
        return StorageRef(storage_type.name, key, length)

    def find_class(self, module: str, name: str):
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict

            return OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return lambda storage, offset, shape, stride, *rest: TensorRef(
                storage, offset, tuple(shape), tuple(stride)
            )
        raise pickle.UnpicklingError(f"unsupported checkpoint value: {module}.{name}")


def load_legacy_checkpoint(path: Path) -> tuple[dict[str, tuple[float, ...]], int]:
    """Load the small, legacy-format local GPT-2 checkpoint using stdlib only."""
    with path.open("rb") as stream:
        unpickler = CheckpointUnpickler(stream)
        magic = unpickler.load()
        version = unpickler.load()
        unpickler.load()  # system information
        if magic != 119547037146038801333356 or version != 1001:
            raise ValueError("Expected a legacy PyTorch checkpoint (magic/version mismatch).")
        state = unpickler.load()
        storage_keys = pickle.load(stream)

        storages: dict[str, tuple[float, ...] | bytes] = {}
        descriptors: dict[str, StorageRef] = {}
        for tensor in state.values():
            if isinstance(tensor, TensorRef):
                descriptors[tensor.storage.key] = tensor.storage
        for key in storage_keys:
            storage = descriptors[key]
            size = 4 if storage.dtype == "FloatStorage" else 1 if storage.dtype == "ByteStorage" else 0
            if size == 0:
                raise ValueError(f"Unsupported storage type: {storage.dtype}")
            raw = stream.read(storage.length * size)
            if len(raw) != storage.length * size:
                raise ValueError(f"Truncated checkpoint storage: {key}")
            storages[key] = (
                tuple(item[0] for item in struct.iter_unpack("<f", raw))
                if storage.dtype == "FloatStorage"
                else raw
            )

    parameters: dict[str, tuple[float, ...]] = {}
    seen: set[tuple[str, int, tuple[int, ...], tuple[int, ...]]] = set()
    aliases = 0
    for name, tensor in state.items():
        if not isinstance(tensor, TensorRef) or "attn.bias" in name or "masked_bias" in name:
            continue
        if tensor.storage.dtype != "FloatStorage":
            continue
        identity = (tensor.storage.key, tensor.offset, tensor.shape, tensor.stride)
        if identity in seen:
            aliases += 1
            continue
        seen.add(identity)
        numel = math.prod(tensor.shape)
        flat: list[float] = []
        for linear_index in range(numel):
            remainder = linear_index
            storage_index = tensor.offset
            for dimension in range(len(tensor.shape) - 1, -1, -1):
                coordinate = remainder % tensor.shape[dimension]
                remainder //= tensor.shape[dimension]
                storage_index += coordinate * tensor.stride[dimension]
            flat.append(storages[tensor.storage.key][storage_index])  # type: ignore[index]
        parameters[name] = tuple(flat)
    return parameters, aliases


def quantize_tensor(
    values: tuple[float, ...], bits: int, group_size: int
) -> tuple[float, float, float, int]:
    """Return absolute-error sum, squared-error sum, max error, and value count."""
    qmax = (1 << (bits - 1)) - 1
    abs_sum = squared_sum = max_error = 0.0
    for start in range(0, len(values), group_size):
        block = values[start : start + group_size]
        absmax = max(abs(value) for value in block)
        scale = absmax / qmax if absmax else 1.0
        for value in block:
            scaled = value / scale
            lower = math.floor(scaled)
            fraction = scaled - lower
            code = lower if fraction < 0.5 else lower + 1 if fraction > 0.5 else lower if lower % 2 == 0 else lower + 1
            code = max(-qmax, min(qmax, code))
            error = code * scale - value
            abs_sum += abs(error)
            squared_sum += error * error
            max_error = max(max_error, abs(error))
    return abs_sum, squared_sum, max_error, len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/tiny-gpt2/pytorch_model.bin"),
    )
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 5, 8])
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[32, 64])
    args = parser.parse_args()
    if any(bits < 2 or bits > 8 for bits in args.bits):
        parser.error("--bits must be between 2 and 8")
    if any(size < 1 for size in args.group_sizes):
        parser.error("--group-sizes must be positive")

    parameters, aliases = load_legacy_checkpoint(args.checkpoint)
    parameter_count = sum(len(value) for value in parameters.values())
    print(f"checkpoint: {args.checkpoint}")
    print(
        f"learned tensors: {len(parameters)}; parameters: {parameter_count:,}; "
        f"tied aliases skipped: {aliases}"
    )
    print("symmetric signed quantization; one float32 absmax scale per group")
    print(
        f"{'bits':>4} {'group':>6} {'MAE':>12} {'RMSE':>12} "
        f"{'max error':>12} {'effective bits/w':>17} {'FP32 ratio':>11}"
    )

    for bits in args.bits:
        for group_size in args.group_sizes:
            abs_error_sum = 0.0
            squared_error_sum = 0.0
            max_error = 0.0
            scale_count = 0
            for value in parameters.values():
                abs_sum, squared_sum, layer_max, count = quantize_tensor(
                    value, bits, group_size
                )
                abs_error_sum += abs_sum
                squared_error_sum += squared_sum
                max_error = max(max_error, layer_max)
                scale_count += math.ceil(count / group_size)

            effective_bits = (bits * parameter_count + 32 * scale_count) / parameter_count
            mae = abs_error_sum / parameter_count
            rmse = math.sqrt(squared_error_sum / parameter_count)
            ratio = 32 / effective_bits
            print(
                f"{bits:>4} {group_size:>6} {mae:12.8f} {rmse:12.8f} "
                f"{max_error:12.8f} {effective_bits:17.4f} {ratio:10.2f}x"
            )


if __name__ == "__main__":
    main()
