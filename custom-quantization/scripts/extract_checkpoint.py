#!/usr/bin/env python3
"""Extract unique learned float tensors from a legacy PyTorch checkpoint.

The script intentionally uses only the Python standard library to read the
pickle metadata, then NumPy to save the named float32 arrays in an NPZ bundle.
"""

from __future__ import annotations

import argparse
import math
import pickle
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np


class StorageType:
    def __init__(self, name: str):
        self.name = name


class StorageRef:
    def __init__(self, dtype: str, key: str, length: int):
        self.dtype, self.key, self.length = dtype, key, length


class TensorRef:
    def __init__(self, storage: StorageRef, offset: int, shape: tuple[int, ...], stride: tuple[int, ...]):
        self.storage, self.offset, self.shape, self.stride = storage, offset, shape, stride


class CheckpointUnpickler(pickle.Unpickler):
    def persistent_load(self, pid: tuple[object, ...]) -> StorageRef:
        if pid[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent object: {pid[0]}")
        storage_type, key, _location, length = pid[1:5]
        return StorageRef(storage_type.name, key, length)

    def find_class(self, module: str, name: str):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return lambda storage, offset, shape, stride, *rest: TensorRef(
                storage, offset, tuple(shape), tuple(stride)
            )
        raise pickle.UnpicklingError(f"unsupported checkpoint value: {module}.{name}")


ELEMENT_WIDTHS = {
    "FloatStorage": 4,
    "DoubleStorage": 8,
    "LongStorage": 8,
    "IntStorage": 4,
    "ShortStorage": 2,
    "ByteStorage": 1,
    "CharStorage": 1,
}


def _tensor_array(tensor: TensorRef, raw_storages: dict[str, bytes]) -> np.ndarray:
    if tensor.storage.dtype != "FloatStorage":
        raise TypeError(f"Expected FloatStorage, got {tensor.storage.dtype}.")
    count = math.prod(tensor.shape)
    values = np.empty(count, dtype=np.float32)
    storage_values = np.frombuffer(raw_storages[tensor.storage.key], dtype="<f4")
    for linear_index in range(count):
        remainder = linear_index
        storage_index = tensor.offset
        for dimension in range(len(tensor.shape) - 1, -1, -1):
            coordinate = remainder % tensor.shape[dimension]
            remainder //= tensor.shape[dimension]
            storage_index += coordinate * tensor.stride[dimension]
        values[linear_index] = storage_values[storage_index]
    return values.reshape(tensor.shape)


def extract_tensors(checkpoint: Path) -> dict[str, np.ndarray]:
    with checkpoint.open("rb") as stream:
        unpickler = CheckpointUnpickler(stream)
        magic, version = unpickler.load(), unpickler.load()
        unpickler.load()  # system information
        if magic != 119547037146038801333356 or version != 1001:
            raise ValueError("Expected a legacy PyTorch checkpoint (magic/version mismatch).")
        state = unpickler.load()
        storage_keys = pickle.load(stream)
        descriptors = {
            value.storage.key: value.storage
            for value in state.values()
            if isinstance(value, TensorRef)
        }
        raw_storages: dict[str, bytes] = {}
        for key in storage_keys:
            storage = descriptors[key]
            if storage.dtype not in ELEMENT_WIDTHS:
                raise TypeError(f"Unsupported storage type {storage.dtype} for {key}.")
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError(f"Missing storage length header for {key}.")
            stored_length = struct.unpack("<Q", header)[0]
            if stored_length != storage.length:
                raise ValueError(
                    f"Storage length mismatch for {key}: checkpoint says {stored_length}, "
                    f"metadata says {storage.length}."
                )
            byte_count = storage.length * ELEMENT_WIDTHS[storage.dtype]
            raw = stream.read(byte_count)
            if len(raw) != byte_count:
                raise ValueError(f"Truncated storage {key}.")
            raw_storages[key] = raw

    tensors: dict[str, np.ndarray] = {}
    seen: set[tuple[str, int, tuple[int, ...], tuple[int, ...]]] = set()
    for name, tensor in state.items():
        if not isinstance(tensor, TensorRef) or tensor.storage.dtype != "FloatStorage":
            continue
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            continue
        identity = (tensor.storage.key, tensor.offset, tensor.shape, tensor.stride)
        if identity in seen:
            continue
        seen.add(identity)
        tensors[name] = _tensor_array(tensor, raw_storages)
    if not tensors:
        raise ValueError("No learned float tensors were found in the checkpoint.")
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tensors = extract_tensors(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() != ".npz":
        parser.error("output must end in .npz so tensor names and shapes are retained")
    np.savez_compressed(args.output, **tensors)
    total = sum(array.size for array in tensors.values())
    print(f"Saved {len(tensors)} unique float tensors ({total:,} values) to {args.output}")


if __name__ == "__main__":
    main()
