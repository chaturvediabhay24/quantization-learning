"""Decode a saved Huffman artifact and validate its weights against source bins."""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def decode_stream(stream: bytes, lengths: dict[int, int], count: int,
                  expected_bits: int, padding_bits: int) -> np.ndarray:
    """Decode canonical Huffman codes, requiring exact framing and zero padding."""
    codes: dict[tuple[int, int], int] = {}
    code, previous = 0, 0
    for symbol, width in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        if width < 1 or width < previous:
            raise ValueError("Invalid Huffman code length")
        code <<= width - previous
        if code >= (1 << width):
            raise ValueError("Oversubscribed canonical Huffman code")
        codes[(width, code)] = symbol
        code += 1
        previous = width
    if len(stream) * 8 != expected_bits + padding_bits:
        raise ValueError("Payload byte count does not match declared bit count and padding")
    if padding_bits and (stream[-1] & ((1 << padding_bits) - 1)):
        raise ValueError("Nonzero trailing padding bits")

    # A short lookup table decodes the common short codes with one lookup per
    # symbol; longer codes use the exact bitwise fallback below.
    lookup_width = min(12, max(lengths.values()))
    lookup: list[tuple[int, int] | None] = [None] * (1 << lookup_width)
    for (code_width, code_value), symbol in codes.items():
        if code_width <= lookup_width:
            base = code_value << (lookup_width - code_width)
            repeats = 1 << (lookup_width - code_width)
            for suffix in range(repeats):
                lookup[base + suffix] = (symbol, code_width)
    max_width = max(lengths.values())
    decoded = np.empty(count, dtype=np.int32)
    out = consumed = bit_buffer = bit_count = 0
    for byte in stream:
        if consumed >= expected_bits:
            break
        valid = min(8, expected_bits - consumed)
        bit_buffer = (bit_buffer << valid) | (byte >> (8 - valid))
        bit_count += valid
        consumed += valid
        while bit_count:
            if bit_count >= lookup_width:
                prefix = (bit_buffer >> (bit_count - lookup_width)) & ((1 << lookup_width) - 1)
                entry = lookup[prefix]
                if entry is not None:
                    symbol, code_width = entry
                    decoded[out] = symbol
                    out += 1
                    bit_count -= code_width
                    bit_buffer &= (1 << bit_count) - 1 if bit_count else 0
                    continue
            # No short code matches the current prefix, or fewer than L bits
            # remain. Decode exactly one bit until a complete code is found.
            if bit_count < max_width:
                break
            matched = None
            for code_width in range(1, min(bit_count, max_width) + 1):
                code_value = (bit_buffer >> (bit_count - code_width)) & ((1 << code_width) - 1)
                matched = codes.get((code_width, code_value))
                if matched is not None:
                    decoded[out] = matched
                    out += 1
                    bit_count -= code_width
                    bit_buffer &= (1 << bit_count) - 1 if bit_count else 0
                    break
            if matched is None:
                if bit_count >= max_width:
                    raise ValueError("Invalid Huffman code in payload")
                break
    # Consume any final incomplete prefix; a valid final code may be shorter
    # than lookup_width, so test the remaining exact bit sequence.
    while bit_count:
        matched = None
        for code_width in range(1, bit_count + 1):
            code_value = (bit_buffer >> (bit_count - code_width)) & ((1 << code_width) - 1)
            matched = codes.get((code_width, code_value))
            if matched is not None:
                decoded[out] = matched
                out += 1
                bit_count -= code_width
                bit_buffer &= (1 << bit_count) - 1 if bit_count else 0
                break
        if matched is None:
            raise ValueError("Truncated Huffman code at end of payload")
    if out != count or consumed != expected_bits:
        raise ValueError(f"Incomplete stream: decoded {out}/{count} symbols, consumed {consumed}/{expected_bits} bits")
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-source", default=None,
                        help="Optional local checkpoint path; artifact metadata retains its canonical model ID")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    with zipfile.ZipFile(args.artifact) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")
        manifest = json.loads(archive.read("manifest.json"))
        stream = archive.read("stream.bin")
    lengths = {int(k): int(v) for k, v in manifest["huffman_code_lengths"].items()}
    expected_freq = Counter({int(k): int(v) for k, v in manifest["symbol_frequencies"].items()})
    nweights = sum(int(t["count"]) for t in manifest["tensors"].values())
    decoded = decode_stream(stream, lengths, nweights,
                            int(manifest["huffman_payload_bits"]),
                            int(manifest["padding_bits"]))
    decoded_freq = Counter(map(int, decoded))
    if decoded_freq != expected_freq:
        raise ValueError("Decoded symbol histogram differs from manifest")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_source or manifest["model"], revision=manifest["revision"],
        torch_dtype=torch.float32, local_files_only=args.local_files_only
    )
    named = dict(model.named_parameters())
    expected_order = [name for name in named if name in manifest["tensors"]]
    if list(manifest["tensor_order"]) != expected_order:
        raise ValueError("Model parameter names/order do not match artifact metadata")
    offset = 0
    max_abs_error = 0.0
    width = float(manifest["bin_width"])
    with torch.no_grad():
        for name in manifest["tensor_order"]:
            metadata = manifest["tensors"][name]
            parameter = named[name]
            count = int(metadata["count"])
            if list(parameter.shape) != metadata["shape"] or parameter.numel() != count:
                raise ValueError(f"Shape/count mismatch for {name}")
            source = parameter.detach().cpu().float().numpy().reshape(-1)
            expected_bins = np.floor(source / width).astype(np.int32)
            actual_bins = decoded[offset:offset + count]
            if not np.array_equal(actual_bins, expected_bins):
                mismatch = int(np.flatnonzero(actual_bins != expected_bins)[0])
                raise ValueError(f"Bin mismatch in {name} at flat index {mismatch}")
            restored = ((actual_bins.astype(np.float32) + 0.5) * width).reshape(metadata["shape"])
            reference = ((expected_bins.astype(np.float32) + 0.5) * width).reshape(metadata["shape"])
            max_abs_error = max(max_abs_error, float(np.max(np.abs(restored - reference), initial=0.0)))
            parameter.copy_(torch.from_numpy(restored).to(parameter.device, parameter.dtype))
            offset += count
    if offset != nweights:
        raise ValueError(f"Tensor metadata covers {offset} weights, expected {nweights}")

    model.to(args.device).eval()
    sample = torch.tensor([[464, 3290, 318, 257]], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        logits = model(sample).logits
    report = {
        "artifact": str(args.artifact), "model": manifest["model"],
        "revision": manifest["revision"], "weights_validated": nweights,
        "occupied_symbols": len(decoded_freq), "payload_bits": manifest["huffman_payload_bits"],
        "padding_bits": manifest["padding_bits"], "max_reconstruction_difference_vs_expected": max_abs_error,
        "sample_logits_finite": bool(torch.isfinite(logits).all()),
        "sample_logits_shape": list(logits.shape), "round_trip": "passed"
    }
    report_path = args.artifact.parent / f"{args.artifact.stem}-validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
