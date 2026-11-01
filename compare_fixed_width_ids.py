"""Compare fixed-width and Huffman coding for identical full-GPT-2 bin IDs."""
from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from validate_huffman_codec import decode_stream


SOURCE = Path("research-results/gpt2-huffman-sweep/gpt2-huffman-0.002.zip")
OUTPUT = Path("research-results/gpt2-huffman-sweep/gpt2-huffman-0.002-fixed-width.zip")


def pack_fixed_codes(codes: np.ndarray, width: int) -> bytes:
    """Pack unsigned labels MSB-first; currently supports widths 1 through 16."""
    if not 1 <= width <= 16:
        raise ValueError("width must be between 1 and 16")
    bit_count = int(codes.size) * width
    output = bytearray((bit_count + 7) // 8)
    accumulator = bits = out = 0
    for start in range(0, codes.size, 1_000_000):
        for raw in codes[start:start + 1_000_000]:
            accumulator = (accumulator << width) | int(raw)
            bits += width
            while bits >= 8:
                bits -= 8
                output[out] = (accumulator >> bits) & 0xFF
                out += 1
                accumulator &= (1 << bits) - 1 if bits else 0
    if bits:
        output[out] = (accumulator << (8 - bits)) & 0xFF
    return bytes(output)


def main() -> None:
    with zipfile.ZipFile(SOURCE) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        huffman_stream = archive.read("stream.bin")
    lengths = {int(k): int(v) for k, v in manifest["huffman_code_lengths"].items()}
    count = sum(int(item["count"]) for item in manifest["tensors"].values())
    symbols = np.asarray(sorted(int(symbol) for symbol in manifest["symbol_frequencies"]), dtype=np.int32)
    indices = decode_stream(huffman_stream, lengths, count,
                            int(manifest["huffman_payload_bits"]), int(manifest["padding_bits"]))
    labels = np.searchsorted(symbols, indices).astype(np.uint16)
    if not np.array_equal(symbols[labels], indices):
        raise ValueError("Fixed-width symbol lookup failed to preserve bin IDs")
    label_width = math.ceil(math.log2(len(symbols)))
    fixed_stream = pack_fixed_codes(labels, label_width)

    fixed_manifest = {key: value for key, value in manifest.items()
                      if key not in {"huffman_code_lengths", "huffman_payload_bits", "padding_bits"}}
    fixed_manifest.update({
        "entropy_coding": "fixed-width sorted-symbol labels",
        "fixed_code_width_bits": label_width,
        "fixed_codebook_symbols": symbols.tolist(),
        "fixed_code_payload_bits": count * label_width,
        "padding_bits": (-count * label_width) % 8,
    })
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("stream.bin", fixed_stream)
        archive.writestr("manifest.json", json.dumps(fixed_manifest, separators=(",", ":")))

    huffman_bytes = SOURCE.stat().st_size
    fixed_bytes = OUTPUT.stat().st_size
    report = {
        "model": manifest["model"], "revision": manifest["revision"],
        "bin_width": manifest["bin_width"], "weights": count,
        "occupied_bins": int(symbols.size), "identical_bin_ids": True,
        "quality_metric_note": "Decoded bin IDs and reconstructed weights are identical, so model perplexity is unchanged by the code assignment.",
        "huffman": {
            "payload_bits": int(manifest["huffman_payload_bits"]), "archive_bytes": huffman_bytes,
            "effective_bits_per_weight": huffman_bytes * 8 / count,
        },
        "fixed_width": {
            "code_width_bits": label_width, "payload_bits": count * label_width,
            "archive_bytes": fixed_bytes, "effective_bits_per_weight": fixed_bytes * 8 / count,
            "artifact": str(OUTPUT),
        },
        "archive_size_reduction_vs_fixed_width_percent": 100 * (1 - huffman_bytes / fixed_bytes),
    }
    result_path = Path("research-results/gpt2-huffman-sweep/fixed-width-coding-comparison.json")
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
