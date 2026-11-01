"""Run the research-roadmap GPT-2 bin-width/perplexity/serialized-size sweep.

Requires torch, transformers, datasets, and numpy. The artifact is a self-contained
ZIP containing the Huffman-coded bin stream, code lengths, bin width, and model
configuration. It is intended as an experimental storage measurement, not a
runtime-ready model format.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
# Transformers probes PyTorch's distributed tensor implementation at import
# time; avoid loading that optional stack for this single-process CPU run.
torch.distributed.is_available = lambda: False
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def huffman_lengths(freq: Counter[int]) -> dict[int, int]:
    if len(freq) == 1:
        return {next(iter(freq)): 1}
    heap = [(n, i, sym) for i, (sym, n) in enumerate(sorted(freq.items()))]
    heapq.heapify(heap)
    serial = len(heap)
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, (a[0] + b[0], serial, (a[2], b[2])))
        serial += 1
    out: dict[int, int] = {}
    def visit(node, depth):
        if isinstance(node, int):
            out[node] = depth
        else:
            visit(node[0], depth + 1); visit(node[1], depth + 1)
    visit(heap[0][2], 0)
    return out


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    result, code, previous = {}, 0, 0
    for sym, length in sorted(lengths.items(), key=lambda p: (p[1], p[0])):
        code <<= length - previous
        result[sym] = (code, length)
        code += 1
        previous = length
    return result


def write_stream(indices: np.ndarray, path: Path) -> tuple[int, Counter[int], dict[int, int]]:
    """Write Huffman bits MSB-first using a bounded byte buffer."""
    # Chunked histogram avoids copying/sorting the full model-sized ID array.
    freq: Counter[int] = Counter()
    chunk_size = 1_000_000
    for start in range(0, len(indices), chunk_size):
        symbols, counts = np.unique(indices[start:start + chunk_size], return_counts=True)
        freq.update({int(symbol): int(count) for symbol, count in zip(symbols, counts)})
    lengths = huffman_lengths(freq)
    codes = canonical_codes(lengths)
    total_bits = sum(freq[s] * lengths[s] for s in freq)
    buf = bytearray()
    acc = nbits = 0
    with path.open("wb") as f:
        for start in range(0, len(indices), chunk_size):
            for raw in indices[start:start + chunk_size]:
                code, width = codes[int(raw)]
                acc = (acc << width) | code
                nbits += width
                while nbits >= 8:
                    nbits -= 8
                    buf.append((acc >> nbits) & 255)
                    acc &= (1 << nbits) - 1 if nbits else 0
            if len(buf) >= 1_000_000:
                f.write(buf); buf.clear()
        if nbits:
            buf.append((acc << (8 - nbits)) & 255)
        f.write(buf)
    return total_bits, freq, lengths


def evaluate(model, tokenizer, device, seq_len: int, max_tokens: int | None) -> float:
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(x for x in data["text"] if x.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tokens:
        ids = ids[:max_tokens]
    ids = ids.to(device)
    model.eval(); total_loss = 0.0; total_count = 0
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, seq_len):
            end = min(start + seq_len, ids.numel() - 1)
            inp = ids[start:end].unsqueeze(0)
            labels = ids[start + 1:end + 1].unsqueeze(0)
            logits = model(inp).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
            )
            total_loss += float(loss); total_count += labels.numel()
    return math.exp(total_loss / total_count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openai-community/gpt2")
    p.add_argument("--model-id", default=None,
                   help="Canonical model identifier to store when --model points to a local cached checkpoint")
    p.add_argument("--resume", action="store_true",
                   help="Skip widths already recorded in results.json and checkpoint after each completed width")
    p.add_argument("--revision", default="main")
    p.add_argument("--widths", type=float, nargs="+", default=[0.001, 0.002, 0.004, 0.008])
    p.add_argument("--output", type=Path, default=Path("research-results/gpt2-huffman-sweep"))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=None, help="Optional smoke-test cap; omit for full WikiText-2 test split")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_model = args.model_id or args.model
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", artifact_model.rstrip("/").split("/")[-1]).lower()
    result_path = args.output / "results.json"
    previous = json.loads(result_path.read_text()) if args.resume and result_path.exists() else None
    completed = {}
    if previous:
        if (previous.get("model") != artifact_model
                or previous.get("dataset") != "Salesforce/wikitext, wikitext-2-raw-v1, test"
                or previous.get("sequence_length") != args.seq_len
                or previous.get("max_tokens") != args.max_tokens):
            raise ValueError("Existing results.json does not match this model/evaluation protocol; use a separate --output directory")
        completed = {float(row["bin_width"]): row for row in previous.get("results", [])}
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    source = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32)
    device = torch.device(args.device); source.to(device)
    resolved_revision = getattr(source.config, "_commit_hash", None) or args.revision
    baseline = evaluate(source, tokenizer, device, args.seq_len, args.max_tokens)
    del source
    if device.type == "mps":
        torch.mps.empty_cache()
    rows = []
    for width in args.widths:
        if args.resume and float(width) in completed:
            print(json.dumps({"status": "already-complete", "bin_width": width}), flush=True)
            continue
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32)
        tensor_metadata = {}
        abs_error = squared_error = max_error = 0.0
        parameter_count = sum(t.numel() for _, t in model.named_parameters())
        # Keep large-model bin IDs in a temporary disk-backed array. A Python
        # list plus concatenate would duplicate many gigabytes for 1B+ models.
        ids_path = args.output / f".{model_slug}-ids-{width:g}.i32"
        flat = np.memmap(ids_path, dtype=np.int32, mode="w+", shape=(parameter_count,))
        offset = 0
        # named_parameters deduplicates GPT-2's tied token embedding / LM head.
        for name, tensor in model.named_parameters():
            values = tensor.detach().cpu().float().numpy()
            bins = np.floor(values / width).astype(np.int32)
            count = int(values.size)
            flat[offset:offset + count] = bins.reshape(-1)
            offset += count
            reconstructed = (bins.astype(np.float32) + 0.5) * width
            errors = values.reshape(-1).astype(np.float64) - reconstructed.reshape(-1).astype(np.float64)
            abs_error += float(np.abs(errors).sum())
            squared_error += float(np.dot(errors, errors))
            max_error = max(max_error, float(np.abs(errors).max(initial=0.0)))
            tensor_metadata[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "count": int(tensor.numel())}
        flat.flush()
        stream_path = args.output / f"stream-{width:g}.bin"
        bits, frequencies, lengths = write_stream(flat, stream_path)
        offset = 0
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                n = tensor.numel()
                # Restore per chunk so a large tensor does not require another
                # full-size float32 temporary alongside the model and ID map.
                target = tensor.view(-1)
                for start in range(0, n, 1_000_000):
                    end = min(start + 1_000_000, n)
                    ids = np.asarray(flat[offset + start:offset + end], dtype=np.float32)
                    restored = (ids + 0.5) * width
                    target[start:end].copy_(torch.from_numpy(restored).to(dtype=tensor.dtype))
                offset += n
        model.to(device)
        ppl = evaluate(model, tokenizer, device, args.seq_len, args.max_tokens)
        del model
        # Include codebook and tensor map in the measured artifact.
        manifest = {"model": artifact_model, "revision": resolved_revision, "bin_width": width,
                    "rounding": "floor(value / width), reconstruct at (bin + 0.5) * width",
                    "symbols": sorted(map(int, frequencies)),
                    "symbol_frequencies": {str(k): int(v) for k, v in frequencies.items()},
                    "huffman_code_lengths": {str(k): int(v) for k, v in lengths.items()},
                    "tensor_order": list(tensor_metadata), "tensors": tensor_metadata,
                    "weight_mae_vs_fp32": abs_error / parameter_count,
                    "weight_mse_vs_fp32": squared_error / parameter_count,
                    "weight_max_abs_error_vs_fp32": max_error,
                    "huffman_payload_bits": bits, "padding_bits": (-bits) % 8}
        manifest_path = args.output / f"manifest-{width:g}.json"
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
        archive = args.output / f"{model_slug}-huffman-{width:g}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
            z.write(stream_path, "stream.bin"); z.write(manifest_path, "manifest.json")
        row = {"bin_width": width, "baseline_perplexity": baseline, "quantized_perplexity": ppl,
               "weight_mae_vs_fp32": abs_error / parameter_count,
               "weight_mse_vs_fp32": squared_error / parameter_count,
               "weight_max_abs_error_vs_fp32": max_error,
               "actual_encoded_bytes": archive.stat().st_size, "huffman_payload_bits": bits,
               "weights": int(flat.size), "occupied_bins": len(frequencies),
               "artifact": str(archive)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        # Write a complete checkpoint after every width so interruption loses
        # at most the currently running width.
        snapshot_rows = {float(r["bin_width"]): r for r in completed.values()}
        snapshot_rows.update({float(r["bin_width"]): r for r in rows})
        result = {"model": artifact_model, "resolved_revision": resolved_revision,
                  "dataset": "Salesforce/wikitext, wikitext-2-raw-v1, test",
                  "sequence_length": args.seq_len, "max_tokens": args.max_tokens,
                  "baseline_perplexity": baseline,
                  "results": [snapshot_rows[k] for k in sorted(snapshot_rows)]}
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        stream_path.unlink(); manifest_path.unlink()
        del flat
        ids_path.unlink()
    result = {"model": artifact_model, "resolved_revision": resolved_revision,
              "dataset": "Salesforce/wikitext, wikitext-2-raw-v1, test",
              "sequence_length": args.seq_len, "max_tokens": args.max_tokens,
              "baseline_perplexity": baseline, "results": rows}
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if (previous.get("model") == result["model"]
                and previous.get("resolved_revision") == result["resolved_revision"]
                and previous.get("dataset") == result["dataset"]
                and previous.get("sequence_length") == result["sequence_length"]
                and previous.get("max_tokens") == result["max_tokens"]):
            combined = {float(row["bin_width"]): row for row in previous.get("results", [])}
            combined.update({float(row["bin_width"]): row for row in rows})
            result["results"] = [combined[width] for width in sorted(combined)]
    result_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
