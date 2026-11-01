"""Compare decoded Huffman model logits with direct in-memory midpoint reconstruction."""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from validate_huffman_codec import decode_stream


ROOT = Path("research-results/gpt2-huffman-sweep")
MODEL_SOURCE = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
WIDTHS = ["0.0003", "0.001", "0.002", "0.004", "0.008", "0.012"]
BATCH_REPORT = ROOT / "logit-parity-batch-progress.json"
PARAMETERS = 124_439_808


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def restore(model_params: dict, base: dict) -> None:
    with torch.no_grad():
        for name, parameter in model_params.items():
            parameter.copy_(base[name])


def main() -> None:
    started = time.perf_counter()
    batch = {"status": "loading_source", "widths": [float(w) for w in WIDTHS], "completed": []}

    def save_batch() -> None:
        batch["elapsed_seconds"] = time.perf_counter() - started
        write(BATCH_REPORT, batch)

    save_batch()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_SOURCE, local_files_only=True, torch_dtype=torch.float32
        ).eval()
        params = dict(model.named_parameters())
        base = {name: parameter.detach().clone() for name, parameter in params.items()}
        sample_tokens = [464, 3290, 318, 257]
        input_ids = torch.tensor([sample_tokens], dtype=torch.long)
        batch.update({"status": "comparing", "source_tensors": len(params),
                      "weights": sum(p.numel() for p in params.values()),
                      "sample_tokens": sample_tokens})
        save_batch()

        for width_text in WIDTHS:
            width_started = time.perf_counter()
            width = float(width_text)
            artifact = ROOT / f"gpt2-huffman-{width_text}.zip"
            output = ROOT / f"gpt2-huffman-{width_text}-logit-parity.json"
            report = {"artifact": str(artifact), "bin_width": width,
                      "status": "reading_artifact"}
            write(output, report)
            batch.update({"current_width": width, "status": "comparing"})
            save_batch()
            with zipfile.ZipFile(artifact) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                stream = archive.read("stream.bin")
            lengths = {int(k): int(v) for k, v in manifest["huffman_code_lengths"].items()}
            nweights = sum(int(t["count"]) for t in manifest["tensors"].values())
            decoded = decode_stream(stream, lengths, nweights,
                                    int(manifest["huffman_payload_bits"]),
                                    int(manifest["padding_bits"]))
            frequencies = Counter(map(int, decoded))
            if frequencies != Counter({int(k): int(v) for k, v in manifest["symbol_frequencies"].items()}):
                raise ValueError(f"Decoded histogram mismatch for width {width}")
            report.update({"status": "decoded", "weights_decoded": nweights,
                           "occupied_symbols": len(frequencies)})
            write(output, report)

            # Path A: compute bin IDs directly from the pinned FP32 source and
            # install those midpoints in memory.
            offset = 0
            tensor_order = manifest["tensor_order"]
            if tensor_order != list(params):
                raise ValueError(f"Source parameter order differs at width {width}")
            with torch.no_grad():
                for name in tensor_order:
                    meta = manifest["tensors"][name]
                    count = int(meta["count"])
                    source = base[name].numpy().reshape(-1)
                    expected_bins = np.floor(source / width).astype(np.int32)
                    actual_bins = decoded[offset:offset + count]
                    if not np.array_equal(expected_bins, actual_bins):
                        raise ValueError(f"Decoded IDs differ from direct bins in {name}")
                    expected_midpoint = ((expected_bins.astype(np.float32) + 0.5) * width).reshape(meta["shape"])
                    params[name].copy_(torch.from_numpy(expected_midpoint))
                    offset += count
            if offset != PARAMETERS:
                raise ValueError(f"Direct path reconstructed {offset} weights, expected {PARAMETERS}")
            report["status"] = "evaluating_direct_reconstruction"
            write(output, report)
            with torch.inference_mode():
                direct_logits = model(input_ids).logits.detach().clone()

            # Path B: install the values reconstructed from the decoded stream.
            offset = 0
            with torch.no_grad():
                for name in tensor_order:
                    meta = manifest["tensors"][name]
                    count = int(meta["count"])
                    decoded_midpoint = ((decoded[offset:offset + count].astype(np.float32) + 0.5) * width)
                    params[name].copy_(torch.from_numpy(decoded_midpoint.reshape(meta["shape"])))
                    offset += count
            report["status"] = "evaluating_decoded_reconstruction"
            write(output, report)
            with torch.inference_mode():
                decoded_logits = model(input_ids).logits.detach().clone()
            differences = torch.abs(direct_logits - decoded_logits)
            max_difference = float(differences.max().item())
            mean_difference = float(differences.mean().item())
            report.update({
                "status": "complete", "sample_sequence_length": len(sample_tokens),
                "logits_shape": list(direct_logits.shape),
                "direct_logits_sha256": hashlib.sha256(direct_logits.cpu().numpy().tobytes()).hexdigest(),
                "decoded_logits_sha256": hashlib.sha256(decoded_logits.cpu().numpy().tobytes()).hexdigest(),
                "logits_bitwise_equal": bool(torch.equal(direct_logits, decoded_logits)),
                "max_abs_logit_difference": max_difference,
                "mean_abs_logit_difference": mean_difference,
                "logit_parity": "passed" if max_difference == 0 else "failed",
                "elapsed_seconds": time.perf_counter() - width_started,
            })
            write(output, report)
            batch["completed"].append({"width": width, "report": str(output),
                                       "logits_bitwise_equal": report["logits_bitwise_equal"],
                                       "max_abs_logit_difference": max_difference})
            save_batch()
            restore(params, base)

        batch.update({"status": "complete", "total_seconds": time.perf_counter() - started})
        save_batch()
        print(json.dumps(batch, indent=2))
    except BaseException as exc:
        batch.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                      "total_seconds": time.perf_counter() - started})
        save_batch()
        raise


if __name__ == "__main__":
    main()
