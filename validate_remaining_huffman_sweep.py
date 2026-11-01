"""Validate the remaining saved Huffman sweep artifacts one at a time."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("research-results/gpt2-huffman-sweep")
MODEL_SOURCE = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
WIDTHS = ["0.001", "0.004", "0.008"]
PROGRESS = ROOT / "validation-batch-progress.json"
VALIDATOR = Path("validate_huffman_codec.py")


def main() -> None:
    started = time.perf_counter()
    progress = {
        "status": "starting", "batch_started_utc": datetime.now(timezone.utc).isoformat(),
        "already_validated_widths": [0.0003, 0.002, 0.012],
        "widths_in_batch": [float(width) for width in WIDTHS],
        "completed": [],
    }

    def save() -> None:
        progress["elapsed_seconds"] = time.perf_counter() - started
        PROGRESS.write_text(json.dumps(progress, indent=2) + "\n")

    save()
    try:
        for width in WIDTHS:
            artifact = ROOT / f"gpt2-huffman-{width}.zip"
            report_path = ROOT / f"gpt2-huffman-{width}-validation.json"
            progress.update({"status": "validating", "current_width": float(width),
                             "current_artifact": str(artifact),
                             "current_started_utc": datetime.now(timezone.utc).isoformat()})
            save()
            command = [sys.executable, "-u", str(VALIDATOR), str(artifact),
                       "--model-source", str(MODEL_SOURCE), "--local-files-only"]
            subprocess.run(command, check=True)
            result = json.loads(report_path.read_text())
            if result.get("round_trip") != "passed":
                raise ValueError(f"Round-trip did not pass for width {width}: {result}")
            progress["completed"].append({
                "width": float(width), "report": str(report_path),
                "weights_validated": result["weights_validated"],
                "occupied_symbols": result["occupied_symbols"],
                "payload_bits": result["payload_bits"],
                "max_reconstruction_difference_vs_expected": result["max_reconstruction_difference_vs_expected"],
                "round_trip": result["round_trip"],
            })
            save()
        progress.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})
        save()
        print(json.dumps(progress, indent=2))
    except BaseException as exc:
        progress.update({"status": "failed", "error_type": type(exc).__name__,
                         "error": str(exc), "failed_utc": datetime.now(timezone.utc).isoformat()})
        save()
        raise


if __name__ == "__main__":
    main()
