"""Replace local-cache model paths in completed Huffman artifacts and refresh sizes."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path


ROOT = Path("research-results/gpt2-huffman-sweep")
MODEL_ID = "openai-community/gpt2"


def main() -> None:
    for artifact in ROOT.glob("gpt2-huffman-*.zip"):
        with zipfile.ZipFile(artifact) as source:
            members = [(info.filename, source.read(info.filename)) for info in source.infolist()]
        updated = []
        for name, data in members:
            if name == "manifest.json":
                manifest = json.loads(data)
                manifest["model"] = MODEL_ID
                data = json.dumps(manifest, separators=(",", ":")).encode()
            updated.append((name, data))
        temporary = artifact.with_suffix(".tmp.zip")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as target:
            for name, data in updated:
                target.writestr(name, data)
        os.replace(temporary, artifact)

    results_path = ROOT / "results.json"
    results = json.loads(results_path.read_text())
    results["model"] = MODEL_ID
    for row in results["results"]:
        artifact = Path(row["artifact"])
        row["actual_encoded_bytes"] = artifact.stat().st_size
        row["effective_bits_per_weight"] = artifact.stat().st_size * 8 / row["weights"]
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    fixed_path = ROOT / "gpt2-huffman-0.002-fixed-width.zip"
    with zipfile.ZipFile(fixed_path) as source:
        members = [(info.filename, source.read(info.filename)) for info in source.infolist()]
    updated = []
    for name, data in members:
        if name == "manifest.json":
            manifest = json.loads(data)
            manifest["model"] = MODEL_ID
            data = json.dumps(manifest, separators=(",", ":")).encode()
        updated.append((name, data))
    temporary = fixed_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as target:
        for name, data in updated:
            target.writestr(name, data)
    os.replace(temporary, fixed_path)

    comparison_path = ROOT / "fixed-width-coding-comparison.json"
    comparison = json.loads(comparison_path.read_text())
    huffman = ROOT / "gpt2-huffman-0.002.zip"
    comparison["model"] = MODEL_ID
    comparison["huffman"]["archive_bytes"] = huffman.stat().st_size
    comparison["huffman"]["effective_bits_per_weight"] = huffman.stat().st_size * 8 / comparison["weights"]
    comparison["fixed_width"]["archive_bytes"] = fixed_path.stat().st_size
    comparison["fixed_width"]["effective_bits_per_weight"] = fixed_path.stat().st_size * 8 / comparison["weights"]
    comparison["archive_size_reduction_vs_fixed_width_percent"] = 100 * (
        1 - huffman.stat().st_size / fixed_path.stat().st_size
    )
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"model": MODEL_ID, "archives_normalized": 6,
                      "huffman_bytes": huffman.stat().st_size,
                      "fixed_width_bytes": fixed_path.stat().st_size,
                      "results_updated": str(results_path)}, indent=2))


if __name__ == "__main__":
    main()
