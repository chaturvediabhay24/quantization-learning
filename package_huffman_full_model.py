"""Add the source GPT-2 config and tokenizer to a Huffman weight archive."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path


REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SOURCE_DIR = Path("/private/tmp/quantization-hf-cache/hub/models--openai-community--gpt2/snapshots") / REVISION
BASE = Path("research-results/gpt2-huffman-sweep/gpt2-huffman-0.012.zip")
OUTPUT = Path("research-results/gpt2-huffman-sweep/gpt2-huffman-0.012-full.zip")
EXTRA_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]


def main() -> None:
    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_STORED) as target:
        for info in source.infolist():
            target.writestr(info.filename, source.read(info.filename))
        included = []
        for filename in EXTRA_FILES:
            path = SOURCE_DIR / filename
            if path.exists():
                target.write(path, filename)
                included.append(filename)
    with zipfile.ZipFile(OUTPUT) as z:
        manifest = json.loads(z.read("manifest.json"))
    size = OUTPUT.stat().st_size
    report = {
        "model": "openai-community/gpt2", "revision": REVISION,
        "bin_width": 0.012, "perplexity": 29.738434110795414,
        "artifact": str(OUTPUT), "artifact_bytes": size,
        "weights": 124439808, "effective_bits_per_weight": size * 8 / 124439808,
        "included_model_files": included,
        "archive_manifest_model_id": manifest["model"],
        "note": "Full-model package size includes source config and tokenizer files; PPL is from in-memory reconstructed weights, with a separate decoder round-trip check required.",
    }
    report_path = Path("research-results/calibrated-baselines/huffman-0.012-full-package.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
