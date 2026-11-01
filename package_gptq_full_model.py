"""Package the GPTQModel MLX checkpoint with ZIP_STORED for size comparison."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path


SOURCE = Path("research-results/calibrated-baselines/gpt2-gptq-int4-g128-mlx")
OUTPUT = Path("research-results/calibrated-baselines/gpt2-gptq-int4-g128-mlx-full.zip")
RESULT = Path("research-results/calibrated-baselines/gpt2-gptq-int4-g128-mlx-package.json")
WEIGHTS = 124_439_808


def main() -> None:
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(p for p in SOURCE.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(SOURCE).as_posix())
    with zipfile.ZipFile(OUTPUT) as archive:
        entries = [{"path": item.filename, "bytes": item.file_size} for item in archive.infolist()]
    result = {
        "source_directory": str(SOURCE), "artifact": str(OUTPUT),
        "packaging": "ZIP_STORED", "artifact_bytes": OUTPUT.stat().st_size,
        "weights": WEIGHTS, "effective_bits_per_weight": OUTPUT.stat().st_size * 8 / WEIGHTS,
        "files": entries,
        "note": "Package size uses the same uncompressed ZIP container mode as the full Huffman archive.",
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
