"""Run optional model dependencies in a local Python environment."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("QUANTIZATION_MODEL_DIR", ROOT.parent / "models" / "tiny-gpt2")).resolve()
RUNS_DIR = Path(os.environ.get("QUANTIZATION_RUNS_DIR", ROOT / "runs")).resolve()
WORKER_LOCK = threading.Lock()


def inference_python() -> str:
    configured = os.environ.get("QUANTIZATION_INFERENCE_PYTHON")
    if configured:
        return configured
    local_python = Path.home() / ".local" / "share" / "quantization-lab" / "venv" / "bin" / "python"
    if local_python.is_file():
        return str(local_python)
    if all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers")):
        return sys.executable
    project_python = ROOT.parent / ".venv" / "bin" / "python"
    return str(project_python) if project_python.is_file() else sys.executable


def runtime_status() -> dict:
    try:
        result = subprocess.run(
            [inference_python(), "-c", "import importlib.util; assert all(importlib.util.find_spec(n) is not None for n in ('torch', 'transformers', 'numpy'))"],
            capture_output=True, timeout=15, check=False,
        )
        if result.returncode:
            return {"ready": False, "detail": "Install requirements-inference.txt in the inference Python environment."}
        if not (MODEL_DIR / "config.json").is_file():
            return {"ready": False, "detail": "The local model configuration is missing."}
        return {"ready": True, "detail": "Local CPU inference · reconstructed FP32 weights"}
    except (OSError, subprocess.TimeoutExpired):
        return {"ready": False, "detail": "The inference Python environment could not be started. Check QUANTIZATION_INFERENCE_PYTHON."}


def run_path(experiment_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    path = RUNS_DIR / experiment_id
    if not (path / "experiment.json").is_file():
        raise HTTPException(status_code=404, detail="Experiment not found")
    return path


def read_experiment(experiment_id: str) -> dict:
    data = json.loads((run_path(experiment_id) / "experiment.json").read_text())
    data["download_url"] = f"/api/experiments/{experiment_id}/download"
    report_path = run_path(experiment_id) / "evaluation.json"
    if report_path.is_file():
        data["evaluation"] = json.loads(report_path.read_text())
    return data


def run_worker(action: str, payload: dict) -> dict:
    if not WORKER_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another model operation is running. Please wait for it to finish.")
    try:
        payload.update(model_dir=str(MODEL_DIR), runs_dir=str(RUNS_DIR))
        result = subprocess.run(
            [inference_python(), "-m", "app.inference", action],
            cwd=ROOT, input=json.dumps(payload), text=True, capture_output=True,
            timeout=180, check=False,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
        )
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise HTTPException(status_code=503, detail="The inference worker could not start. Check the local model and requirements-inference.txt.") from None
        if result.returncode:
            raise HTTPException(status_code=422, detail=response.get("error", "Model operation failed"))
        return response
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="The model operation exceeded 180 seconds. Try a shorter input.") from None
    except OSError:
        raise HTTPException(status_code=503, detail="The inference Python could not be started. Check QUANTIZATION_INFERENCE_PYTHON.") from None
    finally:
        WORKER_LOCK.release()
