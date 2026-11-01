"""Local model worker. Its stdout is a single JSON response for the web app."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "pytorch_model.bin", "model.safetensors"):
        path = model_dir / name
        if path.is_file():
            digest.update(name.encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    model.float().cpu().eval()
    model.tie_weights()
    return model


def apply(payload: dict) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    model_dir = Path(payload["model_dir"])
    model = load_model(model_dir)
    parameters = dict(model.named_parameters())  # Tied parameters occur once.
    with np.load(payload["data_file"], allow_pickle=False) as source:
        if set(source.files) != set(parameters):
            raise ValueError("Dashboard tensors do not match the local model. Use the bundled Tiny GPT-2 dataset for inference.")
        for name, parameter in parameters.items():
            expected = parameter.detach().numpy()
            if source[name].shape != expected.shape or not np.array_equal(source[name], expected):
                raise ValueError(f"Dashboard weights differ from the local model at {name}. Re-extract the matching checkpoint first.")

    settings = payload["settings"]
    names = list(parameters) if settings["tensor"] == "__all__" else [settings["tensor"]]
    absolute_sum = squared_sum = maximum = 0.0
    weight_count = 0
    changed_weights = 0
    with torch.inference_mode():
        for name in names:
            parameter = parameters[name]
            original = parameter.detach().numpy().astype(np.float64)
            restored = ((np.floor(original / settings["bin_width"]) + 0.5) * settings["bin_width"]).astype(np.float32)
            errors = original - restored.astype(np.float64)
            absolute_sum += float(np.abs(errors).sum())
            squared_sum += float(np.square(errors).sum())
            maximum = max(maximum, float(np.abs(errors).max()))
            weight_count += original.size
            changed_weights += int(np.count_nonzero(errors))
            parameter.copy_(torch.from_numpy(restored))
    # Shared embeddings and the LM head must remain tied after the update.
    model.tie_weights()
    experiment_id = payload["id"]
    destination = Path(payload["runs_dir"]) / experiment_id
    temporary = destination.with_name(f".{experiment_id}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        saved_model = temporary / "model"
        model.save_pretrained(saved_model, safe_serialization=True)
        AutoTokenizer.from_pretrained(model_dir, local_files_only=True).save_pretrained(saved_model)
        data = {
            "id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "settings": {**settings, "merge_rare_bins": 0, "max_code_length": None},
            "model": "Tiny GPT-2",
            "source_fingerprint": fingerprint(model_dir),
            "tensor_count": len(names),
            "selected_weight_count": weight_count,
            "changed_weight_count": changed_weights,
            "total_weight_count": sum(p.numel() for p in model.parameters()),
            "weight_metrics": {
                "mae": absolute_sum / weight_count,
                "rmse": math.sqrt(squared_sum / weight_count),
                "max_abs_error": maximum,
            },
            "storage_estimate": payload["storage_estimate"],
            "runtime": {"device": "cpu", "dtype": "float32", "torch_threads": 1},
            "checkpoint_weight_bytes": sum(p.stat().st_size for p in saved_model.glob("*.safetensors")),
            "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "note": "The saved model contains reconstructed FP32 weights. Coding size is an estimate for the selected scope; no entropy-coded checkpoint is written.",
        }
        (temporary / "experiment.json").write_text(json.dumps(data, indent=2, allow_nan=False))
        shutil.make_archive(str(temporary / "checkpoint"), "zip", root_dir=temporary, base_dir="model")
        temporary.rename(destination)
        return data
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def evaluate(payload: dict) -> dict:
    import torch
    from transformers import AutoTokenizer

    model_dir = Path(payload["model_dir"])
    run_dir = Path(payload["runs_dir"]) / payload["id"]
    metadata = json.loads((run_dir / "experiment.json").read_text())
    if fingerprint(model_dir) != metadata["source_fingerprint"]:
        raise ValueError("The original model changed after this experiment was created. Apply quantization again to compare matching checkpoints.")
    original = load_model(model_dir)
    quantized = load_model(run_dir / "model")
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "model", local_files_only=True)
    prompt = tokenizer(payload["prompt"], return_tensors="pt")
    evaluation = tokenizer(payload["evaluation_text"], return_tensors="pt")
    prompt_tokens = prompt.input_ids.shape[1]
    evaluation_tokens = evaluation.input_ids.shape[1]
    context_limit = min(original.config.n_positions, quantized.config.n_positions)
    if prompt_tokens < 1 or prompt_tokens + payload["max_new_tokens"] > context_limit:
        raise ValueError(f"Prompt plus generated tokens must fit within {context_limit} tokens.")
    if not 2 <= evaluation_tokens <= min(context_limit, 256):
        raise ValueError("Evaluation text must contain between 2 and 256 tokens. Shorten or extend the text; it is never silently truncated.")

    models = {"original": original, "quantized": quantized}
    result = {}
    logits = {}
    with torch.inference_mode():
        for label, model in models.items():
            output = model(**evaluation, labels=evaluation.input_ids, use_cache=False)
            loss = float(output.loss)
            if not math.isfinite(loss):
                raise ValueError(f"{label.capitalize()} model produced a non-finite evaluation loss. Try a smaller bin width.")
            logits[label] = output.logits[:, :-1, :].float()
            result[label] = {
                "loss": loss,
                "perplexity": math.exp(loss) if loss < 700 else None,
                "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            }

        generation_options = {
            "max_new_tokens": payload["max_new_tokens"], "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id, "use_cache": True,
        }
        # Clear sampling defaults inherited from this demonstration checkpoint.
        for model in models.values():
            model.generation_config = copy.deepcopy(model.generation_config)
            model.generation_config.temperature = 1.0
            model.generation_config.top_k = 50
            model.generation_config.top_p = 1.0
            model.generate(**prompt, **{**generation_options, "max_new_tokens": 1})
        timings = {label: [] for label in models}
        generated = {}
        for iteration in range(3):
            order = ("original", "quantized") if iteration % 2 == 0 else ("quantized", "original")
            for label in order:
                started = time.perf_counter()
                ids = models[label].generate(**prompt, **generation_options)
                timings[label].append(time.perf_counter() - started)
                generated[label] = ids[0, prompt_tokens:]
        for label in models:
            seconds = sorted(timings[label])[1]
            count = generated[label].numel()
            result[label].update(
                completion=tokenizer.decode(generated[label], skip_special_tokens=True),
                generated_token_ids=generated[label].tolist(),
                generated_tokens=count,
                latency_ms=seconds * 1000,
                tokens_per_second=count / seconds,
                timing_samples_ms=[sample * 1000 for sample in timings[label]],
            )
        difference = logits["original"] - logits["quantized"]
        comparison = {
            "logit_rmse": float(difference.square().mean().sqrt()),
            "next_token_agreement_percent": float((logits["original"].argmax(-1) == logits["quantized"].argmax(-1)).float().mean()) * 100,
            "loss_delta": result["quantized"]["loss"] - result["original"]["loss"],
            "identical_generation": result["original"]["generated_token_ids"] == result["quantized"]["generated_token_ids"],
        }
    report = {
        "experiment_id": payload["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": payload["prompt"], "evaluation_text": payload["evaluation_text"],
        "prompt_tokens": prompt_tokens, "evaluation_tokens": evaluation_tokens,
        "scored_tokens": evaluation_tokens - 1,
        "max_new_tokens": payload["max_new_tokens"],
        "original": result["original"], "quantized": result["quantized"],
        "comparison": comparison,
        "method": "CPU FP32, greedy decoding, one warmup token, median of 3 generation runs, alternating model order. Generation timing includes prefill and decoding; loading and tokenization are excluded. Perplexity scores the supplied text, without an extra BOS token.",
    }
    temporary_report = run_dir / ".evaluation.json.tmp"
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary_report.replace(run_dir / "evaluation.json")
    return report


if __name__ == "__main__":
    try:
        import torch

        torch.set_num_threads(1)
        request = json.load(sys.stdin)
        action = {"apply": apply, "evaluate": evaluate}[sys.argv[1]]
        response = action(request)
        print(json.dumps(response, allow_nan=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
