"""Minimal local inference for models/tiny-gpt2."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(prompt: str, max_new_tokens: int = 30, temperature: float = 0.8) -> str:
    model_dir = Path(__file__).parent / "models" / "tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tiny-GPT-2 locally.")
    parser.add_argument("prompt", nargs="?", default="The smallest transformer model is")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    print(generate(args.prompt, args.max_new_tokens, args.temperature))
