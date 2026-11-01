# Huffman bins vs calibrated GPTQ at near-matched storage

## Protocol

- Model: GPT-2 small, `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Dataset: WikiText-2 raw test, 286,177 input tokens; 286,176 next-token predictions; non-overlapping windows of 1,024 tokens.
- Both MLX evaluations used the same token IDs (SHA-256 `7892b371a41e15356847c03c77abbd2b9d0dee5714aeb92a0013412fa045405e`).
- GPTQ was calibrated on 128 WikiText-2 train windows, each 1,024 tokens, seed 2026; 4-bit, group size 128.
- Huffman used global bins of width 0.012, midpoint reconstruction, a global Huffman stream, and float16 reconstructed weights for MLX evaluation.

## Results

| Method | Perplexity | Change vs dense MLX FP32 | Artifact bytes | Effective bits/weight | MAE vs FP32 | MSE vs FP32 | Calibration / decode time | Test evaluation time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense MLX FP32 reference | 29.0180 | — | Source checkpoint | 32.0000 nominal | 0 | 0 | — | 48.76 s |
| Huffman bins, width 0.012 | 29.7425 | +0.7244 | 89,033,239 | 5.7238 | 0.003002 | 1.2013e-5 | Decode 36.01 s | 36.09 s |
| GPTQ W4, group 128 | 35.5875 | +6.5695 | 86,320,207 | 5.5494 | 0.009801 | 2.1233e-4 | Calibration + quantize + save 265.07 s | 144.76 s |

The Huffman archive is 3.14% larger than the GPTQ artifact by the recorded byte counts, while its perplexity is 5.8451 lower (16.4% lower relative to GPTQ's PPL). These are near-matched storage points, not an exact size match. Both sizes use ZIP_STORED full packages and include model configuration and tokenizer data; their remaining small contents differ (Huffman includes the stream manifest, GPTQ includes its safetensors index and generation config).

The dense reference is MLX evaluation of the full-precision source weights. Huffman evaluation loads the same MLX GPT-2 architecture, installs midpoint weights reconstructed from the archive, and casts those values to float16. GPTQ uses MLX's native packed 4-bit layers. Thus the PPL results use the same runtime, tokens, and evaluation windows, but different weight representations and kernels, as intended for method-level comparison.

Across all unique weights, this GPTQ checkpoint has higher MAE and MSE than the width-0.012 midpoint reconstruction. The 4-bit quantized tensors alone have MAE 0.009811 and MSE 2.1254e-4; the overall result also includes retained lower-precision tensors. The maximum absolute GPTQ weight error is 1.4731, concentrated in `model.h.4.attn.c_attn.weight`; the largest per-layer MSE is in `model.h.3.attn.c_attn.weight`. The full per-tensor breakdown is saved in `gpt2-gptq-int4-g128-mlx-distortion.json`.

This one model/dataset result does not establish that Huffman is generally better than GPTQ. It shows that this Huffman setting preserved more WikiText-2 quality with lower global weight error at a slightly larger measured artifact size for this checkpoint. The PPL evaluation durations are one complete pass; generation latency was measured separately as described below.

## Generation latency in MLX

We also measured one fixed workload on Apple Silicon MLX/Metal: 128 prompt tokens, greedy generation of 64 tokens, one warm-up, and three timed requests per model. The same token IDs were used for each model; tokenization and checkpoint preparation are excluded from request latency. Values below are medians. Prompt processing time is derived from MLX's `prompt_tps` field, which includes the first-token step; TTFT is wall time to the first returned token; decode time is from that first token to completion and covers the remaining 63 tokens.

| Model | Prompt processing (ms) | TTFT (ms) | Decode after first token (ms) | Decode tokens/s | End-to-end 64 tokens (ms) | Preparation time |
|---|---:|---:|---:|---:|---:|---|
| Dense FP32 | 24.37 | 24.71 | 376.94 | 167.1 | 402.50 | Base model load not timed |
| Huffman 0.012, decoded FP16 | 15.90 | 16.06 | 211.92 | 297.3 | 227.27 | Decode/install 28.53 s |
| GPTQ W4, group 128 | 16.10 | 16.24 | 106.63 | 590.8 | 122.80 | Checkpoint load 41.21 s; quantize/save 265.07 s |

GPTQ was faster on this specific generation workload, while Huffman had lower PPL and lower weight MAE/MSE at a slightly larger package size. Huffman inference here uses the fully decoded FP16 model; these measurements do not include per-token decoding from the Huffman stream or prove a compressed-runtime memory benefit. The exact trials and definitions are in `gpt2-huffman-vs-gptq-generation-mlx.json`.

## Reproducibility records

- `gpt2-huffman-0.012-mlx-eval.json`
- `gpt2-gptq-int4-g128-mlx-results.json`
- `gpt2-mlx-dense-reference-results.json`
- `huffman-0.012-full-package.json`
- `gpt2-gptq-int4-g128-mlx-package.json`
- `gpt2-gptq-int4-g128-mlx-distortion.json`
