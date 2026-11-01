# Huffman bins vs FP16, BF16, INT8, and INT4

## Evaluation protocol

- Model: GPT-2 small, `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Test data: full `Salesforce/wikitext`, `wikitext-2-raw-v1`, `test` split; 286,177 tokens.
- Perplexity: non-overlapping 1,024-token chunks. Refreshed FP32 reference PPL: 29.018019, shared by the Huffman sweep and standard-format rows.
- All distortion metrics are measured over the same 124,439,808 unique parameters against original FP32 weights; tied token embeddings are counted once.
- FP16/BF16 and integer checkpoints are evaluated after casting/dequantizing weights to FP32, so this isolates weight-value quantization and does not benchmark lower-precision compute kernels.
- Effective bits/weight use actual ZIP artifact bytes, including stored scales and per-tensor metadata.

## Results

| Method | PPL | Effective bits/weight | MAE vs FP32 | MSE vs FP32 | Prefill ms (128) | Prefill tok/s | TTFT ms | Decode tok/s | E2E ms (64) | Weight prep s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 reference | 29.0180 | 32.0000 | 0 | 0 | 41.84 | 3,059 | 41.88 | 103.0 | 652.71 | 0.00 |
| FP16 cast | 29.0224 | 16.0010 | 1.7954e-05 | 7.8064e-10 | 6,150.15 | 20.8 | 6,150.20 | 18.6 | 9,541.69 | 0.44 |
| BF16 cast | 28.9790 | 16.0010 | 1.4635e-04 | 5.2096e-08 | 6,199.47 | 20.6 | 6,199.54 | 18.6 | 9,582.56 | 0.31 |
| INT8 per-output-channel RTN | 30.6812 | 8.0183 | 1.0549e-03 | 2.0706e-06 | 40.19 | 3,185 | 40.23 | 110.4 | 610.70 | 0.75 |
| INT4 output-channel RTN, group 128 | 8,612.7219 | 4.1261 | 1.3819e-02 | 2.9874e-04 | 40.17 | 3,187 | 40.21 | 109.0 | 618.05 | 4.80 |
| Huffman bins, width 0.0003 | 29.0448 | 10.8717 | 7.4997e-05 | 7.4994e-09 | 39.79 | 3,217 | 39.83 | 110.2 | 611.63 | 0.14 |
| Huffman bins, width 0.001 | 29.0477 | 9.1323 | 2.4990e-04 | 8.3284e-08 | 40.38 | 3,170 | 40.43 | 107.3 | 627.66 | 0.13 |
| Huffman bins, width 0.002 | 29.0367 | 8.1289 | 4.9987e-04 | 3.3315e-07 | 40.00 | 3,200 | 40.04 | 108.1 | 623.19 | 0.13 |
| Huffman bins, width 0.004 | 29.3890 | 7.1270 | 9.9978e-04 | 1.3327e-06 | 40.34 | 3,173 | 40.38 | 109.3 | 616.97 | 0.13 |
| Huffman bins, width 0.008 | 29.8187 | 6.1264 | 2.0008e-03 | 5.3360e-06 | 40.06 | 3,195 | 40.10 | 110.7 | 608.80 | 0.13 |

The INT8 baseline uses symmetric per-output-channel round-to-nearest and FP16
scales. GPT-2's `Conv1D` matrices use their output-channel axis in the second
dimension. INT4 uses symmetric round-to-nearest groups of 128 along each output
channel, with FP16 scales. The INT4 result is a deliberately simple RTN format
baseline and suffers severe quality loss; it is not GPTQ or AWQ.

The Huffman method has lower MAE/MSE and better PPL than the tested INT8 RTN
baseline at similar effective storage (for example, 0.002 bins: 8.1289 b/weight
and PPL 29.0367 vs INT8: 8.0183 b/weight and PPL 30.6812). At INT4's much
smaller 4.1261 b/weight, the original five-point scalar-bin sweep has no
directly matched point: the 0.008 setting is 6.1264 b/weight. A separate
calibrated GPTQ W4 group-128 comparison near 5.55 b/weight is now included
below; AWQ remains unevaluated. FP16/BF16
are reference storage dtypes, while INT8/INT4 here are weight-only RTN
baselines; no inference speed claim is made.

### INT4 outlier diagnosis

The tested all-parameter INT4 group-128 RTN result is an implementation-specific
outlier. The quantizer produced no zero FP16 scales, so scale underflow was not
the cause. Layerwise weight diagnostics show the tied token embedding contributes
34.4% of total squared reconstruction error. Full-test-set ablations yielded:

| INT4 RTN variant | WikiText-2 PPL | FP32-retained parameters | Estimated bits/weight |
|---|---:|---:|---:|
| Quantize every parameter | 8,612.72 | 0 | 4.126 (serialized baseline) |
| Keep only tied embedding in FP32 | 80.14 | 38,597,376 | 12.771 (payload estimate) |
| Keep all 1D parameters and embedding in FP32 | 34.98 | 38,718,720 | about 12.80 (payload estimate) |
| Keep only final layer norm tensors in FP32 | 9,110.92 | 1,536 | not serialized |

These interventions show that the embedding table is a major factor in this
RTN failure, and that biases/normalization tensors also affect quality. The
final-layer-norm-only change did not improve PPL. The mixed-precision variants
are diagnostic; their rates are not matched to the 4.126-bpw baseline and their
artifacts were not serialized.

The full Huffman sweep was rerun as one batch; every row now uses FP32 baseline
PPL 29.018019. FP16, BF16, INT8, and INT4 were also re-evaluated against this
same reference. The earlier standard-format run used 29.017094; its roughly
0.0009 PPL difference from the refreshed reference remains unexplained and is
not mixed into the current table.

### Calibrated GPTQ near-rate comparison

We added a GPTQ W4 group-128 baseline using GPTQModel's MLX route. GPTQ used
128 calibration windows from WikiText-2 train (length 1,024, seed 2026). To
compare model quality, the Huffman 0.012 checkpoint was decoded and its
midpoint weights installed into the MLX GPT-2 model as FP16; GPTQ used its
native packed 4-bit MLX layers. Both were evaluated on the same 286,177 test
tokens and 286,176 next-token predictions (token-ID hash
`7892b371a41e15356847c03c77abbd2b9d0dee5714aeb92a0013412fa045405e`).

| Method | PPL | Effective bits/weight | MAE vs FP32 | MSE vs FP32 | Artifact bytes | Calibration / decode time | Full test PPL pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense MLX FP32 reference | 29.0180 | 32.0000 nominal | 0 | 0 | Source checkpoint | — | 48.76 s |
| Huffman bins, width 0.012, FP16 MLX weights | 29.7425 | 5.7238 | 0.003002 | 1.2013e-5 | 89,033,239 | Decode 36.01 s | 36.09 s |
| GPTQ W4, group 128, native MLX weights | 35.5875 | 5.5494 | 0.009801 | 2.1233e-4 | 86,320,207 | Quantize + save 265.07 s | 144.76 s |

The Huffman artifact is 3.14% larger by recorded bytes and has PPL lower by
5.8451 (16.4% relative to GPTQ's PPL). This is a near-rate result for one
checkpoint and dataset, not a general claim of superiority. Both sizes use
ZIP_STORED full packages and include model configuration and tokenizer files;
their remaining small package contents differ. The PPL evaluation durations
are single full passes. GPTQ's overall max absolute weight error is 1.4731, in
`model.h.4.attn.c_attn.weight`; the largest per-layer MSE is in
`model.h.3.attn.c_attn.weight`. See the per-tensor report in
`calibrated-baselines/gpt2-gptq-int4-g128-mlx-distortion.json`.

### MLX generation latency

The following medians use Apple Silicon MLX/Metal, one warm-up, and three
timed requests per model. Each request uses the same fixed 128-token prompt and
greedily generates 64 tokens. Tokenization and checkpoint preparation are
excluded. Prompt processing is derived from MLX's prompt tokens/s and includes
first-token work; TTFT measures wall time to the first yielded token. Decode
time and throughput cover the 63 tokens after the first. End-to-end measures
the complete model-side request.

| Model | Prompt processing (ms) | TTFT (ms) | Decode after first token (ms) | Decode tokens/s | End-to-end 64 tokens (ms) | Preparation |
|---|---:|---:|---:|---:|---:|---|
| Dense FP32 | 24.37 | 24.71 | 376.94 | 167.1 | 402.50 | Base load not timed |
| Huffman 0.012, decoded FP16 | 15.90 | 16.06 | 211.92 | 297.3 | 227.27 | Decode/install 28.53 s |
| GPTQ W4, group 128 | 16.10 | 16.24 | 106.63 | 590.8 | 122.80 | Checkpoint load 41.21 s; quantize/save 265.07 s |

GPTQ was faster on this workload; Huffman had lower PPL and weight error at a
slightly larger package size. Huffman generation used the fully decoded FP16
model. The benchmark does not measure direct inference from the Huffman stream
or a compressed-runtime memory benefit. Full trials are saved in
`calibrated-baselines/gpt2-huffman-vs-gptq-generation-mlx.json`.

The Huffman in-memory FP32 CPU result at width 0.012 was 29.7384. The decoded
MLX FP16 result is 29.7425; the dense FP32 MLX reference is 29.0180. Use the
MLX trio above for the calibrated comparison, and the main table for the
FP32-reconstructed CPU comparisons.

## Benchmark runtime

The five-width Huffman rerun produced artifacts from 22:00:24 to 22:07:42
local time (7m18s between the first and last completed width). The standard
format refresh ran from shortly after 22:07 to 22:24:44 (about 17 minutes
including dataset/model setup); its first-to-last artifact interval was
10m25s. These are one-off wall-clock observations on the local macOS arm64
CPU, not model inference latencies.

## Inference timing setup and limits

Latency is the median of three measured requests after one warm-up on macOS
arm64 CPU (10 logical CPUs; PyTorch MPS unavailable). Each request uses a
fixed 128-token prompt and greedily generates 64 tokens. TTFT is model-side
time until the first output token, including prefill and argmax; tokenization
and queueing are excluded. Prefill throughput covers the 128 prompt tokens;
decode throughput covers the remaining 63 autoregressive steps. `Weight prep`
is the time to cast or reconstruct weights in memory after checkpoint load; it
is not a serialized-artifact load/decode benchmark. Source-checkpoint load time
is omitted from the comparison because the first model load was cold and later
loads benefited from the operating-system file cache.

FP16 and BF16 were executed at native dtype on CPU and are much slower on this
runtime. INT8, INT4, and Huffman variants were reconstructed to FP32 and use the
same FP32 inference path as the reference. Their similar latency does **not**
demonstrate that packed weights are fast at runtime. A decoder and compatible
quantized inference kernels are needed before reporting compressed-runtime
speed or memory savings.

## Artifacts

- Huffman sweep metrics: `gpt2-huffman-sweep/results.json`
- Huffman weight errors: `gpt2-huffman-sweep/weight-distortion.json`
- Huffman decoder validation: `gpt2-huffman-sweep/gpt2-huffman-*-validation.json` (all six widths passed)
- Huffman sample-logit parity: `gpt2-huffman-sweep/gpt2-huffman-*-logit-parity.json` (all six widths bitwise equal)
- Standard-format metrics: `standard-format-comparison/results.json`
- Standard-format ZIP checkpoints: `standard-format-comparison/gpt2-*.zip`
- Inference timing measurements: `inference-metrics/results.json`
- Reproducible runners: `benchmark_huffman_gpt2.py`, `measure_huffman_distortion.py`, and `benchmark_standard_formats.py`
- Inference timing runner: `benchmark_inference_metrics.py`
- Calibrated baseline results: `calibrated-baselines/near-matched-gptq-comparison.md`
