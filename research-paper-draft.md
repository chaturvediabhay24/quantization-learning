# Uniform Scalar Binning and Huffman Coding for Language Model Weights: GPT-2 and Qwen3-1.7B Evaluations

**Authors:** [Names and affiliations to be added]  
**Status:** Research draft; not ready for submission

## Abstract

We investigate a weight-only compression pipeline that assigns language-model parameters to equal-width scalar bins, reconstructs each parameter at its bin midpoint, and Huffman-codes the bin indices. We evaluate six bin widths on GPT-2 small and Qwen3-1.7B-Base using WikiText-2 raw test. For Qwen3-1.7B, width 0.0003 yields perplexity 10.42295 versus 10.42210 for its FP32 reference at 8.8539 measured bits per weight; width 0.004 yields perplexity 11.0991 at 5.2202 bits per weight. The GPT-2 0.002 setting yields perplexity 29.0367 and 8.1289 bits per weight versus 29.0180 for its FP32 reference. We also compare a 0.012 setting with a calibrated GPTQ W4 group-128 checkpoint using the same MLX runtime and test tokens. The full Huffman package occupies 5.7238 bits per weight and reaches perplexity 29.7425, compared with 5.5494 bits per weight and perplexity 35.5875 for GPTQ. The Huffman archive is 3.14% larger by the recorded byte counts. This near-rate result is promising for this checkpoint and dataset, but it does not establish a general advantage over calibrated quantizers. The evidence remains exploratory across two checkpoints and one dataset, and does not establish general superiority or an inference-speed benefit.

## 1. Introduction

Language-model parameters consume storage and memory, creating interest in representations that reduce the number of bits used per weight while retaining model quality. Scalar quantization replaces a continuous weight with a discrete representative; entropy coding can then assign shorter bit strings to more frequent indices. This paper examines an especially simple version of that pipeline: a common bin width for all selected weights, midpoint reconstruction, and a global Huffman code constructed from the resulting index frequencies.

Quantization followed by Huffman coding is established prior work. In particular, Deep Compression combines trained quantization and Huffman coding in a broader network-compression pipeline [1]. We do not claim that the general combination is new. Our present contribution is a preliminary, reproducible evaluation of this specific uniform-bin configuration on GPT-2 and Qwen3-1.7B-Base, with measured artifact sizes, weight distortion, language-model perplexity, and initial timing observations. Whether the configuration offers a distinct contribution relative to prior methods remains an open research question.

We ask: **At what actual storage rate can global uniform bins and Huffman coding preserve language-model quality, and how does that trade-off compare with basic precision, integer and calibrated quantization baselines?** We report the result as an exploratory study because the available evidence covers two checkpoints, one held-out dataset, and one GPTQ setting.

## 2. Related Work

Deep Compression combines pruning, trained weight sharing and Huffman coding [1]. Our experiment uses fixed-width scalar bins without retraining or pruning, but the underlying idea of entropy coding quantized weights is not novel.

GPTQ uses approximate second-order information for post-training quantization [2]. AWQ uses activation information to scale channels before low-bit weight quantization [3]. We include a calibrated GPTQ W4 group-128 comparison through the GPTQModel MLX route; AWQ remains unevaluated. Prior work also emphasizes evaluating quantized language models across model families, tasks and quality measures [4]. We use perplexity as an initial model-level measure, while recognizing that it does not describe all downstream behavior.

## 3. Method

### 3.1 Uniform bins and reconstruction

For each finite source parameter value \(w\), we convert to FP32 for binning. For a positive bin width \(\Delta\), we compute the integer bin index

\[
k(w)=\left\lfloor\frac{w}{\Delta}\right\rfloor,
\qquad
\hat w=\left(k(w)+\tfrac{1}{2}\right)\Delta.
\]

Flooring applies to negative values as well as positive values. Each bin is the interval \([k\Delta,(k+1)\Delta)\), and reconstruction uses its midpoint. The six-width sweep evaluates \(\Delta\in\{0.0003,0.001,0.002,0.004,0.008,0.012\}\); for GPT-2, width 0.012 was selected to approach the storage rate of the GPTQ W4 group-128 artifact. Both completed model sweeps use FP32 source weights. In each experiment the same width is used across all unique model parameters, including embeddings, projection weights, normalization parameters and biases. Tied parameters are counted once.

### 3.2 Huffman coding and artifact size

The encoder builds one histogram over all bin indices and derives binary Huffman code lengths. It writes a canonical Huffman-coded bitstream and a JSON manifest containing the width, symbol frequencies and code lengths, tensor names, shapes and counts, together with framing information. The stream and manifest are stored in an uncompressed ZIP container. We calculate effective bits per weight as eight times the measured ZIP size divided by the number of unique original parameters. For GPT-2, all six saved artifacts (widths 0.0003, 0.001, 0.002, 0.004, 0.008 and 0.012) passed decoder round-trip validation. Each recovered all 124,439,808 bin IDs from the pinned source checkpoint; the symbol histogram and framing matched the manifest; and midpoint reconstruction had zero difference from the independently computed expected values. A sample forward pass produced finite logits for every artifact.

The sweep ZIPs are experimental weight artifacts, not complete deployable model packages: they reference the model and revision but do not embed the GPT-2 configuration or tokenizer. The 0.012 point also has a full-model package with configuration and tokenizer. Its perplexity was recomputed after decoding the archive and installing the reconstructed weights in MLX. Across all six widths, decoded-stream and direct in-memory midpoint reconstruction also produced bitwise-identical logits for a fixed four-token sample.

## 4. Experimental Setup

### 4.1 Model and data

We use `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` (124,439,808 unique FP32 parameters) and `Qwen/Qwen3-1.7B-Base` at revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (1,720,574,976 unique FP32 parameters). We evaluate the `wikitext-2-raw-v1` test split of WikiText-2 [5]. Non-empty text lines are joined with blank lines and tokenized separately with each model tokenizer. GPT-2 scores 286,176 next-token predictions in independent non-overlapping chunks of up to 1,024 input tokens; Qwen3 uses the same 1,024-token chunk length. Context is reset at each chunk. We do not compare raw perplexity across models because their tokenizers and architectures differ. Scores from other stride or context protocols are not directly comparable. The binning method uses no calibration data.

### 4.2 Baselines and measurement

FP16 and BF16 baselines cast each unique parameter to the corresponding storage dtype. The INT8 baseline uses symmetric per-output-channel round-to-nearest quantization with FP16 scales. The INT4 baseline uses symmetric round-to-nearest quantization in groups of 128 values per output channel, also with FP16 scales. The integer baselines quantize all unique parameters, including tied embeddings, normalization parameters and biases; practical quantized deployments may retain some tensors at higher precision. They are simple weight-only baselines, not GPTQ or AWQ.

For the standard-format table, baseline weights are cast or dequantized back to FP32, and model computation uses FP32 arithmetic. The Huffman-bin variants likewise execute reconstructed FP32 weights. Weight MAE and MSE are calculated over all unique parameters relative to their original FP32 values. The standard-format artifacts are measured ZIP files containing weight data, scales where applicable, tensor metadata and model configuration. Artifact layouts differ: in particular, the sweep Huffman ZIPs omit the model configuration and tokenizer. The separate 0.012 full package includes both.

We reran the full five-width sweep under one 1,024-token chunk protocol. Every width now shares a single FP32 reference perplexity of 29.018019. We refreshed the standard-format quantized scores against that same reference. An earlier standard-format run had reported 29.017094; the small difference from the fresh reference is not explained, so results from the refreshed comparison are used consistently below.

## 5. Results

### 5.1 Rate, perplexity and weight distortion

| Method | Effective bits/weight | WikiText-2 perplexity | Weight MAE | Weight MSE |
|---|---:|---:|---:|---:|
| FP32 reference | 32.0000* | 29.0180 | 0 | 0 |
| FP16 cast | 16.0010 | 29.0224 | 1.7954e-5 | 7.8064e-10 |
| BF16 cast | 16.0010 | 28.9790 | 1.4635e-4 | 5.2096e-8 |
| INT8 per-channel RTN | 8.0183 | 30.6812 | 1.0549e-3 | 2.0706e-6 |
| INT4 group-128 RTN | 4.1261 | 8,612.7219 | 1.3819e-2 | 2.9874e-4 |
| Huffman bins, \(\Delta=0.0003\) | 10.8717 | 29.0448 | 7.4997e-5 | 7.4994e-9 |
| Huffman bins, \(\Delta=0.001\) | 9.1323 | 29.0477 | 2.4990e-4 | 8.3284e-8 |
| Huffman bins, \(\Delta=0.002\) | 8.1289 | 29.0367 | 4.9987e-4 | 3.3315e-7 |
| Huffman bins, \(\Delta=0.004\) | 7.1270 | 29.3890 | 9.9978e-4 | 1.3327e-6 |
| Huffman bins, \(\Delta=0.008\) | 6.1264 | 29.8187 | 2.0008e-3 | 5.3360e-6 |

*The FP32 figure is the raw weight payload rate; other rows use measured experimental ZIP bytes, including their respective metadata. Lower perplexity and weight errors are better.*

At width 0.002, the Huffman artifact occupies 126,444,954 bytes and produces perplexity 29.0367. A fixed-width 12-bit code for the same 2,958 bin symbols produces an artifact of 186,735,952 bytes (12.0049 bits/weight); Huffman reduces that size by 32.29% while preserving exactly the same bin IDs and reconstructed weights. This isolates the coding benefit from the scalar quantizer. The INT8 RTN artifact occupies 124,724,831 bytes and produces perplexity 30.6812. In this specific comparison the Huffman artifact is approximately 1.38% larger while retaining better perplexity and lower weight MAE and MSE. This is not an exact matched-size comparison, and the result does not establish an advantage over stronger INT8 quantizers or calibrated methods.

The widest evaluated Huffman bin, 0.008, produces 6.1264 bits per weight and perplexity 29.8187. It does not reach the 4.1261 bits per weight of the INT4 RTN artifact. To diagnose the INT4 result, we measured per-tensor error and ran full-test-set mixed-precision ablations. The tied token embedding accounts for 34.4% of the all-parameter INT4 reconstruction's total squared error. Keeping only the embedding in FP32 reduces PPL from 8,612.7 to 80.1; keeping all 1D biases and normalization tensors plus the embedding reduces it to 35.0, while keeping only the final layer norm tensors gives 9,110.9. This points to embedding quantization as a major cause, with additional sensitivity to 1D tensors, but does not fully explain the outlier. The mixed-precision variants retain 38.7M of 124.4M weights in FP32 and have an estimated rate around 12.80 bits per weight before full artifact serialization, so they are diagnostic and not fair 4-bit comparisons.

### 5.2 Qwen3-1.7B-Base model expansion

To test whether the pipeline runs beyond GPT-2, we evaluated the pinned Qwen3-1.7B-Base checkpoint on the full WikiText-2 raw test split with 1,024-token chunks. All 1,720,574,976 unique parameters were FP32; no calibration data were used. Its uncompressed FP32 reference perplexity was 10.422098. Results use actual ZIP artifact bytes, including stream and manifest. Weight errors are relative to the FP32 source weights.

| Bin width | Qwen3 PPL | PPL delta vs FP32 | Artifact bytes | Bits/weight | Weight MAE | Weight MSE | Max abs error | Occupied bins |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0003 | 10.422954 | +0.000856 | 1,904,220,504 | 8.8539 | 7.5087e-5 | 7.5189e-9 | 0.0001526 | 2,247 |
| 0.001 | 10.461611 | +0.039513 | 1,552,038,471 | 7.2164 | 2.4991e-4 | 8.3350e-8 | 0.0005035 | 1,817 |
| 0.002 | 10.576752 | +0.154654 | 1,337,480,856 | 6.2188 | 4.9997e-4 | 3.3332e-7 | 0.0010071 | 1,561 |
| 0.004 | 11.099113 | +0.677015 | 1,122,710,932 | 5.2202 | 9.9991e-4 | 1.3331e-6 | 0.0020142 | 1,306 |
| 0.008 | 14.019009 | +3.596911 | 908,771,903 | 4.2254 | 0.0020001 | 5.3336e-6 | 0.0040131 | 1,079 |
| 0.012 | 25.655224 | +15.233126 | 781,548,439 | 3.6339 | 0.0030010 | 1.2006e-5 | 0.0060043 | 960 |

The narrowest setting retains reference perplexity closely at 8.8539 measured bits per weight. Increasing the width reduces artifact size while perplexity degrades, with a marked quality loss at widths 0.008 and 0.012. This shows the pipeline has been exercised on a second, larger checkpoint, but does not establish that its trade-off is better than GPT-2's: model scale, tokenizer, and source-weight distributions differ. The width-0.012 artifact is about 782 MB and is retained locally; its metric record is `research-results/model-expansion/qwen3-1.7b-base/results.json`. The ZIPs exceed GitHub's ordinary file limit and are not in the repository.

### 5.3 Near-matched comparison with calibrated GPTQ

We used GPTQModel's GPT-2 support through its MLX path to build a GPTQ W4 group-128 checkpoint. Calibration used 128 WikiText-2 train windows of length 1,024, selected with seed 2026. For a runtime-matched quality comparison, we decoded the full Huffman 0.012 archive, installed its midpoint weights as FP16 into the MLX GPT-2 model, and evaluated both models with the same MLX implementation and exact token IDs. The test set contains 286,177 tokens and 286,176 predictions, divided into non-overlapping 1,024-token chunks.

| Method | Artifact bytes | Effective bits/weight | WikiText-2 PPL | Weight MAE | Weight MSE | Full-test PPL evaluation |
|---|---:|---:|---:|---:|---:|---:|
| Dense MLX FP32 reference | Source checkpoint | 32.0000 nominal | 29.0180 | 0 | 0 | 48.76 s |
| Huffman bins, \(\Delta=0.012\), FP16 MLX weights | 89,033,239 | 5.7238 | 29.7425 | 0.003002 | 1.2013e-5 | 36.09 s |
| GPTQ W4, group 128, native packed MLX weights | 86,320,207 | 5.5494 | 35.5875 | 0.009801 | 2.1233e-4 | 144.76 s |

The Huffman package is 3.14% larger by the recorded byte counts and has perplexity lower by 5.8451 than GPTQ at this near-matched rate. Both sizes use ZIP_STORED full packages and include model configuration and tokenizer data; the remaining small package contents differ. The rates are near-matched rather than identical. GPTQ used calibrated packed 4-bit layers while Huffman used float16 midpoint weights, so execution kernels differ. Across all unique weights, this GPTQ checkpoint has MAE 0.009801 and MSE 2.1233e-4; the corresponding Huffman errors are 0.003002 and 1.2013e-5. GPTQ's maximum absolute weight error is 1.4731, in the layer 4 attention projection; see the saved per-tensor report. The evaluation durations are single full perplexity passes rather than repeated prompt-generation latency measurements and should not be read as a general inference-speed ranking.

This result is evidence for one checkpoint and one dataset only. It suggests that width 0.012 retained more WikiText-2 quality at a slightly larger package size than this GPTQ setting. It does not establish that Huffman bins are generally better than GPTQ or other calibrated quantizers.

### 5.4 Preliminary inference timing

We also measured one model-side workload on a macOS arm64 CPU with MPS unavailable: a fixed 128-token prompt, greedy generation of 64 tokens, one warm-up and three measured requests. The table reports medians. TTFT includes model prefill and first-token selection, but excludes tokenization and queueing. Decode throughput covers the subsequent 63 autoregressive steps.

| Variant and execution path | Prefill (ms) | TTFT (ms) | Decode (tokens/s) | End-to-end 64-token generation (ms) |
|---|---:|---:|---:|---:|
| FP32 reference, FP32 compute | 41.84 | 41.88 | 103.0 | 652.71 |
| FP16 weights, native FP16 CPU compute | 6,150.15 | 6,150.20 | 18.6 | 9,541.69 |
| BF16 weights, native BF16 CPU compute | 6,199.47 | 6,199.54 | 18.6 | 9,582.56 |
| INT8 RTN, reconstructed FP32 compute | 40.19 | 40.23 | 110.4 | 610.70 |
| Huffman bins, \(\Delta=0.002\), reconstructed FP32 compute | 40.00 | 40.04 | 108.1 | 623.19 |

The execution paths differ. INT8, INT4 and Huffman variants run after weight reconstruction in FP32, so these measurements do not time packed-weight inference, Huffman decoding, or compressed-model memory use. FP16/BF16 use native low-precision CPU operations; their unexpectedly high latency is specific to this environment and has not been profiled to determine the precise kernel-level cause. Model loading is excluded, and only three requests and one workload were measured. These results do not support an inference-speed ranking of the compressed formats.

### 5.5 MLX generation latency comparison

To compare the decoded Huffman model with native GPTQ inference, we used the same Apple Silicon MLX/Metal runtime, fixed 128-token prompt, greedy 64-token generation, one warm-up, and three timed requests. Tokenization and checkpoint preparation are excluded. TTFT is wall time to the first yielded token. Prompt processing is derived from the MLX prompt throughput and includes first-token work. Decode timing starts after the first token and covers the remaining 63 tokens. The table gives medians.

| Model | Prompt processing (ms) | TTFT (ms) | Decode after first token (ms) | Decode tokens/s | End-to-end 64 tokens (ms) | Preparation |
|---|---:|---:|---:|---:|---:|---|
| Dense FP32 | 24.37 | 24.71 | 376.94 | 167.1 | 402.50 | Base load not timed |
| Huffman 0.012, decoded FP16 | 15.90 | 16.06 | 211.92 | 297.3 | 227.27 | Decode/install 28.53 s |
| GPTQ W4, group 128 | 16.10 | 16.24 | 106.63 | 590.8 | 122.80 | Checkpoint load 41.21 s; quantize/save 265.07 s |

GPTQ decoded tokens faster in this specific workload. Huffman retained lower WikiText-2 perplexity and lower weight MAE/MSE at a slightly larger package size. Huffman used a fully decoded FP16 model for inference, so these results do not measure generation directly from the compressed stream or a compressed-runtime memory benefit. They are one prompt workload on one machine and do not support general speed claims.

## 6. Discussion and Limitations

The GPT-2 0.002-bin result suggests that a global scalar grid can preserve WikiText-2 perplexity at around eight measured bits per weight for that checkpoint. The Qwen3-1.7B sweep also has a near-reference point at 8.8539 measured bits per weight, while quality degrades at higher widths. Its addition shows the pipeline has been exercised on a second, larger checkpoint, but it does not establish a general advantage or a directly comparable cross-model rate-quality curve. The measured GPT-2 advantage over our INT8 RTN implementation is a hypothesis for further investigation, not evidence of a general advantage over INT8 or stronger quantizers.

The study has several limitations. The uniform-bin sweep covers two model families and sizes, but only one test split and a non-overlapping perplexity protocol; the integer and calibrated baselines remain GPT-2-only. It has no task-level evaluation, no independent reproduction, no AWQ baseline, and no layerwise activation-sensitivity analysis. The calibrated comparison includes one GPTQ setting at a near-matched but not identical rate. The simple RTN integer baseline quantizes every parameter, while other methods may use mixed-precision policies. All six GPT-2 Huffman archives passed exact bin/weight round-trip checks and sample-logit parity, but decoded-artifact WikiText-2 perplexity was recomputed only for width 0.012; Qwen3 archive round-trip and decoded-artifact evaluations also remain to be checked. Generation latency covers one prompt workload; direct per-token decoding from the Huffman stream and peak-memory measurements remain unmeasured. The INT4 ablations implicate the embedding table and 1D tensors but do not explain all of the RTN failure.

## 7. Planned Experiments

1. Recompute full WikiText-2 perplexity from decoded artifacts at every GPT-2 and Qwen3 sweep width; only GPT-2 width 0.012 currently has a complete decoded-artifact PPL pass.
2. Repeat the method comparison with a fixed sliding-window evaluation protocol and a disjoint calibration/test split. Add AWQ where a supported implementation and GPT-2 checkpoint path are available, and add more GPTQ settings.
3. Sweep widths and quantizer settings to compare at matched measured bits per weight and matched quality. Include all codebooks, scales and container metadata in each artifact size.
4. Isolate the coding contribution by storing the same bin IDs with fixed-width codes, then compare global, per-layer and per-tensor Huffman tables including their overhead.
5. Evaluate additional model sizes and families (the Qwen3-4B run is in progress), datasets and task metrics; report per-layer distortion and repeatability where applicable.
6. Measure actual artifact loading, decoding, peak memory and inference latency with a stated runtime and hardware. A direct compressed-weight kernel would require separate implementation and evaluation.

## 8. Conclusion

Uniform binning followed by Huffman coding produced a measured artifact at 8.1289 bits per weight with perplexity 29.0367 on GPT-2 and, on Qwen3-1.7B, an 8.8539-bit-per-weight point with perplexity 10.42295 versus its 10.42210 FP32 reference. A separate width-0.012 result reached perplexity 29.7425 at 5.7238 bits per weight and was compared with calibrated GPTQ W4 group 128 at 5.5494 bits per weight and perplexity 35.5875. All six Huffman artifacts passed decoder round-trip checks and bitwise sample-logit parity against direct in-memory reconstruction. These results motivate more precise matched-rate comparisons, but do not support a broad claim of superiority. The next experiments should compute full decoded-artifact perplexity at the remaining GPT-2 widths, finish the Qwen3-4B sweep, add AWQ where supported, test more models and tasks, compare more GPTQ settings, and measure runtime and memory on a common workload.

## Reproducibility and Data Availability

The local experiment records include `research-results/model-expansion/qwen3-1.7b-base/results.json`, `research-results/gpt2-huffman-sweep/results.json`, `research-results/gpt2-huffman-sweep/weight-distortion.json`, `research-results/standard-format-comparison/results.json`, `research-results/inference-metrics/results.json`, `research-results/calibrated-baselines/near-matched-gptq-comparison.md`, `research-results/calibrated-baselines/gpt2-gptq-int4-g128-mlx-distortion.json`, and `research-results/calibrated-baselines/gpt2-huffman-vs-gptq-generation-mlx.json`. The corresponding runners include `benchmark_huffman_gpt2.py`, `measure_huffman_distortion.py`, `benchmark_standard_formats.py`, `benchmark_inference_metrics.py`, `benchmark_mlx_gptqmodel.py`, `evaluate_huffman_mlx.py`, `measure_gptq_weight_distortion.py`, and `benchmark_mlx_generation_metrics.py`. The original CPU comparison used Python 3.12.12, PyTorch 2.15.0, Transformers 4.46.3, Datasets 5.0.1 and NumPy 2.5.4. The calibrated MLX experiment used an isolated Python 3.12 environment with GPTQModel 7.7.0, MLX 0.32.4 and mlx-lm 0.32.2. Some scripts reference a temporary local model cache; paths and environment setup must be made portable before public release.

Per-width codec validation and logit-parity outputs are saved in `research-results/gpt2-huffman-sweep/` as `gpt2-huffman-*-validation.json` and `gpt2-huffman-*-logit-parity.json`. The parity runner is `compare_decoded_huffman_logits.py`; the validation runner is `validate_huffman_codec.py`.

## References

[1] S. Han, H. Mao, and W. J. Dally. “Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.” ICLR, 2016. https://arxiv.org/abs/1510.00149

[2] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.” ICLR, 2023. https://arxiv.org/abs/2210.17323

[3] J. Lin et al. “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.” MLSys, 2024. https://arxiv.org/abs/2306.00978

[4] R. Jin et al. “A Comprehensive Evaluation of Quantization Strategies for Large Language Models.” Findings of ACL, 2024. https://aclanthology.org/2024.findings-acl.726/

[5] S. Merity, C. Xiong, J. Bradbury, and R. Socher. “Pointer Sentinel Mixture Models.” ICLR, 2017. https://arxiv.org/abs/1609.07843
