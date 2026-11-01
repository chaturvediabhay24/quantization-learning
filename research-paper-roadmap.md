# Research paper roadmap

Use this file as the paper checklist. Tick a box only when the result, data, and method needed to support it are recorded and reproducible.

## Current checkpoint

- [x] Build a QuantLab prototype that bins weights into equal-width intervals, reconstructs each value at its bin midpoint, and Huffman-codes the resulting bin symbols.
- [x] Report midpoint reconstruction error and Huffman payload and table-size estimates for the bundled Tiny GPT-2 weights.
- [x] Compare the BPE + Huffman experiment with ordinary Huffman and record that the tested 100-merge setting did not improve total estimated size.
- [ ] Treat “bins + Huffman is the best method” as a research conclusion. This remains a working hypothesis until the comparisons below are complete.

Current evidence is exploratory: the bundled checkpoint has 102,714 unique weights and is a Tiny GPT-2 model. The dashboard's size figures are estimates, not measurements from a serialized codec.

## Metrics to report against other quantization methods

Tick these when measured for every method in the comparison, including the unquantized model. Use the same model, evaluation data, tokenizer, hardware, and inference settings.

### Model quality

- [x] **Perplexity or negative log-likelihood:** report the value and the change from the original model on held-out text for the current GPT-2 formats and calibrated GPTQ comparison.
- [ ] **Task performance:** report accuracy, F1, exact match, or another task-appropriate score on several relevant benchmarks. Include instruction following or generation quality for instruction-tuned models.
- [x] **Quality at similar size:** compare actual artifact rates, including a near-rate Huffman 0.012 vs GPTQ W4 group-128 point (5.7238 vs 5.5494 bits/weight). Exact equal-byte settings remain future work.

### Compression and fidelity

- [ ] **Actual model size:** report encoded bytes on disk, including scales, zero points, codebooks, Huffman tables, indexes, and container metadata.
- [ ] **Effective bits per original weight and compression ratio:** calculate both from the actual artifact size and state which original precision is the reference.
- [ ] **Weight distortion:** report MAE, RMSE, and maximum error, including a per-layer breakdown to reveal outliers. Treat these as diagnostic metrics rather than substitutes for model quality.
- [ ] **Size-quality curve:** report several operating points for each method and show the Pareto frontier instead of one hand-picked setting.

### Deployment cost

- [ ] **Runtime memory:** measure peak and steady RAM/VRAM, including any unpacked weights, activations, and KV cache under the stated batch size and context length.
- [ ] **Inference speed:** report prefill throughput, decode tokens per second, time to first token, and end-to-end latency under identical workloads. Partial results are saved for the CPU RTN/standard-format set and the MLX dense/Huffman/GPTQ set; one runtime covering all formats and direct Huffman-stream inference remain.
- [ ] **Conversion and loading cost:** measure quantization/encoding time, decoding or model-loading time, and peak working memory during conversion.
- [ ] **Energy per token:** report it if the paper claims an energy or edge-device benefit, with power measurement conditions.

### Reliability and fair comparison

- [ ] **Generalization:** repeat the comparison across multiple model sizes or families, datasets, and relevant quantization settings.
- [ ] **Variability:** give run counts and uncertainty for noisy speed or task scores; keep deterministic metrics such as artifact bytes exact. The new MLX latency run has three timed requests after one warm-up, but broader runs and uncertainty estimates remain.
- [ ] **Implementation conditions:** state hardware, software versions, kernels, batch sizes, context lengths, calibration data, and whether inference uses compressed weights directly or reconstructs them first.

## Research checkpoints

### 1. Define the method and experiment

- [ ] Write down the binning rule, midpoint reconstruction rule, handling of negative values, Huffman symbol construction, and all metadata needed to decode.
- [ ] Decide the evaluation scopes: whole model, individual tensor, and any grouping used to build Huffman tables.
- [ ] Set the comparison target, such as matching model quality or matching compressed size, before comparing methods.
- [ ] Record the exact checkpoints, tokenizer, evaluation data, software versions, and parameter settings.

### 2. Evaluate larger trained models

- [ ] Run the method on full GPT-2 checkpoints rather than the Tiny GPT-2 prototype.
- [ ] Include at least one model from another model family.
- [ ] Confirm every checkpoint is trained, identify its exact revision, and record parameter count and source.

### 3. Measure model quality after quantization

- [ ] Measure perplexity on a named, held-out language-model evaluation dataset before and after quantization.
- [ ] Measure weight reconstruction MAE, RMSE, and maximum absolute error for each setting.
- [ ] Repeat measurements across several bin widths and report the quality-versus-size curve.
- [ ] Verify that decoding reconstructs the intended midpoint values before evaluating model quality.

### 4. Compare with established baselines

- [x] Compare FP16, BF16, INT8, and INT4 RTN formats on the common GPT-2/WikiText-2 protocol; add a near-rate calibrated GPTQ W4 group-128 point. Broader and exact matched-rate comparisons remain.
- [x] Compare ordinary fixed-width bin IDs with Huffman-coded bin IDs to isolate Huffman's contribution at width 0.002: identical bin IDs; 12.0049 vs 8.1289 effective bits/weight; 32.29% smaller Huffman archive.
- [x] Include calibrated GPTQ W4 group-128 with calibration data and measured package size, alongside the RTN formats.
- [ ] Report both equal-quality and equal-size comparisons where possible.

### 5. Build and measure a real codec

- [ ] Implement an encoder and decoder for the quantized bin stream.
- [x] Round-trip all six saved width artifacts through decode: all 124,439,808 bin IDs matched the pinned source, each reconstructed midpoint tensor matched exactly, and sample logits were finite. Across all six widths, decoded-stream logits matched direct in-memory midpoint logits bitwise for the same four-token sample. Full WikiText-2 PPL was recomputed from decoded width 0.012.
- [ ] Measure the actual file size, including Huffman tree or codebook, bin-width data, tensor shapes, tensor boundaries, and container metadata.
- [ ] Compare measured file size with the dashboard estimate and explain any difference.

### 6. Measure practical costs

- [ ] Measure encoding time, decoding time, peak memory, and inference speed on stated hardware.
- [ ] Report whether inference can use decoded weights directly or requires full reconstruction first.
- [ ] Document any accuracy, speed, or memory trade-offs that affect deployment.

### 7. Run ablations and choose settings

- [ ] Sweep bin width and report compressed size alongside perplexity and reconstruction error.
- [ ] Compare one Huffman table for the whole model with tables per tensor or layer.
- [ ] Measure the effect of table overhead and any optional rare-bin pooling.
- [ ] Keep BPE + Huffman as a comparison only if a measured setting beats ordinary Huffman after merge-rule metadata is included.
- [ ] Select final settings using a stated rule and report the full sweep, not only the best point.

### 8. Establish the research contribution

- [ ] Review prior work on scalar quantization, entropy coding of quantized weights, and model compression.
- [ ] State what is new compared with those methods and identify the strongest competing result.
- [ ] Support every claimed improvement with repeatable measurements and suitable baselines.
- [ ] If results do not show a clear advantage, frame the paper as a careful evaluation or negative result instead of claiming a new best method.

### 9. Prepare the paper and release

- [ ] Draft the paper with sections for motivation, method, experimental setup, results, limitations, and conclusion.
- [ ] Include plots for quality versus actual size and tables for model-level comparisons.
- [ ] Release code, exact configurations, evaluation scripts, and result files needed to reproduce the paper.
- [ ] Choose a suitable venue after the contribution and evidence are clear.
- [ ] Have someone independently reproduce the main result before submission.

## Next checkpoint

- [x] Run a bin-width sweep on a full trained GPT-2 checkpoint and record perplexity plus actual encoded file size. This confirms the Tiny GPT-2 result on GPT-2 small for this dataset and protocol; it does not establish generalization across model families.

## Fresh-paper assessment and priority queue (2026-11-01)

### Model expansion execution log (2026-11-01)

- Prepared `benchmark_huffman_gpt2.py` for model-expansion runs: model-specific artifact names, `--resume`, and a durable `results.json` checkpoint after each completed bin width. Resume is restricted to the same model identifier and evaluation protocol; use a separate output directory for each checkpoint.
- Current hardware snapshot after space was cleared: macOS arm64, 17.2 GB physical RAM, 85+ GiB free on the data volume. The old Python 3.12 environment still fails importing PyTorch because its `libtorch_cpu.dylib` mapping is interrupted. A clean environment at `/private/tmp/quantization-exp-env` now works with PyTorch 2.15.0 (MPS available), Transformers 5.22.1, Datasets 5.1.0, and NumPy 2.5.4.
- Downloaded Qwen3-1.7B-Base. A 2,048-token smoke run completed at width 0.012: FP32 PPL 12.4603, quantized PPL 33.1208, 1,720,574,976 parameters, and 781,548,403 artifact bytes. This is a pipeline check only, not a full-data result; see `research-results/model-expansion/qwen3-1.7b-smoke/results.json`.
- Full WikiText-2 test evaluation of Qwen3-1.7B-Base has started in `research-results/model-expansion/qwen3-1.7b-base/`, using 1,024-token chunks and widths 0.0003, 0.001, 0.002, 0.004, 0.008, and 0.012. The runner writes/merges `results.json` after each width and can resume with `--resume`. The dataset loader now uses `Salesforce/wikitext`, required by the current Datasets/Hugging Face Hub versions.
- Cached model revision: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Full WikiText-2 results at width 0.0003: FP32 reference PPL 10.422098; quantized PPL 10.422954 (delta +0.000856); MAE 7.5087e-5, MSE 7.5189e-9, max error 1.5259e-4; archive 1,904,220,504 bytes (8.8539 bits/weight), 2,247 occupied bins. Result: `research-results/model-expansion/qwen3-1.7b-base/results.json`; artifact: `research-results/model-expansion/qwen3-1.7b-base/qwen3-1.7b-base-huffman-0.0003.zip`. The old run was stopped between widths after writing this checkpoint. Resume the remaining sweep using the pinned revision: `/private/tmp/quantization-exp-env/bin/python -u benchmark_huffman_gpt2.py --model Qwen/Qwen3-1.7B-Base --model-id Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 --widths 0.0003 0.001 0.002 0.004 0.008 0.012 --output research-results/model-expansion/qwen3-1.7b-base --seq-len 1024 --device mps --resume`. The runner now reuses the saved reference PPL on resume and accumulates MPS loss tensors so it only synchronizes after the full evaluation.
- Full WikiText-2 result at width 0.001: quantized PPL 10.461611 (delta +0.039513 vs reference); MAE 2.4991e-4, MSE 8.3350e-8, max error 5.0354e-4; archive 1,552,038,471 bytes (7.2164 bits/weight), 1,817 occupied bins. Result and archive were saved at the same paths under `research-results/model-expansion/qwen3-1.7b-base/`.
- Full WikiText-2 result at width 0.002: quantized PPL 10.576752 (delta +0.154654 vs reference); MAE 4.9997e-4, MSE 3.3332e-7, max error 0.00100708; archive 1,337,480,856 bytes (6.2188 bits/weight), 1,561 occupied bins. Result and archive are saved under `research-results/model-expansion/qwen3-1.7b-base/`.
- Full WikiText-2 result at width 0.004: quantized PPL 11.099113 (delta +0.677015 vs reference); MAE 9.9991e-4, MSE 1.3331e-6, max error 0.00201416; archive 1,122,710,932 bytes (5.2202 bits/weight), 1,306 occupied bins. Result and archive are saved under `research-results/model-expansion/qwen3-1.7b-base/`.
- Full WikiText-2 result at width 0.008: quantized PPL 14.019009 (delta +3.596911 vs reference); MAE 0.00200012, MSE 5.3336e-6, max error 0.00401306; archive 908,771,903 bytes (4.2254 bits/weight), 1,079 occupied bins. Result and archive are saved under `research-results/model-expansion/qwen3-1.7b-base/`.
- Full WikiText-2 result at width 0.012: quantized PPL 25.655224 (delta +15.233126 vs reference); MAE 0.00300102, MSE 1.2006e-5, max error 0.00600433; archive 781,548,439 bytes (3.6339 bits/weight), 960 occupied bins. Result and archive are saved under `research-results/model-expansion/qwen3-1.7b-base/`. All six Qwen3-1.7B widths are now complete.
- The ZIP artifacts are retained locally but are not committed: the repository ignores `*.zip`, and each is over GitHub's ordinary per-file size limit. The small JSON records are committed with the benchmark and roadmap so the measurements remain available from the remote repository.
- An initial run exited with code 137 while advancing from width 0.001 to 0.002. Resuming skipped completed widths, overwrote incomplete temporary ID maps, and completed the remaining widths. Running one width per process after this interruption prevented losing completed results. All six rows are in `results.json`.
- Checked the next roadmap checkpoint, `google/gemma-2-2b`, but the Hugging Face repository is gated and this machine has no authenticated Hugging Face token; the run cannot start until access is granted. Continued to the next accessible model rather than waiting.
- Extended the runner with an explicit `--dtype` option and source-relative distortion metric names so FP16 model runs are measured against the actual loaded checkpoint precision and are not mislabeled FP32. Resume now validates the reference dtype. This is needed for the 17.2 GB machine to attempt Qwen3-4B-Base.
- Confirmed public, unauthenticated access to Qwen3-4B-Base and Qwen3-8B-Base. Pinned Qwen3-4B-Base revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`; the 8B candidate is not a practical FP16 full-model fit in 17.2 GB once evaluation and temporary storage are accounted for. The 4B checkpoint has downloaded, and the FP16 full WikiText-2 baseline (512-token chunks) is actively evaluating; latest observed progress was 352/584 chunks. No 4B metrics have been checkpointed yet. Results will state the source dtype explicitly and should not be compared to the earlier FP32 Qwen3-1.7B distortion numbers as if reference precision matched. Runner PID/session was active when this note was recorded.

### Candidate paper titles and model scope

Suggested title for the expanded empirical study:

> **Revisiting Entropy-Coded Uniform Quantization for Language Model Weight Compression**

Other defensible options, depending on the eventual contribution:

- **How Far Can Calibration-Free Weight Compression Go? A Rate–Quality Study of Language Models** — use if experiments explain where the approach succeeds and fails across models.
- **Uniform Bins, Variable-Length Codes: Measuring the Storage–Quality Trade-off in Language Models** — emphasizes the tested method and measured trade-off.
- **Beyond Fixed-Width Weights: Evaluating Entropy Coding for Language Model Compression** — use if competing quantizers also receive entropy coding.
- **Uniform Quantization and Huffman Coding for GPT-2: A Preliminary Rate–Quality Evaluation** — accurately scopes the current single-model evidence.

Avoid title claims such as “lossless,” “optimal,” “state-of-the-art,” or faster inference: the present pipeline is lossy and reconstructs weights before inference.

Recommended core expansion beyond GPT-2: use pretrained/base checkpoints and pin exact revisions.

| Family | Candidate checkpoint(s) | Purpose |
|---|---|---|
| Qwen3 | 1.7B-Base, 4B-Base, 8B-Base | Three scales within one modern dense family. |
| Llama | Llama-3.1-8B | Cross-family comparison near Qwen3-8B scale. |
| Gemma 2 | 2B, 9B pretrained checkpoints | A third family at smaller and larger scales. |
| Existing GPT-2 | GPT-2 small | Retain as the established starting point and pipeline validation case. |

This gives seven checkpoints across four families, including the existing GPT-2. Compare each quantized checkpoint against its own uncompressed reference; do not rank families by raw perplexity when tokenizers differ. Record tokenizer, revision, original dtype, total and unique parameter counts, tied embeddings, and the share of parameters retained at higher precision. Qwen, Meta Llama, and Gemma checkpoints have different access and license terms; record those and confirm they permit the planned research and release.

Optional scientific-control extension: add Pythia 410M, 1.4B, and 6.9B. Pythia is a family of decoder language models released as a research suite. Its sizes were trained on the same data in the same order, and it includes many intermediate training checkpoints. In simple terms, it is a set of language models built for experiments where you want to study how size or training progress changes a result. This makes it useful for asking whether weight entropy and compression-quality trade-offs change during training, not just across unrelated model families. See the [Pythia project](https://github.com/EleutherAI/pythia) and [Pythia paper](https://arxiv.org/abs/2304.01373).

Optional larger systems extension: Qwen3-30B-A3B-Base is a mixture-of-experts checkpoint with about 30.5B stored parameters and 3.3B activated per token. For compression size, count all stored weights, not only activated weights. This extension can test whether expert-local coding is useful, but should follow the dense-model study.

Suggested model execution order: Qwen3-1.7B-Base and Gemma-2-2B to validate the generalized pipeline; then Qwen3-8B-Base and Llama-3.1-8B for the most important cross-family size comparison; then Qwen3-4B-Base and Gemma-2-9B to complete the core matrix. Use Pythia for controlled scaling/training analysis if resources allow.

### Current assessment

The draft is a careful preliminary evaluation with promising storage-quality points, but the current evidence does not yet establish a distinct or generalizable research contribution. Quantization followed by entropy coding is established prior work. The strongest direction to investigate is whether calibration-free uniform binning remains competitive when competing quantizers receive comparable entropy coding and all methods are measured with complete artifact accounting.

Keep claims scoped to the tested checkpoint, data, and protocol until broader experiments support more. Treat the current points as evidence motivating the next study, rather than a general claim of superiority.

### Completed evidence to preserve

- [x] Full GPT-2 small width sweep, common FP32 reference, measured archive sizes, and WikiText-2 perplexity are recorded above.
- [x] Fixed-width versus Huffman coding of identical width-0.002 bin IDs isolates the coding effect: 12.0049 versus 8.1289 bits/weight, with a 32.29% smaller Huffman archive.
- [x] Six artifacts passed exact bin-ID and midpoint reconstruction checks; direct and decoded paths produced bitwise-identical logits on the recorded four-token sample.
- [x] One near-rate MLX comparison exists for Huffman width 0.012 and GPTQ W4 group-128, including full-package bytes, PPL and a small generation timing run.
- [ ] Do not treat sample-logit parity as a substitute for full decoded-artifact evaluation: full WikiText-2 PPL has been recomputed only for width 0.012.

### Priority 1: validate and strengthen baselines

- [ ] Audit the INT4 RTN and GPTQ paths: projection orientation, group axis and size, packing/dequantization, tied embeddings, biases, normalization tensors, and retained precision. Record the checks and reproduce the reported outlier before using it as a central comparison.
- [ ] Evaluate the dequantized GPTQ checkpoint using the same dense computation path as reconstructed Huffman weights; report native packed-kernel inference separately.
- [ ] Add a dense FP16 MLX reference because the decoded Huffman MLX model uses FP16 weights while the current dense MLX reference is FP32.
- [ ] Sweep supported GPTQ settings and repeat calibration with multiple seeds. Include a controlled common tensor policy and clearly label each method's practical default policy.
- [ ] Compare activation/output distortion or loss sensitivity alongside weight MAE/MSE; weight error alone does not explain model quality.

### Priority 2: compare entropy coding fairly

- [ ] Entropy-code the competing RTN and GPTQ quantized representations as well as the uniform-bin indices. Compare fixed-width and entropy-coded forms at matched quality and measured rate.
- [ ] Add a nonuniform scalar quantizer baseline with fixed-width and entropy-coded indices.
- [ ] Compare Huffman with a stronger entropy coder and a practical general-purpose compressor where appropriate. State whether each result measures model archives, loading, or runtime streams.
- [ ] Count every artifact component consistently: weights/indices, scales, codebooks, Huffman tables, tensor indexes, configuration, tokenizer, container overhead, and any high-precision exceptions.
- [ ] Report empirical symbol entropy, average code length, and metadata overhead separately.

### Priority 3: build comparable rate-quality curves

- [ ] Produce multiple operating points per method across the overlapping rate range. Plot actual bits/weight against excess NLL and PPL; report storage at predefined quality tolerances.
- [ ] Choose widths and method settings on validation data, then evaluate selected settings on test data. Keep calibration, validation, and test roles distinct and document any data overlap.
- [ ] Evaluate all selected artifacts using the same tokenizer, token IDs, context/chunking protocol, runtime, and dtype wherever the comparison is about quality.
- [ ] Resolve why the reported width 0.012 result has slightly lower PPL than width 0.008 by evaluating both under one runtime and dtype.

### Priority 4: explain the behavior and test inexpensive ablations

- [ ] Measure how empirical entropy and average code length change with bin width, and test whether observed rate and MSE trends match high-resolution scalar-quantization approximations. Treat these approximations as hypotheses, not assumed findings.
- [ ] Compare a single global width with per-layer and per-tensor widths under the same total-size budget.
- [ ] Compare global Huffman tables with per-layer/per-tensor tables, including table overhead.
- [ ] Compare the current floor-and-midpoint rule, which reconstructs exact zero as +Delta/2, with a zero-preserving nearest-grid rule such as Delta*round(w/Delta).
- [ ] Analyze layerwise activation/loss sensitivity and the embedding and 1D-tensor INT4 ablations; explain what they establish and what remains unexplained.

### Priority 5: test generality and uncertainty

- [ ] Extend evaluation to multiple model sizes across at least two architecture families, and multiple datasets with different text distributions.
- [ ] Add task-level metrics suited to the chosen models, not only WikiText-2 perplexity.
- [ ] Report uncertainty for quality comparisons by resampling documents or contiguous blocks; do not treat tokens as independent samples. Repeat calibration seeds for calibrated baselines.
- [ ] Have an independent run reproduce the central result after scripts and environments are made portable.

### Priority 6: choose and support the deployment claim

- [ ] Decide whether the paper studies smaller model archives or compressed-weight inference. The current Huffman model is decoded before inference, so current generation timings do not demonstrate direct compressed-stream inference or reduced runtime memory.
- [ ] For an archive/storage study, measure encoding time, cold/warm load time, decode/install time, peak memory during preparation, resident memory, and time to first usable prediction. State which files are included.
- [ ] For an inference/systems claim, implement and separately evaluate direct compressed-weight execution; measure peak memory, prefill, decode tokens/s, TTFT, and end-to-end latency on matched hardware and workloads.
- [ ] Move the unexplained CPU FP16/BF16 slowdown timing out of the main argument unless it is diagnosed and becomes relevant. Keep it as an explicitly limited appendix result if useful.

### Literature, claims, and manuscript cleanup

- [ ] Expand related work beyond Deep Compression to cover entropy-coded neural-network compression (including DeepCABAC), work on compressibility of quantized LLMs, lossless compressed-weight inference (including Huff-LLM), and recent rate-distortion/entropy-coded quantization (including HyperQuant). Verify publication/version status and compare scopes before making novelty claims.
- [ ] State the contribution as a testable finding about conditions, mechanisms, and failure cases. “Uniform quantization plus Huffman coding applied to language models” alone is not a sufficient novelty claim.
- [ ] Remove repeated caveats by consolidating scope and limitations. Move CPU timing and diagnostic INT4 failures out of headline results unless stronger validation makes them central.
- [ ] Resolve or document the historical FP32 reference discrepancy. Replace long local-path lists with a public artifact entry point and portable reproduction instructions before release.
- [ ] Update the existing planned-experiment list when work is completed; fixed-width comparison of width-0.002 IDs is already recorded above.

### Suggested execution order

1. Audit RTN/GPTQ implementations and add dense FP16 MLX control.
2. Entropy-code competing quantizers and produce matched-protocol rate-quality curves.
3. Run zero-preserving and global-versus-local width/coding ablations.
4. Expand to other models, datasets, and task metrics.
5. Measure storage/loading or direct inference costs according to the paper's chosen scope.
6. Rewrite the paper around the strongest finding supported by the results.

## Results log

| Date | Model and revision | Bin width / scope | Quality metric | Actual encoded size | Baseline | Notes |
|---|---|---|---|---|---|---|
| 2026-10-31 | `openai-community/gpt2` `607a30d783dfa663caf39e06633721c8d4cfcd7e` | Global widths 0.0003, 0.001, 0.002, 0.004, 0.008; 124,439,808 unique weights | WikiText-2 raw test PPL: 29.044750, 29.047678, 29.036731, 29.389049, 29.818709 | ZIP bytes: 169,108,948; 142,052,417; 126,444,954; 110,860,215; 95,295,432 | FP32 PPL 29.018019; non-overlapping 1,024-token chunks | Refreshed all widths in one run. All six sweep artifacts, including 0.012, passed exact bin/midpoint decoding validation. See per-width validation reports in `research-results/gpt2-huffman-sweep/`. |
| 2026-10-31 | `openai-community/gpt2` `607a30d783dfa663caf39e06633721c8d4cfcd7e` | Extra width 0.012; GPTQ W4 group 128; 124,439,808 weights | Same MLX WikiText-2 raw test IDs: Huffman PPL 29.742458; GPTQ PPL 35.587507 | Full Huffman ZIP 89,033,239 B (5.7238 b/w); GPTQ full ZIP 86,320,207 B (5.5494 b/w) | Dense MLX PPL 29.018029; 286,176 predictions | GPTQ calibration: 128 WikiText-2 train windows x 1,024, seed 2026; Huffman decode 36.01 s and PPL pass 36.09 s; GPTQ quantize/save 265.07 s and PPL pass 144.76 s. See `research-results/calibrated-baselines/near-matched-gptq-comparison.md`. |
| 2026-10-31 | `openai-community/gpt2` `607a30d783dfa663caf39e06633721c8d4cfcd7e` | Decoder/logit parity across all six bin widths | Exact bin IDs and bitwise-identical four-token logits for direct vs decoded midpoint paths; max absolute logit difference 0 for all widths | Existing six ZIP artifacts | Pinned FP32 source checkpoint | Validation batch reported 275.78 s for the remaining widths and 400.02 s for logit parity across all six. See per-width `*-validation.json` and `*-logit-parity.json` files in `research-results/gpt2-huffman-sweep/`. |
| 2026-10-31 | `openai-community/gpt2` `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 128-token prompt, greedy 64-token generation; 1 warm-up + 3 timed requests | MLX medians: Huffman TTFT 16.06 ms, decode 297.3 tok/s, E2E 227.27 ms; GPTQ TTFT 16.24 ms, decode 590.8 tok/s, E2E 122.80 ms | Huffman decode/install 28.53 s; GPTQ model load 41.21 s | Same Apple Silicon MLX/Metal workload | GPTQ had higher decode speed; Huffman had lower PPL/weight error. Measurements use a decoded FP16 Huffman model, not direct stream decoding. See `research-results/calibrated-baselines/gpt2-huffman-vs-gptq-generation-mlx.json`. |

## Evaluation references

- [GPTQ](https://arxiv.org/abs/2210.17323) reports language-model quality and compression for weight quantization.
- [A Comprehensive Evaluation of Quantization Strategies for Large Language Models](https://aclanthology.org/2024.findings-acl.726/) examines quality beyond perplexity and practical inference effects.
- [EfficientLLM](https://arxiv.org/abs/2505.13840) frames efficiency using memory, latency, throughput, energy, and compression metrics.
