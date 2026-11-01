# Learning Roadmap

Check an item when you can explain it in your own words or have completed the practical exercise. Keep the notes and evidence in `learning-log.md` or a linked experiment document.

## 1. Model and inference foundations

- [x] Inspect the local tiny GPT-2 model structure and checkpoint.
- [x] Trace the basic inference flow: tokenize, run forward passes, choose tokens, decode.
- [ ] Explain tensors, matrix multiplication, logits, softmax, and autoregressive generation.
- [ ] Explain attention, KV caching, normalization, and positional embeddings.
- [ ] Add KV caching to a small generation example and compare output and runtime.
- [ ] Explain the difference between model parameters, activations, and the KV cache.

## 2. Numerical precision and quantization basics

- [ ] Review binary floating point, rounding, clipping, absolute and relative error.
- [ ] Derive uniform affine quantization: scale, zero point, rounding, and clamping.
- [ ] Compare symmetric and asymmetric quantization.
- [ ] Compare per-tensor, per-channel, per-group, and per-token scales.
- [ ] Explain FP32, FP16, BF16, INT8, INT4, and FP8, including useful ranges and tradeoffs.
- [ ] Implement a small uniform quantizer and dequantizer; measure MAE, RMSE, and maximum error.
- [ ] Separate storage bits from effective bits per weight, including scales and metadata.

## 3. Post-training quantization and quality

- [ ] Explain post-training quantization (PTQ), calibration, and quantization-aware training (QAT).
- [ ] Compare static and dynamic activation quantization.
- [ ] Build a representative calibration set and keep evaluation examples separate.
- [x] Quantize weights at multiple bit widths and group sizes on a small model.
- [ ] Compare layer-wise weight error, output/logit error, and task-level behavior.
- [ ] Identify outlier channels and layers that are sensitive to quantization.
- [x] Give every proposed method a fair baseline at similar effective bits per weight.
- [ ] Revisit the existing codec experiments with model-level quality metrics, not only tensor reconstruction error.

## 4. Advanced LLM quantization methods

- [ ] Read [GPTQ](https://arxiv.org/abs/2210.17323); explain its second-order error compensation.
- [ ] Read [AWQ](https://arxiv.org/abs/2306.00978); explain how activation statistics guide weight quantization.
- [ ] Read [SmoothQuant](https://arxiv.org/abs/2211.10438); explain how scaling shifts quantization difficulty.
- [ ] Explain weight-only versus weight-and-activation quantization (W4A16, W8A8, FP8, and related schemes).
- [ ] Explain mixed precision and why some layers may need higher precision.
- [ ] Explain KV-cache quantization and evaluate its effect at longer context lengths.
- [ ] Distinguish a quantization algorithm, numeric format, checkpoint format, and runtime kernel.
- [ ] Optional: read [QLoRA](https://arxiv.org/abs/2305.14314) and explain quantized fine-tuning versus inference quantization.

## 5. Hardware and inference performance

- [ ] Learn GPU memory hierarchy, memory bandwidth, compute throughput, and tensor cores.
- [ ] Estimate model weight memory from parameter count and precision; account for scales and runtime overhead.
- [ ] Estimate KV-cache memory from layer count, batch size, context length, heads, and cache precision.
- [ ] Explain prefill versus decode and identify compute-bound versus bandwidth-bound work.
- [ ] Measure time to first token, inter-token latency, throughput, and peak memory.
- [ ] Benchmark multiple prompt lengths, output lengths, and concurrent requests.
- [ ] Learn profiling basics and identify time spent in kernels, copies, and scheduling.
- [ ] Read [FlashAttention](https://arxiv.org/abs/2205.14135) and explain why reducing data movement matters.
- [ ] Read [PagedAttention](https://arxiv.org/abs/2309.06180) and explain KV-cache paging and serving utilization.

## 6. Runtime and deployment

- [ ] Choose one target runtime based on available hardware: vLLM, TensorRT, llama.cpp, or ONNX Runtime.
- [ ] Check the runtime's current support for the model architecture, quantization format, device, and kernels.
- [ ] Export or prepare a quantized model and verify numerical outputs after loading it in the target runtime.
- [ ] Serve the model through an API and test streaming, cancellation, timeouts, and invalid inputs.
- [ ] Load-test with realistic request lengths and concurrency; track latency percentiles and throughput.
- [ ] Package the service reproducibly and record model, tokenizer, runtime, and quantization versions.
- [ ] Add basic logs and metrics for errors, latency, tokens, queueing, and memory use.
- [ ] Practice safe model version changes and rollback.

## 7. Advanced systems and kernel work

- [ ] Learn batching, continuous batching, prefix caching, and chunked prefill.
- [ ] Explain tensor, pipeline, and data parallel inference and their communication costs.
- [ ] Learn kernel fusion and how quantized matrix multiplication accumulates and rescales values.
- [ ] Complete the [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/) for matrix multiplication and fused operations.
- [ ] Implement or modify a small quantized matrix multiplication kernel and compare against a reference.
- [ ] Explore speculative decoding, sparse or mixture-of-experts execution, or long-context serving.
- [ ] Reproduce one published quantization result and document the setup, result, and limitations.

## 8. Capstone

- [ ] Pick a model, target device, and workload with a measurable quality and latency goal.
- [ ] Establish a full-precision or higher-precision baseline.
- [ ] Compare at least two quantization approaches at recorded effective storage rates.
- [ ] Deploy the best candidate in the chosen runtime and benchmark realistic traffic.
- [ ] Write a short report explaining quality, memory, latency, throughput, cost, and remaining limits.

