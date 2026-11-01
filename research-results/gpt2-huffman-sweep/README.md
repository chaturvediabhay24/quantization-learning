# Full GPT-2 Huffman bin-width sweep

Completed 2026-10-31. Results are in `results.json`; each setting has a ZIP
artifact containing its Huffman-coded parameter-bin stream and decoding
metadata.

## Setup and protocol

- Model: `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` (GPT-2 small).
- Scope: all 124,439,808 unique floating-point parameters; tied parameters are counted once.
- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, `test`; 286,177 tokens after joining non-empty lines.
- Evaluation: perplexity over non-overlapping 1,024-token chunks, on CPU. The refreshed five-width sweep uses a shared baseline PPL of 29.018019.
- Quantization: `bin = floor(weight / width)`; reconstruct at `(bin + 0.5) * width`; one global Huffman code over all parameter bins.
- Artifact size: actual ZIP bytes, including the stream and JSON metadata. FP32 weight-only reference: 497,759,232 bytes.
- Software: Python 3.12.12, PyTorch 2.15.0, Transformers 4.46.3, Datasets 5.0.1, NumPy 2.5.4.

## Results

| Bin width | WikiText-2 PPL | Δ vs original | Actual ZIP bytes | Effective bits/weight | Occupied bins |
|---:|---:|---:|---:|---:|---:|
| 0.0003 | 29.044750 | +0.026731 | 169,108,948 | 10.8717 | 12,484 |
| 0.001 | 29.047678 | +0.029659 | 142,052,417 | 9.1323 | 4,993 |
| 0.002 | 29.036731 | +0.018712 | 126,444,954 | 8.1289 | 2,958 |
| 0.004 | 29.389049 | +0.371030 | 110,860,215 | 7.1270 | 1,756 |
| 0.008 | 29.818709 | +0.800689 | 95,295,432 | 6.1264 | 1,053 |

Wider bins reduce actual storage from 142.1 MB to 95.3 MB as perplexity rises.
At width 0.002, the artifact is about 3.94× smaller than the FP32 weight
payload with a 0.0187 PPL increase. These results support generalization of the
Tiny GPT-2 observation to GPT-2 small for this setting and dataset; they do not
establish superiority over other quantizers. All six width artifacts from
0.0003 through 0.012 have passed decoder round-trip validation. Additional
model families and task metrics remain future work.

The refreshed sweep regenerated each width and now gives every row the same
FP32 reference (29.018019). The 0.0003 point is 29.044750 at 10.8717
bits/weight. The prior standard-format run reported a reference of 29.017094;
the small discrepancy is not explained, so the refreshed comparison uses the
single baseline above throughout.

The primary 0.002 artifact passed decoder validation: all 124,439,808 bin
IDs and the manifest histogram matched, the padding/framing was valid, and
reconstructed FP32 weights matched the independently computed midpoint values
exactly. The 0.012 artifact also passed round-trip validation and was evaluated
after decoding in MLX/Metal at PPL 29.742458 using FP16 weights. The other four
width artifacts passed the same exact bin and midpoint checks; per-width
reports are saved beside each ZIP. All six widths also produced bitwise-identical
logits between decoded-stream weights and direct in-memory midpoint weights on
a shared four-token sample; see the `*-logit-parity.json` reports.

For width 0.002, a fixed-width alternative stores 12-bit labels for the same
2,958 bin symbols. Its ZIP is 186,735,952 bytes (12.0049 effective bits/weight)
versus 126,444,954 bytes (8.1289 bits/weight) for Huffman, a 32.29% reduction.
Both encodings decode to identical bin IDs.

## Near-rate calibrated GPTQ comparison

An extra width 0.012 point was evaluated against GPTQ W4 group 128 using
GPTQModel's MLX path. Both full packages use ZIP_STORED and include configuration
and tokenizer files.

| Method | WikiText-2 PPL | Full-package bytes | Effective bits/weight |
|---|---:|---:|---:|
| Huffman bins, width 0.012 | 29.742458 | 89,033,239 | 5.7238 |
| GPTQ W4, group 128 | 35.587507 | 86,320,207 | 5.5494 |

Both results use the same 286,177 WikiText-2 raw test token IDs and non-
overlapping 1,024-token chunks in MLX/Metal. Huffman is 3.14% larger by bytes
and has PPL lower by 5.8451 for this single model and dataset; this does not
establish a general advantage. Calibration, decode, evaluation, and container
details are in `../calibrated-baselines/near-matched-gptq-comparison.md`.

## Reproduction

```sh
HF_HOME=/private/tmp/quantization-hf-cache \
  /private/tmp/quantization-bench-env/bin/python -u benchmark_huffman_gpt2.py \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --seq-len 1024 --widths 0.0003 0.001 0.002 0.004 0.008
```
