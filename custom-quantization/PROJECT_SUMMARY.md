# Custom Quantization Lab — Project Summary

## Purpose

A local FastAPI application with a browser UI for exploring fixed-width
quantization on bundled Tiny GPT-2 weights and real Qwen safetensors checkpoints.
It shows reconstruction error, frequency distributions, variable-length codes,
and estimated storage without running inference.

## Quantization and model scope

- Each weight is assigned to `k = floor(value / bin_width)` and reconstructed
  at the bin midpoint `(k + 0.5) × bin_width`.
- The scope selector switches between all unique learned model tensors and
  one selected tensor/layer.
- The bundled dataset contains 28 unique learned float tensors and 102,714
  weights. Attention masks are excluded and tied aliases are counted once.
- The extractor matches exact mask-buffer suffixes, preserving the two learned
  attention-projection bias tensors that an earlier broad filter omitted.
- The UI reports MAE, RMSE, and maximum absolute reconstruction error.

## Binning and frequency controls

- **Bin width** is controlled by a numeric input and logarithmic slider.
- **Merge least frequent bins (K)** pools the K lowest-frequency occupied input
  bins into one shared bucket reconstructed at the mean of its source weights.
- The frequency table supports ascending and descending frequency order,
  numbered rows, and virtualized scrolling over all occupied bins.
- The histogram visualizes occupied bins in value order, highlights shared
  buckets, and omits empty ranges.

## Coding methods

- **Huffman** assigns optimal binary prefix codes from output-bucket
  frequencies.
- **Unary** sorts output buckets by descending frequency (with a stable symbol
  tie-break) and assigns `1`, `01`, `001`, and so on. More frequent buckets get
  shorter codes.
- Both methods use the same bins, rare-bin pooling, frequency table, loss
  calculations, and storage reporting.
- **Max code length (L)** is optional and accepts limits from 1 to 64 bits.
  Huffman repeatedly pools overlength symbols into a shared overflow bucket and
  rebuilds the tree. Unary retains the most frequent buckets that fit and pools
  the remaining tail into one overflow bucket. Overflow buckets use the mean
  of their source weights for reconstruction.
- The frequency table shows codewords; very long unary codewords use compact
  zero-run notation such as `0^40 1`.

## Metrics and comparison

- The storage panel shows payload size, code-table estimate, estimated total
  size, and payload rate.
- **Average code bits / original weight** is the weighted average code length
  over the selected source weights:

  `sum(output_bucket_frequency × output_bucket_code_length) / selected_weight_count`

  Source weights pooled into shared buckets retain their contribution through
  the bucket frequency. This rate is payload-only; the table estimate is shown
  separately.
- The UI compares **Current settings** against **No K/L pooling**. Both columns
  use the same model/layer scope, bin width, and coding method. The baseline sets
  `K = 0` and disables the code-length limit, so every occupied input bin keeps
  its own output symbol.
- The comparison includes pooled-bin counts, output buckets, code lengths,
  MAE/RMSE/maximum error, average bits per original weight, entropy lower bound,
  payload/table/total size, and the estimated reduction relative to FP32.
- Storage estimates are not a serialized codec size. They exclude
  tensor-shape metadata, file/container framing, byte padding, and an actual
  encoder or decoder. The analysis panels measure weight reconstruction.
- **Explore bin-width trade-offs** opens a separate view with synchronized
  MAE, MSE, occupied-input-bin, and average-code-rate plots. It uses 17
  logarithmically spaced widths from 1/4× to 4× the selected width (clipped to
  the supported range), holding scope, coding method, K, and L constant. Hover
  or use the slider to inspect all four values at one width. Input-bin count is
  measured before K/L pooling.

## Real checkpoint analysis

- The switchable **Real checkpoints** dashboard supports official Qwen3-0.6B,
  Qwen3-1.7B, and Qwen3-4B-Instruct-2507 safetensors checkpoints.
- Checkpoints download from their public Hugging Face repositories on first use
  into the ignored `model-cache/` folder; checkpoints already in the standard
  Hugging Face cache are reused.
- The analyzer reads actual floating-point tensor values from safetensors in
  bounded CPU chunks. It does not instantiate a transformer, run a forward pass,
  generate text, or calculate language-model scores.
- Whole-model and individual-tensor scopes report exact MAE, MSE, RMSE, maximum
  error, input/output occupied bins, code lengths, average code rate, entropy,
  Huffman/Unary payload estimates, table estimates, and FP32/source-size comparisons.
- The selected checkpoint weights are analyzed in their saved dtype. Applying
  midpoint reconstruction happens while calculating the exact quantization
  errors; the dashboard does not save a reconstructed checkpoint or bitstream.
- Bin widths run from `1e-5` to `1` to bound the occupied symbol map. K rare-bin
  pooling and L code-length limits are supported.
- Source checkpoint bytes include safetensors files. Estimated coding bytes are
  a model-size estimate, not a serialized codec; table, tensor metadata, framing,
  and alignment limitations are described in the dashboard.

## Running the app

From the `custom-quantization` directory:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The FastAPI docs are at
`/docs`.

## Main files

- `app/main.py` — FastAPI routes, quantization, pooling, coding, error and
  storage calculations.
- `app/real_model_analysis.py` — Hugging Face checkpoint resolution and
  chunked, real-weight Qwen analysis.
- `app/static/index.html` — application layout and controls.
- `app/static/app.js` — bundled-weight API requests, UI rendering, comparison table, and
  histogram drawing.
- `app/static/real-models.js` — checkpoint selection, analysis job progress, and
  result rendering.
- `app/static/styles.css` — visual styling and responsive layout.
- `README.md` — setup, usage, API, and dataset documentation.
- `data/tiny_gpt2_weights.npz` — bundled model weights.
