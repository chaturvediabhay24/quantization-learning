# Custom Quantization Lab

A local FastAPI app for exploring fixed-width midpoint quantization and
frequency-based Huffman or unary coding across a model or one selected tensor. The
bundled Tiny GPT-2 checkpoint data contains 28 unique learned float tensors and
102,714 weights. Attention masks are excluded, and tied aliases are counted
once.

## Run locally

```bash
cd custom-quantization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Choose **Full model** or
any learned tensor from **Model scope**. The bin-width field and logarithmic
slider update the histogram, frequency table, reconstruction errors, and
storage estimates for the selected scope. Set **Merge least frequent bins** to
`K` to pool the `K` lowest-frequency occupied input bins into one shared
output bucket. The chart marks that bucket in orange, and the expandable
preview lists the source bins selected for pooling. The frequency table can
sort the original bins in descending (most-to-least) or ascending
(least-to-most) frequency order. Rows include rank numbers; pooled source bins
are marked and show their shared Huffman code. The scrollable table covers every
occupied bin, with a virtualized row window so large bin counts remain usable.
Choose **Huffman** for optimal binary prefix codes or **Unary** to assign
frequency-ranked codes (`1`, `01`, `001`, …), with the most frequent output
bucket receiving the shortest code. **Max code length** accepts a value from 1
to 64 bits for either method. Huffman bins whose tree codes exceed the limit are
pooled into an overflow bucket and the tree is rebuilt. Unary keeps the most
frequent buckets that fit and pools the remaining tail into one visible
overflow bucket. Leave the field blank to disable the limit.

Beside **Bin frequency**, **Explore bin-width trade-offs** opens a separate
view with four synchronized plots: MAE, MSE, occupied input bins, and average
code bits per original weight. It samples 17 logarithmically spaced widths
from one quarter to four times the selected width, clipped to the supported
range. The current scope, coding method, K, and L are held constant. Hover over
a plot or use the slider to inspect all four values at one width. Occupied input
bins are counted before K/L pooling; code rate reflects the selected method and
pooling settings.

When Huffman is selected, click **Review Huffman codes** beside the symbol map
to open a separate, read-only code map for the current scope, bin width, `K`,
and `L`. It shows the coding summary and a searchable, virtualized mapping
from every source bin to its Huffman code. The preview does not change the
model or write a Huffman-coded bitstream.

## Quantization and storage estimates

For each weight `x` and bin width `Δ`, the app assigns an integer symbol
`k = floor(x / Δ)` and reconstructs the weight at `(k + 0.5) × Δ`. Bins are
half-open intervals `[kΔ, (k+1)Δ)`, including for negative values.

The backend optionally maps the `K` least-frequent occupied bins to one shared
bucket, reconstructed at the mean of the original values in those bins. Huffman
mode builds an optimal binary tree. Unary mode sorts output buckets by
descending frequency (stable on symbol index) and gives rank `r` the code
`0^(r-1)1`. If a maximum code length is set, each method pools bins into an
overflow bucket as needed to keep every output code within the limit. Pooled
bucket reconstructions use the mean of their source weights. The app reports
payload bits and codewords for the selected method, and estimates the decoder
table using the appropriate tree or code map, fixed-width bin IDs, scalar
headers, and pooled-bucket representatives. It measures MAE, RMSE, and maximum
absolute error against the source float32 weights. The storage panel reports
the weighted average code length per original selected weight:
`sum(bucket_frequency × code_length) / selected_weight_count`. Rare-bin and
length-limited pools retain the frequencies of all source weights they contain.
The estimated total size separately includes the code table. A comparison table
shows the current K/L settings beside a baseline with K set to 0 and the code
length limit disabled, using the same scope, bin width, and coding method.

## BPE + Huffman experiment

Choose a layer or the whole model and open **Explore BPE merges**. Set the
number of BPE merge rules, then run **BPE + Huffman**. QuantLab starts with the
existing equal-width midpoint bins, learns ordered adjacent-pair merges, and
applies Huffman coding to the resulting sequence of original and merged
symbols. The merge table records the adjacent pair frequency before each
replacement and the number of non-overlapping replacements actually made.

The same analysis is available at
`GET /api/bpe-huffman-analysis?bin_width=0.00035712&tensor=__all__&merges=100`.

The final symbol table includes each symbol frequency and Huffman code. Merged
symbols expand to their two source symbols and ultimately to the original bin
IDs; they do not create new scalar midpoint values. Thus quantization MAE,
RMSE, and maximum error remain unchanged. Bits per original weight and the
estimated total size are compared with ordinary Huffman at the same bin width.
The size estimate includes the Huffman tree, symbol IDs, framing, and an
estimate for the BPE merge rules.

Tree/table bytes are an estimate, not a serialized codec size. The estimate
excludes tensor-shape metadata, file/container framing, byte padding, and an
The analysis dashboard includes a **Real checkpoints** view for Qwen3-0.6B,
Qwen3-1.7B, and Qwen3-4B-Instruct-2507. It analyzes the actual floating-point
values in their public safetensors checkpoints, with whole-model or individual
tensor scope. Checkpoint files download from Hugging Face on first use into the
ignored `model-cache/` directory; an already cached checkpoint is reused.

Weights are read in bounded CPU chunks. The analyzer does not instantiate a
transformer, run inference, or calculate language-model scores. It applies the
midpoint reconstruction rule to the actual selected weights and reports exact
MAE, MSE, RMSE, maximum error, occupied bins, Huffman or Unary code rate, entropy,
estimated table and payload bytes, and comparisons with the original checkpoint
and FP32 storage. K rare-bin pooling and an optional L code-length limit are
supported. Bin widths are limited to `1e-5`–`1` to bound the number of symbols.

The dashboard does not save a reconstructed checkpoint or encoded bitstream.
Coding sizes are estimates; source checkpoint bytes are the actual safetensors
file sizes for a full-model analysis. Tensor metadata, container framing, byte
alignment, and a real encoder/decoder are not included in the coding estimate.

## API

- `GET /api/layers` returns all selectable tensor names, shapes, and sizes.
- `GET /api/analysis?bin_width=0.00035712&tensor=__all__&merge_rare_bins=0&frequency_order=desc&coding_method=unary&max_code_length=8` analyzes the full bundled model with Unary coding and an 8-bit code-length cap. Use `coding_method=huffman` for Huffman; omit `max_code_length` to leave codes unconstrained. The old `max_huffman_length` query remains accepted for compatibility.
- `GET /api/bin-width-sweep?center_bin_width=0.00035712&tensor=__all__&merge_rare_bins=0&coding_method=huffman` returns the 17-point log-spaced trade-off sweep. `max_code_length` is optional.
- `GET /api/real-models` lists available Qwen3 checkpoints and whether their weights are already cached.
- `POST /api/real-models/qwen3-0.6b/analysis?bin_width=0.00035712&merge_rare_bins=0&coding_method=huffman` starts a real-weight analysis job. Optional settings include `max_code_length` and `tensor` (an exact tensor name; defaults to the whole model).
- `GET /api/real-analysis-jobs/{job_id}` returns progress or the completed metrics.
- Pass a tensor name in `tensor` to analyze one layer, for example `tensor=transformer.wte.weight`.
- FastAPI interactive docs are available at `/docs`.

## Replace or re-extract the dataset

Set `QUANTIZATION_DATA_FILE` to another `.npz` containing named floating-point
arrays. A single `.npy` array also works and appears as **custom tensor**.

To extract all unique learned float tensors from another compatible legacy
PyTorch checkpoint:

```bash
python scripts/extract_checkpoint.py /path/to/pytorch_model.bin data/tiny_gpt2_weights.npz
```
