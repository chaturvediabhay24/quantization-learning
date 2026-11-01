# Tiny Transformer Quantization Playground

This project contains a very small GPT-2-style Transformer for learning and experimentation.

## Model

The model is `sshleifer/tiny-gpt2`, stored locally in `models/tiny-gpt2/`.

Its architecture is:

- 2 Transformer decoder blocks
- 2 attention heads per block
- hidden size of 2
- maximum sequence length of 1,024 tokens
- vocabulary size of 50,257 tokens
- GELU activations
- PyTorch checkpoint format

The checkpoint is only a few megabytes, but the model is intentionally too small for useful text generation. Its main purpose is to make model inspection, inference, and quantization easy to understand.

## Environment setup

The project uses a local Python virtual environment at `.venv/`.

Activate it with:

```bash
source .venv/bin/activate
```

The environment contains:

- Python 3.14
- PyTorch
- Transformers
- NumPy

## Run inference

From the project root:

```bash
source .venv/bin/activate
python infer.py "The smallest transformer model is"
```

Useful options:

```bash
python infer.py "Hello" --max-new-tokens 30 --temperature 0.8
```

Arguments:

- `prompt`: input text; defaults to `The smallest transformer model is`
- `--max-new-tokens`: number of tokens to generate; default is `30`
- `--temperature`: sampling temperature; use `0` for greedy decoding

Example deterministic run:

```bash
python infer.py "The smallest transformer model is" \
  --max-new-tokens 12 \
  --temperature 0
```

## How inference works

1. Load the local tokenizer from `models/tiny-gpt2/`.
2. Convert the prompt into token IDs.
3. Load the local GPT-2 model weights.
4. Run a forward pass without gradients.
5. Select the next token repeatedly until the requested number of tokens is generated.
6. Decode the resulting token IDs back into text.

The implementation is in [infer.py](infer.py).

## Expected output quality

Output may be repetitive or nonsensical, for example:

```text
The smallest transformer model is stairs stairs stairs ...
```

This is expected. The model has an unusually small hidden size of 2 and is useful for demonstrations rather than meaningful language generation.

## Model inspection

The architecture can be summarized as:

```text
Text → BPE tokenizer → token embeddings
     → Transformer block 1
     → Transformer block 2
     → language-model head
     → next-token probabilities
```

The model files are intentionally kept local so later experiments can inspect weights, calculate weight statistics, and test quantization without downloading another checkpoint.

## Rank-and-function reconstruction experiment

`rank_function_codec.py` tests the proposed sorted-rank compression method. For
each contiguous slice, it sorts the values, fits a line to sorted rank, and
reconstructs original positions from the inverse sorting permutation.

Run it on the token embedding tensor:

```bash
source .venv/bin/activate
python rank_function_codec.py --tensor transformer.wte.weight --slice-size 16
```

Run it across every learned floating-point checkpoint tensor:

```bash
python rank_function_codec.py --tensor all --slice-size 16
```

The first baseline uses `f(rank) = slope * rank + intercept`, with two float32
parameters per slice. On `transformer.wte.weight`, 16-value slices produce:

| Metric | Result |
| --- | ---: |
| Mean absolute error | 0.00376851 |
| RMSE | 0.00513108 |
| Maximum absolute error | 0.05019430 |
| Theoretical packed storage | 6.8130 bits/weight |
| Compression against float32 | 4.697× |

The storage figure assumes an optimally packed permutation, requiring 45 bits
for each complete 16-value slice, plus two 32-bit line parameters. The script
does not yet serialize a compressed checkpoint or modify the model.

## Matched-input codec comparison

`compare_codecs.py` compares three reconstruction algorithms using the same
original tensor and the same contiguous 16-value blocks:

```bash
python compare_codecs.py --tensor transformer.wte.weight --block-size 16
```

| Method | MAE | RMSE | Maximum error | Bits/weight |
| --- | ---: | ---: | ---: | ---: |
| Rank + fitted line | 0.00376851 | 0.00513108 | 0.05019430 | 6.8130 |
| Rank + normal quantile | 0.00338568 | 0.00444764 | 0.04039343 | 6.8130 |
| Uniform INT4 | 0.00129200 | 0.00158239 | 0.00551944 | 6.0003 |
| NF4 | 0.00121399 | 0.00161236 | 0.01024074 | 6.0003 |

INT4 and NF4 use four bits per weight plus one float32 absolute-maximum scale
per 16-value block. The rank-function method uses a packed 16-value permutation
plus two float32 line parameters per block. This comparison measures only
weight-reconstruction error; it does not yet compare model logits or generation.

The normal-quantile version fits each sorted block with
`scale * normal_quantile((rank + 0.5) / block_size) + location`. It uses the
same two parameters and storage as the line, and improved the baseline error.

## Normal-quantile slice sweep

The fitted coefficients are now rounded to their declared stored precision
before reconstruction. Sweep sizes and coefficient precision with:

```bash
python sweep_rank_codec.py --tensor transformer.wte.weight --function normal-quantile
```

| Slice size | Coefficients | RMSE | Bits/weight |
| ---: | --- | ---: | ---: |
| 16 | float16 | 0.00444764 | 4.8127 |
| 32 | float16 | 0.00350721 | 4.6877 |
| 64 | float16 | 0.00267024 | 5.1249 |
| 128 | float16 | 0.00200354 | 5.8512 |
| 256 | float16 | 0.00147384 | 6.7022 |

The 256-value normal-quantile reconstruction is lower-error than the earlier
16-value INT4 and NF4 baseline, but its 6.70-bit storage estimate is higher.
The methods do not yet use the same block size at that point, so a final
comparison should sweep block size for INT4 and NF4 as well.

## Matched 256-value block comparison

The following comparison uses the same embedding tensor and the same 256-value
block boundaries for every method. The rank-normal method stores float16 fitted
coefficients; each baseline stores a float32 absolute-maximum scale per block.

```bash
python compare_codecs.py --tensor transformer.wte.weight --block-size 256 \
  --rank-coefficient-dtype float16
```

| Method | RMSE | Maximum error | Bits/weight |
| --- | ---: | ---: | ---: |
| Rank + normal quantile | 0.00147384 | 0.02976903 | 6.7022 |
| Uniform INT4 | 0.00235008 | 0.00629594 | 4.1251 |
| NF4 | 0.00196618 | 0.01140713 | 4.1251 |
| Uniform INT6 | 0.00055956 | 0.00148748 | 6.1251 |
| Uniform INT7 | 0.00027693 | 0.00072846 | 7.1251 |

The current rank-normal method improves over 4-bit methods in RMSE but has a
larger maximum error and needs more bits. Uniform INT6 has lower error while
using fewer bits, so this version of the proposed codec does not beat a simple
nearby-rate baseline on raw embedding reconstruction.

## Exact-value exception sweep

`sweep_exceptions.py` starts from the 256-value, float16-coefficient
rank-normal reconstruction. Any value whose absolute reconstruction error is
above a chosen threshold is stored as its original float32 value, alongside its
8-bit within-block position. This needs 40 bits per exception before metadata
such as per-block exception counts.

```bash
python sweep_exceptions.py --tensor transformer.wte.weight --slice-size 256
```

| Error threshold | Exact exceptions | RMSE | Maximum error | Estimated bits/weight |
| ---: | ---: | ---: | ---: | ---: |
| 0.0025 | 5,767 (5.738%) | 0.00097380 | 0.00249998 | 8.9972 |
| 0.0050 | 1,059 (1.054%) | 0.00120861 | 0.00499855 | 7.1236 |
| 0.0100 | 156 (0.155%) | 0.00136906 | 0.00999425 | 6.7643 |
| 0.0200 | 8 (0.008%) | 0.00145860 | 0.01983098 | 6.7054 |

Exact exceptions control maximum error, but the current scheme still does not
beat uniform INT6 on both error and storage. The sizes are lower bounds because
they exclude exception-count and stream-framing metadata.

## Nearby-value mean clustering and function-generated codebooks

`compare_cluster_codecs.py` removes the unique-permutation requirement. Each
256-value block assigns weights to shared nearby-value groups. The mean-codebook
variant stores a float16 mean for each group. The normal-codebook variant
generates the group values from two float16 parameters and fixed normal-quantile
levels.

```bash
python compare_cluster_codecs.py --tensor transformer.wte.weight \
  --block-size 256 --levels 16 32
```

| Method | RMSE | Maximum error | Bits/weight |
| --- | ---: | ---: | ---: |
| Uniform INT4 | 0.00235008 | 0.00629594 | 4.1251 |
| NF4 | 0.00196618 | 0.01140713 | 4.1251 |
| Normal codebook, 16 levels | 0.00196773 | 0.03092233 | 4.1251 |
| Mean codebook, 16 levels | 0.00183057 | 0.02858689 | 5.0009 |
| Uniform INT5 | **0.00113423** | **0.00303845** | 5.1251 |
| Normal codebook, 32 levels | 0.00114921 | 0.02182934 | 5.1251 |
| Mean codebook, 32 levels | 0.00098909 | 0.02158330 | 7.0019 |

The function-generated 32-level codebook gives nearly the same RMSE as uniform
INT5 at the same storage rate. Its average absolute error is lower, but its
tail errors are larger. The 32-level mean codebook achieves lower RMSE but its
stored mean table makes it substantially more expensive.

## Normal-codebook exact exceptions

`sweep_codebook_exceptions.py` keeps an original float32 value when the 32-level
normal-codebook reconstruction error is above a selected threshold. Each exact
exception is estimated as an 8-bit within-block position and a 32-bit value.

```bash
python sweep_codebook_exceptions.py --tensor transformer.wte.weight \
  --block-size 256 --levels 32
```

| Threshold | Exact exceptions | RMSE | Maximum error | Estimated bits/weight |
| ---: | ---: | ---: | ---: | ---: |
| 0.0025 | 3,130 (3.114%) | 0.00076403 | 0.00249981 | 6.3707 |
| 0.0050 | 629 (0.626%) | 0.00094726 | 0.00499761 | 5.3754 |
| 0.0100 | 129 (0.128%) | 0.00105233 | 0.00992150 | 5.1765 |
| 0.0200 | 2 (0.002%) | 0.00114513 | 0.01936565 | 5.1259 |

At a 0.005 threshold, the normal-codebook-with-exceptions method improves on
uniform INT5's 0.00113423 RMSE, using about 0.25 additional bits per weight.
These estimates still exclude exception-count and stream-framing metadata.

## Matched exception-budget benchmark

`compare_exception_benchmarks.py` gives uniform INT5 the same number of exact
float32 exceptions as our normal-codebook method. INT5's exceptions are chosen
from its largest errors, making this a strong same-storage baseline.

```bash
python compare_exception_benchmarks.py --tensor transformer.wte.weight \
  --block-size 256 --levels 32
```

| Storage budget | Method | RMSE | Maximum error |
| ---: | --- | ---: | ---: |
| 5.1765 bits/weight | Our normal32 + 129 exceptions | **0.00105233** | 0.00992150 |
| 5.1765 bits/weight | INT5 + 129 exceptions | 0.00113019 | **0.00250548** |
| 5.3754 bits/weight | Our normal32 + 629 exceptions | **0.00094726** | 0.00499761 |
| 5.3754 bits/weight | INT5 + 629 exceptions | 0.00111773 | **0.00228556** |
| 6.3707 bits/weight | Our normal32 + 3,130 exceptions | **0.00076403** | 0.00249981 |
| 6.3707 bits/weight | INT5 + 3,130 exceptions | 0.00106720 | **0.00200225** |

At equal exception counts, our codec has lower average and RMS error. INT5 has
the lower remaining maximum error. The current normal codebook therefore
distributes its remaining error more unevenly, with a few larger tail misses.

## Huffman and byte-pair coding on weight streams

`compare_entropy_codecs.py` compares fixed-width quantized IDs with Huffman
coding, and applies the same lossless coders directly to the raw little-endian
float32 bytes. It reports reconstruction RMSE, payload bits, estimated
codebook/dictionary metadata, and encoder time. The raw-float rows reconstruct
exactly; the quantized rows retain the quantizer's error.

```bash
./.venv/bin/python compare_entropy_codecs.py \
  --tensor transformer.wte.weight --block-size 256 \
  --bpe-merges 16 --min-pair-count 3
```

On the 100,514-value embedding matrix, with 16 greedy BPE merges:

| Method | Bits/weight | RMSE | Encoder ms |
| --- | ---: | ---: | ---: |
| FP32 raw | 32.0000 | exact | 0.69 |
| FP32 + Huffman | 29.5549 | exact | 14.48 |
| FP32 + BPE + Huffman | 29.6269 | exact | 712.08 |
| Uniform INT4 fixed | 4.1251 | 0.00235008 | 6.75 |
| Uniform INT4 + Huffman | **3.5361** | 0.00235008 | 9.95 |
| Uniform INT4 + BPE + Huffman | 3.7103 | 0.00235008 | 137.04 |
| NF4 fixed | 4.1251 | **0.00196618** | 6.99 |
| NF4 + Huffman | 3.9182 | 0.00196618 | 10.25 |
| Uniform INT5 fixed | 5.1251 | 0.00113423 | 7.91 |
| Uniform INT5 + Huffman | **4.5743** | 0.00113423 | 11.09 |

The estimates include one float32 scale per 256-value block, canonical
Huffman code-length metadata, and BPE pair-table metadata. On this matrix,
Huffman coding saves storage after quantization, while the extra BPE pass does
not pay for its dictionary and is much slower. The result is specific to this
tensor and traversal order; a structured or sparse matrix with repeated ID
sequences could make BPE worthwhile. The benchmark is a storage experiment,
not a GPU inference implementation, so it does not claim that compressed
variable-length streams can be consumed directly by a matrix-multiply kernel.

## Normalized and bucketed weights

`compare_normalized_buckets.py` checks whether normalizing a tensor and fitting
shared scalar buckets improves quantization. It reports error, estimated bits
including bucket-table and normalization metadata, the number of distinct
reconstructed values, and the exact-zero fraction.

```bash
./.venv/bin/python compare_normalized_buckets.py \
  --tensor transformer.wte.weight --block-size 256 --buckets 32
```

The embedding tensor has no exact zeros; only 3.986% of its values have
absolute magnitude below 0.001. Results for 32 buckets were:

| Method | RMSE | Bits/weight | Exact zeros |
| --- | ---: | ---: | ---: |
| Uniform INT4 / block absmax | 0.00235008 | 4.1251 | 0.000% |
| NF4 / block absmax | 0.00196618 | 4.1251 | 10.399% |
| Uniform INT5 / block absmax | 0.00113423 | 5.1251 | 0.000% |
| Normalized normal buckets / block | 0.00114921 | 5.1251 | 0.000% |
| Learned buckets / block | **0.00098909** | 7.0019 | 0.000% |
| Learned buckets / global | 0.00111014 | 5.0051 | 0.000% |
| Z-score + learned buckets / global | 0.00111021 | 5.0057 | 0.000% |
| Z-score + forced zero bucket / global | 0.00112525 | 5.0057 | 3.270% |

Normalization itself makes almost no difference: the global learned-bucket
results with and without z-scoring are effectively identical. The useful part
is sharing a bucket value across many weights. A global 32-bucket codebook is a
promising compact option, matching INT5's error at slightly under 5.01
bits/weight. Per-block learned buckets reduce error further, but their separate
tables cost 7 bits/weight. A forced zero bucket increases exact sparsity; it
does not remove sparsity from the original FP32 tensor and slightly increases
RMSE here.

## Frequent-symbol Huffman with an escape symbol

`compare_top_huffman.py` tests a two-part entropy code. After quantization, the
most frequent `K` IDs receive Huffman codes. Every other ID becomes one ESC
symbol. In the lossless version, ESC is followed by the original fixed-width
quantized ID. A second variant collapses all ESC values to one zero-nearest
reconstruction value, which is lossy.

```bash
./.venv/bin/python compare_top_huffman.py \
  --tensor transformer.wte.weight --block-size 256 --top-k 4 8 12
```

Selected results on the embedding matrix:

| Method | Bits/weight | RMSE | Rare IDs |
| --- | ---: | ---: | ---: |
| Uniform INT4 + full Huffman | **3.5361** | 0.00235008 | — |
| Uniform INT4 top-4 + ESC, lossless | 3.9642 | 0.00235008 | 41.909% |
| Uniform INT4 top-8 + ESC, lossless | 3.6922 | 0.00235008 | 11.156% |
| Uniform INT4 top-12 + ESC, lossless | 3.5736 | 0.00235008 | 1.848% |
| Uniform INT4 top-16 + ESC, lossless | 3.5371 | 0.00235008 | 0.000% |
| Uniform INT4 top-12 collapsed | 3.4997 | 0.00750412 | 1.848% |
| NF4 + full Huffman | **3.9182** | 0.00196618 | — |
| NF4 top-12 + ESC, lossless | 4.0620 | 0.00196618 | 6.329% |

Restricting Huffman to frequent IDs does not help here. The alphabet already
has only 16 or 32 quantized IDs, so ordinary Huffman coding is cheap and can
encode every ID directly. ESC adds an extra code and fixed-width side data for
rare IDs. Collapsing rare IDs saves bits, but rare IDs are not all close to one
value; mapping them to one fallback value causes a large error increase.

With the proposed INT4 limit of `2^4 = 16` Huffman symbols, all 16 INT4
quantized IDs are already covered. Therefore no INT4 IDs remain for a second
quantization path; the result is effectively full Huffman coding, with a small
extra metadata cost for the ESC format.
