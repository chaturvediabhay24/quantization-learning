# Huffman and INT4 hybrid embedding compression

## Research question

Can a token-embedding tensor be compressed by preserving a small set of rounded
values and using INT4 quantization for the remaining weights, with Huffman coding
applied to the resulting symbols?

The measurements below use the local token embedding tensor
`transformer.wte.weight`, with shape `(50,257, 2)` and 100,514 FP32 weights.
Weights are rounded to four decimal places when selecting special values.

## Encoding definitions

The simple baseline is uniform blockwise INT4 quantization. Each contiguous
256-weight block stores a FP32 absolute-maximum scale and a 4-bit level ID.
That gives 4 bits per weight for IDs plus 32/256 = 0.125 bits per weight for
scales.

The unified hybrid stream assigns one symbol to every weight:

- `T0` through `T14` represent fifteen selected rounded values.
- `Q0` through `Q15` represent the sixteen INT4 levels.

A single Huffman codebook is built over all 31 symbols. A decoded `Ti` looks up
the stored rounded value; a decoded `Qj` uses the INT4 level and the scale for
the current block. There is no separate ESC marker or fixed 4-bit payload after
an escape symbol.

For the estimates below, metadata includes the block scales and a small codebook
estimate. The exact number depends on the final file format, so payload bits and
the practical estimate are both shown.

## Results

| Method | Huffman payload (bits/weight) | Including scales and metadata | RMSE |
| --- | ---: | ---: | ---: |
| Simple uniform INT4 | 4.0000 | 4.1251 | 0.00235008 |
| Unified stream with 15 most-frequent rounded values | 3.6509 | ~3.781 | 0.00231105 |
| Unified stream with 15 least-frequent rounded values | 3.4135 | ~3.544 | 0.00235005 |

The fifteen most-frequent rounded values account for 3,304 weights (3.29% of
the tensor). They reduce RMSE by about 1.66% relative to simple INT4, while
the estimated storage is about 8.3% lower than the INT4 baseline.

After rounding to four decimals, the fifteen least-frequent selected values all
occur only once. They affect just 15 weights, so RMSE is effectively unchanged;
the relative improvement is about 0.001%. Their lower bit estimate comes
mostly from Huffman coding the INT4 IDs, not from preserving the rare values.

## Why a separate ESC design differs

If the stream is split into “top value” and “INT4 remainder” paths, a remainder
needs an ESC marker plus its 4-bit INT4 ID. For most weights that is about five
bits, so the separate design is approximately 5.13 bits per weight after scale
and metadata overhead. The unified stream avoids that extra per-weight escape
structure and lets common INT4 IDs receive short variable-length codes.

## Paper-worthiness assessment

These results are not yet sufficient for a research paper. They are a useful
prototype and support a testable hypothesis: a shared Huffman stream over
special representatives and quantized IDs can reduce storage while preserving
or slightly improving raw weight-reconstruction error.

The current evidence is limited in several ways:

- It uses one very small GPT-2 embedding tensor.
- The bit counts are estimates rather than serialized files.
- The codebook, scale, framing, and decoder costs need a precise format.
- RMSE is measured on weights only; perplexity, logits, generation quality,
  memory use, decoding speed, and training compatibility are unmeasured.
- The result must be compared with strong baselines such as INT4/NF4 with
  entropy coding, learned codebooks, GPTQ, AWQ, and relevant published methods.

The strongest next step is to implement a packed encoder and decoder, evaluate
the complete model on several checkpoints and layers, and report a rate-quality
curve against matched-storage baselines. If the method retains its advantage on
larger models and model-level metrics, it could become a paper contribution;
the current experiment is best described as an early research note.
