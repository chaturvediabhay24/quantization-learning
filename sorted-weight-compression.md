# Sorted-weight function compression

This note records our proposed compression experiment for the local
`sshleifer/tiny-gpt2` model. The method is an idea to test; we have not yet fitted
a function or measured compression quality.

## Core idea

Given a one-dimensional array of weights, sort the values and fit a compact
function to weight versus sorted rank. Store each weight's rank instead of its
original floating-point value. Reconstruct the original array by evaluating the
function at those ranks.

Assume initially that all weights are unique. For N weights, ranks run from
0 through N - 1. Let:

- `w[j]` be the original weight at position j.
- `s[i]` be the weight at sorted rank i.
- `p[i]` be the original position of sorted weight i.
- `r[j]` be the sorted rank of original weight j (the inverse permutation).
- `f(i; theta)` be the fitted function, with stored parameters theta.

Then:

```text
s[i] = w[p[i]]
r[p[i]] = i
reconstructed_w[j] = f(r[j]; theta)
```

## Mapping example

Our example stores original positions alongside the sorted values:

```text
Sorted values:             [f(0), f(1), f(2)]
Original positions p:      [   1,    2,    0]

Reconstructed weights:    [f(2), f(0), f(1)]
```

For direct lookup at each original position, store the inverse mapping instead:

```text
Ranks at original positions r: [2, 0, 1]
reconstructed_w[j] = f(r[j])
```

Either mapping is sufficient; we do not need to store both. Sorted rank and
original position are different concepts and must not be interchanged.

## Applying it to our embedding matrix

The token embedding tensor `transformer.wte.weight` has shape `(50257, 2)`.
Treat its two columns separately. Each column contains 50,257 weights and gets
its own rank mapping and fitted function.

Sorting each column independently means the same sorted rank in the two columns
can refer to different token rows. Reconstruction must restore both columns to
the original token-row order before the model uses them.

The earlier sorted bar plots showed only the first 32 token rows. A compression
experiment should use all 50,257 rows per dimension.

## Exact versus approximate reconstruction

A compact fitted function generally does not pass through every sorted weight:

```text
w[j] = f(r[j]; theta) + residual[j]
```

Keeping the rank mapping preserves placement, but not necessarily the exact
original values. Evaluating f in float32 does not guarantee float32-level
agreement with the original weights.

Exact recovery would require an exact representation of the sorted values, or
sufficient residual information. A function that interpolates every point may
need many coefficients; high-degree interpolation is not automatically compact
or numerically stable. Residual storage and coefficient precision count toward
the total size. There is no guarantee that arbitrary unique weights admit a
compact lossless representation.

Duplicate weights do not prevent the method: a stable sort can assign separate
ranks to ties. Sharing a rank or representative value is another option, with
a different encoding scheme.

## Storage estimate

A fixed-width rank requires `ceil(log2(N))` bits. For N = 50,257, that is 16 bits.

| Representation | Bytes per dimension | Bytes for both dimensions |
| --- | ---: | ---: |
| Original float32 values | 201,028 | 402,056 |
| Packed uint16 ranks only | 100,514 | 201,028 |

The rank-only estimate is half the original float32 payload. Actual storage is:

```text
rank mapping + function parameters + metadata + optional residuals
```

Thus roughly 2x compression is plausible if the function is small and its error
is acceptable, but it has not been demonstrated. The saving is relative to
float32: float16 already uses 16 bits per value without a rank mapping.

PyTorch sorting returns int64 indices by default. Storing those directly would
use 64 bits per index and defeat this estimate. A saved encoding must actually
pack ranks into uint16 (or another compact representation). Temporary int64
indices during reconstruction are separate from compressed file size.

These figures concern embedding values, not the entire model checkpoint or
runtime memory. Reconstructing a full float32 tensor before inference restores
its original memory footprint. The language-model head shares token-embedding
weights in GPT-2, so any model integration must preserve that sharing.

## Candidate fitted functions

If the weights are approximately normally distributed, one candidate is:

```text
f(i) = mu + sigma * inverse_normal_cdf((i + 0.5) / N)
```

The midpoint rank convention avoids evaluating the inverse CDF at 0 or 1.
Estimate mu and sigma separately for each embedding dimension. This is a
starting hypothesis, not a verified fit. Tail values may have larger errors.

A sorted Gaussian sample follows a normal quantile curve: relatively gentle
in the middle and steep toward the tails. Our earlier suggestion of tanh or a
sigmoid needs correction: those functions flatten at the extremes, so they do
not naturally capture the same tail behavior.

Other candidates include a monotone piecewise-linear function or a monotone
spline. Their knots and coefficients must be included in the storage budget.

## Connection to quantization

Unique ranks retain a separate index for every distinct weight. If nearby
sorted values can share a reconstructed value, fewer indices are needed.
For example, 256 representative values fit into 8-bit codes, and 16 fit into
4-bit codes. This connects the proposal to nonuniform quantization.

Reducing the number of representative values can save more space, but introduces
additional approximation error. An explicit codebook also has a storage cost;
a fitted function could generate its representative values instead.

## Proposed experiment

1. Load the original token embeddings and retain them as a reference.
2. Sort each full column, retaining the sorting and inverse mappings.
3. Fit a compact function separately to each column.
4. Reconstruct weights in their original token-row order.
5. Measure mean absolute error, mean squared error, and maximum absolute error,
   including errors at the tails.
6. Measure serialized bytes for indices, parameters, metadata, and any residuals.
7. Compare with float16 and simple 8-bit quantization at their actual sizes.
8. Compare model logits and deterministic generation before and after replacing
   embeddings. Weight error alone does not establish inference quality.

The decision is whether reconstruction quality justifies the total storage and
decoding cost. No model weights have been changed as part of this note.

## First baseline result

`rank_function_codec.py` implements the initial test on the local checkpoint.
It splits a flattened tensor into contiguous 16-value slices, fits a least
squares line to sorted rank in each slice, then restores values to their
original flattened positions using the inverse sorting permutation.

For `transformer.wte.weight` (100,514 values), the test measured:

| Metric | Result |
| --- | ---: |
| Mean absolute error | 0.00376851 |
| RMSE | 0.00513108 |
| Maximum absolute error | 0.05019430 |
| Theoretical packed storage | 6.8130 bits per weight |
| Compression ratio versus float32 | 4.697× |

The 6.8130 bits per weight comprise 45 packed permutation bits and 64 bits for
the line parameters per complete 16-value slice. The final slice has two values
and is counted using its own 1-bit permutation cost. This is a size estimate,
not a serialized file size. Model quality after reconstruction has not yet been
measured.

## Normal-quantile function result

We also tested the two-parameter monotone function:

```text
f(rank) = scale * normal_quantile((rank + 0.5) / slice_size) + location
```

It has the same packed-size estimate as the fitted line: 6.8130 bits per weight
for 16-value slices. On `transformer.wte.weight`, it reduced MAE from 0.00376851
to 0.00338568 and RMSE from 0.00513108 to 0.00444764. Its maximum absolute error
reduced from 0.05019430 to 0.04039343. It still has greater raw reconstruction
error than the matched 16-value INT4 and NF4 baselines.

## Slice-size and coefficient-precision sweep

`sweep_rank_codec.py` evaluates the normal-quantile function with the same
embedding input and slice sizes 16, 32, 64, 128, and 256. It simulates actual
coefficient storage by casting fitted parameters to float32 or float16 before
reconstruction.

The float16 results were:

| Slice size | RMSE | Bits per weight |
| ---: | ---: | ---: |
| 16 | 0.00444764 | 4.8127 |
| 32 | 0.00350721 | 4.6877 |
| 64 | 0.00267024 | 5.1249 |
| 128 | 0.00200354 | 5.8512 |
| 256 | 0.00147384 | 6.7022 |

The coefficient precision made negligible error difference for this experiment.
The total bit cost is not monotonic in slice size: coefficient overhead falls
with larger slices, but the information required for a permutation rises with
slice size. The lowest tested bit cost is at 32 values, while the lowest tested
error is at 256 values.

## Exact-value exceptions

An extension stores exact original values when rank-normal reconstruction error
exceeds a threshold. For 256-value slices, an exception is estimated as an
8-bit within-slice position plus a 32-bit float32 original value, or 40 bits.
This estimate excludes count and framing metadata.

| Threshold | Exceptions | RMSE | Bits per weight |
| ---: | ---: | ---: | ---: |
| 0.0025 | 5,767 (5.738%) | 0.00097380 | 8.9972 |
| 0.0050 | 1,059 (1.054%) | 0.00120861 | 7.1236 |
| 0.0100 | 156 (0.155%) | 0.00136906 | 6.7643 |
| 0.0200 | 8 (0.008%) | 0.00145860 | 6.7054 |

Exceptions reduce error and cap the remaining maximum error at the threshold,
but the tested version does not beat uniform INT6 in both size and raw weight
error. Possible later variants could encode residuals instead of full float32
values, but those need a separate storage and reconstruction design.
