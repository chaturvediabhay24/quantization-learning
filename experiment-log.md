# Quantization experiment log

This is the running record of experiments on the local `sshleifer/tiny-gpt2`
checkpoint. It focuses on `transformer.wte.weight`, the token embedding tensor
with shape `(50,257, 2)` and 100,514 float32 values. The tensor is useful for
rapid experiments but is not sufficient to establish model-level quality.

All reported measurements are raw weight-reconstruction error unless noted.
The primary measures are mean absolute error (MAE), root mean squared error
(RMSE), maximum absolute error, and estimated bits per weight.

## Original proposal

The initial idea was to sort values in a slice, fit a compact function to the
sorted values, then preserve each value's original placement through its sorted
rank. For an original vector `w`:

```text
sorted_values[i] = w[sorted_position[i]]
rank[sorted_position[i]] = i
reconstructed_w[j] = f(rank[j])
```

The `rank` mapping is a permutation. For 16 unique values, a packed permutation
needs `ceil(log2(16!)) = 45` bits. This is more compact than sixteen separate
4-bit positions, which need 64 bits.

## Functions tested with unique ranks

### Linear rank function

Each sorted slice was fitted with:

```text
f(rank) = slope * rank + intercept
```

For 16-value embedding slices with float32 parameters:

| Metric | Result |
| --- | ---: |
| MAE | 0.00376851 |
| RMSE | 0.00513108 |
| Maximum error | 0.05019430 |
| Estimated bits/weight | 6.8130 |

### Normal-quantile rank function

The next function used the inverse normal CDF:

```text
f(rank) = scale * normal_quantile((rank + 0.5) / slice_size) + location
```

It has the same two-parameter form but represents the steep tails of a
Gaussian-like sorted distribution better.

For 16-value slices it improved RMSE to `0.00444764` at the same 6.8130-bit
estimate. Coefficients were then cast to their declared stored precision before
reconstruction. Float16 coefficients produced negligible additional error.

## Slice-size sweep: normal-quantile rank codec

The following uses float16 scale and location parameters.

| Slice size | RMSE | Bits/weight |
| ---: | ---: | ---: |
| 16 | 0.00444764 | 4.8127 |
| 32 | 0.00350721 | 4.6877 |
| 64 | 0.00267024 | 5.1249 |
| 128 | 0.00200354 | 5.8512 |
| 256 | 0.00147384 | 6.7022 |

Larger slices improve the curve fit, but a permutation becomes more expensive
per value as slice size grows. The unique-rank version did not outperform
uniform INT6 at nearby storage.

## Baseline quantizers

Baselines use blockwise absolute-maximum scaling and shared levels. NF4 is the
standard 16-level normal-float codebook. Uniform integer baselines use evenly
spaced levels. For 256-value blocks, the scale overhead is 32/256 = 0.125 bits
per weight.

| Method | RMSE | Maximum error | Bits/weight |
| --- | ---: | ---: | ---: |
| Uniform INT4 | 0.00235008 | 0.00629594 | 4.1251 |
| NF4 | 0.00196618 | 0.01140713 | 4.1251 |
| Uniform INT5 | 0.00113423 | 0.00303845 | 5.1251 |
| Uniform INT6 | 0.00055956 | 0.00148748 | 6.1251 |
| Uniform INT7 | 0.00027693 | 0.00072846 | 7.1251 |

## Exact exceptions on the unique-rank codec

For high-error values, we tested keeping the original float32 value plus an
8-bit within-256-block position. Each exception therefore costs an estimated
40 bits before exception-count or stream metadata.

This reduced error but did not beat uniform INT6 on both storage and RMSE. It
also demonstrated that full float32 exceptions can become expensive quickly.

## Shift from unique ranks to shared value IDs

The key change was to remove the unique permutation. Instead, multiple weights
can share a nearby representative value. This stores a 4-bit or 5-bit cluster
ID per weight, similar to scalar quantization.

Two methods were tested.

### Function-generated normal codebook

A block uses two float16 parameters, `scale` and `location`, to generate 16 or
32 levels from fixed normal-quantile positions. Each weight stores the nearest
level ID. This is our current main method.

### Nearby-value mean codebook

One-dimensional k-means / Lloyd-Max clustering groups nearby values and stores
the float16 mean of each group. Each weight stores its group ID. This gives the
best value fit for the chosen number of groups, but the mean table costs storage.

| Method | RMSE | Maximum error | Bits/weight |
| --- | ---: | ---: | ---: |
| Normal codebook, 16 levels | 0.00196773 | 0.03092233 | 4.1251 |
| Mean codebook, 16 levels | 0.00183057 | 0.02858689 | 5.0009 |
| Normal codebook, 32 levels | 0.00114921 | 0.02182934 | 5.1251 |
| Mean codebook, 32 levels | 0.00098909 | 0.02158330 | 7.0019 |

At 32 levels, the function-generated codebook nearly matches uniform INT5's
RMSE at the same 5.1251 bits per weight. Its remaining error is concentrated in
the tails, which creates a higher maximum error.

## Exact exceptions on the 32-level normal codebook

We then preserved original float32 values for normal-codebook reconstruction
errors above a threshold. This adds a 40-bit exception estimate as above.

| Threshold | Exact values | RMSE | Maximum error | Bits/weight |
| ---: | ---: | ---: | ---: | ---: |
| 0.0025 | 3,130 (3.114%) | 0.00076403 | 0.00249981 | 6.3707 |
| 0.0050 | 629 (0.626%) | 0.00094726 | 0.00499761 | 5.3754 |
| 0.0100 | 129 (0.128%) | 0.00105233 | 0.00992150 | 5.1765 |

## Fair comparison: same exception budget

To avoid giving only our method exceptions, uniform INT5 was given the exact
same number of float32 exceptions. Its exceptions were chosen from its largest
errors, which makes it a strong same-storage comparison.

| Bits/weight | Method | RMSE | Maximum error |
| ---: | --- | ---: | ---: |
| 5.1765 | Our normal32 + 129 exceptions | **0.00105233** | 0.00992150 |
| 5.1765 | INT5 + 129 exceptions | 0.00113019 | **0.00250548** |
| 5.3754 | Our normal32 + 629 exceptions | **0.00094726** | 0.00499761 |
| 5.3754 | INT5 + 629 exceptions | 0.00111773 | **0.00228556** |
| 6.3707 | Our normal32 + 3,130 exceptions | **0.00076403** | 0.00249981 |
| 6.3707 | INT5 + 3,130 exceptions | 0.00106720 | **0.00200225** |

Our current method has lower MAE and RMSE than INT5 with the same exact-value
exception budget. Its maximum error is higher, showing that the normal codebook
still misses some tail values more severely.

## What we learned

1. A unique sorted rank per weight is too costly. It leaves too few bits for an
   accurate fitted curve.
2. Float16 function parameters are sufficient for these experiments.
3. Shared code IDs are the useful direction. They remove permutation storage
   and make the method a function-generated quantizer.
4. A normal-quantile codebook models the dense central distribution well.
5. A small set of exact values improves average error efficiently, but must be
   evaluated with the same exception allowance given to baselines.
6. The promising configuration is 256-value blocks, 32 normal-quantile levels,
   float16 scale/location, and a controlled number of float32 exceptions.

## Important limitations

- Bit counts are estimates; no packed bitstream or real file size has been
  produced yet.
- Exception counts and framing metadata are excluded from the storage estimates.
- All results concern one embedding tensor from a tiny model.
- We have not measured logits, generated output quality, perplexity, latency,
  or model memory after actual decoding.
- GPTQ and AWQ are calibration-based methods and are not represented by these
  raw reconstruction comparisons.

## Recommended next experiment

Jointly optimize the normal codebook and exceptions per block:

1. Fit the normal codebook.
2. Reserve a fixed count of highest-error values as exceptions.
3. Refit scale and location using only remaining values.
4. Reassign codes and update the exception set until stable.
5. Give uniform INT5 the same fixed exception budget and refit its block scale.

Then write an actual packed encoder and decoder, measure serialized bytes, and
replace the embedding weights during inference to test logits and output.

## Scripts

- `rank_function_codec.py`: unique-rank linear and normal-quantile reconstruction.
- `sweep_rank_codec.py`: slice-size and coefficient-precision sweep.
- `compare_codecs.py`: rank codec versus uniform integer and NF4 baselines.
- `sweep_exceptions.py`: exceptions on the unique-rank codec.
- `compare_cluster_codecs.py`: shared mean and normal codebooks.
- `sweep_codebook_exceptions.py`: exceptions on the normal codebook.
- `compare_exception_benchmarks.py`: same-exception-budget comparison against INT5.

## Research-paper readiness assessment — 2026-10-14

Current assessment: a useful research prototype with a testable hypothesis,
but insufficient evidence for a strong research paper yet. This is not a
conclusion that the idea lacks novelty or cannot become publishable.

### Evidence we currently have

On this embedding tensor, our 32-level normal codebook with 629 exact float32
exceptions achieved RMSE 0.00094726 versus 0.00111773 for uniform INT5 with
629 exceptions: approximately 15.25% lower RMSE at the same estimated
5.3754 bits per weight. Our maximum error was worse: 0.00499761 versus
0.00228556. This is a limited reconstruction result, not a model-quality result.

### What remains unproven

- **Novelty:** Fitted codebooks, nonuniform quantization, and high-precision
  exceptions have related prior work. We have not established a new technical
  contribution in this combination. A limited literature search neither proves
  novelty nor proves the exact method already exists.
- **Generality:** Our main comparisons use one embedding tensor from an
  extremely tiny model. We need results across layers and realistic trained
  models, including weight distributions that are less Gaussian.
- **Baseline strength:** Our normal-codebook parameters are optimized, while
  the uniform baseline uses absmax scaling. Equal exception counts do not make
  optimization effort equal. Optimizing the uniform scale and jointly choosing
  exceptions could shrink or eliminate the observed advantage.
- **Actual storage:** Reported rates exclude exception counts, framing, and
  other metadata. A packed encoder/decoder must establish the complete byte cost.
- **Practical value:** We have not established improvements in perplexity,
  task accuracy, decoding latency, inference speed, or memory use. Lower weight
  RMSE does not by itself imply better model behavior.

### Validation milestones

1. Compare equally optimized normal and uniform codebooks. Give both methods
   the same exception budget and opportunity to refit after selecting exceptions.
   Retain the best measured solution if alternating optimization regresses.
2. Build and verify packed encoders/decoders; compare at matched actual storage,
   including all metadata, parameter precision, code packing, and exceptions.
3. Test multiple tensors, layers, and realistic trained models with a documented
   evaluation protocol. Separate tuning choices from final evaluation data.
4. Evaluate model perplexity or task accuracy and practical decode/runtime costs.
   Preserve tied weights and account for them once in model storage comparisons.
5. Run ablations for the function shape, parameter fitting, block size,
   coefficient precision, and exception selection. Identify which component
   explains any improvement and report configurations where it loses.
6. Review closely related literature and state precisely what the proposed
   contribution adds. Do not equate a familiar-technique combination with
   established novelty.

### Decision criteria for revisiting publication

Continue toward a paper if the method shows a reproducible advantage over
strong baselines at matched actual storage, holds across realistic settings,
and has a clearly articulated contribution relative to prior work. Explain the
accuracy/storage/runtime tradeoff rather than claiming universal superiority.

If the advantage disappears after optimizing baselines or counting metadata,
record that outcome and reconsider the design before making broader claims.
The current tiny-model experiments alone do not justify a state-of-the-art claim.

## Conversation findings and corrected Huffman-bin analysis — 2026-10-28

This section records the follow-up discussion about normal codebooks, fixed
width bins, Huffman coding, rare bins, whole-model estimates, and research
readiness. Later direct-checkpoint calculations in this section use a corrected
legacy-checkpoint reader, described below.

### Terminology and normal-codebook algorithm

Keep these two sizes separate:

- **Group/block size** is the number of weights sharing scale or codebook
  metadata. Common reference choices include 64 for bitsandbytes NF4 and 128
  for GPTQ/AWQ configurations. See the [bitsandbytes implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py),
  [GPTQ configuration docs](https://huggingface.co/docs/transformers/quantization/gptq),
  and [AWQ examples](https://github.com/mit-han-lab/llm-awq/blob/main/README.md).
- **Level/bucket count** is the number of reconstruction values. 16 levels
  require a 4-bit ID; 32 levels require a 5-bit ID.

The project's **generated normal codebook** is blockwise. For a block of
weights, generate standard-normal quantiles
`z_i = Phi^-1((i + 0.5) / K)`, fit a location `mu` and scale `s`, and use
`mu + s*z_i` as the K nonuniform reconstruction levels. The fit iterates
between assigning each weight to its nearest level and refitting `mu` and `s`.
The implementation stores the two fitted parameters in FP16 per block, plus
the level ID per weight. Levels are denser near the center and farther apart in
the tails. This generated-per-block codebook is related to, but distinct from,
NF4's fixed 16-value table and block scaling.

### Corrected embedding MAE and bin-width sweep

The input tensor is `transformer.wte.weight`, shape `(50,257, 2)`, with
100,514 weights. The fixed-bin experiment uses zero-aligned half-open intervals
`[k*delta, (k+1)*delta)` and reconstructs each weight as the bin midpoint
`(k + 0.5)*delta`. The initial width was `delta = 0.00000279`, based on the
rounded FP32-to-FP16 MAE. This is a chosen fixed scalar step; FP16 itself has
magnitude-dependent spacing, so the two are not equivalent quantizers.

The following sweep is the corrected result. The Huffman payload is the
optimal binary-Huffman data-stream bit count for the observed bin frequencies.
The estimated table adds a simple serialized-tree cost: one marker bit per tree
node (`2*M - 1`) plus a fixed-width bin index per leaf. It excludes small file
headers, byte padding, and an actual packed encoder/decoder.

| Width multiple | Bin width | Occupied bins | Midpoint MAE | Huffman payload (bits) | Estimated table (bits) | Estimated total (bits) | Total / FP32 | Reduction vs FP32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x | 0.00000279 | 29,448 | 0.000000696176 | 1,464,284 | 530,063 | 1,994,347 | 62.00% | 38.00% |
| 2x | 0.00000558 | 17,147 | 0.00000139738 | 1,378,058 | 291,498 | 1,669,556 | 51.91% | 48.09% |
| 4x | 0.00001116 | 9,565 | 0.00000279562 | 1,285,516 | 153,039 | 1,438,555 | 44.72% | 55.28% |
| 8x | 0.00002232 | 5,226 | 0.00000559183 | 1,189,346 | 78,389 | 1,267,735 | 39.41% | 60.59% |
| 16x | 0.00004464 | 2,818 | 0.0000111511 | 1,091,157 | 39,451 | 1,130,608 | 35.15% | 64.85% |
| 32x | 0.00008928 | 1,497 | 0.0000222905 | 991,814 | 19,460 | 1,011,274 | 31.44% | 68.56% |
| 64x | 0.00017856 | 793 | 0.0000447481 | 892,021 | 9,515 | 901,536 | 28.03% | 71.97% |
| 128x | 0.00035712 | 422 | 0.0000893765 | 791,988 | 4,641 | 796,629 | 24.77% | 75.23% |

At the 1x width, occupied embedding-bin frequencies include 7,676 bins with
one weight, 5,289 bins with two weights, and 4,377 bins with three weights.
The most frequent 1x bin has 16 weights. Prior top-bin rows shown in the
conversation used the faulty reader and should not be reused.

### Corrected whole-model result at 128x width

The corrected reader finds 26 unique learned FP32 tensors and one tied alias
that is counted once, for 102,702 scalar weights total. Attention mask buffers
(`attn.bias` and `masked_bias`) are excluded. Thus the FP32 reference is
3,286,464 bits. A single global bin histogram and Huffman tree are assumed for
all tensors, with `delta = 0.00035712`.

| Measure | Result |
| --- | ---: |
| Occupied bins | 423 |
| Midpoint reconstruction MAE | 0.0000893883 |
| Huffman payload | 809,262 bits |
| Estimated tree and bin-index table | 5,921 bits |
| Estimated total | 815,183 bits |
| Total / FP32 | 24.8043% |
| Reduction from FP32 | 75.1957% |

The top 10 whole-model bins at this width were:

| Rank | Bin interval | Count |
| ---: | --- | ---: |
| 1 | [0, 0.00035712) | 805 |
| 2 | [0.00071424, 0.00107136) | 785 |
| 3 | [-0.00142848, -0.00107136) | 782 |
| 4 | [-0.00499968, -0.00464256) | 773 |
| 5 | [-0.00285696, -0.00249984) | 772 |
| 6 | [-0.00071424, -0.00035712) | 753 |
| 7 | [0.00428544, 0.00464256) | 749 |
| 8 | [0.00535680, 0.00571392) | 742 |
| 9 | [-0.00249984, -0.00214272) | 741 |
| 10 | [0.00178560, 0.00214272) | 741 |

### Rare-bin collapse and secondary quantization

At the whole-model 128x width, 34 bins have frequency one, representing 34
weights (0.0331% of all weights). Their values span approximately
`[-0.0839552, 0.0944713]`.

| Treatment of singleton bins | MAE | Estimated total bits | Difference from separate-bin baseline |
| --- | ---: | ---: | ---: |
| Keep singleton bins separate | 0.0000893883 | 815,183 | — |
| Map every singleton to zero | 0.000113973 (+27.50%) | 814,549 | 634 bits fewer |
| Map every singleton to the median representative (`-0.067564`) | 0.000112661 (+26.04%) | 814,565* | 618 bits fewer |
| Escape symbol + 4-level FP16 secondary codebook | 0.0000903730 (+1.10%) | 814,681 | 502 bits fewer |
| Escape symbol + 8-level FP16 secondary codebook | 0.0000897327 (+0.38%) | 814,779 | 404 bits fewer |
| Escape symbol + 16-level FP16 secondary codebook | 0.0000895177 (+0.15%) | 814,941 | 242 bits fewer |

`*` The earlier quick median estimate omitted storage for its representative;
this corrected estimate adds one 16-bit scalar.

For the secondary-codebook estimates, the main Huffman stream uses one escape
symbol for singleton bins. A side stream stores a fixed-width secondary ID for
each escaped weight; the secondary codebook is learned by 1D clustering and
stores FP16 centers. These are bit-count estimates, not serialized artifacts.
Because there are only 34 singleton weights, the net savings are tiny (about
30–63 bytes). Mapping them all to one representative loses substantially more
accuracy than a small secondary codebook. A Huffman escape symbol is a symbol
whose bit pattern is assigned by the tree; it is not necessarily the literal
bit `0`.

### Corrected direct MAEs for the embedding layer

These values were recomputed from correctly read checkpoint values with a
small standard-library reproduction of the quantizers; the grouping is 256
weights for the blockwise methods. Float32 operation details can cause tiny
last-digit differences from the PyTorch implementations. Treat these as
corrected working figures and verify them in the PyTorch runner before
publication.

| Method | MAE |
| --- | ---: |
| FP32 to FP16 conversion | 0.00000279561 |
| Symmetric INT4, absmax per 256 weights, q range -7 to 7 | 0.00215727 |
| Symmetric INT8, absmax per 256 weights, q range -127 to 127 | 0.000119111 |
| Uniform 16-level grid on [-1, 1], per-block absmax | 0.00201747 |
| NF4, per-block absmax | 0.00160048 |
| Uniform 32-level grid on [-1, 1], per-block absmax | 0.000972528 |
| Generated normal codebook, 16 levels | 0.00143795 |
| Generated normal codebook, 32 levels | 0.000744903 |

The current codebook comparison in the README reports RMSE, not these MAEs.
That `torch.load`-based comparison is separate from the flawed raw reader and
its existing RMSE results remain the recorded baseline. Lower weight MAE alone
does not establish better perplexity, task accuracy, or inference behavior.

### Checkpoint-reader correction and superseded results

The old-format PyTorch checkpoint writes an 8-byte storage-element-count
header before each storage's raw bytes. `quantize_weights.py`'s lightweight
`load_legacy_checkpoint` reader skipped this header. The corrected direct
calculations above verified and skipped one 8-byte header per storage; all
32 storage headers matched the tensor metadata. The existing reader in
`quantize_weights.py` should be fixed before using its output for future
experiments.

This bug affects direct calculations that used that reader: earlier embedding
histogram rankings, frequency-1/2/3 counts, Huffman-table-size estimates,
bin-width sweeps, and the first direct FP16/INT4/INT8 MAEs in this conversation
are superseded by this section. In particular, corrected embedding 1x figures
are 29,448 occupied bins, 1,464,284 Huffman payload bits, 530,063 estimated
table bits, and 1,994,347 total bits (62.00% of the embedding's FP32 size).
The existing README results produced through `torch.load` were not computed
with this faulty reader and remain distinct.

### Publication and patent assessment

The conversation's preliminary assessment was that the work is a useful
exploration, not yet a strong paper or patent case. Weight quantization/weight
sharing followed by Huffman coding is established prior work, including
[*Deep Compression*](https://arxiv.org/abs/1510.00149). A specific contribution
would need to be stated and compared against that and other prior art.

For publication, priorities are an actual encoder/decoder with all metadata
counted, fair matched-storage baselines (including NF4/INT8 and calibration-
based methods where practical), tests across realistic full models, and
perplexity/task-quality and decode/runtime measurements. For a patent, the
current high-level combination appears weak without a specific novel,
non-obvious technical mechanism. Patentability requires a claim-specific prior
art search and jurisdiction-specific legal review; USPTO criteria include
novelty, non-obviousness, usefulness, and adequate disclosure
([USPTO guidance](https://www.uspto.gov/patents/basics/using-legal-services/pro-se-assistance-program)).

The intended future claim to test is: "At the same actual storage, this method
consistently preserves model quality better, and we can explain why." At present,
the supported claim is only a promising small-scale reconstruction experiment.

### 2026-10-31 GPTQ near-rate benchmark and saved follow-up

The 0.012 Huffman artifact was decoder-validated and evaluated after decoding
in MLX/Metal with FP16 midpoint weights. On the same 286,177 WikiText-2 test
tokens as the GPTQ W4 group-128 run, Huffman measured PPL 29.742458 at 5.7238
bits/weight; GPTQ measured PPL 35.587507 at 5.5494 bits/weight. Both full
packages use ZIP_STORED and include model/tokenizer metadata. Their sizes are
89,033,239 B (Huffman) and 86,320,207 B (GPTQ). Huffman was 3.14% larger and
had lower PPL on this single model/dataset; no broad superiority claim follows.

Weight errors vs the pinned FP32 source were also measured: Huffman 0.012 with
FP16 storage had MAE 0.00300237, MSE 1.20128e-5, and max error 0.007851;
GPTQ had MAE 0.00980140, MSE 2.12330e-4, and max error 1.47307. GPTQ's
largest per-layer MSE is in `model.h.3.attn.c_attn.weight`; its largest maximum
error is in `model.h.4.attn.c_attn.weight`. Full records are in
`research-results/calibrated-baselines/near-matched-gptq-comparison.md` and
the adjacent JSON result/distortion files. A first distortion pass stopped on
GPTQ's packed embedding shape and saved a failed status; the corrected pass
handles both MLX QuantizedLinear and QuantizedEmbedding and completed.

For generation latency, three measured requests after one warm-up used the
same 128-token prompt and greedy 64-token generation in MLX/Metal. Medians:
Huffman FP16 TTFT 16.06 ms, prompt processing 15.90 ms, decode 297.3 tok/s,
end-to-end 227.27 ms; GPTQ W4 TTFT 16.24 ms, prompt processing 16.10 ms,
decode 590.8 tok/s, end-to-end 122.80 ms. GPTQ was faster on this workload;
Huffman retained lower PPL and weight errors. Huffman decode/install took
28.53 s; GPTQ checkpoint load took 41.21 s. See
`research-results/calibrated-baselines/gpt2-huffman-vs-gptq-generation-mlx.json`
for every trial and the metric definitions.

All six Huffman sweep widths (0.0003, 0.001, 0.002, 0.004, 0.008, 0.012)
have now passed decoder round-trip validation. Each report confirms the full
124,439,808 bin sequence and histogram, zero midpoint reconstruction
difference, valid framing/padding, and finite sample logits. Reports are
`research-results/gpt2-huffman-sweep/gpt2-huffman-*-validation.json`; the
remaining-width batch status is in `validation-batch-progress.json`. That
batch reported 275.78 seconds elapsed; time included File Provider downloads
for on-demand workspace files.

Direct output parity was then checked for all six widths. For each archive,
the decoder-restored model and the independent in-memory midpoint model
produced bitwise-identical logits on token IDs `[464, 3290, 318, 257]`; max
and mean absolute logit differences were both zero. The batch reported
400.02 seconds. Per-width reports are `research-results/gpt2-huffman-sweep/*-logit-parity.json`;
batch status is in `logit-parity-batch-progress.json`.
