# Learning Log

Add one dated entry after each study session. Keep the result short and include measurements or links when relevant.

## Entry template

### YYYY-MM-DD — Topic

- **Learned:**
- **Tried:**
- **Result or question:**
- **Next:**
- **Related code or notes:**

## Starting point — 2026-10-25

- **Learned:** The repository already contains a tiny GPT-2 inference example and several exploratory weight-codec comparisons.
- **Tried:** Reviewed the current project README, inference script, and experiment log to shape the roadmap around existing work.
- **Result or question:** Existing codec results focus mainly on reconstruction error and estimated storage. The next useful step is to measure model-output quality and then benchmark an actual inference path.
- **Next:** Start the model-level evaluation task in `roadmap.md`; document baseline logits or outputs and compare them with one reconstructed checkpoint.
- **Related code or notes:** [Project README](../README.md), [inference script](../infer.py), [quantization experiment log](../experiment-log.md).

### 2026-10-27 — Symmetric weight quantization by bit width and group size

- **Learned:** Lower bit widths and larger groups reduce storage overhead, while coarser quantization generally increases reconstruction error.
- **Tried:** Quantized all unique learned parameters in the local tiny GPT-2 checkpoint with symmetric INT4 and INT8, using group sizes 32 and 64. One float32 absolute-maximum scale was stored per group. The tied output embedding was counted once.
- **Result or question:** Across 102,702 parameters, INT4/group 32 measured MAE 0.00161767 and RMSE 0.00194273 at 5.0064 effective bits/weight. INT4/group 64 measured RMSE 0.00217753 at 4.5069 bits/weight. INT8/group 32 measured RMSE 0.00012532 at 9.0064 bits/weight; INT8/group 64 measured RMSE 0.00014030 at 8.5069 bits/weight. Larger groups reduced scale overhead and increased RMSE in both comparisons.
- **Next:** Compare errors by layer, then compare model logits and generated outputs against the original checkpoint.
- **Related code or notes:** [Quantization comparison script](../quantize_weights.py).

### 2026-10-27 — Compare at a similar storage rate

- **Learned:** Comparing quantizers at the same nominal bit width can be misleading when group scales add storage. Match effective bits per weight instead.
- **Tried:** Compared symmetric INT4 with group size 32 against INT5 with group size 512 on the same 102,702 unique tiny-GPT-2 parameters, counting one float32 scale per group.
- **Result or question:** INT4/group 32 used 5.0064 effective bits/weight with RMSE 0.00194273. INT5/group 512 used 5.0701 bits/weight with RMSE 0.00170476. The latter used 1.3% more storage and had 12.2% lower weight reconstruction RMSE. Storage rates are close, not identical; these results do not measure logits or generated text.
- **Next:** Compare weight reconstruction error by layer, then measure model-output differences.
- **Related code or notes:** [Quantization comparison script](../quantize_weights.py), [fair-rate comparison visual](../../../../.codex/visualizations/2026/10/26/01a178f6-2521-7691-8de8-75aee0b7990b/fair-quantization-baseline.html).

