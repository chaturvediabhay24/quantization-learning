# Quantization, Inference, and Deployment Learning Hub

This folder is the home for the learning plan, progress notes, and next steps. The plan focuses on language-model quantization and deployment, while building on the tiny GPT-2 playground already in this repository.

## How to use this folder

- Work through `roadmap.md` in order. Check off a task when you can explain it or demonstrate it, not just after reading about it.
- Add a dated entry to `learning-log.md` after each study session: what you learned, what you tried, what surprised you, and what you will do next.
- Keep detailed experiment results beside the relevant code. The existing [experiment log](../experiment-log.md) records the codec experiments; add a link here when a new learning project begins.
- We can update the checkboxes and notes together as each topic is completed.

## Existing work to build on

- [Project README](../README.md): local tiny GPT-2 setup, model structure, and inference walkthrough.
- [Inference implementation](../infer.py): simple autoregressive generation loop.
- [Quantization experiment log](../experiment-log.md): weight reconstruction experiments, codebooks, exceptions, and entropy coding.
- [Huffman INT4 research note](../huffman-int4-research-note.md): current research note.

The existing codec experiments mostly measure weight reconstruction. The next major step is to measure the effect on model outputs and real inference performance.

## Current focus

Start with the first unchecked task in `roadmap.md`. The recommended next milestone is **model-level evaluation**: compare the original tiny GPT-2 checkpoint with a reconstructed or quantized checkpoint by measuring logits and generated outputs, then record the result.

