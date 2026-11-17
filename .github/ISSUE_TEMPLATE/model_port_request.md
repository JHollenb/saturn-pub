---
name: Model port request
about: Ask for a model family to be supported by saturn-pub adapters or the mdb engine
title: "[port] "
labels: model-port
---

**Family**

- Hugging Face `model_type` (from the checkpoint's `config.json`):
- An example checkpoint (name and size):
- Where you want it: saturn-pub (run, fork, capture, replay) / mdb (port-level stepping) / both

**How it differs from a registered family**

saturn-pub registers gpt2, gpt_neox, phi, llama, mistral, mixtral, gemma, gemma2, qwen2, qwen3 and mamba; mdb's port program covers qwen2, qwen3, llama and mistral. Name the nearest one and what is different: activation, norms (placement, per-head q/k norm), attention (sliding window, softcap, scaling), biases, RoPE variant or scaling, tied or scaled embeddings, MoE routing, anything computed outside the decoder layer.

**What you tried**

The refusal message you got (adapters name the rule a config breaks), or the error.

**Will you contribute it?**

See "Adding a model family" in CONTRIBUTING.md. Ports are developed and tested on random tiny CPU models; the maintainers run the real-weight validation.
