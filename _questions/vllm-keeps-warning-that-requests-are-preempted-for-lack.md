---
title: "vLLM keeps warning that requests are preempted for lack of KV cache space. Does it matter for a batch job?"
section: inference
order: 11
requires: /blog/ai-map/
sitemap: false
description: "Preemption happens when outputs grow past the free KV and vLLM must recompute evicted requests later; for a batch job it costs less than it looks. In our AI.MAP post, Quail reserved KV for each request's expected output length and evicted nothing, while reserving only for prompts caused 103-884 evictions per query. Removing them made queries at most 1.09x faster."
sources:
  - title: "Why Summarizing a Dataset with an LLM Is Slow, and How We Made It 6x Faster"
    url: https://fsdatalab.github.io/blog/ai-map/
asked:
  - https://discuss.vllm.ai/t/vllm-latency-ttft-spike-on-load-tesing/1396
  - https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html
---
