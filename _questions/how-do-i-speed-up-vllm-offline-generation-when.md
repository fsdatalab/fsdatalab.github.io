---
title: "How do I speed up vLLM offline generation when I run several prompts over each long document, such as summaries and field lists?"
section: inference
order: 10
requires: /blog/ai-map/
sitemap: false
description: "Keep each document's prompts next to each other so prefix caching reuses its KV, and share the document's KV reads across those prompts in attention. In our AI.MAP post, sorting alone cut wasted tokens 24-81x compared with random order. Quail, which also schedules by learned output lengths and reads each document's KV once per batch, reached 2.0-6.2x the throughput of naive vLLM with Qwen3 4B on one H100."
sources:
  - title: "Why Summarizing a Dataset with an LLM Is Slow, and How We Made It 6x Faster"
    url: https://fsdatalab.github.io/blog/ai-map/
asked:
  - https://github.com/vllm-project/vllm/issues/23083
  - https://discuss.vllm.ai/t/suggestion-to-improve-inferencing-speed/2080
---
