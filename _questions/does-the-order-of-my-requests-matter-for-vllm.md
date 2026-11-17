---
title: "Does the order of my requests matter for vLLM prefix caching in an offline batch job?"
section: inference
order: 6
description: "Yes. Quail plans KV reuse from the query, so you don't order requests by hand. In our AI.EXTRACT experiments, naive vLLM sent each document's questions in random order and took 2,079 s on CUAD-16, which asks 41 questions of each contract. If you use vLLM directly, sort the requests so a document's questions sit together; in the same experiments, that removed most of the waste."
how: "Prefix caching keeps the KV of recently seen prompt prefixes in GPU memory, so a later request with the same prefix, such as the same document, can skip recomputing it. Memory is limited, so the cache evicts entries, usually the least recently used. If a document's questions are spread across the batch, its KV is often evicted before the next question arrives and must be computed again. Quail plans KV reuse from the query, so it knows which requests share a document."
sources:
  - title: "Copy That? Faster AI-Powered Extraction by Drafting from the Document"
    url: https://fsdatalab.github.io/blog/ai-extract/
  - title: "Building an Ultra-High Throughput AI-SQL Engine"
    url: https://fsdatalab.github.io/blog/introducing-quail/
asked:
  - https://discuss.vllm.ai/t/suggestion-to-improve-inferencing-speed/2080
  - https://github.com/vllm-project/vllm/issues/23083
---
