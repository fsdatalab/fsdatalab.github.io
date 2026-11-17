---
title: "What is the fastest, cheapest way to run an LLM over every row of a large dataset?"
section: inference
order: 1
description: "Use Quail, our open-source engine for AI functions in SQL: it runs an open model as an offline batch job, keeps the GPU full, and plans the query and the inference together to reuse work across rows. Two AI filters over all 100,000 IMDB reviews cost $0.37 on one H100 at Modal's price, including model startup. The same workload at GPT-5 nano prices would cost about $1.75."
sources:
  - title: "Building an Ultra-High Throughput AI-SQL Engine"
    url: https://fsdatalab.github.io/blog/introducing-quail/
asked:
  - https://www.reddit.com/r/LocalLLaMA/comments/1ohgf9t/batch_inference_locally_on_4080/
  - https://discuss.ray.io/t/offline-inference-vllm-map-batches-vs-build-llm-processor/23369
---
