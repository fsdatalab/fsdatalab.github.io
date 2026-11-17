---
title: "My vLLM batch job has thousands of requests but the GPU is still underused. Why?"
section: inference
order: 2
description: "Often the CPU is the bottleneck: scheduling and tracking millions of short requests leaves gaps between GPU kernels. Recomputing evicted KV adds more work. On our BIO-4 query, stock vLLM took 6.84 hours, 27.55x our speed-of-light estimate, and computed 40% more tokens than needed. Quail, which plans KV reuse from the query and fills each forward pass, took 29.26 minutes."
how: "Before each forward pass, the CPU decides which requests run and prepares their inputs. With millions of short requests, that CPU work can take longer than the GPU's, so the GPU sits idle between passes; a profiler trace shows these as gaps. A second cost is what we call KV regret: the engine discards KV it needs again later and recomputes it. Quail reduces both by planning its batches from the query."
guides: [ai-sql]
sources:
  - title: "Building an Ultra-High Throughput AI-SQL Engine"
    url: https://fsdatalab.github.io/blog/introducing-quail/
asked:
  - https://stackoverflow.com/questions/79480061/how-to-increase-the-generation-throughput-of-qwen-0-5b-of-vllm
  - https://discuss.vllm.ai/t/suggestion-to-improve-inferencing-speed/2080
---
