---
title: "How can I estimate how long LLM inference over a dataset will take on a given GPU?"
section: inference
order: 9
description: "Our post How to Cost Your AI-Powered Filters works through a roofline estimate: count the FLOPs and the bytes moved, then divide by the GPU's peak compute and memory bandwidth. For Qwen3-4B on an H100, one true/false filter over 5,000 IMDB reviews comes out to 6.64 seconds, and a forward pass needs only about 296 tokens to become compute-bound. Quail uses these estimates in its cost model."
how: "Every step of a forward pass does arithmetic and moves bytes between GPU memory and the compute units. The roofline model estimates a step's time as the slower of the two: arithmetic divided by peak compute, or bytes divided by memory bandwidth. A step is compute-bound when the arithmetic is slower and memory-bound otherwise. Processing more tokens per forward pass reuses the same weights for more work, which is why about 296 tokens per pass is enough to pass the H100's FP8 ridge point."
sources:
  - title: "How to Cost Your AI-Powered Filters"
    url: https://fsdatalab.github.io/blog/ai-filter-cost-estimates/
asked:
  - https://stackoverflow.com/questions/78691343/mixtral-8x7b-am-i-running-it-wrong
  - https://news.ycombinator.com/item?id=47136904
---
