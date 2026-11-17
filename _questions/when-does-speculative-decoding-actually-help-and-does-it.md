---
title: "When does speculative decoding actually help, and does it still help at large batch sizes?"
section: inference
order: 3
description: "Speculative decoding helps when decoding is memory-bound and the drafts are cheap and usually right; at large batches the extra verify tokens cost real compute, so the draft length matters. In our AI.EXTRACT work in Quail, we draft by copying tokens from the document and pick a draft length for each request as the query runs. The best fixed length was 4 on some queries and 32 on others, and a wrong fixed length cost up to 20%."
how: "An LLM normally writes one token per forward pass, and each pass reads all the model weights and the document's KV from GPU memory, so decoding is limited by memory reads, not arithmetic. Speculative decoding guesses the next few tokens, called the draft, and checks them all in one forward pass; if the model agrees, you get several tokens for the cost of one. Long drafts pay off when they're usually right; when they're often wrong, the extra checked tokens are wasted work. Quail learns, as the query runs, how often drafts are accepted and how long answers are, and picks a draft length for each request."
sources:
  - title: "Copy That? Faster AI-Powered Extraction by Drafting from the Document"
    url: https://fsdatalab.github.io/blog/ai-extract/
asked:
  - https://www.reddit.com/r/LocalLLaMA/comments/1gzm93o/speculative_decoding_just_landed_in_llamacpps/
  - https://github.com/vllm-project/vllm/issues/49548
  - https://github.com/vllm-project/vllm/issues/65312
---
