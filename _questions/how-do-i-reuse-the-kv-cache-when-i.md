---
title: "How do I reuse the KV cache when I ask many questions about the same long document?"
section: inference
order: 5
description: "Quail handles this for you: it plans KV reuse from the query and keeps a document's KV only while a later operator will read it. On AGENT-5, which runs three classifications over the same traces, 64% of the tokens vLLM computed were recomputations. With a general engine such as vLLM, put the document first and the question last, send a document's questions together, and turn on prefix caching; an LRU cache can still evict the document before the next question arrives."
how: "When an LLM reads a document, it stores intermediate results for every token, called the KV cache, so it doesn't redo that work when it writes the answer. If you ask a second question about the same document, the engine can reuse that KV instead of reading the document again, but only if it's still in GPU memory. Quail knows from the query which operators will read each document, so it keeps the KV exactly until the last one runs."
sources:
  - title: "To Classify, Try a Trie Tree"
    url: https://fsdatalab.github.io/blog/ai-classification/
  - title: "Building an Ultra-High Throughput AI-SQL Engine"
    url: https://fsdatalab.github.io/blog/introducing-quail/
asked:
  - https://github.com/vllm-project/vllm/issues/23083
  - https://news.ycombinator.com/item?id=41350076
---
