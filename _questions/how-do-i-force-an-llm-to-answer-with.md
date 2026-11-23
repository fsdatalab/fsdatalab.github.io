---
title: "How do I force an LLM to answer with only one of my class labels?"
section: inference
order: 7
description: "Write the query with Quail's AI.CLASSIFY, which restricts the model to your labels: it builds a trie of the label tokens and runs the model only where labels branch apart. With 10 support-ticket labels, that averages 1.5 decode requests per document, and on 12 classification queries Quail was 1.8x faster per query than vLLM generating and parsing the label. Without Quail, you can use structured or constrained decoding in vLLM or an LLM API to limit the output to your labels."
how: "A trie, or prefix tree, stores each label as a path of tokens; labels that start with the same words share the start of their path. Quail only asks the model to choose where paths branch apart. For example, with the labels \"shipping delay\", \"shipping damage\", and \"shipping address change\", once the model picks \"shipping\" and then \"address\", the word \"change\" is already determined, so Quail skips that decode step. Constrained decoding in vLLM or an API also limits output to valid labels, but it typically still decodes every token."
sources:
  - title: "To Classify, Try a Trie Tree"
    url: https://fsdatalab.github.io/blog/ai-classification/
asked:
  - https://github.com/vllm-project/vllm/issues/288
  - https://discuss.vllm.ai/t/extract-class-log-probabilities-from-llm-classifier-using-async-vllm-and-aws-lmi-containers/543
---
