---
title: "How do I get a probability for each label from an LLM classifier, when labels are more than one token?"
section: inference
order: 8
description: "Quail's AI.CLASSIFY can return a probability for every label. Its trie_tree method scores each label only up to the word where it branches away from the other labels, then normalizes over all labels, so long labels aren't penalized. Quail also has a method that puts a letter before each label, so each label is one token, and it picks the cheaper method for your model and labels."
how: "A label like \"refund request\" spans several tokens, and the model gives a probability for each one. Multiplying all of them gives longer labels lower scores even when the model favors them. trie_tree scores each label only up to the point where it splits from the other labels, since later tokens add no information, then divides by the sum over all labels so the probabilities add up to 1."
sources:
  - title: "To Classify, Try a Trie Tree"
    url: https://fsdatalab.github.io/blog/ai-classification/
asked:
  - https://discuss.vllm.ai/t/can-i-get-log-prob-with-classify/1632
  - https://github.com/sgl-project/sglang/issues/523
  - https://community.openai.com/t/classification-with-generative-models/600277
---
