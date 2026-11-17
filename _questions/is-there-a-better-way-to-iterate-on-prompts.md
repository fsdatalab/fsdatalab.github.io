---
title: "Is there a better way to iterate on prompts when an LLM pipeline runs over many documents?"
section: interfaces
order: 3
description: "We recommend looking at outputs across many documents, noting what goes wrong, and turning those notes into prompt changes; a single example hides most failures. Our DocWrangler IDE supports this: you annotate documents and outputs in place, and an LLM turns your notes into revised prompts or suggests splitting an operation that is too hard. In our study and public deployment, users often turned open-ended operations into classifiers so they were easier to check."
sources:
  - title: "Steering Semantic Data Processing with DocWrangler (UIST 2025)"
    url: https://arxiv.org/abs/2504.14764
asked:
  - https://www.reddit.com/r/PromptEngineering/comments/1jmdm2x/software_to_support_querying_multiple_models_and/
  - https://www.reddit.com/r/ArtificialInteligence/comments/1s41acd/autoresearch_promptfoo_autoprompter_open_source/
---
