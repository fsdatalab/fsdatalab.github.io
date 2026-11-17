---
title: "Do long-context models make RAG obsolete?"
section: database-systems
order: 29
description: "No. Even with a 10M-token window you still have to choose what goes in it. In What We've Learned From a Year of Building with LLMs (O'Reilly Radar), we note there was little evidence that models reason well over very long inputs beyond needle-in-a-haystack tests. Irrelevant context can distract the model, and inference time grows with context length, so reading everything for every question costs more."
sources:
  - title: "What We've Learned From a Year of Building with LLMs (O'Reilly Radar)"
    url: https://applied-llms.org
asked:
  - https://news.ycombinator.com/item?id=47668359
  - https://reddit.com/r/Rag/comments/1sdyqdp/rag_vs_finetuning_for_business_ai_when_does_each/
---
