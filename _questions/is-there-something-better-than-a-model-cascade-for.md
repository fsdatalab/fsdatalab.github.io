---
title: "Is there something better than a model cascade for LLM classification over documents?"
section: database-systems
order: 7
description: "Task cascades, from our SIGMOD 2026 paper, vary more than the model: an LLM agent proposes simpler or decomposed tasks and the relevant parts of each document, and we assemble them into a cascade, optionally with an accuracy guarantee. At a 90% target accuracy over eight tasks, task cascades cut cost by 36% on average compared with model cascades."
how: "A model cascade varies only the model. A task cascade can also vary the task: a cheap model might answer an easier related question, or read only the part of the document that matters, and settle most rows that way. An LLM agent proposes these simpler tasks and document parts, and the cascade tries them from cheapest to most expensive, sending only the rows that remain uncertain to the full task."
guides: [document-processing]
sources:
  - title: "Task Cascades for Efficient Unstructured Data Processing (SIGMOD 2026)"
    url: https://arxiv.org/abs/2601.05536
asked:
  - https://www.reddit.com/r/mlops/comments/1rcp0ad/broke_down_our_32k_llm_bill_68_was_preventable/
  - https://news.ycombinator.com/item?id=38968751
---
