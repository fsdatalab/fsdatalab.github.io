---
title: "How do I use an LLM to find the top k records that best match a question?"
section: database-systems
order: 31
requires: /blog/ai-score/
description: "Ask the LLM to score each record on its own and sort by the score, instead of comparing records in pairs. In our semantic top-k post, Quail's AI.SCORE turns the TRUE and FALSE logits of Qwen3-4B into a score from 0 to 1 for each record. On three tasks, this matched or beat pairwise comparison with Quickselect on Precision@25 (for example, 96% vs. 80% on contracts) and ran 8.2x faster per task (geometric mean)."
guides: [ai-sql]
sources:
  - title: "Supporting LLM-Powered Top-k Queries in Quail"
    url: https://fsdatalab.github.io/blog/ai-score/
asked:
  - https://community.openai.com/t/retrieve-a-top-x-list-out-of-a-large-list/627588
---
