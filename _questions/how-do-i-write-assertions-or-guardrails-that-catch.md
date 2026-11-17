---
title: "How do I write assertions or guardrails that catch bad LLM outputs in production?"
section: database-systems
order: 22
description: "Start from the mistakes you already fixed: each instruction you added to your prompt to stop a mistake is a candidate check. Our SPADE system does this automatically: it mines a pipeline's history of prompt edits to write candidate assertion functions, then picks a small set that meets coverage and accuracy targets. Across nine real pipelines, SPADE needed 14% fewer assertions and had 21% fewer false failures than simpler baselines. It is offered in LangSmith."
guides: [llm-evals]
sources:
  - title: "SPADE: Synthesizing Data Quality Assertions for Large Language Model Pipelines (VLDB 2024)"
    url: https://arxiv.org/abs/2401.03038
asked:
  - https://www.reddit.com/r/LLMDevs/comments/1sq1xox/deterministic_vs_probabilistic_guardrails_for/
  - https://www.reddit.com/r/LLMDevs/comments/1szuqv7/how_are_people_making_llm_outputs_reliable_enough/
  - https://www.reddit.com/r/LLMDevs/comments/1rwgwok/how_are_you_validating_llm_behavior_before/
---
