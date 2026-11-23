---
title: "Can an LLM generate the test criteria or assertions for my prompt?"
section: database-systems
order: 23
description: "It can propose them, but review the list, since picking the right checks for a task is hard. Our PromptEvals dataset has 2,087 production pipeline prompts with 12,623 assertion criteria, collected from developers using our open-source tools. Open models we fine-tuned on it beat GPT-4o at generating relevant assertions by 20.93% on average."
guides: [llm-evals]
sources:
  - title: "PromptEvals: A Dataset of Assertions and Guardrails for Custom Production Large Language Model Pipelines (NAACL 2025)"
    url: https://arxiv.org/abs/2504.14738
asked:
  - https://www.reddit.com/r/AI_Agents/comments/1rp0c04/learnings_from_building_guardrails_for_ai_systems/
  - https://www.reddit.com/r/LLMDevs/comments/1rwgwok/how_are_you_validating_llm_behavior_before/
---
