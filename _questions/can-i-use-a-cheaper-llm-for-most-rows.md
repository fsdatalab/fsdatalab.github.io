---
title: "Can I use a cheaper LLM for most rows without losing accuracy?"
section: database-systems
order: 6
description: "Use BARGAIN, our method for LLM-powered data processing with guarantees: a cheap model handles a row when it can be trusted, and BARGAIN decides when that is with adaptive sampling and statistical estimates, with guarantees on accuracy, precision, or recall. Across 8 datasets, BARGAIN reduced cost by up to 86% more than prior methods."
how: "A model cascade sends each row to a cheap model first and keeps its answer only when the model's confidence is above a threshold; the other rows go to the expensive model. The hard part is the threshold: set it too low and accuracy drops, too high and you save little. BARGAIN checks the cheap model on an adaptively chosen sample of rows and uses statistical estimates to pick the threshold that saves the most while still meeting the accuracy, precision, or recall target you set."
sources:
  - title: "Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees (SIGMOD 2026)"
    url: https://arxiv.org/abs/2509.02896
asked:
  - https://www.reddit.com/r/AI_Agents/comments/1rtvqkn/how_do_large_ai_apps_manage_llm_costs_at_scale/
  - https://www.reddit.com/r/LangChain/comments/1rtqraf/how_are_you_handling_llm_costs_in_production/
---
