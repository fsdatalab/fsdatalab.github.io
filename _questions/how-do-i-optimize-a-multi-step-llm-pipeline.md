---
title: "How do I optimize a multi-step LLM pipeline for both accuracy and cost, not just one prompt at a time?"
section: database-systems
order: 5
description: "Use MOAR, our optimizer for DocETL: it searches over whole-pipeline rewrites with over 30 rewrite rules and a multi-armed bandit, because operators interact and tuning each alone can give a worse plan. Across six workloads, MOAR reached 27% higher accuracy than the next-best optimizer, and matched that optimizer's best accuracy at 55% of its cost."
sources:
  - title: "Multi-Objective Agentic Rewrites for Unstructured Data Processing (VLDB 2026)"
    url: https://arxiv.org/abs/2512.02289
asked:
  - https://www.reddit.com/r/LangChain/comments/1rtqraf/how_are_you_handling_llm_costs_in_production/
  - https://www.reddit.com/r/mlops/comments/1rcp0ad/broke_down_our_32k_llm_bill_68_was_preventable/
---
