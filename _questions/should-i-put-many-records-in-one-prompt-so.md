---
title: "Should I put many records in one prompt so the LLM sees context when classifying?"
section: database-systems
order: 11
description: "It can save cost, but it didn't improve accuracy in our tests, because long inputs hurt LLM quality. In HoldUp, batching records into one prompt averaged 43% classification accuracy, compared with 66% one row at a time, and asking the LLM to write a rubric from a sample reached 59%. HoldUp instead gives the LLM small random samples many times, and reached 80%."
guides: [document-processing]
sources:
  - title: "Semantic Data Processing with Holistic Data Understanding (Preprint)"
    url: https://arxiv.org/abs/2604.02655
asked:
  - https://reddit.com/r/MachineLearning/comments/1q5e8k3/d_llms_for_classification_task/
  - https://arxiv.org/abs/2604.03684
---
