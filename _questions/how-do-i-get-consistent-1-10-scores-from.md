---
title: "How do I get consistent 1-10 scores from an LLM across many items?"
section: database-systems
order: 12
description: "Scoring each item alone on an absolute scale drifts; comparing items gives steadier results. In HoldUp, we group similar records, ask the LLM which group should score higher, and sort the groups with an integer linear program. On 5 scoring datasets, row-by-row scoring averaged 48% accuracy and HoldUp 64%."
sources:
  - title: "Semantic Data Processing with Holistic Data Understanding (Preprint)"
    url: https://arxiv.org/abs/2604.02655
asked:
  - https://reddit.com/r/LocalLLM/comments/1t81626/llmasjudge_scoring_is_noisier_than_i_expected/
  - https://community.openai.com/t/ensuring-gpt-consistently-follows-a-defined-scoring-system-tips-and-best-practices/454333
---
