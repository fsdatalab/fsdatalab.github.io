---
title: "Why is my LLM classifier inaccurate when I label records one at a time, even with a good prompt?"
section: database-systems
order: 10
description: "Often the right label depends on the rest of the dataset, and a model that sees one row can't know that. For example, a ticket can be urgent next to minor issues and routine next to security bugs. In HoldUp, labeling rows one at a time averaged under 70% accuracy on 10 classification datasets. HoldUp clusters related records with the LLM and labels the clusters jointly, reaching 80% on average and up to 33% higher."
sources:
  - title: "Semantic Data Processing with Holistic Data Understanding (Preprint)"
    url: https://arxiv.org/abs/2604.02655
asked:
  - https://reddit.com/r/MachineLearning/comments/1q5e8k3/d_llms_for_classification_task/
  - https://community.openai.com/t/ensuring-gpt-consistently-follows-a-defined-scoring-system-tips-and-best-practices/454333
---
