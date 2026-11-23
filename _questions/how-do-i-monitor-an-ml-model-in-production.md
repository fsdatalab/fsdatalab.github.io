---
title: "How do I monitor an ML model in production when labels arrive late or not at all?"
section: database-systems
order: 25
description: "Track inputs and predictions right away, and compute accuracy-style metrics as labels arrive, knowing each window is incomplete until then. In our position paper Rethinking Streaming Machine Learning Evaluation, we argue that tracking accuracy alone on unbounded, time-ordered data misses problems, partly because labels arrive late, and we recommend additional metrics. The Modern ML Monitoring Mess series works through why."
sources:
  - title: "Rethinking Streaming Machine Learning Evaluation (ICLR 2022: Workshop on ML Evaluation Standards)"
    url: https://arxiv.org/abs/2205.11473
  - title: "The Modern ML Monitoring Mess: Rethinking Streaming Evaluation (1/4)"
    url: https://www.sh-reya.com/blog/rethinking-ml-monitoring-1/
asked:
  - https://www.reddit.com/r/MLQuestions/comments/1s65sjp/best_way_to_monitor_online_ml_models_in_production/
  - https://stackoverflow.com/questions/79337323/azure-machine-learning-monitoring-mapping-ground-truth-data-for-model-performa
  - https://www.reddit.com/r/mlops/comments/1t7y4j1/how_do_you_actually_catch_when_your_production/
---
