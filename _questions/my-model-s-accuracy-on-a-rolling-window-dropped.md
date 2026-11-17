---
title: "My model's accuracy on a rolling window dropped. Is the model actually getting worse?"
section: database-systems
order: 26
description: "Not necessarily: a windowed metric can move even when the model's predictive power hasn't changed. The Modern ML Monitoring Mess (part 1) gives three causes: class ratios differ across windows, windows hold different numbers of examples (fewer requests on a Sunday than a Monday), and labels arrive late, so recent predictions lack labels. We recommend checking these before triggering a retrain."
sources:
  - title: "The Modern ML Monitoring Mess: Rethinking Streaming Evaluation (1/4)"
    url: https://www.sh-reya.com/blog/rethinking-ml-monitoring-1/
asked:
  - https://www.reddit.com/r/mlops/comments/1t7y4j1/how_do_you_actually_catch_when_your_production/
  - https://www.reddit.com/r/MLQuestions/comments/1s65sjp/best_way_to_monitor_online_ml_models_in_production/
---
