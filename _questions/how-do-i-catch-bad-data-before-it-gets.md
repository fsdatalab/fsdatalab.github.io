---
title: "How do I catch bad data before it gets into a retraining run without drowning in false alerts?"
section: database-systems
order: 24
description: "We recommend validating each new batch or partition against recent history and tuning the checks for precision, since per-feature thresholds catch most problems but alert too often. Our GATE system summarizes each data partition with quality metrics and flags corrupted partitions before retraining. At a large tech company, GATE gave a 2.1x average precision gain over prior work."
sources:
  - title: "Automatic and Precise Data Validation for Machine Learning (CIKM 2023)"
    url: https://dl.acm.org/doi/10.1145/3583780.3614786
asked:
  - https://www.reddit.com/r/mlops/comments/1ta021b/how_are_you_guys_catching_upstream_schema_drift/
  - https://www.reddit.com/r/mlops/comments/1t7y4j1/how_do_you_actually_catch_when_your_production/
---
