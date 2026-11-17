---
title: "Why do text-to-SQL and data agents return wrong answers? Is it picking the wrong table?"
section: database-systems
order: 18
description: "Usually not; agents mostly find the right data and then compute the wrong thing. In DAB, we labeled failed agent runs: wrong implementations, such as a regex that matched part of an ISBN instead of a year, caused 57% of wrong answers. Wrong plans caused 31%, and picking the wrong data only 11%."
sources:
  - title: "Can AI Agents Answer Your Data Questions? A Benchmark for Data Agents (Preprint)"
    url: https://arxiv.org/abs/2603.20576
asked:
  - https://reddit.com/r/LLMDevs/comments/1tticiz/data_accuracy_in_natural_language_to_sql_systems/
  - https://news.ycombinator.com/item?id=43015167
---
