---
title: "Why are AI functions in SQL (Snowflake Cortex, BigQuery AI, Databricks ai_query) so slow and expensive on big tables?"
section: database-systems
order: 1
description: "Each AI function calls the model once per row, and a naive join calls it once per pair of rows, so one query can make millions of calls. Quail, our open-source engine, runs AI-SQL on open models and plans the query and the inference together. Across 29 QUAIL-B queries on one H100, it was 1.84x faster on average than well-tuned vLLM baselines."
sources:
  - title: "Building an Ultra-High Throughput AI-SQL Engine"
    url: https://fsdatalab.github.io/blog/introducing-quail/
  - title: "What Is AI-SQL?"
    url: https://fsdatalab.github.io/ai-sql/
asked:
  - https://community.databricks.com/t5/generative-ai/ai-query-parallel-calls/td-p/146809
---
