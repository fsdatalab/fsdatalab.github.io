---
title: "How do I run an LLM over a table in my data warehouse (Snowflake, BigQuery, Databricks) on my own GPUs?"
section: database-systems
order: 32
description: "Use Quail, our open-source engine for AI functions in SQL. Read the table as an Apache Iceberg table with PyIceberg, or export the table to Parquet, CSV, or JSON files, and register the table in a Quail session. Quail then runs a SQL query with AI functions, such as AI.IF, on an open source LLM on your GPU or on a GPU rented from Modal. Two AI filters over 100,000 IMDB reviews cost $0.37 on one H100 at Modal's price, and the same filters would cost about $1.75 at GPT-5 nano prices."
how: "Snowflake, BigQuery, and Databricks can each store a table in Apache Iceberg format, an open table format that other engines can read. PyIceberg loads the Iceberg table as an Arrow table, and session.register makes the Arrow table available to SQL. Quail runs each AI function with the open source LLM you choose, so you pay for GPU time instead of for each call to a hosted model."
guides: [ai-sql]
sources:
  - title: "When to use Quail (with a complete Iceberg example)"
    url: https://fsdatalab.github.io/quail/docs/when-to-use
  - title: "Building an Ultra-High Throughput AI-SQL Engine"
    url: https://fsdatalab.github.io/blog/introducing-quail/
---
