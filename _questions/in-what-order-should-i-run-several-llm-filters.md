---
title: "In what order should I run several LLM filters on the same data?"
section: database-systems
order: 9
description: "Run first the filters that reject the most rows for the least cost, so later filters see fewer rows. In How to Cost Your AI-Powered Filters, we rank each later filter by its cost per row divided by the fraction it rejects, following Hellerstein and Stonebraker, and also try each filter first. For three filters over 5,000 IMDB reviews, the chosen order scored 6.87 s compared with 7.30 s for the worst candidate."
sources:
  - title: "How to Cost Your AI-Powered Filters"
    url: https://fsdatalab.github.io/blog/ai-filter-cost-estimates/
asked:
  - https://community.databricks.com/t5/generative-ai/ai-query-parallel-calls/td-p/146809
---
