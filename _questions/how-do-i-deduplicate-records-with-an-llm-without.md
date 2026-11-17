---
title: "How do I deduplicate records with an LLM without missing too many true matches?"
section: database-systems
order: 15
description: "Asking an LLM about each pair alone tends to give high precision and low recall; using the other answers to fix mistakes helps. In Revisiting Prompt Engineering via Declarative Crowdsourcing (CIDR 2024), gpt-3.5-turbo matched citation pairs with 0.952 precision but 0.503 recall. Marking two records as duplicates when the LLM linked them through other records raised F1 by over 6%."
sources:
  - title: "Revisiting Prompt Engineering via Declarative Crowdsourcing (CIDR 2024)"
    url: https://www.cidrdb.org/cidr2024/papers/p67-parameswaran.pdf
asked:
  - https://reddit.com/r/LangChain/comments/1sdeyw8/using_ai_to_untangle_10000_property_titles_in/
  - https://reddit.com/r/Rag/comments/1te3fb0/graphrag_entity_deduplication/
---
