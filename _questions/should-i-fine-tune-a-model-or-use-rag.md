---
title: "Should I fine-tune a model or use RAG to give an LLM my company's knowledge?"
section: database-systems
order: 28
description: "We recommend starting with RAG for new knowledge, and fine-tuning only after you have shown that prompting and retrieval fall short. In What We've Learned From a Year of Building with LLMs, we cite studies where RAG beat fine-tuning on both known and new facts. Retrieval indexes are also cheaper to update, easy to purge of bad documents, and can be split so each customer sees only its own data."
sources:
  - title: "What We've Learned From a Year of Building with LLMs (O'Reilly Radar)"
    url: https://applied-llms.org
asked:
  - https://reddit.com/r/Rag/comments/1sdyqdp/rag_vs_finetuning_for_business_ai_when_does_each/
  - https://news.ycombinator.com/item?id=40687491
---
