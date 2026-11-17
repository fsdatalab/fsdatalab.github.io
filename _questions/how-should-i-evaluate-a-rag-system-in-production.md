---
title: "How should I evaluate a RAG system in production?"
section: database-systems
order: 30
description: "We recommend evaluating retrieval and generation separately, using failures you find in real traces. For pipelines with several LLM steps, Data Flywheels for LLM Applications recommends placing validators at each step and computing per-step accuracy, so you can see where errors compound."
guides: [llm-evals]
sources:
  - title: "How should I approach evaluating my RAG system? (AI Evals FAQ, Hamel Husain and Shreya Shankar)"
    url: https://hamel.dev/blog/posts/evals-faq/how-should-i-approach-evaluating-my-rag-system.html
  - title: "Data Flywheels for LLM Applications"
    url: https://www.sh-reya.com/blog/ai-engineering-flywheel/
asked:
  - https://datascience.stackexchange.com/questions/134522/in-production-how-do-you-evaluate-the-quality-of-the-response-generated-by-a-ra
  - https://stackoverflow.com/questions/79488543/what-to-include-in-context-precision-recall-for-rag-llm-evaluation
  - https://news.ycombinator.com/item?id=43588464
---
