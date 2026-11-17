---
title: "My RAG answers are wrong. How do I tell whether retrieval or generation is the problem?"
section: interfaces
order: 5
description: "We recommend testing the pieces separately; for example, give the model known-good context and see whether the answer improves. In RAG Without the Lag (CHI 2026 Best Paper), we studied how 12 engineers debug RAG pipelines and built RAGGY, which pairs composable RAG building blocks with an interface for fast, interactive debugging. We found the settings that matter most often need hours of preprocessing after each change, which slows debugging."
guides: [data-interfaces]
sources:
  - title: "RAG Without the Lag: Interactive Debugging for Retrieval-Augmented Generation Pipelines (CHI 2026)"
    url: https://arxiv.org/abs/2504.13587
asked:
  - https://www.reddit.com/r/Rag/comments/1s2c1ci/rag_question_retrieval_looks_correct_but_answers/
  - https://www.reddit.com/r/LangChain/comments/1skx1d6/how_is_everyone_debugging_retrieval_quality_in/
  - https://stackoverflow.com/questions/78881336/best-approach-to-evaluate-a-graph-rag-pipeline-using-metrics
---
