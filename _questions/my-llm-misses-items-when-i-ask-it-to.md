---
title: "My LLM misses items when I ask it to find every instance of something in a long document. Should I split the prompt or the document?"
section: database-systems
order: 3
description: "Usually split: models miss instances in long inputs even when the input fits in the context window. DocETL, our open-source system, rewrites pipelines to decompose the document or the operation; our DocETL paper uses this exact case, finding every force majeure or indemnification clause in long contracts. In MOAR, our later optimizer for DocETL, one example rewrite turns one compound filter into two simpler filters."
guides: [document-processing]
sources:
  - title: "DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing (VLDB 2025)"
    url: https://arxiv.org/abs/2410.12189
  - title: "Multi-Objective Agentic Rewrites for Unstructured Data Processing (VLDB 2026)"
    url: https://arxiv.org/abs/2512.02289
asked:
  - https://www.reddit.com/r/Bard/comments/1pv5phb/document_extraction_accuracy_and_recall_tips/
  - https://www.reddit.com/r/Bard/comments/1t1xl5m/how_to_stop_llms_being_lazy_on_api/
  - https://community.openai.com/t/trouble-extracting-all-information-from-long-context-document/739300
---
