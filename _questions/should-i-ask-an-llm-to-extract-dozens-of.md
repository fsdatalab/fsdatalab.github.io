---
title: "Should I ask an LLM to extract dozens of fields from a long document in one prompt, or split it up?"
section: database-systems
order: 4
description: "For many unrelated fields over long documents, split the work: one long prompt with every field tends to miss some of them. In LLM-Powered Proactive Data Systems (IEEE Data Engineering Bulletin 2025), we argue the system should do this rewriting itself: break \"extract fields f1 to fn\" into one focused prompt per field, send the LLM only the relevant part of a long document, and leave arithmetic such as averages to SQL."
guides: [document-processing]
sources:
  - title: "LLM-Powered Proactive Data Systems (IEEE Data Engineering Bulletin 2025)"
    url: https://arxiv.org/abs/2502.13016
asked:
  - https://reddit.com/r/PromptEngineering/comments/1jllcvf/extracting_thousands_of_knowledge_points_from_pdf/
---
