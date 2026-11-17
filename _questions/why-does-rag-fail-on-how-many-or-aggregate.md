---
title: "Why does RAG fail on \"how many\" or aggregate questions across a whole document collection?"
section: database-systems
order: 14
description: "Retrieval returns a few chunks, but a count needs a value from every document. ZenDB answers SQL queries, including aggregations, by locating the right section in each document through its template. On three real document collections, it beat RAG baselines by up to 61% in precision and 80% in recall, at slightly higher cost."
sources:
  - title: "Querying Templatized Document Collections with Large Language Models (ICDE 2025)"
    url: https://www.computer.org/csdl/proceedings-article/icde/2025/360300c422/26FZAUkb1ra
asked:
  - https://reddit.com/r/LLMDevs/comments/1u108nr/why_topk_retrieval_structurally_cant_answer/
  - https://reddit.com/r/AI_Agents/comments/1sefvp5/is_this_a_dumb_idea/
  - https://community.openai.com/t/rag-is-failing-when-the-number-of-documents-increase/578498
---
