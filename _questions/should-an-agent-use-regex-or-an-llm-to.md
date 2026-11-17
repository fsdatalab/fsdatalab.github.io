---
title: "Should an agent use regex or an LLM to pull values like dates out of free-text columns?"
section: database-systems
order: 19
description: "Regex works for a few fixed formats and breaks on varied natural language. In DAB, all three agents we tested defaulted to regex for text extraction. On the patents dataset, dates appear as \"dated 5th March 2019\" and \"March the 18th, 2019\"; no agent's regex covered them, and every agent scored 0%. A regex for MALE also matched inside FEMALE."
sources:
  - title: "Can AI Agents Answer Your Data Questions? A Benchmark for Data Agents (Preprint)"
    url: https://arxiv.org/abs/2603.20576
asked:
  - https://reddit.com/r/OpenAI/comments/1r88f0g/we_need_to_talk_about_using_opus_46_for_tasks/
---
