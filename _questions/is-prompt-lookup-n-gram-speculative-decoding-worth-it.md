---
title: "Is prompt lookup (n-gram) speculative decoding worth it for extraction or other tasks that copy from the input?"
section: inference
order: 4
description: "It works best when the output copies the input, as in extraction, and helps little for freely written text. In Quail's AI.EXTRACT, every answer is a substring of the document, so we copy the draft from the first match in the document and learn, per field, how often drafts are accepted and how long answers run. Quail was 1.8x faster on average than vLLM with n-gram speculation and sorted prompts, and 4.7x faster than naive vLLM."
how: "Prompt lookup decoding drafts tokens by finding the text written so far in the prompt and guessing that what followed it there comes next. In extraction, every answer is copied from the document, so these guesses are usually right. Quail also constrains decoding so the answer must stay a substring of the document, and it sets the draft length for each request from what it has learned about that field."
sources:
  - title: "Copy That? Faster AI-Powered Extraction by Drafting from the Document"
    url: https://fsdatalab.github.io/blog/ai-extract/
asked:
  - https://github.com/hybridindie/rec/issues/10
  - https://github.com/jundot/omlx/issues/2673
---
