---
title: "How do I add a summary of each user's history and feedback to my prompts without slowing down responses?"
section: database-systems
order: 16
description: "We recommend updating the summary in the background and letting requests read the latest finished version, even if it is slightly stale. Our Motion system (SIGMOD 2024 demo) treats a prompt's parts as views that are updated as new feedback arrives, under a weak consistency model. In our shopping demo, a fresh GPT-4 summary could take over thirty seconds, while Motion returned recommendations in 1.5 seconds."
sources:
  - title: "Building Reactive Large Language Model Pipelines with Motion (SIGMOD 2024 (Demo))"
    url: https://dl.acm.org/doi/10.1145/3626246.3654734
asked:
  - https://guide.mem0.ai/faq/we-are-getting-killed-in-response-latency-because-of-how-much-context-we-load-per-request-which-plat-1977a6
  - https://dev.to/mrbolo/how-we-built-chat-memory-that-actually-works-lessons-from-shipping-to-100k-users-24cd
---
