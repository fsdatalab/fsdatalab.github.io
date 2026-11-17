---
title: "How do I let an AI agent edit a document while I keep typing, without it overwriting my changes?"
section: interfaces
order: 8
requires: /blog/introducing-docwriter/
sitemap: false
description: "Check each agent edit against the current text before applying it, and reject edits based on text the user has since changed. In our DocWriter post, the agent's edit tool names the exact text it replaces. If that text is gone or appears more than once, the tool returns an error, and the agent rereads the document and retries or drops the edit. The user's own keystrokes always apply right away."
guides: [data-interfaces]
sources:
  - title: "Don't Make Me Wait! Enabling Concurrency in Human-Agent Collaboration"
    url: https://fsdatalab.github.io/blog/introducing-docwriter/
asked:
  - https://github.com/microsoft/vscode/issues/279589
  - https://github.com/zed-industries/zed/issues/39738
  - https://www.moment.dev/blog/collab-with-ai-is-hard
---
