---
title: "Can I just make the AI agent another CRDT peer, such as with Yjs, in my collaborative editor?"
section: interfaces
order: 9
requires: /blog/introducing-docwriter/
sitemap: false
description: "You can, but a CRDT merges characters, not intent, and agents edit from a read that may be a minute old. In our DocWriter post, an agent rewrote a sentence the user had just changed, and the CRDT left the user's words stranded next to the agent's new sentence. DocWriter instead borrows dependency checks and tentative writes from Bayou: agent edits land as suggestions that only the user can accept."
guides: [data-interfaces]
sources:
  - title: "Don't Make Me Wait! Enabling Concurrency in Human-Agent Collaboration"
    url: https://fsdatalab.github.io/blog/introducing-docwriter/
asked:
  - https://dev.to/norfolkd/when-an-ai-agent-joins-your-yjs-room-three-assumptions-break-50h8
  - https://www.sim.ai/blog/agent-as-yjs-peer
---
