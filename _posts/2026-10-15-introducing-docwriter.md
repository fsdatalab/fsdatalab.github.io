---
layout: post
title: "Don't Make Me Wait! Enabling Concurrency in Human-Agent Collaboration"
date: 2026-10-15
author: "Shreya Shankar, Aditya G. Parameswaran"
permalink: /blog/introducing-docwriter/
unlisted: true
sitemap: false
description: "DocWriter is an open-source text editor where you and your agent can write concurrently. Here, we discuss the concurrency challenges we faced in designing DocWriter."
image:
  path: /assets/blog/docwriter/social.png
  width: 2400
  height: 1260
  alt: "Introducing DocWriter: write a document concurrently with your agent, without waiting. A DocWriter snippet shows an agent suggestion with Accept and Reject buttons."
---

<aside class="tldr"><strong>TL;DR:</strong> We built <a href="https://github.com/docwriter-org/docwriter">DocWriter</a>, an open-source text editor where you and your agent can write a document <em>concurrently</em>. Think of DocWriter as "Cursor for writing," except that DocWriter is not chat-based: the agent runs ambiently in the background, "watching" you write and suggesting edits as you go, and you decide which of the agent's edits to keep. DocWriter works with your existing Claude Code or Codex subscription. <a href="https://github.com/docwriter-org/docwriter">Star us on GitHub</a>, try DocWriter, and <a href="TODO">sign up for our upcoming user study</a> (TODO create link)!</aside>

<nav class="post-toc" aria-label="Table of contents">
<strong>Contents</strong>
<ol>
  <li><a href="#1-introduction">Introduction</a></li>
  <li><a href="#2-new-challenges-that-agents-bring">New Challenges That Agents Bring</a>
    <ol>
      <li><a href="#21-how-agents-differ-from-human-writers">How Agents Differ from Human Writers</a></li>
      <li><a href="#22-some-naive-designs">Some Naive Designs</a></li>
    </ol>
  </li>
  <li><a href="#3-taking-inspiration-fromyou-guessed-it-databases">Taking Inspiration from...You Guessed It: Databases</a></li>
  <li><a href="#4-how-docwriter-handles-concurrent-edits">How DocWriter Handles Concurrent Edits</a>
    <ol>
      <li><a href="#41-the-agent-merges-on-the-flywithout-bothering-the-user">The Agent Merges on the Fly...Without Bothering the User</a></li>
      <li><a href="#42-the-user-commits-agent-editswith-interface-assistance">The User Commits Agent Edits...With Interface Assistance</a></li>
      <li><a href="#43-other-implementation-details">Other Implementation Details</a></li>
    </ol>
  </li>
  <li><a href="#5-open-questions">Open Questions</a></li>
  <li><a href="#6-conclusion">Conclusion</a></li>
</ol>
</nav>

# 1. Introduction

Interest in human-agent collaboration has grown very quickly. But most of this work is about coding and other long tasks, where the human gives one instruction and then steps away while the agent works.

Writing is different. Usually, it is not a good idea to ask an agent to write a document in such a "one-shot" manner, because people clarify their thoughts as they write. So, writing with an agent is a collaboration, with many exchanges between the human and the agent.

Today, almost all human-agent collaboration happens through chat: the user sends a message, and the agent replies or edits the document. A strawman argument against chat is that a chat box is not where people usually write documents. But people can learn to work in a chat box if truly necessary (in fact, a growing share of scientific papers is now written with AI).<sup><a href="#note-1">1</a></sup> The real problem is that chat interfaces are _inefficient_ for effective writing, where there is (ideally) a high frequency of human-agent interaction. Chat is synchronous: users have to wait for the agent to respond. Therefore, the onus is on the user to be efficient: they have to synchronize their "think time" with the agent's "think time," which is impossible. So, even generously assuming that people want to stay involved in their writing, it is not surprising that many people run out of patience and let the agent write nearly the whole thing!

The honest story behind DocWriter is a little embarrassing. Shreya wanted to write her PhD dissertation as quickly as possible, but could not figure out how to use any frontier LLM effectively in the process, without ending up with a dissertation full of slop. Therefore, we built DocWriter, an editor where a person and their agent write concurrently. Think of DocWriter as a custom harness (i.e., system prompt and tools) for a frontier agent that runs ambiently in the background, coupled with an IDE- or text-editor-like UI where the user can write normally.

In the rest of this article, we dive into the concurrency challenges we faced in designing DocWriter. We discuss:

- Why agents bring new challenges for concurrency control ([Section 2](#2-new-challenges-that-agents-bring)).
- Our inspiration from databases: Bayou, a storage system from the 1990s ([Section 3](#3-taking-inspiration-fromyou-guessed-it-databases)).
- How DocWriter handles concurrent edits, plus other implementation details ([Section 4](#4-how-docwriter-handles-concurrent-edits)).
- Open questions that we are thinking about ([Section 5](#5-open-questions)).

# 2. New Challenges That Agents Bring

In DocWriter, a user writes in a normal text editor, and an agent, e.g., Claude Code or Codex, edits the same document at the same time. The agent has three kinds of tools: it can read the document, edit the text, and write or reply to comments.

The user can call the agent through chat, as in other tools, and the user's messages wait in a queue while the agent works. DocWriter also calls the agent automatically, with a simple rule: when the user stops typing for about 3 seconds and the agent is idle, DocWriter sends the agent the changes since the agent's last turn.

Now, because the user and the agent edit the document concurrently, their edits can conflict. Imagine that a user writes the first three lines of a paper, where line 3 is a note that asks the agent for a sentence ([Figure 1a](#figure-1)). The user pauses, so the agent reads the document and starts to think. While the agent thinks, the user changes "most tools" on line 2 to "chat interfaces", and then starts line 4 ([Figure 1b](#figure-1)).

<figure id="figure-1" style="width: min(52rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/docwriter/example-doc.svg' | relative_url }}" alt="Two views of the document paper.md. (a) Three lines when the agent reads: two sentences and a note to the agent in double brackets. (b) About a minute later: line 2 now says chat interfaces instead of most tools, and the user has started line 4.">
  <figcaption>Figure 1. The running example. (a) The document when the agent reads it. Line 3 is a note to the agent. (b) The document about a minute later, before the agent writes its edits. The user has changed line 2 and started line 4.</figcaption>
</figure>

A minute later, the agent edits lines 1 to 3. But the agent read line 2 before the user changed it, so the agent's new line 2 still says "most tools". If DocWriter applies this edit, the user's change is lost ([Figure 2](#figure-2)).

<figure id="figure-2" style="width: min(52rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/docwriter/timeline.svg' | relative_url }}" alt="A timeline of the user and the agent. The agent reads, thinks for about a minute, and then edits lines 1 to 3. While it thinks, the user changes line 2 and writes line 4.">
  <figcaption>Figure 2. While the agent thinks, the user changes "most tools" to "chat interfaces" on line 2. The agent's edit to line 2 is based on its old read.</figcaption>
</figure>

So how should DocWriter handle the agent's edits? We first discuss why an agent should be treated differently from a human writer in concurrency control ([Section 2.1](#21-how-agents-differ-from-human-writers)). Then, we walk through a few naive concurrency control ideas and their problems ([Section 2.2](#22-some-naive-designs)).

## 2.1 How Agents Differ from Human Writers

Real-time collaborative editors, such as Google Docs, let multiple writers edit the same document simultaneously. Each writer maintains a local copy. When a writer types, the change appears immediately, without waiting for approval from a server or other writers. The editor then sends the edit to the other copies. If two writers edit concurrently, their copies may temporarily differ. The editor uses concurrency control to make the copies consistent again, typically through one of two methods:

- **Operational transformation (OT).** In OT, every edit goes through a central server, which decides the order of all edits. Each edit specifies a position, e.g., "insert `a` at position 5." Because a writer may not have seen earlier edits in the sequence, the server transforms the edit to account for them. For example, if another writer inserted three characters before position 5, the server changes the operation to "insert `a` at position 8." [Google Docs uses OT](https://en.wikipedia.org/wiki/Google_Docs#Collaboration_and_revision_history).
- **Conflict-free replicated data type (CRDT).** A CRDT is a data structure that each writer can edit on their own copy, with no central server. The CRDT defines how to merge copies so that, no matter the order in which edits arrive, all copies end up the same. For text, a CRDT gives each character a unique ID, and an insertion refers to the ID of the character before the insertion, not to a position. For example, given "cat," writer A inserts "s" after "t," while writer B inserts "!" after "t." Both insertions refer to "t," so the CRDT uses a fixed rule (e.g., the insertion from the writer with the lower ID goes first) to order them, and all copies show "cats!". [Notion uses CRDTs](https://www.notion.com/blog/how-notion-handles-concurrent-editing-with-crdts).

Both methods are optimistic; that is, writers can edit without waiting for one another, and the system reconciles their changes afterward. Unfortunately, neither method checks whether the resulting text makes any sense. For example, if two writers rewrite the same sentence concurrently, the merged sentence may contain words from both versions --- and turn out to be gibberish.

Human writers usually avoid conflicts by coordinating socially, in addition to relying on concurrency control.<sup><a href="#note-2">2</a></sup> For example, a writer can see another writer's cursor and avoid editing the same region. Unfortunately, agents don't work this way. Moreover, the "shape" of agent work looks quite different from human work:

**Agents run long, "disconnected" transactions.** A transaction is a unit of work consisting of reads and writes, often treated as atomic. A human writer's transactions are small: they read a few words and type a few characters, repeatedly ([Figure 3a](#figure-3)). In contrast, an agent reads the document, spends seconds to minutes generating a response, and then makes potentially large edits based on its earlier read ([Figure 3b](#figure-3)). We call this a *disconnected transaction*, borrowing from disconnected operation in the Coda file system.<sup><a href="#note-3">3</a></sup> Like a laptop working offline, the agent operates on a snapshot that becomes increasingly stale. During this time, the user may change text that the agent plans to edit, without the agent seeing those changes. More subtly, the user may change text outside the agent's intended edit that still affects whether the edit makes sense. For example, in [Figure 2](#figure-2), the user begins line 4 with "We built DocWriter, an editor where," potentially making the agent's new sentence on line 3 redundant.

<figure id="figure-3" style="width: min(52rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/docwriter/transaction-shapes.svg' | relative_url }}" alt="Two human writers make many short transactions over time. A user also makes many short transactions, while an agent makes one long transaction that reads at the start and writes at the end.">
  <figcaption>Figure 3. Reads are green, and writes are blue. (a) Two concurrent human writers read and write small parts of the document all the time. (b) The agent makes one long transaction: it reads, thinks, and then writes.</figcaption>
</figure>

**Agent edits have lower priority than user edits.** In human collaboration, writers generally have equal authority over the document, or they socially agree on a lead author. Real-time editors reflect this: OT and CRDTs treat every writer's edits the same way. In DocWriter, the relationship is asymmetric: the user owns the document, and the agent assists. At the same time, the agent needs autonomy to be useful. For example, when asked to write an abstract, the agent should be able to read the paper and draft the abstract without asking for permission at every step. So the concurrency control must treat the two writers differently. The agent should be free to make edits, even many edits, but those edits must never overwrite newer user changes or stop the user from typing, and the user should have the final say over what enters the document --- without feeling overwhelmed.

Wow, what a challenging set of constraints to design around! To summarize, concurrency control for human-agent writing should meet four requirements:

1. **Never block the user.** The user should never have to wait for the agent. So the mechanism should be optimistic, like OT and CRDTs, and should not lock the user out of any text.
2. **Never lose a user change.** An agent edit that is based on a stale read must not overwrite newer text from the user.
3. **Let the agent act on its own.** The agent should be able to make large edits without asking for permission at every step.
4. **Give the user the final say.** The user decides which agent edits stay in the document, without feeling overwhelmed.

## 2.2 Some Naive Designs

We considered several ways to handle concurrent user and agent edits. Each meets some of our four requirements, but none meets all of them.

**Treat the agent as another writer.** The simplest approach is to use the same concurrency control as a real-time editor, e.g., a CRDT such as [Yjs](https://yjs.dev/). Both the user and agent can edit freely, and the CRDT merges their changes. But it merges characters, not intent. In our example ([Figure 2](#figure-2)), the user's "chat interfaces" gets stranded next to the agent's replacement sentence on line 2. The user's change survives technically, but its meaning is lost (requirement 2). Also, the agent's edits appear directly in the document without user review (requirement 4).

**Isolate the agent's work.** We could lock the document, or just the text the agent reads, until it finishes. This prevents conflicts, but blocks the user from typing (requirement 1). Alternatively, we could give the agent a separate copy, as in a Git branch or pull request. Both writers can then work independently, but the user must resolve conflicts and review a potentially large set of changes at the end. This gives the user control, but makes review burdensome (requirement 4).

**Reject stale edits instead of merging them.** Rather than merging concurrent edits, we could check whether the document has changed since the agent's disconnected transaction began. The agent records a document version when it reads, and the editor rejects its edits if the version has changed. This protects the user's changes without blocking either writer. However, the check is too coarse. In our example, the user's change to line 2 causes all three agent edits to fail, including those to lines 1 and 3, which do not touch the changed text. We need to distinguish stale edits from edits that are still valid, rather than rejecting everything because the document changed.

Moreover, even the edits that are still valid should wait for the user's approval, instead of going straight into the document.

# 3. Taking Inspiration from...You Guessed It: Databases

We reached into our arsenal of database techniques and found a surprisingly good fit for disconnected transactions: Bayou, a storage system from Xerox PARC (1995).<sup><a href="#note-4">4</a></sup> Bayou was built for laptops that are mostly offline. Each laptop keeps its own copy of shared data, users write to their copy while disconnected, and the copies sync when laptops reconnect. So an offline write is based on an old read, just like the agent's disconnected transaction.

Bayou is a full replicated storage system: laptops sync with each other in pairs, one laptop, designated ahead of time as the *primary*, decides the final order of writes, each laptop can undo and redo writes when the order changes, and clients get guarantees about what they see as they move between copies. We don't need most of this machinery, because DocWriter has one server. But two of Bayou's ideas port over to DocWriter directly.

**Dependency checks.** In Bayou, each write has three parts:

- **The update:** the change to the data.
- **A dependency check:** a query, plus the result that the write expects.
- **A merge procedure:** code that runs if the check fails.

Here is an example write from the Bayou paper, which books a meeting room in a shared calendar. The check looks for overlapping meetings, and the merge procedure tries backup time slots:

```python
Bayou_Write(
    update = ("insert", "Meetings", "12/18/95", "1:30pm", "60min", "Budget Meeting"),
    dependency_check = {
        "query": """SELECT key FROM Meetings
                    WHERE day = 12/18/95 AND start < 2:30pm AND end > 1:30pm""",
        "expected_result": [],
    },
    merge_procedure = try_alternates(
        ["12/18/95 3:00pm", "12/19/95 9:30am"],
        else_log_to = "ErrorLog",
    ),
)
```

Suppose Bob and then Alice each book the room for 1:30 pm, on their own laptops, while offline ([Figure 4](#figure-4)). Both checks pass, because neither laptop has the other's meeting yet. Later, the laptops sync, and both laptops order the writes by timestamp, with Bob's write first. On Bob's laptop, Alice's check now finds Bob's meeting and fails, so her merge procedure moves her to 3:00 pm. Alice's laptop undoes her write, applies Bob's write, and runs hers again, with the same result. Both calendars end up the same.

<figure id="figure-4" style="width: min(52rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/docwriter/bayou-example.svg' | relative_url }}" alt="A timeline with one lane for Bob's laptop and one for Alice's. Both book 1:30 pm offline. After the laptops sync, Alice's check fails on both laptops, and her meeting moves to 3:00 pm.">
  <figcaption>Figure 4. Bob's laptop is the primary. Bob and Alice both book the room for 1:30 pm while offline. At the first sync, Bob's laptop runs Alice's write after Bob's write, and Alice's laptop replays its log with Bob's write first. On both laptops, Alice's check fails, so her meeting moves to 3:00 pm. Bob's laptop then commits Alice's write. At the next sync, Alice's laptop learns about the commit, and her 3:00 pm meeting becomes final.</figcaption>
</figure>

Note that Alice's check only looks at what her write depends on: whether the room is free at 1:30 pm. Other changes to the calendar do not make it fail. E.g., if Bob had booked the room on a different day instead, Alice's check would pass on Bob's laptop, and her meeting would stay at 1:30 pm. DocWriter needs the same behavior: an agent edit should fail only if the user changed the text that the edit depends on, not whenever the user changes anything.

**Tentative writes.** When a laptop first performs a write, the write is *tentative*: the write can still change, because the laptop cannot know if another laptop has a write with an earlier timestamp that has not arrived yet. E.g., in [Figure 4](#figure-4), Alice's meeting first appears on her calendar at 1:30 pm, and then moves to 3:00 pm after the sync.

So how does a tentative write become final? Laptops sync whenever they connect, e.g., on the same network, and send each other the writes that they are missing. When a write reaches the primary, the primary *commits* the write: the write gets a fixed place in the log, so the result of the write can no longer change. In [Figure 4](#figure-4), Bob's laptop is the primary, so Bob's write commits right away. Alice's write commits at the first sync, when the write reaches Bob's laptop. Each laptop can then show two views of the data: a *committed view*, with only the committed writes, and a *full view*, which also includes the tentative writes.

In the next section, we show how DocWriter adapts these two components.

# 4. How DocWriter Handles Concurrent Edits

To handle the agent's disconnected transactions, DocWriter takes inspiration from two ideas in Bayou: dependency checks and tentative writes. But, we had to make two simple yet key adaptations. First, rather than define a merge procedure ahead of time, we let the agent write the merge on the fly, after the agent's dependency check fails ([Section 4.1](#41-the-agent-merges-on-the-flywithout-bothering-the-user)). Second, the user should have the final say over what enters the document, so the user, not the agent, commits each tentative edit ([Section 4.2](#42-the-user-commits-agent-editswith-interface-assistance)). Because the user must commit every agent edit, the interface must make committing many edits easy. We briefly describe each adaptation in turn.

## 4.1 The Agent Merges on the Fly...Without Bothering the User

How does the agent propose edits to the document? The agent calls a tool, `edit_doc`, with the text to replace (`old`) and the new text (`new`). Before DocWriter executes the `edit_doc` call, DocWriter runs a dependency check: the text to replace must still be in the document, unchanged, and appear only once. This simple check worked well for us. Here is pseudocode for `edit_doc`, including the check:

```python
def edit_doc(old: str, new: str) -> str:
    doc = current_document()
    # Dependency check: the text that the agent read is still there.
    if doc.count(old) != 1:
        return "Error: old not found. The text may have changed. Call read_doc and retry."
    doc.replace(old, new)  # shown as a suggestion (Section 4.2)
    return "Proposed the edit."
```

If the check fails, that is, the text to replace is no longer in the document because the user edited the text, then DocWriter returns an error. The error tells the agent to call `read_doc` and write a new edit. In other words, the agent writes the merge on the fly. The agent can also decide to drop the edit. All of the merging happens without the user knowing, which is quite nice: the user keeps writing in the editor as normal.

For example, in [Figure 5](#figure-5), the agent's edit to line 2 fails, because the user changed "most tools" to "chat interfaces" during the agent's disconnected transaction. The agent calls `read_doc`, retries with "chat interfaces", and the check passes.

<figure id="figure-5" style="width: min(41.36rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/docwriter/edit-checks.svg' | relative_url }}" alt="The agent's tool calls. edit_doc for line 2 fails, because its old text says &quot;most tools&quot; and the user changed it. The agent calls read_doc, then calls edit_doc again with &quot;chat interfaces&quot;, and the check passes. Then edit_doc for the note on line 3 passes.">
  <figcaption>Figure 5. The agent's tool calls for two of its edits from Figure 2. The edit to line 2 fails its check, because its old text still says "most tools". The agent reads the document again and retries with "chat interfaces".</figcaption>
</figure>

While implementing, we had to overcome one small obstacle. We observed that the agent's text to replace (`old`) tended to be quite large, even when the actual change was small, so the dependency check failed more often. We easily fixed the problem by prompting the agent to make multiple smaller-scoped edits rather than large blocks of edits.

Overall, the dependency check keeps agent edits from overwriting user changes (Requirement 2), and merging on the fly lets the agent keep working without the user (Requirement 3).

## 4.2 The User Commits Agent Edits...With Interface Assistance

What happens to an agent edit that passes the dependency check? The edit enters the document as a tentative edit, or *suggestion*. Only the user can commit a suggestion, by accepting the suggestion, or remove a suggestion, by rejecting the suggestion. Only agent edits are tentative. The user's keystrokes commit immediately, so the user never waits for the agent (Requirement 1).

Accepting or rejecting every agent edit can quickly overwhelm the user. To make each decision fast, the DocWriter interface helps the user in four ways ([Figure 6](#figure-6)):

- **Diffs in the document.** DocWriter strikes through the text that a suggestion deletes. When the user clicks on the struck-through text, DocWriter shows the new text in green ([Figure 6](#figure-6), steps 2 and 3).
- **The agent's reason.** Each suggestion has a comment thread, where the agent explains the edit ([Figure 6](#figure-6), step 2).
- **Replies.** The user may not immediately love an edit. So, the user can reply in the thread, and the agent will revise the suggestion ([Figure 6](#figure-6), steps 5 and 6).
- **One suggestion per phrase.** One phrase should not have many different diffs. To change text that already has a suggestion, the agent must revise the existing suggestion and add a comment to the same thread, for continuity ([Figure 6](#figure-6), step 6).

<figure id="figure-6" class="stepper" style="width: min(70rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/docwriter/suggestions-1.svg' | relative_url }}" data-base="{{ '/assets/blog/docwriter/suggestions-' | relative_url }}" alt="DocWriter with agent suggestions. Use the Next button to step through how the user reviews them.">
  <div class="stepper-bar">
    <button type="button" class="stepper-prev" aria-label="Previous step" disabled>&larr;</button>
    <div class="stepper-dots"></div>
    <button type="button" class="stepper-next" aria-label="Next step">&rarr;</button>
  </div>
  <p class="stepper-text" aria-live="polite">The user types on line 4 (the purple "You" flag) while the agent works.</p>
  <figcaption>Figure 6. Reviewing agent suggestions in DocWriter.</figcaption>
</figure>

<style>
#figure-6 .stepper-bar { display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin: 0.9rem auto 0.4rem; }
#figure-6 .stepper-dots { display: flex; gap: 0.35rem; }
#figure-6 .stepper-bar button { width: 2rem; height: 2rem; padding: 0; font: 500 0.875rem/1 var(--body, sans-serif); border: 1px solid #d0d5dd; border-radius: 50%; background: #fff; color: #344054; cursor: pointer; }
#figure-6 .stepper-bar button[aria-current="step"] { background: #101828; border-color: #101828; color: #fff; }
#figure-6 .stepper-bar button:disabled { opacity: 0.35; cursor: default; }
#figure-6 .stepper-text { max-width: 40rem; min-height: 3.3em; margin: 0 auto 1.25rem !important; font-size: 0.875rem; line-height: 1.65; text-align: center; color: #101828; }
</style>

<script>
document.querySelectorAll("figure.stepper").forEach(function (fig) {
  var steps = [
    "The user types on line 4 (the purple &quot;You&quot; flag) while the agent works.",
    "The agent's edits appear as suggestions, with deleted text struck through. The orange numbers match each suggestion to its thread.",
    "The user clicks the suggestion on line 2. DocWriter shows the new text in green, with Accept and Reject buttons.",
    "The user accepts the suggestion on line 2.",
    "The user replies on the thread for line 1.",
    "The agent revises the suggestion on line 1. The suggestion on line 3 waits until the user is ready."
  ];
  var img = fig.querySelector("img"), text = fig.querySelector(".stepper-text");
  var prev = fig.querySelector(".stepper-prev"), next = fig.querySelector(".stepper-next");
  var dots = fig.querySelector(".stepper-dots"), n = steps.length, k = 1;
  steps.forEach(function (_, i) {
    var d = document.createElement("button");
    d.type = "button"; d.textContent = i + 1; d.setAttribute("aria-label", "Step " + (i + 1));
    d.addEventListener("click", function () { k = i + 1; show(); });
    dots.appendChild(d);
    new Image().src = img.dataset.base + (i + 1) + ".svg";
  });
  function show() {
    img.src = img.dataset.base + k + ".svg";
    text.innerHTML = steps[k - 1];
    dots.querySelectorAll("button").forEach(function (d, i) {
      if (i === k - 1) d.setAttribute("aria-current", "step"); else d.removeAttribute("aria-current");
    });
    prev.disabled = k === 1; next.disabled = k === n;
  }
  prev.addEventListener("click", function () { if (k > 1) { k--; show(); } });
  next.addEventListener("click", function () { if (k < n) { k++; show(); } });
  show();
});
</script>

With these helpers, the user has the final say over what enters the document, without feeling overwhelmed (Requirement 4).

## 4.3 Other Implementation Details

**Client and server.** DocWriter runs locally, on the user's machine. The `docwriter` command starts a server, a Node.js process written in [TypeScript](https://www.typescriptlang.org/) with [SvelteKit](https://svelte.dev/docs/kit), and opens the editor in the browser. We built the editor with [Tiptap](https://tiptap.dev/), on [ProseMirror](https://prosemirror.net/). The server runs the agent, holds the live documents, and syncs each document with the client over WebSockets ([Hocuspocus](https://tiptap.dev/docs/hocuspocus/introduction)).

**Document model.** Each document is a [Yjs](https://yjs.dev/) CRDT. Suggestions and comments are marks on the text, in the same Yjs document. The DocWriter server holds the live document in memory, and the agent's tools (`read_doc`, `edit_doc`) read and change that live document. The design in Sections 4.1 and 4.2 definitely does not (and probably should not) need a CRDT. DocWriter has one server, and the dependency check only needs that server to apply edits one at a time. We picked Yjs for its robust TypeScript support, and because prior work on AI agents in shared documents also uses Yjs.<sup><a href="#note-5">5</a></sup>

**Storage.** DocWriter keeps its state in [SQLite](https://www.sqlite.org/), so DocWriter does not have to maintain many copies of `.md` files. SQLite keeps an append-only log of every change to each document, and each change records whether the user or the agent made the change, so DocWriter knows the lineage of every edit. SQLite also stores agent sessions, rules, and hooks. Every 500 ms, DocWriter writes the committed view of each document to the `.md` file on disk, so the user's file stays a normal Markdown file.

**Undo.** Ctrl+Z undoes the user's own actions: typing, and accepting or rejecting a suggestion. E.g., if the user accepts an agent edit, Ctrl+Z brings back the suggestion and reopens the suggestion's thread, in one step. Ctrl+Z does not undo a pending agent edit. To remove a pending suggestion, the user rejects the suggestion.

**Agent harness.** DocWriter gives the agent its tools as an [MCP](https://modelcontextprotocol.io/) server, and runs the agent through each provider's SDK, e.g., the [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview), the [Codex SDK](https://developers.openai.com/codex/sdk), and the Cursor SDK. So, the user can use an existing subscription.

# 5. Open Questions

We are excited to release DocWriter into the world! But we are far from done! Here are some questions that we have been thinking about. The first two questions follow directly from the design in this article. The other questions are more about the people who write with agents.

1. **The agent can't be interrupted.** The agent does long, disconnected transactions with many tool calls ([Section 2.1](#21-how-agents-differ-from-human-writers)). Some tool calls are slow, especially search calls.<sup><a href="#note-6">6</a></sup> However, during a long transaction, the user can change other parts of the document, and those changes can affect what the agent should write. The dependency check does not catch these changes, so the agent continues to work from outdated information. How can DocWriter quickly find user changes that are important to the agent's work, and tell the agent before the agent completes the work?

2. **Atomicity is per edit, not per request.** A request is one thing that the user asks the agent to do, e.g., one `[[ ]]` note or one comment. A user might naturally expect that one request is one transaction: all of the request's edits apply together, or none apply. But in DocWriter, each `edit_doc` call is atomic alone. One request can become many edits, and some of the edits can pass the dependency check while others fail. The user can also accept some of the edits and reject others. Either way, only part of the request applies. How can DocWriter make a whole request atomic, while each edit stays small, and the agent does not have to redo a lot of work or block its other edits?

3. **Reviewing agent edits is still a cognitive burden.** The current DocWriter interface is intended to make each review decision fast ([Section 4.2](#42-the-user-commits-agent-editswith-interface-assistance)). But the user must still accept or reject every agent edit, and each new suggestion pulls the user's attention away from their own writing. Deciding when the agent should act, when the agent should interrupt the user, and when the agent should wait is the classic problem of designing a good mixed-initiative interface.<sup><a href="#note-7">7</a></sup> How can DocWriter decrease the cognitive burden of review, e.g., by choosing which edits can commit without the user, or when to show a suggestion?

4. **The agent does not write like the user.** Review takes even more effort when the agent's text does not sound like the user. Off-the-shelf models produce slop: text with the same words and sentence shapes, which readers and detectors can easily identify as AI-written.<sup><a href="#note-8">8</a></sup> We take some steps to decrease slop in DocWriter. Users can add `skill.md` files, e.g., with examples of how the user writes, and users can define rules, e.g., "never say 'delve'". But skills and rules are only instructions in the prompt, so the agent does not always follow them. How can an agent learn to write like one specific person, or at least write text that does not give that person an allergic reaction?

5. **More people, more agents, more applications.** Our design assumes one human and one agent (which, in theory, can spawn subagents) in one text document. With many writers, e.g., a first author and the PIs, it is not clear who commits an agent's edit, or whose instructions an agent should follow. Prior work has found that humans themselves struggle to agree on how to set up agents for a team.<sup><a href="#note-5">5</a></sup> And beyond writing, much knowledge work involves communicating ideas in other mediums, such as slides and presentations. How can many people and many agents effectively collaborate asynchronously, across many different applications?

6. **One editing session is only one part of writing.** DocWriter supports one stage of writing: drafting and editing one document. But writing has many stages, such as planning, drafting, and revising, and each stage probably needs a different interface. We are excited to design these interfaces with folks at CMU, building on the large body of HCI work on what people want from AI writing assistants.<sup><a href="#note-9">9</a>, <a href="#note-10">10</a></sup> Stay tuned!

As agents take on more of our work, these questions become increasingly important to tackle, lest people rely on agents so much that they lose the ability to think and communicate on their own.<sup><a href="#note-11">11</a></sup> On the flip side, however, the AI revolution presents an opportunity to help people think _more_ clearly, if we manage to design the right interfaces and systems for human-agent collaboration.


# 6. Conclusion

Agents can now handle long tasks, so people will give agents more and more of their work. But for writing, and other work where people clarify their own thinking as they participate, people should not just delegate. People should stay involved and shape the work with their own taste, together with their agents. Such collaboration needs new interfaces and mechanisms for humans and agents to work asynchronously, and excitingly, these may look quite different from the ones we built for human-human collaboration.

In this article, we discussed how we might design such mechanisms for a concurrent writing application with one human and one agent. We fished out some old ideas from the weak consistency canon and arrived at a surprisingly simple design, with lessons to share: let the agent author its own merge logic, and design interfaces that speed up the user's review of the agent's tentative edits.

Looking forward, there is so much to explore in co-designing interfaces and systems for effective human-agent collaboration. In the meantime, [try DocWriter](https://github.com/docwriter-org/docwriter), and [sign up for our upcoming user study](TODO) (TODO create link)!

# Acknowledgements

We thank the [Laude Institute](https://www.laude.org/) for supporting DocWriter through a Slingshots grant.

# Notes

<span id="note-1"><strong>1.</strong></span> E.g., Weixin Liang, Yaohui Zhang, Zhengxuan Wu, et al. "Mapping the Increasing Use of LLMs in Scientific Papers." arXiv:2404.01268, 2024 ([link](https://arxiv.org/abs/2404.01268)). In computer science papers, the authors estimate that as much as 17.5% of the content was modified by LLMs by early 2024.

<span id="note-2"><strong>2.</strong></span> Greenberg and Marwood make this point for real-time groupware: people mediate their own actions through social protocols, so concurrency conflicts are rare, and people repair the ones that occur ([Greenberg and Marwood, CSCW 1994](https://doi.org/10.1145/192844.193011)).

<span id="note-3"><strong>3.</strong></span> In disconnected operation, a laptop keeps working on its own copy of the files while it is offline, and the system reconciles its changes when it reconnects ([Kistler and Satyanarayanan, TOCS 1992](https://doi.org/10.1145/146941.146942)).

<span id="note-4"><strong>4.</strong></span> Douglas B. Terry, Marvin M. Theimer, Karin Petersen, Alan J. Demers, Mike J. Spreitzer, and Carl H. Hauser. "Managing Update Conflicts in Bayou, a Weakly Connected Replicated Storage System." SOSP 1995.

<span id="note-5"><strong>5.</strong></span> Florian Lehmann, Krystsina Shauchenka, and Daniel Buschek. "Collaborative Document Editing with Multiple Users and AI Agents." CHI 2026 ([doi](https://doi.org/10.1145/3772318.3790648)).

<span id="note-6"><strong>6.</strong></span> For example, to make citations, the agent must search the web for each paper, and each search can take many seconds.

<span id="note-7"><strong>7.</strong></span> Eric Horvitz. "Principles of Mixed-Initiative User Interfaces." CHI 1999 ([doi](https://doi.org/10.1145/302979.303030)).

<span id="note-8"><strong>8.</strong></span> E.g., Bradley Emi and Max Spero. "Technical Report on the Pangram AI-Generated Text Classifier." arXiv:2402.14873, 2024 ([link](https://arxiv.org/abs/2402.14873)).

<span id="note-9"><strong>9.</strong></span> E.g., Mina Lee, Katy Ilonka Gero, John Joon Young Chung, et al. "A Design Space for Intelligent and Interactive Writing Assistants." CHI 2024 ([doi](https://doi.org/10.1145/3613904.3642697)).

<span id="note-10"><strong>10.</strong></span> Katy Ilonka Gero, Tao Long, and Lydia B. Chilton. "Social Dynamics of AI Support in Creative Writing." CHI 2023 ([doi](https://doi.org/10.1145/3544548.3580782)).

<span id="note-11"><strong>11.</strong></span> Hao-Ping Lee, Advait Sarkar, Lev Tankelevitch, et al. "The Impact of Generative AI on Critical Thinking: Self-Reported Reductions in Cognitive Effort and Confidence Effects From a Survey of Knowledge Workers." CHI 2025 ([doi](https://doi.org/10.1145/3706598.3713778)).

# Cite this article

<div class="bibtex-block" markdown="1">
<button class="copy-bibtex" type="button">Copy BibTeX</button>

```bibtex
@misc{shankar2026docwriter,
  title = {Don't Make Me Wait! Enabling Concurrency in Human-Agent Collaboration},
  author = {Shankar, Shreya and Parameswaran, Aditya G.},
  year = {2026},
  month = oct,
  url = {https://fsdatalab.github.io/blog/introducing-docwriter/}
}
```

</div>
