---
layout: post
title: "Five Principles for Generating Readable AI Text in Bulk"
date: 2026-10-29
author: "Shreya Shankar"
permalink: /blog/agent-text-at-scale/
typora-root-url: ..
description: "Five principles for making large amounts of AI-generated text, e.g., in dashboards or websites, easy for humans to read, with real examples and fixes."
image:
  path: /assets/blog/agent-text-at-scale/social-preview.png
  width: 1200
  height: 630
  alt: "Five Principles for Generating Readable AI Text in Bulk: the five principles, next to a sketch of a list of inconsistent text rows and the same list with every row in one pattern."
---

<nav class="post-toc" aria-label="Table of contents">
<strong>Contents</strong>
<ol>
  <li><a href="#the-problem">The Problem.</a></li>
  <li><a href="#checklist">Checklist.</a>
    <ol>
      <li><a href="#1-match-how-human-readers-parse-sentences">Match How Human Readers Parse Sentences.</a></li>
      <li><a href="#2-put-the-most-important-words-first">Put the Most Important Words First.</a></li>
      <li><a href="#3-use-the-same-structure-and-the-same-vocabulary">Use the Same Structure and the Same Vocabulary.</a></li>
      <li><a href="#4-use-variation-only-to-show-a-meaningful-difference">Use Variation Only to Show a Meaningful Difference.</a></li>
      <li><a href="#5-align-the-connotation-of-verbs-with-the-purpose-of-the-dashboard">Align the Connotation of Verbs with the Purpose of the Dashboard.</a></li>
    </ol>
  </li>
  <li><a href="#takeaways">Takeaways.</a></li>
</ol>
</nav>

# The Problem

Thanks to modern AI, we humans find ourselves reading AI-generated text at an unprecedented scale! Many of our daily driver products show feeds of such text, e.g., in Claude Code or Codex, there is a sidebar with AI-generated titles of our past agent sessions; on an iPhone, Apple Intelligence now groups and summarizes "stacks" of notifications. People also generate many temporary dashboards and web pages with AI, and share them broadly --- for example, Google publishes a [list of the bugs](https://issuetracker.google.com/issues?q=componentid:1836411&s=type:desc&s=issue_id:desc) found by Big Sleep, its AI bug-finding agent.

Unfortunately, such feeds of AI-generated texts are quite difficult for humans to read. This isn't very surprising, as it's well-known that text items need to be _predictable_ and _consistent_ for humans to read many of them quickly.

In this blog post, informed by our recent experience working with AI to generate text dashboards (and consume lots of AI-generated text in dashboards), we present a checklist of 5 principles for effective feeds of AI-generated text.

# Checklist

## 1. Match How Human Readers Parse Sentences

Interestingly, humans parse text quite predictably and simply, and we have known this for quite some time. In 1970, Thomas Bever, a psycholinguist who studies how people understand sentences, found that English readers use the "NVN" strategy: they take the first noun as the subject, the next verb as its action, and the next noun as its object.<sup><a href="#note-1">1</a></sup> For example, in "The agent fixed the login bug," readers take "agent" as the subject, "fixed" as the action, and "bug" as the object.

As a consequence, if text does not follow the NVN pattern, humans need to do multiple passes over the text, which is quite annoying.<sup><a href="#note-2">2</a></sup> For example, in "Login bug fixed by the agent," readers first take "bug" as the subject and "fixed" as its action, as if the bug fixed something. Only at "by the agent" do readers find the real subject, and then they parse the text again.

To render a list of text items with as little cognitive effort for readers as possible, here is a rough procedure to follow:

1. Start each item with its subject or its verb. If the subject is the same in all items, drop it and start with the verb.
2. Put the first verb of each item in the tense that best matches the reader's expectations: past, continuous, or future.

**Example.** Many of us use a coding agent app every day, e.g., Claude or Codex. The sidebar of the app shows a list of agent sessions. [Figure 1](#figure-1) (left) shows a real example from the Claude sidebar. The titles mix nouns ("KV persistence"), gerunds ("Switching to main"), and imperatives ("Fine-tune model on personal writing"). On the right are the same titles, rewritten with the procedure above.

All sessions have the same subject, the agent, so by the procedure above, each title should start with a verb. Then, because each session is active, each verb is in the continuous form, e.g., "Measuring vLLM performance."

<figure id="figure-1" style="width: min(52rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/agent-text-at-scale/session-titles.svg' | relative_url }}" alt="Seven session titles as written in a chat sidebar, next to the same titles rewritten to start with a verb.">
  <figcaption>Figure 1. Seven session titles from the author's Claude sidebar, stylized (left), and the same titles rewritten to start with a verb in the continuous form (right).</figcaption>
</figure>

## 2. Put the Most Important Words First

Jakob Nielsen, a very famous usability researcher who studies how people read websites, used eye tracking to find that readers scan web pages in an "F" shape: they read the start of each line and skip most of the rest<sup><a href="#note-3">3</a></sup> --- and, readers of a list often see only the first two words of each item.<sup><a href="#note-4">4</a></sup> As such, we should:

1. Find the words that make each item different from the other items.
2. Move those words as close to the start of the item as possible.

**Example.** As shown in [Figure 2](#figure-2), a bad way to write order email subjects is to put the status last, e.g., "Your order #1001 has shipped." Then every subject starts with the same words. But Amazon puts the status first, e.g., `Ordered: "Bounty Essentials..."`.

<figure id="figure-2" style="width: min(59.5rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/agent-text-at-scale/order-emails.svg' | relative_url }}" alt="Four order emails in an inbox with the status last, next to four order emails with the status first.">
  <figcaption>Figure 2. Order emails in an inbox, stylized. When the status is last (left), the first words are the same in every email. When the status is first, as in Amazon order emails (right), the first word tells the emails apart.</figcaption>
</figure>

## 3. Use the Same Structure and the Same Vocabulary

In another interesting psycholinguistic study from the 1980s, Lyn Frazier and her colleagues measured how fast people read sentences with two clauses joined by "and." Readers read the second clause faster when it had the same structure as the first clause, e.g., when both clauses were active or both were passive.<sup><a href="#note-5">5</a></sup> One can directly apply this idea to a list of texts: each item is faster to read when it has the same structure as the items before it.<sup><a href="#note-6">6</a></sup>

Moreover, it is well-known that a piece of text should not use two different words for the same concept.<sup><a href="#note-7">7</a></sup> So, in a list of texts:

1. Use one structure in every item.
2. Use one word for each concept.

**Example.** We present one example that does _not_ follow this principle, followed by one that does. For our bad example, we show the AI autofill feature in Notion --- which writes one summary for each row of a table in Notion. We made a table of 10 public GitHub issues from Ollama and let Notion AI fill in a Summary column.<sup><a href="#note-8">8</a></sup> As shown in [Figure 3](#figure-3), the summaries are actually hard to quickly skim. The summaries inconsistently start with "The issue reports that," "The page reports," or an action ("Pressing Ctrl+C"), and they call the same thing an "issue" in some rows and a "page" in another.

<figure id="figure-3" style="width: min(59.75rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/agent-text-at-scale/notion-summaries.svg' | relative_url }}" alt="A Notion table of four GitHub issues with summaries that Notion AI wrote. The summaries start in different ways.">
  <figcaption>Figure 3. Four rows of a Notion table, stylized. Notion AI wrote each summary from the text of the issue. The highlights mark how each summary starts.</figcaption>
</figure>

In contrast, Perplexity's Discover feed nicely follows this checklist item #3 (as well as item #1).<sup><a href="#note-9">9</a></sup> As shown in [Figure 4](#figure-4), every headline has the same structure: the subject, a verb in the present tense, and then the object, e.g., "Bank of Canada hikes rate to 2.5%" and "Brewers beat Rays in Game 5 for first World Series title."

<figure id="figure-4" style="width: min(40rem, calc(100vw - 3rem));">
  <img src="{{ '/assets/blog/agent-text-at-scale/discover-headlines.svg' | relative_url }}" alt="Four stories from the Perplexity Discover feed, each with a headline that has a subject, a verb in the present tense, and an object.">
  <figcaption>Figure 4. Four stories from the Perplexity Discover feed on October 28, 2026, stylized. Every headline has the same structure.</figcaption>
</figure>

## 4. Use Variation Only to Show a Meaningful Difference

In 1986, Jock Mackinlay, a visualization researcher (and pioneer of modern data visualization), wrote a paper on how to design charts automatically.<sup><a href="#note-10">10</a></sup> One of its main ideas is *expressiveness*: a chart should express all of the facts in the data, and only the facts in the data. E.g., if one bar in a bar chart is red, readers will assume that the color means something --- so make sure any variation in bars represents some meaningful variable.<sup><a href="#note-11">11</a></sup> Now, although this research is about charts of structured (i.e., tabular) data, it is not so far-fetched to apply it to unstructured text: if one item in a list differs from the others, e.g., in its length, its structure, or its words, readers will assume that the difference means something.

**Example.** LLMs are often used to summarize the "key themes" in user feedback. We ran Claude's data analysis plugin on 100 recent Duolingo App Store reviews, to see what a frontier model would do.<sup><a href="#note-12">12</a></sup> On the left of [Figure 5](#figure-5), each card adds a different kind of detail. Only the bugs card names an app version (7.146.0), and only the energy card says that some reviews are in Russian and Turkish. So readers cannot tell whether these details are significant: is 7.146.0 the only version with bugs, and do the other themes come only from English reviews?

On the right, we rewrote every card so that each one has the same components: a sentence about what users say, followed by one example from the reviews. We also show the number of reviews, an important variable, for each theme in a bar chart. Of course, the rewritten cards drop some detail, but readers can still open the reviews of any theme they care about.

<figure id="figure-5" class="figure-themes">
<div class="td" data-theme-dashboard>
<div class="td-col"><div class="td-head">Claude's version</div>
<button type="button" class="td-card td-orig" data-i="0"><span class="td-row"><span class="td-th">Fun, easy, and effective learning</span><span class="td-n">74 reviews</span></span><span class="td-sum">Most reviews praise Duolingo as fun, easy, and helpful for learning, often in short 5-star notes. Users cite new vocabulary, pronunciation help, self-paced lessons, and being able to talk with family or friends. Spanish is the most named language, followed by Japanese, Italian, Korean, German, and others. Several of these reviewers still raise a complaint, such as energy limits or cost.</span></button>
<button type="button" class="td-card td-orig" data-i="1"><span class="td-row"><span class="td-th">Chess, music, and math courses</span><span class="td-n">12 reviews</span></span><span class="td-sum">Chess is a popular new draw, and some users mention music, piano, and math too. Feedback is mostly positive, but there are specific complaints: a checkmate rule that seems wrong, no way for skilled players to skip ahead, a distracting new owl voice and hard-to-see pieces in the chess puzzle, and many clicks between games.</span></button>
<button type="button" class="td-card td-orig" data-i="2"><span class="td-row"><span class="td-th">Bugs and technical problems</span><span class="td-n">9 reviews</span></span><span class="td-sum">Reports include voice answers not registering, the app freezing (one user lost a streak), the app crashing on launch, finished lessons not counting, leaderboard members disappearing, a new event missing for one family-plan member, and a shared iPad starting a new learner at the wrong level. Most come from the current version, 7.146.0.</span></button>
<button type="button" class="td-card td-orig" data-i="3"><span class="td-row"><span class="td-th">Energy and battery limits on lessons</span><span class="td-n">8 reviews</span></span><span class="td-sum">The energy (battery) system, and the older hearts system, limits how much users can practice. Users say they run out fast, must wait or watch ads, and see it as a push to buy Super. Some reviews are in Russian and Turkish, so the complaint shows up across markets. One user suggests earning energy by doing chess, music, or math.</span></button>
<button type="button" class="td-card td-orig" data-i="4"><span class="td-row"><span class="td-th">New language requests</span><span class="td-n">3 reviews</span></span><span class="td-sum">Users ask for new courses. Two ask for Icelandic, noting that Duolingo already offers other Nordic languages, and one asks for Elvish from The Lord of the Rings.</span></button>
</div><div class="td-col"><div class="td-head">Our version</div>
<button type="button" class="td-card td-new" data-i="0"><span class="td-row"><span class="td-th">Fun, easy, and effective learning</span><span class="td-n">74 reviews</span></span><span class="td-sum">Users love that the app is fun and helps them learn, e.g., new vocabulary and pronunciation.</span></button>
<button type="button" class="td-card td-new" data-i="1"><span class="td-row"><span class="td-th">Chess, music, and math courses</span><span class="td-n">12 reviews</span></span><span class="td-sum">Users enjoy the new courses but complain about problems, e.g., a checkmate rule that seems wrong.</span></button>
<button type="button" class="td-card td-new" data-i="2"><span class="td-row"><span class="td-th">Bugs and technical problems</span><span class="td-n">9 reviews</span></span><span class="td-sum">Users complain about bugs that cost them progress, e.g., a freeze that lost a daily streak.</span></button>
<button type="button" class="td-card td-new" data-i="3"><span class="td-row"><span class="td-th">Energy and battery limits on lessons</span><span class="td-n">8 reviews</span></span><span class="td-sum">Users complain that they run out of energy fast and must wait to keep practicing.</span></button>
<button type="button" class="td-card td-new" data-i="4"><span class="td-row"><span class="td-th">New language requests</span><span class="td-n">3 reviews</span></span><span class="td-sum">Users request new courses, e.g., Icelandic.</span></button>
<div class="td-sub">Reviews per theme</div><div class="td-bars">
<button type="button" class="td-bar" data-i="0"><span class="td-bl">Fun, easy, and effective learning</span><span class="td-track"><span class="td-fill" style="width:100.0%"></span></span><span class="td-bn">74</span></button>
<button type="button" class="td-bar" data-i="1"><span class="td-bl">Chess, music, and math courses</span><span class="td-track"><span class="td-fill" style="width:16.2%"></span></span><span class="td-bn">12</span></button>
<button type="button" class="td-bar" data-i="2"><span class="td-bl">Bugs and technical problems</span><span class="td-track"><span class="td-fill" style="width:12.2%"></span></span><span class="td-bn">9</span></button>
<button type="button" class="td-bar" data-i="3"><span class="td-bl">Energy and battery limits on lessons</span><span class="td-track"><span class="td-fill" style="width:10.8%"></span></span><span class="td-bn">8</span></button>
<button type="button" class="td-bar" data-i="4"><span class="td-bl">New language requests</span><span class="td-track"><span class="td-fill" style="width:4.1%"></span></span><span class="td-bn">3</span></button>
</div></div></div>
<figcaption>Figure 5. A dashboard of key themes in 100 Duolingo App Store reviews, generated by Claude via its data analysis plugin, stylized (left; 5 of 11 themes shown). On the right, the same themes rewritten with one template, with the number of reviews per theme as a bar chart. Click a card or a bar to highlight the same theme in all three views.</figcaption>
</figure>

<script src="{{ '/assets/js/theme-dashboard.js' | relative_url }}" defer></script>

## 5. Align the Connotation of Verbs with the Purpose of the Dashboard

Many dashboards exist to surface one kind of information, e.g., what customers dislike, or what customers like and what is going well. The verbs in a dashboard should make its purpose obvious. For example, a dashboard of customer complaints should use clearly negative verbs, like "complain," and a dashboard of praise should use clearly positive verbs, like "love." Readers skim dashboards with very little attention, so a verb with a strong connotation helps readers grasp each item quickly.<sup><a href="#note-13">13</a></sup>

**Example.** In [Figure 5](#figure-5), Claude's version starts the chess card with "Chess is a popular new draw," even though most of the card lists complaints. So readers cannot tell at a glance whether the theme is good or bad news. In our version, each card starts with a verb that has an obvious sentiment, e.g., "Users love" for praise and "Users complain" for problems, so readers can tell from the verb alone whether a theme is praise or a complaint.

# Takeaways

In this post, we discussed five principles for making lots of AI-generated text easy to read. While we only discussed simple dashboard-like examples, we believe these principles apply to _any_ artifact or product that AI generates, e.g., websites. So please, feed this checklist into your coding agents, or make a skill out of it, so your AI-generated texts are easy for humans to read. Page by page, let's move towards a world free of slop.

# Notes

<span id="note-1"><strong>1.</strong></span> Thomas G. Bever. The Cognitive Basis for Linguistic Structures. In J. R. Hayes, editor, *Cognition and the Development of Language*, pages 279–362. Wiley, New York, 1970. Bever describes the strategy as reading a noun, a verb, and a noun as the actor, the action, and the object of the action.

<span id="note-2"><strong>2.</strong></span> Lyn Frazier and Keith Rayner. [Making and Correcting Errors During Sentence Comprehension: Eye Movements in the Analysis of Structurally Ambiguous Sentences](https://doi.org/10.1016/0010-0285(82)90008-1). *Cognitive Psychology*, 14(2):178–210, 1982. In an eye-tracking study, readers of sentences with a misleading start moved their eyes back to earlier words to parse the sentence again.

<span id="note-3"><strong>3.</strong></span> Jakob Nielsen. [F-Shaped Pattern for Reading Web Content (Original Study)](https://www.nngroup.com/articles/f-shaped-pattern-reading-web-content-discovered/). Nielsen Norman Group, April 16, 2006. The study tracked the eyes of 232 users as they read thousands of web pages.

<span id="note-4"><strong>4.</strong></span> Jakob Nielsen. [First 2 Words: A Signal for the Scanning Eye](https://www.nngroup.com/articles/first-2-words-a-signal-for-scanning/). Nielsen Norman Group, April 6, 2009.

<span id="note-5"><strong>5.</strong></span> Lyn Frazier, Lori Taft, Tom Roeper, Charles Clifton, and Kate Ehrlich. [Parallel Structure: A Source of Facilitation in Sentence Comprehension](https://link.springer.com/article/10.3758/BF03198303). *Memory & Cognition*, 12(5):421–430, 1984. In 2010, Sturt, Keller, and Dubey found that readers also read a phrase faster when it has the same structure as an earlier phrase, even when "and" does not join the two phrases: Patrick Sturt, Frank Keller, and Amit Dubey. [Syntactic Priming in Comprehension: Parallelism Effects with and without Coordination](https://doi.org/10.1016/j.jml.2010.01.001). *Journal of Memory and Language*, 62(4):333–351, 2010.

<span id="note-6"><strong>6.</strong></span> Funnily enough, modern data compression algorithms work in a similar way, by exploiting repeated patterns. E.g., gzip stores a pattern the first time it appears and replaces each later copy with a short pointer back to it. So a list with one pattern is easy to compress, both for gzip and for a reader.

<span id="note-7"><strong>7.</strong></span> E.g., a core rule of [ASD Simplified Technical English](https://www.asd-ste100.org/about.html), a writing standard for aircraft maintenance manuals, is "one word, one meaning." Also, Jakob Nielsen's fourth usability heuristic says that users "should not have to wonder whether different words, situations, or actions mean the same thing." See Jakob Nielsen. [10 Usability Heuristics for User Interface Design](https://www.nngroup.com/articles/ten-usability-heuristics/). Nielsen Norman Group, 1994.

<span id="note-8"><strong>8.</strong></span> We copied 10 open issues from [ollama/ollama](https://github.com/ollama/ollama/issues) on October 28, 2026, and added a Notion AI autofill property named "Summary" with its default settings. Figure 4 shows the summaries of issues [#19359](https://github.com/ollama/ollama/issues/19359), [#19342](https://github.com/ollama/ollama/issues/19342), [#19375](https://github.com/ollama/ollama/issues/19375), and [#19340](https://github.com/ollama/ollama/issues/19340), without changes.

<span id="note-9"><strong>9.</strong></span> [Perplexity Discover](https://www.perplexity.ai/discover), accessed October 28, 2026.

<span id="note-10"><strong>10.</strong></span> Jock Mackinlay. [Automating the Design of Graphical Presentations of Relational Information](https://doi.org/10.1145/22949.22950). *ACM Transactions on Graphics*, 5(2):110–141, 1986.

<span id="note-11"><strong>11.</strong></span> Jeff Heer. [CSE 512: Data Visualization](https://courses.cs.washington.edu/courses/cse512/24sp/). University of Washington, Spring 2024. See the lectures on data and image models and on visualization design.

<span id="note-12"><strong>12.</strong></span> We downloaded the 100 most recent US App Store reviews of Duolingo from Apple's public review feed on October 28, 2026; they are from October 26 and 27, 2026. Claude ran the `data:analyze` skill of Claude's data analysis plugin on them and wrote 11 theme cards. Figure 5 shows five of them, without changes. The word counts in the text are ours; the plugin did not report them.

<span id="note-13"><strong>13.</strong></span> It turns out, after asking Claude, that there is related prior work, which we have not read closely. E.g., Kousta, Vinson, and Vigliocco found that people recognize emotion words, positive or negative, faster than neutral words. See Stavroula-Thaleia Kousta, David P. Vinson, and Gabriella Vigliocco. [Emotion Words, Regardless of Polarity, Have a Processing Advantage over Neutral Words](https://doi.org/10.1016/j.cognition.2009.06.007). *Cognition*, 112(3):473–481, 2009.

