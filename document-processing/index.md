---
layout: default
title: How to Process Documents with LLMs Cheaply
description: "Running an LLM over every document is expensive. How query optimizers, model cascades with accuracy guarantees, and cheaper semantic joins cut the cost, from our work on DocETL, MOAR, BARGAIN, Task Cascades, FDJ, and ZenDB."
permalink: /document-processing/
guide: true
key: document-processing
---
<article class="content-narrow prose" markdown="1">

<div class="post-header"><h1>How to process documents with LLMs cheaply</h1></div>

LLMs can classify, extract from, and summarize documents that no rule-based
pipeline could handle. Running one over every document in a large collection
is expensive, though, and the cheapest model is often not accurate enough.
Our work cuts the cost in five ways, most of it with Aditya Parameswaran and
Sepanta Zeighami.

## Let an optimizer rewrite the pipeline

You describe the pipeline as a sequence of LLM operations, such as map,
filter, and reduce, and an optimizer rewrites it into a cheaper or more
accurate plan.

- [DocETL](https://arxiv.org/abs/2410.12189) (VLDB 2025), with Tristan
  Chambers, Tarak Shah, Aditya Parameswaran, and Eugene Wu: an agent rewrites
  the pipeline, for example by splitting long documents or a hard operation,
  and evaluates each plan. Its plans were 25 to 80% more accurate than
  well-engineered baselines. The code is at [docetl.org](https://www.docetl.org/).
- [MOAR](https://arxiv.org/abs/2512.02289) (VLDB 2026), led by Lindsey Linxi
  Wei, with Sepanta Zeighami, Yeounoh Chung, Fatma Ozcan, and Aditya
  Parameswaran: an optimizer for DocETL that trades off accuracy and cost.
  Across six workloads, it was 27% more accurate than ABACUS, and matched
  ABACUS's best accuracy at 55% of its cost.

## Use a cheaper model where it is good enough

A cascade answers each document with a cheap model first, and sends only the
documents the cheap model is unsure about to an expensive one.

- [BARGAIN](https://arxiv.org/abs/2509.02896) (SIGMOD 2026), led by Sepanta
  Zeighami, with Aditya Parameswaran: a cascade with accuracy guarantees. On 8
  real-world datasets, it cut cost by up to 86% more than the previous state
  of the art, with stronger guarantees.
- [Task Cascades](https://arxiv.org/abs/2601.05536) (SIGMOD 2026), with
  Sepanta Zeighami and Aditya Parameswaran: a cascade that varies the task, not
  only the model: a cheap model might answer an easier related question, or
  read only the part of the document that matters. At a 90% target
  accuracy on eight document tasks, it cut cost by 36% on average compared
  with model cascades.

## Join documents without comparing every pair

A semantic join asks an LLM whether two rows match, so a naive join makes one
call for every pair of rows.

- [Featurized-Decomposition Join](https://arxiv.org/abs/2512.05399) (VLDB
  2026), led by Sepanta Zeighami, with Aditya Parameswaran: an LLM extracts features
  from each row, such as a product's brand, and combines them into rules that
  rule out most pairs with cheap comparisons. It cost up to 10 times less than
  prior methods at the same quality guarantees.

## Use the structure documents share

Many collections, such as civic agenda reports, follow one template.
[ZenDB](https://www.computer.org/csdl/proceedings-article/icde/2025/360300c422/26FZAUkb1ra)
(ICDE 2025), led by Yiming Lin, detects the shared section structure of each
document. You query the collection with SQL, and ZenDB sends the LLM only the
sections that matter.

## Run it faster on GPUs

When the pipeline runs on open-weight models, the inference engine decides
much of the cost. [Quail]({{ '/ai-sql/' | relative_url }}), our engine for
AI functions in SQL, plans the query and the LLM inference together.

## Questions

{% include question-list.html guide="document-processing" %}

</article>
