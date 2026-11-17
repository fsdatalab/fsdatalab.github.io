---
layout: default
title: Interfaces for Data Work with LLMs
description: "How people steer LLM pipelines over many documents, debug RAG, find datasets, and read AI-generated text at scale, and the interfaces we build for each."
permalink: /data-interfaces/
guide: true
---
<article class="content-narrow prose" markdown="1">

<div class="post-header"><h1>Interfaces for data work with LLMs</h1></div>

LLMs now run over whole datasets: they classify documents, extract fields,
summarize transcripts, and answer questions over retrieved text. The model is
only part of the work. People still decide what to ask, check whether the
answers are right, and read the results. We study how people do that work and
build interfaces that make it faster.

## Steering an LLM pipeline over many documents

A prompt that works on one example often fails on the hundredth document.
[DocWrangler](https://arxiv.org/abs/2504.14764) (UIST 2025), an IDE for
[DocETL](https://docetl.org) pipelines, lets you annotate documents and
outputs in place; an LLM turns your notes into revised prompts, or suggests
splitting an operation that is too hard. In our study and public deployment,
users often turned open-ended operations into classifiers so they were easier
to check.

## Debugging retrieval-augmented generation

When a RAG answer is wrong, the fault can be in retrieval or in generation.
[RAGGY](https://arxiv.org/abs/2504.13587) (CHI 2026) pairs composable RAG
building blocks with an interface for fast, interactive debugging, based on a
study of how 12 engineers debug RAG pipelines.

## Finding a dataset

Keyword search sees only titles and tags, so you often can't tell whether a
dataset fits your task until you inspect its rows.
[DataScout](https://arxiv.org/abs/2507.18971) (UIST 2025) suggests query
rewrites, searches dataset contents, and shows how relevant each dataset is to
your task.

## Reading AI-generated text at scale

A column of AI summaries is hard to skim when every row uses different words
and structure.
[Five Principles for Generating Readable AI Text in Bulk]({{ '/blog/agent-text-at-scale/' | relative_url }})
draws on reading research for rules such as starting each item with its
subject or verb.

## Questions

{% include question-list.html section="interfaces" %}

</article>
