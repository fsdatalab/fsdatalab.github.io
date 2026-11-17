---
layout: default
title: What Are LLM Evals?
description: "LLM evals are code checks and LLM judges that tell you whether an LLM app's outputs are good enough. How to start, from reading outputs to aligning a judge with your own labels."
permalink: /llm-evals/
guide: true
---
<article class="content-narrow prose" markdown="1">

<div class="post-header"><h1>What are LLM evals?</h1></div>

LLM evals are checks that tell you whether an LLM application's outputs are
good enough for the people who use it. Each eval is either a code check, such
as "the output parses as JSON", or an LLM judge, a prompt that asks a model
whether an output meets a criterion, such as "the reply answers the
customer's question".

## How to start

1. **Read many real outputs, and write down what goes wrong.** Pick metrics
   from those failures, not from a list of generic metrics.
2. **Turn recurring failures into criteria, and expect them to change.** In
   our [EvalGen study](https://arxiv.org/abs/2404.12272), we named this
   *criteria drift*: grading outputs changes the criteria people want.
3. **Check simple criteria with code, and subjective criteria with an LLM
   judge.**
4. **Use pass/fail instead of 1-to-5 scores.** Binary metrics are easier to
   align with your judgment and to label consistently over time.
5. **Check each judge against your own labels before you trust it.** Grade a
   sample of real outputs yourself, compare the judge's answers, and check
   again as the outputs change.

## Further reading

- [AI Evals: Everything You Need to Know](https://hamel.dev/blog/posts/evals-faq/),
  by Hamel Husain and Shreya Shankar: answers to the questions people ask
  most about evals, such as what a minimum viable setup looks like.
- [Data Flywheels for LLM Applications](https://www.sh-reya.com/blog/ai-engineering-flywheel/):
  how evals, monitoring, and prompt changes fit into one loop.
- [Who Validates the Validators?](https://arxiv.org/abs/2404.12272) (UIST
  2024): the EvalGen paper, on aligning LLM judges with human grades.
- [What We've Learned From a Year of Building with LLMs](https://applied-llms.org):
  practical advice on LLM judges, such as pairwise comparisons.

## Questions

{% include question-list.html section="evals" %}

</article>
