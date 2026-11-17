---
layout: default
title: What Are LLM Evals?
description: "LLM evals are code checks and LLM judges that tell you whether an LLM app's outputs are good enough. Where to start in the AI Evals FAQ, and our research on aligning LLM judges."
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

## Start with the AI Evals FAQ

Hamel Husain and Shreya Shankar answer the questions people ask most about
evals in [AI Evals: Everything You Need to Know](https://hamel.dev/blog/posts/evals-faq/). Good first reads:

- [What are AI evals?](https://hamel.dev/blog/posts/evals-faq/what-are-llm-evals.html)
- [What's a minimum viable evaluation setup?](https://hamel.dev/blog/posts/evals-faq/whats-a-minimum-viable-evaluation-setup.html)
- [Why is error analysis so important, and how is it performed?](https://hamel.dev/blog/posts/evals-faq/why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed.html)
- [Why binary (pass/fail) evaluations instead of 1-5 ratings?](https://hamel.dev/blog/posts/evals-faq/why-do-you-recommend-binary-passfail-evaluations-instead-of-1-5-ratings-likert-scales.html)
- [How do I know if I can trust my automated eval?](https://hamel.dev/blog/posts/evals-faq/how-do-i-know-if-i-can-trust-my-automated-eval.html)
- [How many examples do I need for an eval?](https://hamel.dev/blog/posts/evals-faq/how-many-examples-do-i-need-for-an-eval.html)

## Our research

- [Who Validates the Validators?](https://arxiv.org/abs/2404.12272) (UIST
  2024): EvalGen, a tool that uses your grades on a sample of outputs to pick
  LLM judges that agree with you. Our user study named *criteria drift*:
  grading outputs changes the criteria people want.
- [Data Flywheels for LLM Applications](https://www.sh-reya.com/blog/ai-engineering-flywheel/):
  how evals, monitoring, and prompt changes fit into one loop.
- [What We've Learned From a Year of Building with LLMs](https://applied-llms.org):
  practical advice on LLM judges, such as pairwise comparisons.

## Questions

{% include question-list.html section="evals" %}

</article>
