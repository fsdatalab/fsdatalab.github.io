---
layout: default
title: "Working With Us (If You Have a Non-Traditional Background)"
description: "How people from non-traditional backgrounds (in industry, outside CS, or in high school) can get involved with the Full Stack Data Lab: use Quail, our open-source AI-SQL engine, to analyze a dataset, and send us a video walking through what you found."
permalink: /working-with-us/
---
<article class="content-narrow prose" markdown="1">

<div class="post-header"><h1>Working with us (if you have a non-traditional background)</h1></div>

Interested in working with our group, but coming from a non-traditional
background, e.g., industry, a field outside CS, or high school? You don't
need a CS degree or research experience. (If you're a CMU
student or a prospective PhD student, reach out to us directly instead.)

Here's a taste of the kinds of problems we work on. One of the systems we
are building is [Quail]({{ '/blog/introducing-quail/' | relative_url }}),
a SQL query engine with AI functions: you write prompts inside a SQL query,
and Quail runs them over your data with open-source LLMs.

To get started, try Quail yourself: pick a dataset (see the [ideas
below](#ideas)) and analyze it with Quail. The [open-source repo](https://github.com/fsdatalab/quail)
has documentation, and you can run Quail on [Modal](https://modal.com/pricing),
whose free plan includes $30 of compute per month (see the [Modal
quickstart](https://github.com/fsdatalab/quail/blob/d8d31f14f9d5c40d6cf683a74ba860788ce0d500/demos/quickstart_modal.py)).

Then, email [shreyashankar@cmu.edu](mailto:shreyashankar@cmu.edu)
a short video (under 10 minutes) of you explaining your analysis, along with
your CV. Feel free to present it with a dashboard, slides, or another visual
tool. You don't need to share your code.

## What makes a project good

A good project does at least one of these:

- Applies Quail to an interesting or impactful problem.
- Shows a kind of query people need that Quail can't run yet, and how far
  you get with what Quail can do today.
- Identifies ways to make Quail faster, cheaper, or more accurate.
- Finds a better way to write Quail queries, or to communicate their results
  to people, e.g., an interface, a dashboard, or a story.

## Ideas

These are just some ideas for inspiration. If you have your own data or
questions, explore those! Pick something you're interested in, or a domain
you already know well.

**Federal rulemaking.** [Regulations.gov](https://www.regulations.gov/) has
public comments on proposed rules, sometimes hundreds of thousands per rule.
What do people argue for and against? How much comes from organized
campaigns? Do final rules respond to the most common arguments?

**Consumer finance.** The [CFPB complaint
database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
has millions of complaints about banks, lenders, and debt collectors, many
with the consumer's own description. Which practices hurt people the most?
Which ones show up across many companies? What changed after new rules?

**Vehicle safety.** NHTSA publishes [owner complaints about
vehicles](https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-complaints-complaints-flat-file)
going back decades. Which defects showed up in complaints long before a
recall? Which problems are common now but haven't been recalled?

**Courts.** [CourtListener](https://www.courtlistener.com/help/api/bulk-data/)
has millions of court opinions. How are courts responding to lawyers who
cite cases that AI made up? How do outcomes on the same legal issue differ
across courts?

**Policing.** The [Police Records Access
Project](https://bids.berkeley.edu/california-police-records-access-project)
has about 1.5 million pages of records from 12,000 misconduct and
use-of-force cases in California. What are officers accused of? How does
discipline for the same allegation differ across agencies?

**State legislation.** [Open States](https://open.pluralpolicy.com/data/)
has bills from all 50 state legislatures. What are states doing about AI,
housing, or data privacy? Which bills copy each other, and where does model
legislation come from?

**Clinical trials.** [ClinicalTrials.gov](https://clinicaltrials.gov/) has
close to 600,000 registered studies. Which trials could a given patient
join? Why do trials stop early? Who gets excluded from trials, and for what
reasons?

**Corporate disclosures.** [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus)
has the annual reports of all US public companies from 1993 to 2020. How
have companies described climate, cyber, or supply chain risks over time?
Which risks did companies warn about before a crisis hit?

**History.** [American Stories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)
has articles from nearly 20 million pages of American newspapers from 1774
to 1963. How did coverage of an event, like the 1918 flu, spread across
the country? How did papers in different regions describe the same events?

**AI use.** [WildChat](https://huggingface.co/datasets/allenai/WildChat-4.8M)
has millions of real conversations with chatbots, and Nebius has [80,036
coding agent trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories).
What do people actually use chatbots for, and where do chatbots fail them?
Why do coding agents fail?

</article>
